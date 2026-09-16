"""Live Minecraft reward for ES training (M5, live phase).

Reward is earned purely by what the bot does in a real episode against a
real running server - no admin shortcuts, no handed-out items, no
teleports. An episode starts wherever the bot naturally already is
(including its own respawn point if it died in a previous episode) and is
scored only from observable, self-earned progress: staying alive, keeping
health, moving under its own power, and gathering resources by its own
mining.

This is deliberately a separate, slower path from
train_interface.py's evaluate_situations() proxy task - a real episode
takes real wall-clock Minecraft ticks (no speeding that up), so this
exists for validating the training mechanism against genuine survival
reward, not for the bulk of iteration during algorithm development.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.goals import MILESTONE_REWARDS, milestone_score, reached_steps  # noqa: E402
from agent.movement import movement_command_from_actions  # noqa: E402
from agent.self_actions import perform  # noqa: E402
from agent.loop import ATTACKABLE_ENTITY_KINDS, LIF_PARAMS, SIM_TICKS_PER_ACTION  # noqa: E402
from agent.task_manager import FreeWillTaskManager, TaskManager  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import SensoryEncoder  # noqa: E402
from sim.lif import LIFNetwork  # noqa: E402
from training.train_interface import unflatten_params  # noqa: E402

EPISODE_STEPS = 150
STEP_SLEEP_S = 0.1
MELEE_RANGE = 4.0

# --- Displeasure: not every point of health lost is equally the bot's
# fault, and one undifferentiated "ouch" is equally true for a stubbed toe
# and walking straight into lava. Each cause below gets its own weight
# (per HP lost) so the reward actually teaches something actionable
# (surface for air, back away from fire, don't starve) instead of just
# "something hurt, once, for unknown reasons." Combat is weighted lowest on
# purpose: taking a hit while fighting something is normal and expected,
# and punishing it as harshly as carelessness would just retrain the
# "stand perfectly still" collapse this reward already had to be rescued
# from once (see the note further down).
DAMAGE_WEIGHTS = {
    "lava": 0.30,
    "void": 0.50,
    "fire": 0.20,
    "drowning": 0.20,
    "starvation": 0.15,
    "fall": 0.15,
    "mob": 0.05,
    "unknown": 0.10,
}
VOID_Y_THRESHOLD = -20.0

# Recovering from a mistake is its own skill, worth reinforcing separately
# from "never get hurt at all" - but the weight must stay strictly below
# every DAMAGE_WEIGHTS entry above (including the cheapest, "mob"), or
# hurting itself on purpose to farm healing would be a profitable exploit.
HEAL_REWARD_PER_HP = 0.03

# "Doing something bad" that isn't health loss at all: every other bot in
# this shared world shows up as a Player entity, so an unprovoked attack on
# one is real friendly fire against the whole training run, not just a
# wasted swing - weighted worse than attacking a passive animal for no
# resource gain.
BAD_ATTACK_PENALTY = {"player": 1.0, "passive": 0.3}

HOSTILE_HIT_REWARD = 0.15
HOSTILE_KILL_REWARD = 1.0
HAZARD_FREE_BONUS = 0.1  # * fraction of steps spent outside any hazard

# Items are only actually lost one way in vanilla survival: dying drops the
# whole inventory. Layered on top of the flat death penalty below so dying
# empty-handed and dying with a hard-won stockpile aren't scored the same -
# protect what you've earned, don't take needless risks once you're
# carrying a lot.
RESOURCE_LOSS_PENALTY_PER_ITEM = 0.05

# An ongoing "reckless play" tax distinct from the flat death penalty -
# rewards learning to retreat/heal at near-zero health instead of
# continuing to tank hits, even in episodes that happen to survive anyway.
CRITICAL_HEALTH_THRESHOLD = 4.0
CRITICAL_HEALTH_PENALTY_PER_STEP = 0.015

# Per *distinct* task verb that succeeds at least once this episode (not
# per success - spamming one verb for repeated reward would defeat the
# point). Intermediate verbs like craft_planks/craft_sticks aren't on the
# milestone ladder at all (agent/goals.py only scores named end-items), so
# without this they gave zero reward signal of their own, leaving the
# whole chain to be learned purely from the one sparse eventual payoff at
# crafting_table.
TASK_SUCCESS_REWARD = 0.1

# A one-time kicker the first time *any* worker reaches a given rung of the
# ladder during this training run, on top of the normal milestone reward -
# credits genuine breakthrough distinctly from a later episode routinely
# repeating it. Workers are threads in one process (train_live.py's
# ThreadPoolExecutor), so a lock-guarded module-level set is enough to
# share this safely - it does not need to survive past this run.
FIRST_EVER_MILESTONE_BONUS = 1.0
_ever_reached_lock = threading.Lock()
_ever_reached_milestones: set[str] = set()


def _claim_first_ever(ids: set[str]) -> set[str]:
    """Returns the subset of `ids` this training run has never reached
    before, and marks all of them as reached from now on - so a second
    episode (even a concurrent one) that touches the same milestone never
    claims the bonus twice."""
    with _ever_reached_lock:
        new_ids = ids - _ever_reached_milestones
        _ever_reached_milestones.update(ids)
    return new_ids


def classify_damage_cause(prev_obs: dict, obs: dict) -> str | None:
    """Attributes a health drop between two consecutive observations to a
    cause. Order matters: a specific, certain cause (standing in lava) is
    checked before a coincidental one (airborne last tick, so "fall").
    Only a rough approximation for the task-manager path, where two
    samples can be many real seconds apart (a `goto` or `mineBlock` action
    can run up to ~130s) and several different things could have happened
    in between - there is no event stream, only polling."""
    if obs["health"] >= prev_obs["health"]:
        return None
    if prev_obs.get("inLava") or obs.get("inLava"):
        return "lava"
    if prev_obs.get("blockAtFeet") == "fire" or obs.get("blockAtFeet") == "fire":
        return "fire"
    if prev_obs.get("inWater") and obs.get("oxygen") is not None and obs["oxygen"] <= 0:
        return "drowning"
    if prev_obs.get("food", 20) <= 0:
        return "starvation"
    if obs.get("position", {}).get("y", 0.0) < VOID_Y_THRESHOLD:
        return "void"
    if prev_obs.get("onGround") is False:
        return "fall"
    if any(
        e.get("kind") == "Hostile mobs" and e.get("distance", 999.0) <= MELEE_RANGE
        for e in prev_obs.get("nearbyEntities", [])
    ):
        return "mob"
    return "unknown"


def classify_attack_target(entity: dict) -> str | None:
    """None means a legitimate target (an actual threat). Anything else
    names which displeasure bucket an unprovoked attack falls into."""
    kind = entity.get("kind")
    if kind == "Player":
        return "player"
    if kind in ("Passive mobs", "Ambient mobs", "Water creature"):
        return "passive"
    return None


def _hostile_health_by_id(observation: dict) -> dict[int, float]:
    return {
        e["id"]: e["health"]
        for e in observation.get("nearbyEntities", [])
        if e.get("kind") == "Hostile mobs" and e.get("health") is not None
    }


def _combat_outcome(target_id: int, prev_health: dict[int, float], curr_health: dict[int, float]) -> str | None:
    """Whether an attack issued against `target_id` last step landed, given
    the hostile-entity health snapshots from just before and just after.
    A target that was in melee range and has now vanished from the nearby-
    entity list entirely is treated as a kill - safe to assume, since
    anything actually still alive within melee range one step ago can't
    have wandered outside the (much larger) entity-sensing radius by now."""
    if target_id not in prev_health:
        return None
    if target_id not in curr_health:
        return "kill"
    if curr_health[target_id] < prev_health[target_id]:
        return "hit"
    return None


def _is_hazardous(observation: dict) -> bool:
    if observation.get("inLava") or observation.get("blockAtFeet") == "fire":
        return True
    oxygen = observation.get("oxygen")
    return bool(observation.get("inWater")) and oxygen is not None and oxygen <= 2


def _total_resources(observation: dict) -> int:
    return sum(i["count"] for i in observation["inventory"])


def _mine_ahead(observation: dict, client) -> None:
    """Digs whatever the bot is facing. Without this the self-play path had
    no way to acquire *any* material at all - the decoder could fire
    mine_ahead all it liked and nothing happened, which quietly made the
    whole progression ladder unreachable in self mode."""
    ahead = [
        b for b in observation.get("nearbyBlocks", [])
        if b["y"] in (0, 1) and abs(b["x"]) <= 1 and abs(b["z"]) <= 1 and (b["x"] != 0 or b["z"] != 0)
    ]
    if not ahead:
        return
    target = ahead[0]
    pos = observation["position"]
    try:
        client.do_action(
            {
                "type": "dig",
                "x": pos["x"] + target["x"],
                "y": pos["y"] + target["y"],
                "z": pos["z"] + target["z"],
            }
        )
    except BridgeError:
        pass  # out of reach, unbreakable, or gone - all normal


def run_episode(
    theta: np.ndarray,
    encoder: SensoryEncoder,
    decoder: MotorDecoder,
    weights,
    bridge_url: str,
    use_task_manager: bool = False,
    free_will: bool = False,
    episode_steps: int = EPISODE_STEPS,
) -> float:
    """Runs one real episode against a live bot-bridge instance and returns
    an earned reward. Mutates `encoder`/`decoder` in place (via
    unflatten_params) to whatever `theta` this evaluation is for.

    `use_task_manager`/`free_will` pick the task manager the same way as
    agent/loop.py's `run()`: no task manager, the full scripted TaskManager,
    or the safety-only FreeWillTaskManager (see agent/task_manager.py).

    Never raises: a long unattended run must survive one worker's bot-
    bridge process crashing or refusing a connection without taking down
    the whole training run - such a rollout just scores 0.0 rather than
    propagating the exception.
    """
    try:
        return _run_episode_inner(theta, encoder, decoder, weights, bridge_url, use_task_manager, free_will, episode_steps)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"rollout against {bridge_url} failed ({exc!r}); scoring 0.0")
        return 0.0


def _run_episode_inner(
    theta: np.ndarray,
    encoder: SensoryEncoder,
    decoder: MotorDecoder,
    weights,
    bridge_url: str,
    use_task_manager: bool,
    free_will: bool,
    episode_steps: int,
) -> float:
    unflatten_params(theta, encoder, decoder)
    net = LIFNetwork(weights, LIF_PARAMS)
    if not use_task_manager:
        task_manager = None
    elif free_will:
        task_manager = FreeWillTaskManager()
    else:
        task_manager = TaskManager()

    with BridgeClient(bridge_url) as client:
        try:
            start_obs = client.get_observation()
        except BridgeError:
            return 0.0  # bridge not ready this instant - treat as a wasted rollout, not a crash

        start_pos = start_obs["position"]
        start_resources = _total_resources(start_obs)
        start_milestone = milestone_score(start_obs)
        # Tracked as a running max rather than read at the end: dying drops
        # the whole inventory, and an episode that crafted a stone pickaxe
        # and *then* got killed still demonstrated the capability we're
        # selecting for. Scoring only the final inventory would throw that
        # signal away entirely.
        best_milestone = start_milestone
        max_distance = 0.0
        died = False
        final_health = start_obs["health"]
        task_attempts: dict[str, list[int]] = {}  # verb -> [attempts, successes]

        # Displeasure/reward bookkeeping - see the DAMAGE_WEIGHTS etc. block
        # at the top of this module for what each bucket means and why it's
        # weighted the way it is.
        prev_obs: dict | None = None
        last_attack_target_id: int | None = None
        damage_by_cause: dict[str, float] = {}
        heal_total = 0.0
        bad_attacks: dict[str, int] = {}
        hostile_hits = 0
        hostile_kills = 0
        hazard_free_steps = 0
        observed_steps = 0
        critical_health_steps = 0
        pre_death_resources = 0
        distinct_task_successes: set[str] = set()
        episode_reached_ids: set[str] = {step.id for step in reached_steps(start_obs)}

        for _ in range(episode_steps):
            try:
                observation = client.get_observation()
            except BridgeError:
                break  # kicked/disconnected mid-episode; score what was earned so far

            final_health = observation["health"]
            best_milestone = max(best_milestone, milestone_score(observation))
            episode_reached_ids.update(step.id for step in reached_steps(observation))

            if prev_obs is not None:
                cause = classify_damage_cause(prev_obs, observation)
                if cause is not None:
                    damage_by_cause[cause] = damage_by_cause.get(cause, 0.0) + (prev_obs["health"] - observation["health"])
                elif observation["health"] > prev_obs["health"]:
                    heal_total += observation["health"] - prev_obs["health"]

                if last_attack_target_id is not None:
                    outcome = _combat_outcome(
                        last_attack_target_id, _hostile_health_by_id(prev_obs), _hostile_health_by_id(observation)
                    )
                    if outcome == "hit":
                        hostile_hits += 1
                    elif outcome == "kill":
                        hostile_kills += 1
                    last_attack_target_id = None

            if final_health <= 0:
                died = True
                # prev_obs is the last observation before death - the
                # inventory that's about to be wiped, since this one may
                # already reflect the post-death/respawn state depending on
                # exactly when the server sent each update.
                pre_death_resources = _total_resources(prev_obs) if prev_obs is not None else 0
                break

            observed_steps += 1
            if not _is_hazardous(observation):
                hazard_free_steps += 1
            if final_health <= CRITICAL_HEALTH_THRESHOLD:
                critical_health_steps += 1

            dx = observation["position"]["x"] - start_pos["x"]
            dz = observation["position"]["z"] - start_pos["z"]
            max_distance = max(max_distance, (dx**2 + dz**2) ** 0.5)

            handled = task_manager.step(observation, client) if task_manager is not None else False
            if not handled:
                ext_current = encoder.encode(observation, net.n)
                spike_window = np.zeros((SIM_TICKS_PER_ACTION, net.n))
                for tick in range(SIM_TICKS_PER_ACTION):
                    spike_window[tick] = net.step(ext_current)
                actions = decoder.decode(spike_window)
                move_command = movement_command_from_actions(actions, observation)
                try:
                    client.do_action(move_command)
                except BridgeError:
                    pass

                # Whichever task verb the network asked for most strongly,
                # if any. Nothing here decides what it "should" be doing -
                # see agent/self_actions.py. Counted regardless of whether
                # the attempt actually succeeds: a failed craft (missing
                # materials) is completely normal early on, and only
                # counting successes made every attempt invisible in the
                # log - it looked like the network never even tried,
                # when live data (see training/README.md) shows it reliably
                # does; it just usually can't afford the recipe yet.
                task_action = decoder.decode_task_action(spike_window)
                if task_action is not None:
                    succeeded = perform(task_action, observation, client)
                    counts = task_attempts.setdefault(task_action, [0, 0])
                    counts[0] += 1
                    counts[1] += 1 if succeeded else 0
                    if succeeded:
                        distinct_task_successes.add(task_action)

                if actions.get("mine_ahead"):
                    _mine_ahead(observation, client)

                # Self-play previously had no way to fight back at all -
                # "attack" decoded to a real action in the live control loop
                # (agent/loop.py) but was silently dropped here, so no
                # combat reward could ever fire in training. Mirrors
                # loop.py's own restriction to ATTACKABLE_ENTITY_KINDS and
                # melee range (attacking a dropped item or a far-off entity
                # gets the bot kicked for "attacking an invalid entity").
                if actions.get("attack"):
                    attackable = sorted(
                        (e for e in observation.get("nearbyEntities", []) if e.get("kind") in ATTACKABLE_ENTITY_KINDS),
                        key=lambda e: e["distance"],
                    )
                    if attackable and attackable[0]["distance"] <= MELEE_RANGE:
                        target = attackable[0]
                        try:
                            client.do_action({"type": "attack", "entityId": target["id"]})
                            bad = classify_attack_target(target)
                            if bad is not None:
                                bad_attacks[bad] = bad_attacks.get(bad, 0) + 1
                            elif target.get("kind") == "Hostile mobs":
                                last_attack_target_id = target["id"]
                        except BridgeError:
                            pass  # target may have died/left range between observation and action

            prev_obs = observation
            time.sleep(STEP_SLEEP_S)

        try:
            final_obs = client.get_observation()
            if prev_obs is not None and not died:
                cause = classify_damage_cause(prev_obs, final_obs)
                if cause is not None:
                    damage_by_cause[cause] = damage_by_cause.get(cause, 0.0) + (prev_obs["health"] - final_obs["health"])
                elif final_obs["health"] > prev_obs["health"]:
                    heal_total += final_obs["health"] - prev_obs["health"]
            final_health = final_obs["health"]
            final_resources = _total_resources(final_obs)
            best_milestone = max(best_milestone, milestone_score(final_obs))
            episode_reached_ids.update(step.id for step in reached_steps(final_obs))
        except BridgeError:
            final_resources = start_resources

        try:
            client.do_action({"type": "stop"})
        except BridgeError:
            pass

    # Reward shape, rewritten after live data showed the old one collapsing
    # the population into "stand perfectly still": a flat survival bonus
    # plus full health scored a guaranteed ~2.0 for doing nothing, while
    # any attempt to explore risked losing all of it to a zombie. ES quite
    # rationally learned to freeze. Now progression dominates, staying
    # alive is worth much less than making progress, and dying is a real
    # but survivable cost - well below the value of a single tier of tools.
    health_fraction = final_health / 20.0
    resources_gained = max(0, final_resources - start_resources)
    milestone_gain = best_milestone - start_milestone

    damage_penalty = sum(DAMAGE_WEIGHTS[cause] * hp for cause, hp in damage_by_cause.items())
    heal_bonus = HEAL_REWARD_PER_HP * heal_total
    bad_attack_penalty = sum(BAD_ATTACK_PENALTY[kind] * count for kind, count in bad_attacks.items())
    combat_bonus = HOSTILE_HIT_REWARD * hostile_hits + HOSTILE_KILL_REWARD * hostile_kills
    hazard_free_fraction = (hazard_free_steps / observed_steps) if observed_steps else 1.0
    hazard_bonus = HAZARD_FREE_BONUS * hazard_free_fraction
    resource_loss_penalty = RESOURCE_LOSS_PENALTY_PER_ITEM * pre_death_resources
    critical_health_penalty = CRITICAL_HEALTH_PENALTY_PER_STEP * critical_health_steps
    task_success_bonus = TASK_SUCCESS_REWARD * len(distinct_task_successes)

    first_ever_ids = _claim_first_ever(episode_reached_ids)
    first_ever_bonus = FIRST_EVER_MILESTONE_BONUS * len(first_ever_ids)

    reward = (
        milestone_gain
        + 0.5 * health_fraction
        + 0.05 * max_distance
        + 0.25 * resources_gained
        + heal_bonus
        + combat_bonus
        + hazard_bonus
        + task_success_bonus
        + first_ever_bonus
        - damage_penalty
        - bad_attack_penalty
        - resource_loss_penalty
        - critical_health_penalty
        - (3.0 if died else 0.0)
    )

    attempted = ",".join(f"{k}x{tries}({ok}ok)" for k, (tries, ok) in sorted(task_attempts.items())) or "none"
    damage_str = ",".join(f"{c}:-{hp:.1f}hp" for c, hp in sorted(damage_by_cause.items())) or "none"
    bad_str = ",".join(f"{k}x{n}" for k, n in sorted(bad_attacks.items())) or "none"
    first_ever_str = ",".join(sorted(first_ever_ids)) or "none"
    print(
        f"  episode [{bridge_url}]: reward={reward:.3f} "
        f"(died={died}, health={final_health}/20, distance={max_distance:.1f}, "
        f"resources_gained={resources_gained}, milestones=+{milestone_gain:.0f}, "
        f"damage={damage_str}, healed={heal_total:.1f}hp, "
        f"combat(hits={hostile_hits},kills={hostile_kills}), bad_attacks={bad_str}, "
        f"hazard_free={hazard_free_fraction * 100:.0f}%, critical_hp_steps={critical_health_steps}, "
        f"resources_lost_on_death={pre_death_resources if died else 0}, "
        f"task_successes={len(distinct_task_successes)}, first_ever={first_ever_str}, "
        f"tasks_tried={attempted}, "
        f"status={task_manager.status if task_manager else 'self'})"
    )
    return reward
