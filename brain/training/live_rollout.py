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
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.goals import MILESTONE_REWARDS, milestone_score  # noqa: E402
from agent.movement import movement_command_from_actions  # noqa: E402
from agent.self_actions import perform  # noqa: E402
from agent.loop import LIF_PARAMS, SIM_TICKS_PER_ACTION  # noqa: E402
from agent.task_manager import TaskManager  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import SensoryEncoder  # noqa: E402
from sim.lif import LIFNetwork  # noqa: E402
from training.train_interface import unflatten_params  # noqa: E402

EPISODE_STEPS = 150
STEP_SLEEP_S = 0.1


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
    episode_steps: int = EPISODE_STEPS,
) -> float:
    """Runs one real episode against a live bot-bridge instance and returns
    an earned reward. Mutates `encoder`/`decoder` in place (via
    unflatten_params) to whatever `theta` this evaluation is for.

    Never raises: a long unattended run must survive one worker's bot-
    bridge process crashing or refusing a connection without taking down
    the whole training run - such a rollout just scores 0.0 rather than
    propagating the exception.
    """
    try:
        return _run_episode_inner(theta, encoder, decoder, weights, bridge_url, use_task_manager, episode_steps)
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
    episode_steps: int,
) -> float:
    unflatten_params(theta, encoder, decoder)
    net = LIFNetwork(weights, LIF_PARAMS)
    task_manager = TaskManager() if use_task_manager else None

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

        for _ in range(episode_steps):
            try:
                observation = client.get_observation()
            except BridgeError:
                break  # kicked/disconnected mid-episode; score what was earned so far

            final_health = observation["health"]
            best_milestone = max(best_milestone, milestone_score(observation))
            if final_health <= 0:
                died = True
                break

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

                if actions.get("mine_ahead"):
                    _mine_ahead(observation, client)

            time.sleep(STEP_SLEEP_S)

        try:
            final_obs = client.get_observation()
            final_health = final_obs["health"]
            final_resources = _total_resources(final_obs)
            best_milestone = max(best_milestone, milestone_score(final_obs))
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
    reward = (
        milestone_gain
        + 0.5 * health_fraction
        + 0.05 * max_distance
        + 0.25 * resources_gained
        - (3.0 if died else 0.0)
    )

    attempted = ",".join(f"{k}x{tries}({ok}ok)" for k, (tries, ok) in sorted(task_attempts.items())) or "none"
    print(
        f"  episode [{bridge_url}]: reward={reward:.3f} "
        f"(died={died}, health={final_health}/20, distance={max_distance:.1f}, "
        f"resources_gained={resources_gained}, milestones=+{milestone_gain:.0f}, "
        f"tasks_tried={attempted}, "
        f"status={task_manager.status if task_manager else 'self'})"
    )
    return reward
