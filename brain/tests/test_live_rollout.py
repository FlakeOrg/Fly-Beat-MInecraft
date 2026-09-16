"""Tests for the displeasure/reward attribution helpers in
training/live_rollout.py - the pure, bridge-free pieces of the reward shape
that decide *why* a health change happened and whether an attack was
aimed at something legitimate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.live_rollout import (  # noqa: E402
    BAD_ATTACK_PENALTY,
    CRITICAL_HEALTH_PENALTY_PER_STEP,
    DAMAGE_WEIGHTS,
    FIRST_EVER_MILESTONE_BONUS,
    HEAL_REWARD_PER_HP,
    RESOURCE_LOSS_PENALTY_PER_ITEM,
    TASK_SUCCESS_REWARD,
    _claim_first_ever,
    _combat_outcome,
    _hostile_health_by_id,
    _is_hazardous,
    classify_attack_target,
    classify_damage_cause,
)


def make_obs(**overrides) -> dict:
    obs = {
        "health": 20,
        "food": 20,
        "position": {"x": 0.0, "y": 64.0, "z": 0.0},
        "onGround": True,
        "inWater": False,
        "inLava": False,
        "oxygen": 20,
        "blockAtFeet": "grass_block",
        "nearbyEntities": [],
    }
    obs.update(overrides)
    return obs


def test_no_damage_returns_none():
    prev = make_obs(health=20)
    obs = make_obs(health=20)
    assert classify_damage_cause(prev, obs) is None


def test_healing_is_not_classified_as_damage():
    prev = make_obs(health=10)
    obs = make_obs(health=15)
    assert classify_damage_cause(prev, obs) is None


def test_lava_damage():
    prev = make_obs(health=20, inLava=True)
    obs = make_obs(health=14, inLava=True)
    assert classify_damage_cause(prev, obs) == "lava"


def test_fire_damage():
    prev = make_obs(health=20, blockAtFeet="fire")
    obs = make_obs(health=17, blockAtFeet="fire")
    assert classify_damage_cause(prev, obs) == "fire"


def test_drowning_damage():
    prev = make_obs(health=20, inWater=True, oxygen=1)
    obs = make_obs(health=18, inWater=True, oxygen=0)
    assert classify_damage_cause(prev, obs) == "drowning"


def test_no_false_drowning_when_oxygen_is_merely_unreported():
    """A None oxygen reading must never masquerade as 'suffocating' -
    only an actual oxygen<=0 sample while in water counts."""
    prev = make_obs(health=20, inWater=True, oxygen=None)
    obs = make_obs(health=18, inWater=True, oxygen=None)
    assert classify_damage_cause(prev, obs) != "drowning"


def test_starvation_damage():
    prev = make_obs(health=20, food=0)
    obs = make_obs(health=19, food=0)
    assert classify_damage_cause(prev, obs) == "starvation"


def test_void_damage():
    prev = make_obs(health=20, position={"x": 0.0, "y": -10.0, "z": 0.0})
    obs = make_obs(health=16, position={"x": 0.0, "y": -25.0, "z": 0.0})
    assert classify_damage_cause(prev, obs) == "void"


def test_fall_damage():
    prev = make_obs(health=20, onGround=False)
    obs = make_obs(health=17, onGround=True)
    assert classify_damage_cause(prev, obs) == "fall"


def test_mob_damage():
    prev = make_obs(
        health=20,
        nearbyEntities=[{"id": 1, "kind": "Hostile mobs", "distance": 2.0}],
    )
    obs = make_obs(health=16)
    assert classify_damage_cause(prev, obs) == "mob"


def test_unknown_damage_when_nothing_explains_it():
    prev = make_obs(health=20)
    obs = make_obs(health=18)
    assert classify_damage_cause(prev, obs) == "unknown"


def test_lava_takes_priority_over_a_coincidental_hostile_nearby():
    prev = make_obs(
        health=20,
        inLava=True,
        nearbyEntities=[{"id": 1, "kind": "Hostile mobs", "distance": 2.0}],
    )
    obs = make_obs(health=10, inLava=True)
    assert classify_damage_cause(prev, obs) == "lava"


def test_every_damage_cause_has_a_weight():
    for cause in ("lava", "void", "fire", "drowning", "starvation", "fall", "mob", "unknown"):
        assert cause in DAMAGE_WEIGHTS


def test_heal_reward_is_strictly_below_every_damage_weight():
    """Otherwise hurting itself on purpose to farm the healing bonus would
    be a profitable exploit - this must never regress."""
    assert all(HEAL_REWARD_PER_HP < weight for weight in DAMAGE_WEIGHTS.values())


def test_classify_attack_target_hostile_is_legitimate():
    assert classify_attack_target({"kind": "Hostile mobs"}) is None


def test_classify_attack_target_player_is_bad():
    assert classify_attack_target({"kind": "Player"}) == "player"


def test_classify_attack_target_passive_is_bad():
    assert classify_attack_target({"kind": "Passive mobs"}) == "passive"
    assert classify_attack_target({"kind": "Ambient mobs"}) == "passive"
    assert classify_attack_target({"kind": "Water creature"}) == "passive"


def test_bad_attack_penalty_covers_every_bad_target_kind():
    for entity in ({"kind": "Player"}, {"kind": "Passive mobs"}):
        kind = classify_attack_target(entity)
        assert kind in BAD_ATTACK_PENALTY


def test_hostile_health_by_id_only_tracks_hostiles_with_known_health():
    obs = make_obs(
        nearbyEntities=[
            {"id": 1, "kind": "Hostile mobs", "health": 10, "distance": 2.0},
            {"id": 2, "kind": "Passive mobs", "health": 10, "distance": 2.0},
            {"id": 3, "kind": "Hostile mobs", "health": None, "distance": 2.0},
        ]
    )
    assert _hostile_health_by_id(obs) == {1: 10}


def test_combat_outcome_hit_when_health_drops():
    prev_health = {1: 10}
    curr_health = {1: 6}
    assert _combat_outcome(1, prev_health, curr_health) == "hit"


def test_combat_outcome_kill_when_target_vanishes():
    prev_health = {1: 4}
    curr_health = {}
    assert _combat_outcome(1, prev_health, curr_health) == "kill"


def test_combat_outcome_none_when_untouched():
    prev_health = {1: 10}
    curr_health = {1: 10}
    assert _combat_outcome(1, prev_health, curr_health) is None


def test_combat_outcome_none_for_a_target_not_previously_tracked():
    assert _combat_outcome(99, {}, {}) is None


def test_is_hazardous_lava_and_fire():
    assert _is_hazardous(make_obs(inLava=True)) is True
    assert _is_hazardous(make_obs(blockAtFeet="fire")) is True


def test_is_hazardous_low_oxygen_underwater():
    assert _is_hazardous(make_obs(inWater=True, oxygen=1)) is True


def test_not_hazardous_when_safe():
    assert _is_hazardous(make_obs()) is False
    assert _is_hazardous(make_obs(inWater=True, oxygen=20)) is False


def test_resource_loss_penalty_is_below_the_resource_gain_weight():
    """Otherwise gathering would stop being worthwhile in expectation the
    moment any real risk of death exists - the 0.25 gain weight lives in
    the reward formula in live_rollout.py, not exported as a constant, so
    this is pinned directly against the number itself."""
    assert RESOURCE_LOSS_PENALTY_PER_ITEM < 0.25


def test_critical_health_penalty_is_small_per_step():
    """A whole 150-step episode spent critical should still cost meaningfully
    less than one death (-3.0) - it's a nudge toward retreating, not a
    second death penalty."""
    assert CRITICAL_HEALTH_PENALTY_PER_STEP * 150 < 3.0


def test_task_success_reward_is_below_the_smallest_milestone_reward():
    """Dense sub-goal shaping must never outweigh the sparse milestone it's
    building toward, even summed across every distinct verb there is."""
    from agent.goals import TRAINING_STEPS
    from interface.motor_decoder import TASK_ACTIONS

    smallest_milestone = min(step.reward for step in TRAINING_STEPS)
    assert TASK_SUCCESS_REWARD * len(TASK_ACTIONS) < smallest_milestone


def test_claim_first_ever_only_pays_out_once():
    unique = {"test_marker_a_9f3", "test_marker_b_9f3"}
    first_claim = _claim_first_ever(unique)
    second_claim = _claim_first_ever(unique)
    assert first_claim == unique
    assert second_claim == set()


def test_claim_first_ever_pays_only_for_the_new_ones():
    base = {"test_marker_c_9f3"}
    _claim_first_ever(base)
    mixed = base | {"test_marker_d_9f3"}
    assert _claim_first_ever(mixed) == {"test_marker_d_9f3"}


def test_first_ever_bonus_is_a_meaningful_one_time_amount():
    assert FIRST_EVER_MILESTONE_BONUS > 0
