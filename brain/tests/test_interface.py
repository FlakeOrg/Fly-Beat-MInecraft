"""Tests for brain/interface/{sensory_encoder,motor_decoder}.py using
fabricated observations/spike traces - no live bot-bridge or Minecraft
server needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "interface"))

from motor_decoder import ACTIONS, MotorDecoder  # noqa: E402
from sensory_encoder import FEATURE_NAMES, SensoryEncoder, extract_features  # noqa: E402


def make_observation(**overrides) -> dict:
    obs = {
        "health": 20,
        "food": 20,
        "yaw": 0.0,
        "onGround": True,
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "nearbyEntities": [],
        "nearbyBlocks": [],
    }
    obs.update(overrides)
    return obs


def test_extract_features_shape():
    features = extract_features(make_observation())
    assert features.shape == (len(FEATURE_NAMES),)


def test_extract_features_health_and_food_fractions():
    obs = make_observation(health=10, food=5)
    features = extract_features(obs)
    assert features[FEATURE_NAMES.index("health_frac")] == pytest.approx(0.5)
    assert features[FEATURE_NAMES.index("food_frac")] == pytest.approx(0.25)


def test_extract_features_hostile_vs_passive_entity():
    near_zombie = make_observation(nearbyEntities=[{"name": "zombie", "distance": 3.0}])
    near_cow = make_observation(nearbyEntities=[{"name": "cow", "distance": 3.0}])

    hostile_idx = FEATURE_NAMES.index("hostile_near")
    entity_idx = FEATURE_NAMES.index("entity_near")

    zombie_features = extract_features(near_zombie)
    cow_features = extract_features(near_cow)

    assert zombie_features[hostile_idx] == 1.0
    assert zombie_features[entity_idx] == 1.0
    assert cow_features[hostile_idx] == 0.0
    assert cow_features[entity_idx] == 1.0


def test_extract_features_far_entity_does_not_count():
    obs = make_observation(nearbyEntities=[{"name": "zombie", "distance": 100.0}])
    features = extract_features(obs)
    assert features[FEATURE_NAMES.index("hostile_near")] == 0.0
    assert features[FEATURE_NAMES.index("entity_near")] == 0.0


def test_extract_features_block_ahead_when_facing_south():
    # yaw=0 faces +Z (south) in Mineflayer's convention.
    obs = make_observation(yaw=0.0, nearbyBlocks=[{"x": 0, "y": 0, "z": 1, "name": "stone"}])
    features = extract_features(obs)
    assert features[FEATURE_NAMES.index("block_ahead")] == 1.0


def test_extract_features_falling():
    obs = make_observation(onGround=False, velocity={"x": 0.0, "y": -0.5, "z": 0.0})
    features = extract_features(obs)
    assert features[FEATURE_NAMES.index("is_falling")] == 1.0
    assert features[FEATURE_NAMES.index("on_ground")] == 0.0


def test_sensory_encoder_only_touches_input_neurons():
    input_idx = np.array([2, 5, 8])
    encoder = SensoryEncoder(input_idx, seed=0)
    ext_current = encoder.encode(make_observation(), n_neurons_total=20)

    assert ext_current.shape == (20,)
    non_input_mask = np.ones(20, dtype=bool)
    non_input_mask[input_idx] = False
    assert np.all(ext_current[non_input_mask] == 0.0)


def test_sensory_encoder_gain_scales_output():
    input_idx = np.array([0, 1, 2])
    obs = make_observation(health=10)
    low_gain = SensoryEncoder(input_idx, gain=1.0, seed=0).encode(obs, n_neurons_total=10)
    high_gain = SensoryEncoder(input_idx, gain=5.0, seed=0).encode(obs, n_neurons_total=10)
    np.testing.assert_allclose(high_gain, low_gain * 5.0)


def test_motor_decoder_output_keys():
    output_idx = np.arange(30)
    decoder = MotorDecoder(output_idx, seed=0)
    spike_window = np.zeros((10, 30))
    actions = decoder.decode(spike_window)
    assert set(actions.keys()) == set(ACTIONS)


def test_motor_decoder_silent_network_produces_no_actions():
    output_idx = np.arange(30)
    decoder = MotorDecoder(output_idx, seed=0)
    spike_window = np.zeros((10, 30))
    actions = decoder.decode(spike_window)
    assert not any(actions.values())


def test_motor_decoder_fully_active_pool_triggers_all_actions():
    output_idx = np.arange(30)
    decoder = MotorDecoder(output_idx, seed=0)
    spike_window = np.ones((10, 30))
    actions = decoder.decode(spike_window)
    assert all(actions.values())
