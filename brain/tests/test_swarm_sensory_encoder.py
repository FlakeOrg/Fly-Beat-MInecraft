"""Tests for interface/swarm_sensory_encoder.py - the FlyBrainMain-only
encoder variant that adds a live swarm-summary channel on top of the same
base senses every other bot has.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interface.sensory_encoder import FEATURE_NAMES, extract_features  # noqa: E402
from interface.swarm_sensory_encoder import (  # noqa: E402
    SWARM_FEATURE_NAMES,
    SwarmSensoryEncoder,
    extract_swarm_features,
)


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


def test_extract_swarm_features_defaults_to_zero_when_missing():
    assert np.array_equal(extract_swarm_features(None), np.zeros(len(SWARM_FEATURE_NAMES)))
    assert np.array_equal(extract_swarm_features({}), np.zeros(len(SWARM_FEATURE_NAMES)))


def test_extract_swarm_features_reads_the_expected_keys():
    summary = {"progress_avg": 0.4, "progress_best": 0.9, "danger_frac": 0.25}
    features = extract_swarm_features(summary)
    assert features[SWARM_FEATURE_NAMES.index("swarm_progress_avg")] == pytest.approx(0.4)
    assert features[SWARM_FEATURE_NAMES.index("swarm_progress_best")] == pytest.approx(0.9)
    assert features[SWARM_FEATURE_NAMES.index("swarm_danger_frac")] == pytest.approx(0.25)


def test_encoder_feature_count_includes_swarm_features():
    encoder = SwarmSensoryEncoder(np.array([0, 1, 2]), seed=0)
    assert encoder.n_features == len(FEATURE_NAMES) + len(SWARM_FEATURE_NAMES)
    assert encoder.weights.shape == (3, len(FEATURE_NAMES) + len(SWARM_FEATURE_NAMES))


def test_encode_without_swarm_summary_matches_zeroed_swarm_features():
    input_idx = np.array([0, 1])
    encoder = SwarmSensoryEncoder(input_idx, seed=0)
    obs = make_observation()

    no_summary = encoder.encode(obs, n_neurons_total=5, swarm_summary=None)
    zero_summary = encoder.encode(obs, n_neurons_total=5, swarm_summary={"progress_avg": 0.0, "progress_best": 0.0, "danger_frac": 0.0})
    assert np.allclose(no_summary, zero_summary)


def test_swarm_features_actually_influence_the_injected_current():
    """Isolates the swarm columns with a hand-built weight matrix so a
    change in swarm_summary is guaranteed to move the output - otherwise a
    random default init could coincidentally give them ~zero weight and
    this would pass for the wrong reason."""
    input_idx = np.array([0])
    n_features = len(FEATURE_NAMES) + len(SWARM_FEATURE_NAMES)
    weights = np.zeros((1, n_features))
    weights[0, len(FEATURE_NAMES) + SWARM_FEATURE_NAMES.index("swarm_progress_best")] = 1.0
    encoder = SwarmSensoryEncoder(input_idx, weights=weights, gain=1.0)

    obs = make_observation()
    low = encoder.encode(obs, n_neurons_total=1, swarm_summary={"progress_best": 0.1})
    high = encoder.encode(obs, n_neurons_total=1, swarm_summary={"progress_best": 0.9})
    assert high[0] > low[0]


def test_base_features_unaffected_by_swarm_summary():
    """The ordinary senses must not shift just because the swarm channel
    changed - only its own dedicated columns should respond."""
    input_idx = np.array([0])
    n_features = len(FEATURE_NAMES) + len(SWARM_FEATURE_NAMES)
    weights = np.zeros((1, n_features))
    weights[0, FEATURE_NAMES.index("health_frac")] = 1.0
    encoder = SwarmSensoryEncoder(input_idx, weights=weights, gain=1.0)

    obs = make_observation(health=10)
    a = encoder.encode(obs, n_neurons_total=1, swarm_summary={"progress_best": 0.1})
    b = encoder.encode(obs, n_neurons_total=1, swarm_summary={"progress_best": 0.9})
    assert a[0] == pytest.approx(b[0])


def test_extract_features_still_matches_base_shape():
    """Sanity check that this module didn't accidentally mutate the shared
    base encoder's own feature extraction."""
    assert extract_features(make_observation()).shape == (len(FEATURE_NAMES),)
