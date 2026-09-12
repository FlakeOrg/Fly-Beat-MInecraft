"""Tests for brain/connectome/graph.py using fabricated neuron/connection
tables shaped like real neuPrint output - no network access or auth token
needed. Validates the graph-building logic itself; fetch_hemibrain.py's
live fetch is exercised separately once a real NEUPRINT_TOKEN is available.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "connectome"))

from graph import build_adjacency, identify_pools, neuron_signs  # noqa: E402


@pytest.fixture
def sample_tables():
    neurons_df = pd.DataFrame(
        {
            "bodyId": [100, 200, 300, 400],
            "type": ["LC10", "PN_generic", "PN_generic", "DNp09"],
            "predictedNt": ["acetylcholine", "gaba", "glutamate", "acetylcholine"],
        }
    )
    conn_df = pd.DataFrame(
        {
            "bodyId_pre": [100, 200, 300, 999],  # 999 is not in neurons_df
            "bodyId_post": [200, 300, 400, 400],
            "weight": [5, 3, 7, 42],
        }
    )
    return neurons_df, conn_df


def test_neuron_signs_from_predicted_nt():
    neurons_df = pd.DataFrame(
        {"bodyId": [1, 2, 3, 4], "predictedNt": ["acetylcholine", "gaba", "glutamate", "dopamine"]}
    )
    signs = neuron_signs(neurons_df)
    np.testing.assert_array_equal(signs, [1.0, -1.0, -1.0, 1.0])


def test_neuron_signs_prefers_consensus_over_predicted():
    neurons_df = pd.DataFrame(
        {
            "bodyId": [1],
            "consensusNt": ["gaba"],
            "predictedNt": ["acetylcholine"],
        }
    )
    signs = neuron_signs(neurons_df)
    assert signs[0] == -1.0


def test_neuron_signs_defaults_to_excitatory_when_unknown():
    neurons_df = pd.DataFrame({"bodyId": [1], "predictedNt": [None]})
    signs = neuron_signs(neurons_df)
    assert signs[0] == 1.0


def test_build_adjacency_shape_and_signs(sample_tables):
    neurons_df, conn_df = sample_tables
    weights, neuron_meta = build_adjacency(neurons_df, conn_df)

    assert weights.shape == (4, 4)
    assert len(neuron_meta) == 4
    np.testing.assert_array_equal(neuron_meta["sign"].to_numpy(), [1.0, -1.0, -1.0, 1.0])


def test_build_adjacency_drops_unknown_bodies(sample_tables):
    neurons_df, conn_df = sample_tables
    weights, _ = build_adjacency(neurons_df, conn_df)
    # 4 input rows, but the one referencing bodyId 999 (not a traced neuron)
    # should have been dropped rather than raising or silently corrupting data.
    assert weights.nnz == 3


def test_build_adjacency_values_and_convention(sample_tables):
    neurons_df, conn_df = sample_tables
    weights, neuron_meta = build_adjacency(neurons_df, conn_df)
    dense = weights.toarray()

    idx = {body_id: i for i, body_id in enumerate(neuron_meta["bodyId"])}

    # 100 (excitatory) -> 200, weight 5: positive entry at [post=200, pre=100]
    assert dense[idx[200], idx[100]] == pytest.approx(5.0)
    # 200 (inhibitory, gaba) -> 300, weight 3: negative entry
    assert dense[idx[300], idx[200]] == pytest.approx(-3.0)
    # 300 (inhibitory, glutamate) -> 400, weight 7: negative entry
    assert dense[idx[400], idx[300]] == pytest.approx(-7.0)
    # no edge from 400 -> 100
    assert dense[idx[100], idx[400]] == 0.0


def test_identify_pools_matches_type_prefixes(sample_tables):
    neurons_df, conn_df = sample_tables
    _, neuron_meta = build_adjacency(neurons_df, conn_df)

    input_idx, output_idx = identify_pools(neuron_meta, r"^LC", r"^DN")

    assert list(neuron_meta.loc[input_idx, "bodyId"]) == [100]
    assert list(neuron_meta.loc[output_idx, "bodyId"]) == [400]


def test_identify_pools_empty_when_no_match(sample_tables):
    neurons_df, conn_df = sample_tables
    _, neuron_meta = build_adjacency(neurons_df, conn_df)

    input_idx, output_idx = identify_pools(neuron_meta, r"^NoSuchType", r"^NoSuchType")

    assert len(input_idx) == 0
    assert len(output_idx) == 0
