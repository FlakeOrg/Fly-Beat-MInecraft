"""Builds the sparse, signed, weighted connectivity graph the LIF simulator
runs on, from cached hemibrain data (see fetch_hemibrain.py). This is the
"real fly brain" part of the project - fixed, not trained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

# The hemibrain dataset records a predicted neurotransmitter per neuron
# under one of a few column names depending on how it was fetched/curated;
# check them in this priority order and use whichever is present first.
NT_COLUMN_PRIORITY = ["consensusNt", "predictedNt", "celltypePredictedNt"]

# GABA and glutamate are treated as inhibitory in the fly CNS - the
# convention used in prior hemibrain-simulation work (e.g. Shiu et al.
# 2024's leaky-integrate-and-fire connectome model). Acetylcholine is the
# majority excitatory transmitter. The aminergic/neuromodulatory ones
# (dopamine, serotonin, octopamine) are treated as excitatory here for lack
# of a better default - their real effect is receptor-dependent and far
# more context-specific than a single sign can capture; revisit if that
# matters for a particular neuron pool.
INHIBITORY_NT = {"gaba", "glutamate"}


def _best_nt(neurons_df: pd.DataFrame) -> pd.Series:
    nt = pd.Series(pd.NA, index=neurons_df.index, dtype=object)
    for col in NT_COLUMN_PRIORITY:
        if col in neurons_df.columns:
            nt = nt.fillna(neurons_df[col])
    return nt.fillna("unknown").str.lower()


def neuron_signs(neurons_df: pd.DataFrame) -> np.ndarray:
    """+1.0 (excitatory) / -1.0 (inhibitory) per neuron, from the best
    available predicted-neurotransmitter column."""
    nt = _best_nt(neurons_df)
    return np.where(nt.isin(INHIBITORY_NT), -1.0, 1.0)


def build_adjacency(neurons_df: pd.DataFrame, conn_df: pd.DataFrame) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Build the signed weighted adjacency matrix + aligned neuron metadata.

    Returns (weights, neuron_meta):
      - weights[i, j] is the synapse-count-weighted, sign-adjusted synaptic
        strength from pre-synaptic neuron j to post-synaptic neuron i
        (matching the convention `sim/lif.py` expects).
      - neuron_meta is `neurons_df` reindexed to 0..n-1 (row order matches
        the matrix) with an added `sign` column.

    Connections referencing a bodyId not present in `neurons_df` (e.g. an
    untraced or excluded neuron) are dropped.
    """
    neuron_meta = neurons_df.reset_index(drop=True).copy()
    neuron_meta["sign"] = neuron_signs(neuron_meta)

    body_id_to_index = pd.Series(neuron_meta.index, index=neuron_meta["bodyId"])
    pre_idx = conn_df["bodyId_pre"].map(body_id_to_index)
    post_idx = conn_df["bodyId_post"].map(body_id_to_index)

    valid = pre_idx.notna() & post_idx.notna()
    pre_idx = pre_idx[valid].astype(int).to_numpy()
    post_idx = post_idx[valid].astype(int).to_numpy()
    weight = conn_df.loc[valid, "weight"].to_numpy()

    signed_weight = weight * neuron_meta["sign"].to_numpy()[pre_idx]

    n = len(neuron_meta)
    weights = sp.coo_matrix((signed_weight, (post_idx, pre_idx)), shape=(n, n)).tocsr()
    return weights, neuron_meta


def identify_pools(
    neuron_meta: pd.DataFrame,
    input_type_pattern: str,
    output_type_pattern: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (into neuron_meta / the adjacency matrix) for candidate
    input (sensory) and output (descending) neuron pools, matched by regex
    against the hemibrain `type` column.

    Starting-point patterns (hemibrain cell-type nomenclature):
      - input (visual projection neurons): r"^(LC|LT|LPLC|MeTu)"
      - output (descending neurons to the VNC): r"^DN"

    Inspect `neuron_meta['type'].value_counts()` to refine these for
    whatever sensory/motor mapping is actually needed.
    """
    types = neuron_meta["type"].fillna("")
    input_idx = neuron_meta.index[types.str.match(input_type_pattern)].to_numpy()
    output_idx = neuron_meta.index[types.str.match(output_type_pattern)].to_numpy()
    return input_idx, output_idx
