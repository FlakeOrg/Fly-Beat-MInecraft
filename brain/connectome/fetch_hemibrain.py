"""Fetches and caches the real hemibrain Drosophila connectome via neuPrint.

Requires a free neuPrint account and personal auth token:

    1. Sign in at https://neuprint.janelia.org (Google account is enough).
    2. Go to Account (top right) and copy your personal auth token.
    3. Set it as the NEUPRINT_TOKEN environment variable - never commit it
       (the repo's .gitignore already excludes .env / *.token).

The first fetch downloads the full traced connectome (~25k neurons, ~20M
synapses; takes a few minutes and ~300MB) and caches it under data/, so
later runs are instant and offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_SERVER = "https://neuprint.janelia.org"
DEFAULT_DATASET = "hemibrain:v1.2.1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "hemibrain"


def get_client(token: str | None = None, server: str = DEFAULT_SERVER, dataset: str = DEFAULT_DATASET):
    """Build a neuprint.Client, reading the token from NEUPRINT_TOKEN if not given."""
    from neuprint import Client

    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "No neuPrint auth token found. Create a free account at "
            f"{server}, copy your personal token from Account, and set it "
            "as the NEUPRINT_TOKEN environment variable."
        )
    return Client(server, dataset=dataset, token=token)


def fetch_and_cache(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    client=None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (or load from cache) the traced hemibrain connectome.

    Returns (neurons_df, conn_df):
      - neurons_df: one row per traced neuron (bodyId, type, predictedNt, ...)
      - conn_df: one row per (bodyId_pre, bodyId_post) pair with a summed
        synapse-count `weight` (collapsed across ROI - see graph.py for how
        this becomes a signed weighted adjacency matrix)
    """
    cache_dir = Path(cache_dir)
    neurons_path = cache_dir / "neurons.parquet"
    conn_path = cache_dir / "connections.parquet"

    if not force and neurons_path.exists() and conn_path.exists():
        return pd.read_parquet(neurons_path), pd.read_parquet(conn_path)

    from neuprint import fetch_traced_adjacencies

    if client is None:
        client = get_client()

    neurons_df, roi_conn_df = fetch_traced_adjacencies(client=client)
    conn_df = roi_conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()

    cache_dir.mkdir(parents=True, exist_ok=True)
    neurons_df.to_parquet(neurons_path)
    conn_df.to_parquet(conn_path)
    return neurons_df, conn_df


if __name__ == "__main__":
    neurons_df, conn_df = fetch_and_cache()
    print(f"{len(neurons_df)} traced neurons, {len(conn_df)} directed connections")
    print(neurons_df[["bodyId", "type", "predictedNt"]].head())
