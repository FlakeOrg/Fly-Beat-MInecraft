"""Fetches and caches the real hemibrain Drosophila connectome via neuPrint.

Requires a free neuPrint account and personal auth token:

    1. Sign in at https://neuprint.janelia.org (Google account is enough).
    2. Go to Account (top right) and copy your personal auth token.
    3. Set it as the NEUPRINT_TOKEN environment variable - never commit it
       (the repo's .gitignore already excludes .env / *.token).

The first fetch downloads the full traced connectome (~22k neurons, ~3.5M
directed connections; takes a couple minutes) and caches it under data/, so
later runs are instant and offline.

Neuron-level predicted neurotransmitter (needed for excitatory/inhibitory
sign - see graph.py) is NOT available as a property on this neuPrint
dataset's Neuron nodes (confirmed by querying `keys(n)` directly - it's
simply not part of this deployment's schema). It comes from a separate,
independently published source instead: Eckstein & Bates et al., Cell
(2024), which aggregated their synapse-level transmitter classifier per
neuron. supplemental_data_3.csv on that paper's Zenodo record
(https://zenodo.org/records/10593546) is exactly that: one row per hemibrain
body ID with a `top_nt` predicted class. This is fetched and merged in here
too, so `graph.py` downstream never needs to know the two data sources are
different.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_SERVER = "https://neuprint.janelia.org"
DEFAULT_DATASET = "hemibrain:v1.2.1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "hemibrain"

# Eckstein & Bates et al., Cell (2024) - per-neuron aggregated predicted
# neurotransmitter for hemibrain (among other datasets in the same record).
NT_ZENODO_URL = "https://zenodo.org/records/10593546/files/supplemental_data_3.csv?download=1"


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


def _fetch_predicted_nt(cache_dir: Path) -> pd.DataFrame:
    """Downloads (or loads from cache) the Eckstein & Bates et al. per-neuron
    predicted-neurotransmitter table and returns it indexed by bodyId with
    a single `predictedNt` column (lowercase, e.g. "acetylcholine", "gaba";
    "unknown"/"neither" become NaN so callers fall back consistently)."""
    raw_path = cache_dir / "supplemental_data_3.csv"
    if not raw_path.exists():
        import requests

        cache_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(NT_ZENODO_URL, timeout=120)
        response.raise_for_status()
        raw_path.write_bytes(response.content)

    nt_df = pd.read_csv(raw_path, usecols=["bodyid", "top_nt"])
    nt_df = nt_df.rename(columns={"bodyid": "bodyId", "top_nt": "predictedNt"})
    nt_df["predictedNt"] = nt_df["predictedNt"].replace({"unknown": pd.NA, "neither": pd.NA})
    return nt_df.set_index("bodyId")


def fetch_and_cache(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    client=None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (or load from cache) the traced hemibrain connectome.

    Returns (neurons_df, conn_df):
      - neurons_df: one row per traced neuron (bodyId, type, predictedNt)
      - conn_df: one row per (bodyId_pre, bodyId_post) pair with a summed
        synapse-count `weight` (collapsed across ROI - see graph.py for how
        this becomes a signed weighted adjacency matrix)
    """
    cache_dir = Path(cache_dir)
    neurons_path = cache_dir / "neurons.parquet"
    conn_path = cache_dir / "connections.parquet"

    if not force and neurons_path.exists() and conn_path.exists():
        return pd.read_parquet(neurons_path), pd.read_parquet(conn_path)

    from neuprint import NeuronCriteria, fetch_adjacencies

    if client is None:
        client = get_client()

    criteria = NeuronCriteria(status="Traced", cropped=False, client=client)
    neurons_df, roi_conn_df = fetch_adjacencies(
        criteria,
        criteria,
        include_nonprimary=False,
        properties=["type", "instance"],
        client=client,
    )
    conn_df = roi_conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()

    nt_by_body = _fetch_predicted_nt(cache_dir)
    neurons_df = neurons_df.join(nt_by_body, on="bodyId")

    cache_dir.mkdir(parents=True, exist_ok=True)
    neurons_df.to_parquet(neurons_path)
    conn_df.to_parquet(conn_path)
    return neurons_df, conn_df


if __name__ == "__main__":
    neurons_df, conn_df = fetch_and_cache()
    print(f"{len(neurons_df)} traced neurons, {len(conn_df)} directed connections")
    print(f"predictedNt coverage: {neurons_df['predictedNt'].notna().sum()}/{len(neurons_df)}")
    print(neurons_df[["bodyId", "type", "predictedNt"]].head())
