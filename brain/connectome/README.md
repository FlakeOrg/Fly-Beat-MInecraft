# brain/connectome

Loads and caches the real hemibrain Drosophila connectome (via `neuprint-python`)
and builds the sparse weighted graph the simulator runs on. Fixed / not trained.

Implemented (M2) and **the real data has now been fetched and validated**:

- `fetch_hemibrain.py` — `fetch_and_cache()`: pulls the full traced hemibrain
  connectome via `neuprint-python`, caches it as parquet under
  `data/hemibrain/` (gitignored). Requires a free neuPrint account +
  personal auth token (`NEUPRINT_TOKEN` env var).

  Neuron-level predicted neurotransmitter is **not** a property on this
  neuPrint dataset's Neuron nodes (confirmed directly via `keys(n)` - it's
  simply absent from this deployment's schema, not just unrequested). It's
  fetched from a separate, independently published source instead: Eckstein
  & Bates et al., *Cell* (2024), whose per-neuron aggregated transmitter
  predictions (`supplemental_data_3.csv` on
  [Zenodo record 10593546](https://zenodo.org/records/10593546)) are
  downloaded and merged in by body ID.

  **Real numbers from the actual fetch**: 21,738 traced neurons, 3,550,061
  directed connections, 21,562 (99.2%) with a real predicted-neurotransmitter
  match.

- `graph.py` — `build_adjacency()`: signed weighted sparse adjacency +
  aligned metadata. On the real data: **13,617 excitatory / 8,121
  inhibitory neurons** (63%/37%, consistent with known fly-brain biology -
  cholinergic majority, substantial GABAergic/glutamatergic inhibition).
  `identify_pools()` with `r"^(LC|LT|LPLC|MeTu)"` / `r"^DN"` finds **2,412**
  real visual-projection neurons (types like `LC40a`, `LT72`) and **101**
  real descending neurons (types like `DNp09`, `DNg30`, `DN1a`) - genuine,
  individually-identified cell types from the connectomics literature.

## Wiring the real graph into the simulator: harder than the synthetic case

`sim/demo.py`'s tuning (M3) doesn't transfer directly. The real graph has
~163 average connections per neuron (vs. ~16 in the synthetic test graph)
and a heavy-tailed weight distribution (median synapse-count weight 1, mean
4, max 4299) - raw weights need normalizing (e.g. dividing each neuron's
incoming weights by its total incoming synaptic magnitude) before a
dimensionless LIF model is usable at all.

More importantly: across a wide sweep of gain (`syn_scale` 2 to 10,000),
spike-frequency adaptation (`b_adapt`/`tau_adapt`), and short-term synaptic
depression (`depression_frac`/`tau_depression` - see `sim/lif.py`, added
specifically for this), driving any meaningful subset of the real visual-PN
pool consistently settles into a **persistent, self-sustaining activity
state that does not depend on whether the input is still present** - unlike
the synthetic graph, where the same mechanisms found a clean input-tracking
regime. This looks like a genuine structural property of the real
connectome's recurrent circuitry (real fly brains do support persistent/
attractor-like states - e.g. central-complex heading representation), not
a bug to be tuned away with 2-3 global scalars.

Whether that's actually a problem depends on framing: tested whether
*different* driven inputs produce *different* persistent states (not just
"some elevated activity" vs. silence) - driving two disjoint 150-neuron
subsets of the visual-PN pool gives output-pool firing-rate vectors with
**0.84 correlation** to each other (vs. 1.0 for repeating the same input
twice), so there is real, repeatable, input-specific structure in the
result even though the network doesn't return to a zero baseline. That
makes this closer to a **liquid-state-machine / reservoir-computing**
setup - a fixed, complex, recurrent substrate with rich (if persistent)
dynamics, decoded by a trained readout - than a simple reflex arc, which
may be a perfectly workable framing for M5's trained decoder rather than a
blocker. Not yet validated with an actual trained decoder, though - this is
the honest state of things, not a solved problem.

Tests: [../tests/test_graph.py](../tests/test_graph.py) (fabricated data,
still passes unchanged).

See [../../docs/architecture.md](../../docs/architecture.md).
