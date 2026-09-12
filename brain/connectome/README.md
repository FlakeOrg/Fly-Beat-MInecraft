# brain/connectome

Loads and caches the real hemibrain Drosophila connectome (via `neuprint-python`)
and builds the sparse weighted graph the simulator runs on. Fixed / not trained.

Implemented (M2):

- `fetch_hemibrain.py` — `fetch_and_cache()`: pulls the full traced hemibrain
  connectome via `neuprint-python`'s `fetch_traced_adjacencies` and caches it
  as parquet under `data/hemibrain/` (gitignored) so later runs are instant
  and offline. **Requires a free neuPrint account + personal auth token**
  (see the module docstring) set as the `NEUPRINT_TOKEN` env var — this is
  the one step in this project only you can do, since it's a personal
  account signup.
- `graph.py` — `build_adjacency()`: turns the cached neuron/connection
  tables into a signed, weighted sparse adjacency matrix (excitatory
  cholinergic vs. inhibitory GABAergic/glutamatergic, per the predicted
  neurotransmitter columns neuPrint provides) plus aligned neuron metadata.
  `identify_pools()`: picks candidate input (sensory, e.g. visual projection
  neurons) and output (descending neuron) pools by matching the hemibrain
  `type` naming convention. Fully unit-tested with fabricated data — doesn't
  need the real connectome or a token to validate its own logic.

Tests: [../tests/test_graph.py](../tests/test_graph.py).

Once a token is available, `fetch_and_cache()` + `build_adjacency()` feed
directly into `sim.lif.LIFNetwork`, replacing `sim.synthetic.make_synthetic_graph`
— the simulator itself doesn't change, just its input. Expect to re-run the
`sim/demo.py`-style parameter sweep (`syn_scale`, `b_adapt`, `tau_adapt`)
against the real graph's very different weight scale (raw synapse counts,
not a hand-tuned lognormal) before trusting its dynamics.

See [../../docs/architecture.md](../../docs/architecture.md).
