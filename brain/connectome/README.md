# brain/connectome

Loads and caches the real hemibrain Drosophila connectome (via `neuprint-python`)
and builds the sparse weighted graph the simulator runs on. Fixed / not trained.

Planned (M2):

- `fetch_hemibrain.py` — pulls data via `neuprint-python` (needs a free
  neuPrint account + token) and caches it under `data/`
- `graph.py` — builds the sparse weighted adjacency matrix + neuron metadata,
  identifies candidate input pools (sensory) and output pools (descending
  neurons)

See [../../docs/architecture.md](../../docs/architecture.md).
