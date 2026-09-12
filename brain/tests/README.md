# brain/tests

Sanity tests for the connectome graph and simulator.

- `test_lif.py` — implemented (M3): stability (no NaNs/saturation), a silent
  network stays silent, the output pool responds to driven input, and
  activity decays back down once the drive is removed rather than
  self-sustaining forever. Run with `pytest` from `brain/`.
- `test_graph.py` — implemented (M2): signed-weight assignment from
  predicted neurotransmitter, adjacency matrix shape/values/convention,
  dropping connections to unknown bodies, and pool identification by cell
  type — all against fabricated tables, no network/token needed.
