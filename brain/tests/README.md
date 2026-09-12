# brain/tests

Sanity tests for the connectome graph and simulator.

- `test_lif.py` — implemented (M3): stability (no NaNs/saturation), a silent
  network stays silent, the output pool responds to driven input, and
  activity decays back down once the drive is removed rather than
  self-sustaining forever. Run with `pytest` from `brain/`.
- `test_graph.py` — planned (M2): connectome loads, shapes and metadata line up.
