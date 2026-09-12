# brain/sim

The leaky-integrate-and-fire (LIF) spiking simulator that runs over the
connectome graph. Implemented (M3):

- `lif.py` — `LIFNetwork`/`LIFParams`: integrates injected current + weighted
  synaptic input + leak per tick, emits spikes on threshold crossing.
  Includes spike-frequency adaptation (needed to keep small recurrent
  excitatory/inhibitory networks from either staying silent or locking into
  permanent self-sustained firing regardless of input — see the comments in
  `lif.py` and `demo.py`) and short-term synaptic depression
  (`depression_frac`/`tau_depression`, disabled by default) — added while
  wiring up the real hemibrain graph (M2), where adaptation alone wasn't
  enough to stop reverberating loops between populations from settling into
  permanent activity; see `connectome/README.md` for what did and didn't
  work there.
- `synthetic.py` — `make_synthetic_graph`: a random Dale's-law-respecting
  directed graph used as a stand-in until the real hemibrain graph (M2)
  exists. Not the real fly brain — just enough structure to validate the
  simulator.
- `demo.py` — run with `python demo.py` from this directory. Drives a
  synthetic graph's input pool with constant current, then cuts it, and
  prints per-tick firing rates for the input/other/output pools. Confirms
  stability, that signal propagates from input to output, and that activity
  quiets back down once the drive stops rather than self-sustaining forever.

Tests: [../tests/test_lif.py](../tests/test_lif.py).

Once M2 lands, `brain/agent/loop.py` will point `LIFNetwork` at the real
hemibrain graph instead of `synthetic.make_synthetic_graph` — the simulator
itself doesn't change, just its input.

See [../../docs/architecture.md](../../docs/architecture.md).
