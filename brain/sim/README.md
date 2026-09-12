# brain/sim

The leaky-integrate-and-fire (LIF) spiking simulator that runs over the
connectome graph.

Planned (M3):

- `lif.py` — LIF simulator: integrates injected current + weighted synaptic
  input + leak per tick, emits spikes on threshold crossing

See [../../docs/architecture.md](../../docs/architecture.md).
