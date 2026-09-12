# brain/training

Evolution-strategy training of the sensory encoder and motor decoder (and
optionally a per-neuron-type neuromodulatory gain) against a survival/progress
reward, since the spiking sim + live game loop isn't cleanly differentiable
end to end.

Planned (M5):

- `train_interface.py` — ES training harness

See [../../docs/architecture.md](../../docs/architecture.md).
