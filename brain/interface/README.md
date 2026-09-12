# brain/interface

The trainable boundary between Minecraft game state and the fixed connectome.

Planned (M4/M5):

- `sensory_encoder.py` — game-state feature vector -> injected current into
  chosen input neuron pools (trainable)
- `motor_decoder.py` — output (descending neuron) firing rates -> discrete
  Minecraft action (trainable)

See [../../docs/architecture.md](../../docs/architecture.md).
