# brain/interface

The trainable boundary between Minecraft game state and the fixed connectome.

Implemented (M4, hand-built weights — not yet trained):

- `sensory_encoder.py` — `extract_features()` turns a bot-bridge observation
  into a fixed 8-feature vector (health, hunger, nearby hostiles/entities,
  a block directly ahead, falling, on-ground). `SensoryEncoder.encode()`
  projects that through a hand-built sparse receptive-field matrix into
  injected current on the connectome's input neuron pool.
- `motor_decoder.py` — `MotorDecoder.decode()` reads firing rates over the
  output (descending neuron) pool through a hand-built partition into action
  scores, producing booleans for `forward`/`left`/`right`/`jump`/`attack`/
  `mine_ahead`.

These weights are arbitrary (random receptive fields / an even partition) —
good enough to validate that a real observation can drive the network and
that its output can drive believable-looking bot commands (see
`agent/loop.py`), but not tuned for good play. M5 replaces `_default_weights`
in both files with something learned (evolution strategy) against a
survival/progress reward.

See [../../docs/architecture.md](../../docs/architecture.md).
