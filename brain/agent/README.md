# brain/agent

Ties everything together: talks to `bot-bridge`, runs the real-time control
loop, and hosts the (explicitly non-biological) task manager that sequences
Minecraft progression.

- `bridge_client.py` — implemented (M1): `BridgeClient`, a WebSocket client
  for `bot-bridge` (`get_observation()`, `do_action(command)`, `ping()`).
  Needs `websockets>=12` (for the `websockets.sync.client` API) — install
  this in `brain/`'s own venv, not globally, to avoid version clashes with
  unrelated projects.

Planned (M4/M6):

- `task_manager.py` — subgoal stack / state machine for progression (wood ->
  stone -> iron -> diamond -> nether -> stronghold -> Ender Dragon)
- `loop.py` — the real-time control loop (observation -> task manager ->
  encoder -> sim step -> decoder -> action)

See [../../docs/architecture.md](../../docs/architecture.md).
