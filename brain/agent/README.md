# brain/agent

Ties everything together: talks to `bot-bridge`, runs the real-time control
loop, and hosts the (explicitly non-biological) task manager that sequences
Minecraft progression.

Planned (M4/M6):

- `bridge_client.py` — WebSocket client for `bot-bridge`
- `task_manager.py` — subgoal stack / state machine for progression (wood ->
  stone -> iron -> diamond -> nether -> stronghold -> Ender Dragon)
- `loop.py` — the real-time control loop (observation -> task manager ->
  encoder -> sim step -> decoder -> action)

See [../../docs/architecture.md](../../docs/architecture.md).
