# brain/agent

Ties everything together: talks to `bot-bridge`, runs the real-time control
loop, and hosts the (explicitly non-biological) task manager that sequences
Minecraft progression.

- `bridge_client.py` — implemented (M1): `BridgeClient`, a WebSocket client
  for `bot-bridge` (`get_observation()`, `do_action(command)`, `ping()`).
  Needs `websockets>=12` (for the `websockets.sync.client` API) — install
  this in `brain/`'s own venv, not globally, to avoid version clashes with
  unrelated projects. Every call has a receive timeout (default 15s) as a
  hard backstop on top of bot-bridge's own per-action timeout, so a wedged
  bridge process can't hang an automated rollout loop forever (matters a
  lot once M5 is running many unattended training rollouts).
- `loop.py` — implemented (M4): the real-time control loop (observation ->
  encoder -> `SIM_TICKS_PER_ACTION` sim steps -> decoder -> action), still
  on the synthetic stand-in graph. Validated live end-to-end against a real
  server; along the way found and fixed a real concurrency bug (see
  bot-bridge's README) where digging while moving could leave a request
  permanently unanswered.

Planned (M5/M6):

- `task_manager.py` — subgoal stack / state machine for progression (wood ->
  stone -> iron -> diamond -> nether -> stronghold -> Ender Dragon)

See [../../docs/architecture.md](../../docs/architecture.md).
