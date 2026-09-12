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
- `loop.py` — implemented (M4, now on the real connectome): the real-time
  control loop. Each step, `task_manager.decide()` gets first say; only
  when it returns `None` (nothing specific to do) does control fall
  through to the fly-brain's trained reflexes. `load_real_graph()` is
  shared with `training/live_rollout.py` so live training scores the same
  graph the live loop actually runs on.
- `task_manager.py` — implemented (M6, first slice): explicitly
  non-biological subgoal stack. Currently covers gather wood -> craft
  planks -> craft + place a table -> craft basic wooden tools -> hand off
  to open-ended fly-brain exploration, plus an always-on low-health flee
  override. Deliberately scripted, precise, deterministic logic for things
  a fly brain fundamentally can't do (crafting, navigating to a specific
  known block) — the fly brain still does the open-ended
  movement/exploration/threat-response in between. Later progression
  stages (stone/iron/diamond tools, Nether travel, stronghold, the Ender
  Dragon fight) follow the same stage-machine pattern, not implemented yet
  — this is a first working slice, not the full game. Unit-tested against
  fabricated observations in
  [../tests/test_task_manager.py](../tests/test_task_manager.py).

See [../../docs/architecture.md](../../docs/architecture.md).
