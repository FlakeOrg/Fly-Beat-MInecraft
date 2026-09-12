# bot-bridge

Node.js service that connects a [Mineflayer](https://github.com/PrismarineJS/mineflayer)
bot to a Minecraft server and exposes it over a WebSocket JSON API so the
Python `brain/` process never has to touch the Minecraft protocol directly.
Implemented (M1) and validated end-to-end against a local Paper server.

- `src/index.js` — connects the bot, runs the WebSocket server. Protocol:
  client sends `{id, method, params}`, server replies `{id, result}` or
  `{id, error}`; unsolicited `{type: "event", event: "spawn"|"death"|"kicked"|"end"}`
  messages are broadcast for lifecycle changes.
- `src/observation.js` — `buildObservation(bot)`: position, health/food,
  nearby blocks (small radius around the bot) and entities, inventory.
- `src/actions.js` — `performAction(bot, mcData, command)`: `move`, `stop`,
  `look`, `lookAt`, `dig`, `place`, `attack`, `equip`, `craft`, `chat`.
  `dig` has its own timeout that cancels the in-progress dig (`bot.stopDigging()`)
  if the target goes out of reach mid-action — discovered live in M4 when
  digging while moving left a request permanently unanswered. `index.js`
  also wraps every `doAction` call in a blanket timeout as a safety net for
  any other action type that might not settle cleanly.

## Run it

```
npm install
npm start
```

Env vars: `MC_HOST` (default `localhost`), `MC_PORT` (`25565`), `MC_USERNAME`
(`FlyBrain`), `MC_VERSION` (auto-detect), `MC_AUTH` (`offline`),
`BRIDGE_PORT` (`8081`). See [../scripts/start_server.md](../scripts/start_server.md)
for setting up a local test server to point it at.

Python side: [../brain/agent/bridge_client.py](../brain/agent/bridge_client.py).

See [../docs/architecture.md](../docs/architecture.md).
