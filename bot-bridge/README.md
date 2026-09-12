# bot-bridge

Node.js service that connects a [Mineflayer](https://github.com/PrismarineJS/mineflayer)
bot to a local Minecraft server and exposes it over a small WebSocket JSON API
so the Python `brain/` process never has to touch the Minecraft protocol
directly.

Planned layout (M1):

- `package.json` — deps: `mineflayer`, `ws`
- `src/index.js` — connects the bot, runs the WebSocket server
- `src/observation.js` — builds an observation JSON from bot state
  (position, health, hunger, nearby blocks/entities, inventory)
- `src/actions.js` — translates action commands (move, turn, jump, mine,
  place, attack, craft) into Mineflayer calls

Not yet implemented — see [../docs/architecture.md](../docs/architecture.md)
and the M1 milestone.
