# Fly-Beat-Minecraft

An agent whose control core is a simulation of a **real fly brain** — a real
Drosophila connectome, the kind of data hosted at
[virtualflybrain.org](https://www.virtualflybrain.org/) — wired up to pilot a
Minecraft bot. The end goal: kill the Ender Dragon.

## The honest pitch

A real fly brain, however faithfully simulated, has no capacity for multi-hour
planning, crafting trees, or "go find a stronghold." A fly's brain evolved for
flight, feeding, courtship, and escape reflexes — not Minecraft progression.
No amount of simulation fidelity changes that.

So this project is a **hybrid**:

- **The real connectome is the reflexive/sensorimotor core.** It handles
  moment-to-moment control: walking, turning, obstacle and threat response,
  target pursuit and attack. The neuron identities and synapse weights/signs
  come straight from real connectome data and are not touched by training.
- **A separate, explicitly non-biological task manager handles long-horizon
  sequencing** — what to mine, when to craft, where to go, when to fight the
  dragon. This part makes no claim to be biologically real; it's a planner
  that uses the fly brain as its "motor cortex."

This is called out up front rather than overselling "the fly beats the game
unaided" — it doesn't, and no real fly could.

## Architecture

```
Minecraft server (local, vanilla/Paper)
        |
   bot-bridge/  (Node.js, mineflayer)
     - connects a bot to the server
     - exposes a WebSocket JSON API: getObservation(), doAction()
        |  (WebSocket, localhost)
        v
   brain/  (Python)
     agent/task_manager.py   <- non-biological planner: subgoal stack
                                 (wood -> stone -> iron -> diamond ->
                                  nether -> find stronghold -> kill dragon)
     agent/loop.py           <- real-time control loop
        |
        v  (task manager biases which sensory channels are "hot")
     interface/sensory_encoder.py  <- trainable: game state -> injected
                                       current into input neuron pools
     connectome/ (graph.py, fetch_hemibrain.py)  <- real hemibrain weighted
                                                     graph (fixed, not trained)
     sim/lif.py               <- leaky-integrate-and-fire sim over that graph
     interface/motor_decoder.py    <- trainable: descending-neuron firing
                                       rates -> discrete Minecraft actions
     training/train_interface.py  <- evolution-strategy training of the
                                       encoder + decoder
```

See [docs/architecture.md](docs/architecture.md) for the full technical design.

## What's real biology vs. engineered

| Piece | Real / fixed | Engineered / trained |
|---|---|---|
| Neuron identities, synapse weights & signs | ✅ from the hemibrain connectome | |
| Spiking dynamics (leaky-integrate-and-fire) | ✅ standard neuroscience model | |
| Sensory encoder (game state → current injection) | | ✅ trained |
| Motor decoder (firing rates → actions) | | ✅ trained |
| Task manager (crafting/progression logic) | | ✅ scripted/engineered |

## Repo layout

```
bot-bridge/     Node.js — mineflayer bot + WebSocket bridge to the brain
brain/          Python — connectome, spiking sim, encoder/decoder, task manager, training
data/           cached connectome data (gitignored)
docs/           architecture and design notes
scripts/        setup helpers (e.g. starting a local test server)
```

## Status

Early scaffold. See [docs/architecture.md](docs/architecture.md) for the
milestone roadmap (M0 scaffold → M1 bot-bridge → M2 connectome pipeline →
M3 standalone simulator → M4 wire brain to bot → M5 train the interface →
M6 task manager → M7 scale + go for the dragon).
