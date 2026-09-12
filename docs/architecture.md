# Architecture

## Why the hemibrain connectome

[Virtual Fly Brain](https://www.virtualflybrain.org/) is primarily an
atlas/query/cross-reference layer — 3D neuron morphology, cross-dataset neuron
identity matching, a Python/R/Cypher query API. It's the right tool for
*looking up* neuron types (e.g. "which neurons are optic-lobe visual
projection neurons" or "which are descending neurons to the ventral nerve
cord"), but it isn't the primary source for a full weighted connectivity
matrix.

[neuPrint](https://neuprint.janelia.org/) and the **hemibrain** dataset (Janelia),
accessed via the `neuprint-python` client, is the practical source of a real,
densely-reconstructed connectome: ~25,000 neurons of the central adult
Drosophila brain, ~20M synapses, with a downloadable compact connection
matrix, cell-type annotations, and a predicted neurotransmitter
(excitatory/inhibitory) per neuron. That's what we simulate.

FlyWire's full-brain connectome (~140k neurons, whole CNS including optic
lobes) is a natural scale-up once the hemibrain pipeline is working end to
end — noted as a stretch goal (M7), not a starting point.

## Layers

### 1. Connectome (`brain/connectome/`)

- `fetch_hemibrain.py`: pulls data via `neuprint-python` (requires a free
  neuPrint account + auth token from the user) and caches it locally under
  `data/`.
- `graph.py`: builds a sparse weighted adjacency matrix (pre-synaptic →
  post-synaptic, weight = synapse count, sign = predicted neurotransmitter),
  plus neuron metadata (cell type, brain region/neuropil). Also identifies
  candidate **input pools** (e.g. optic-lobe visual projection neurons or
  other convenient large sensory populations) and **output pools**
  (identified descending neurons — real cell types documented in the
  connectomics literature, e.g. the giant fiber escape circuit, forward-
  walking DNs).

This layer is **fixed** — it is not touched by training. It's the "real fly
brain" part of the claim.

### 2. Simulator (`brain/sim/lif.py`)

A leaky-integrate-and-fire (LIF) spiking simulator run directly over the
connectome graph. Each simulation tick integrates injected current, synaptic
input from connected neurons (weighted by the real synapse counts/signs), and
leak, then produces spikes when a neuron crosses threshold. This is a
standard, well-understood neuroscience model — the tradeoff for "real
biology" is stability/tuning work (see M3: sanity-check the sim doesn't blow
up or flatline, and firing rates are in a plausible ballpark before trusting
it downstream).

### 3. Interface (`brain/interface/`)

- `sensory_encoder.py`: **trainable**. Maps a compact game-state feature
  vector (health, hunger, nearby blocks/entities/threats, current subgoal
  bias from the task manager) to injected current into the chosen input
  neuron pools.
- `motor_decoder.py`: **trainable**. Maps firing rates of the output
  (descending neuron) pools to a discrete Minecraft action (forward, turn,
  jump, mine-ahead, attack, place, etc).

This is the layer that gets trained — real animals also have plastic sensory
transduction and motor gain even though the wiring diagram itself is largely
fixed, so this is a reasonably honest split.

### 4. Agent (`brain/agent/`)

- `bridge_client.py`: WebSocket client talking to `bot-bridge`.
- `task_manager.py`: **explicitly non-biological.** A subgoal stack/state
  machine (later possibly a lightweight RL policy) implementing Minecraft
  progression: gather wood → stone → iron → diamond → nether → find
  stronghold → kill the Ender Dragon. It doesn't control the body directly;
  it biases which sensory channels are "hot" (e.g. "current subgoal: find
  trees" boosts the encoder's weighting toward wood-block detection) and
  reads back progress from observations.
- `loop.py`: the real-time control loop — pulls an observation from
  bot-bridge, feeds it through the task manager and encoder, steps the LIF
  sim, decodes an action, sends it back.

### 5. Training (`brain/training/train_interface.py`)

The connectome's spiking dynamics aren't cleanly differentiable end-to-end
through a live Minecraft rollout, so the encoder/decoder (and optionally a
per-neuron-type neuromodulatory gain) are trained with an **evolution
strategy** (e.g. CMA-ES or a simple ES) against a reward signal — survival
time, distance traveled, resources gathered, progression milestones reached.
ES is a good fit here: it doesn't need gradients through the simulator or the
game, and it parallelizes trivially across many rollouts.

### 6. Bot bridge (`bot-bridge/`, Node.js)

[Mineflayer](https://github.com/PrismarineJS/mineflayer) connects a real bot
to a local vanilla/Paper Minecraft server and gives structured world state
(blocks, entities, inventory, health) without needing pixel perception — much
easier to map onto a small set of sensory channels than raw vision would be.
`bot-bridge` wraps this in a small WebSocket JSON API:

- `getObservation()` → `{ position, health, hunger, nearbyBlocks, nearbyEntities, inventory, ... }`
- `doAction(cmd)` → move/turn, jump, mine, place, attack, craft

keeping the game-facing code in JS (where Mineflayer lives) and everything
else (connectome, sim, training) in Python's ML ecosystem, talking over a
plain local WebSocket.

## Milestone roadmap

- **M0 — Scaffold** *(this session)*: repo structure, README, this doc.
- **M1 — bot-bridge**: bot joins a local server; WebSocket API round-trips an
  observation and accepts actions. Manually drive it to confirm the plumbing.
- **M2 — connectome pipeline**: fetch + cache hemibrain data, build the graph,
  pick initial input/output neuron pools. Requires a free neuPrint token.
- **M3 — standalone simulator**: run the LIF sim with no Minecraft involved;
  confirm stability and plausible firing rates.
- **M4 — wire brain to bot-bridge**: hand-built simple encoder/decoder;
  confirm signal flows connectome → action and produces *some* coherent
  reaction. This milestone validates plumbing, not intelligence.
- **M5 — train the interface**: ES training of encoder/decoder against a
  survival/progress reward in a fast local server.
- **M6 — task manager**: layer Minecraft progression logic on top, using the
  trained connectome for moment-to-moment execution of each subgoal.
- **M7 — scale + iterate**: grow neuron pools, consider the FlyWire
  full-brain connectome, keep training, work toward an actual Ender Dragon
  kill run.
