# brain/training

Evolution-strategy (ES) training of the sensory encoder and motor decoder,
since the spiking sim + an eventual live Minecraft rollout aren't cleanly
differentiable end to end.

Implemented (M5) — `train_interface.py`:

- `run_es()` — a simple mirrored-sampling ES (OpenAI-ES style): sample
  symmetric Gaussian perturbations around the current weights, evaluate
  fitness for each, take a step in the fitness-weighted direction. Generic
  over `fitness_fn` - it doesn't know or care whether that's scoring hand-
  specified situations or an eventual live Minecraft rollout through
  bot-bridge. Unit-tested against a trivial quadratic target
  ([../tests/test_training.py](../tests/test_training.py)), independent of
  the connectome.
- `flatten_params()` / `unflatten_params()` — pack/unpack
  `SensoryEncoder.weights` + `MotorDecoder.weights` into a single vector ES
  can optimize over.
- `evaluate_situations()` + `SITUATIONS` — a **proxy validation task**, not
  the real reward: 5 hand-specified (observation, correct-action) pairs
  (explore when clear, mine what's ahead, attack a nearby hostile, do
  nothing while falling, keep exploring even when hungry). Fitness = the
  fraction of individual action booleans matched across all 5 situations
  (30 total). This exists to validate that ES can actually train something
  useful on the real connectome's dynamics *before* committing to the much
  slower live-rollout reward - live Minecraft episodes are far too slow to
  evaluate thousands of times per training run the way this proxy can be.

## Real result, against the real hemibrain connectome

Random-initialized (hand-built) encoder/decoder weights: **0.667** fitness.
After 25 generations of ES (population 15, mirrored -> 30 evals/gen, sigma
0.5, lr 0.3): **0.900**, converged by around generation 3-4 and stable
afterward (no collapse/overfitting across the remaining ~20 generations).

This is real evidence the training mechanism works on the actual
connectome, not just in principle - directly following up on
`connectome/README.md`'s finding that the real graph's persistent
"reservoir" dynamics carry genuine, multi-dimensional, input-specific
structure (participation ratio ~5.3 across 15 conditions): here, a trained
readout demonstrably *uses* that structure to do a concrete task better
than random weights do.

## Live reward, against the real game — implemented and run

`live_rollout.py` + `multi_bridge.py` + `train_live.py` are the live phase:
real Minecraft episodes, not the proxy task. Important constraint the user
was explicit about: **reward must be earned by the bot's own actions only**
— no admin commands, no handed-out items, no teleports during training.
`run_episode()` in `live_rollout.py` only ever reads observations and
sends normal player-equivalent actions through bot-bridge; the fitness
score is survival + health retained + distance traveled + resources
actually gathered (measured as an inventory count increase over the
episode), nothing else.

`multi_bridge.py` launches N separate bot-bridge processes (each its own
port + username, e.g. `FlyBrain0`, `FlyBrain1`, `FlyBrain2`), all joining
the same running server — genuinely concurrent bots, not a simulated
population. `run_es()` in `train_interface.py` was extended with an
optional `batch_fitness_fn` (a whole generation's perturbed weight-sets at
once, in addition to the original one-at-a-time `fitness_fn`) so
`train_live.py` can dispatch a generation's population across the N live
bots via a thread pool — many fly brains playing at once, all of their
real, separately-earned outcomes combining into one shared weight update.

**Real result** (`N_PARALLEL_BOTS=3`, 5 generations, population 4): fitness
**2.413 → 5.139**, improving by generation 2 and holding stable afterward.
Saved to `data/trained_interface_live.npz` (separate from
`trained_interface.npz`, which is still the proxy-task result — this
live-trained one hasn't had nearly enough generations/population to be
assumed better yet).

## What's honestly still true

This was a small, first end-to-end validation that live parallel trial-
and-error training works at all — not a training run anywhere near
dragon-killing competence. Real Minecraft ticks can't be sped up, so each
episode costs real wall-clock time regardless of how many bots run in
parallel; getting meaningfully further needs far more generations/
population/parallel bots than fits in one interactive session, most
realistically as a long unattended run. Scale `N_PARALLEL_BOTS` and
`ESConfig`'s `generations`/`population_size` in `train_live.py` up once
ready to commit to that.

See [../../docs/architecture.md](../../docs/architecture.md).
