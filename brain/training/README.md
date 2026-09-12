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

## What's not yet done

This is still a proxy task with 5 hand-specified situations, not real
Minecraft reward. The natural next step is a `fitness_fn` that runs an
actual episode through `agent/loop.py` + bot-bridge and returns something
like survival time / distance traveled / resources gathered - much slower
per evaluation (real wall-clock Minecraft ticks, one server), so it'll need
either a much smaller population/generation count, parallelizing rollouts
across multiple local server instances, or both.

See [../../docs/architecture.md](../../docs/architecture.md).
