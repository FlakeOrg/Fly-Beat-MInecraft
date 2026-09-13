"""Evolution-strategy training of the sensory encoder + motor decoder (M5).

The connectome's spiking dynamics aren't differentiable end to end (discrete
spikes, and eventually a live non-differentiable Minecraft rollout), so the
encoder/decoder weights are trained with a simple OpenAI-ES-style algorithm:
sample symmetric (mirrored) Gaussian perturbations around the current
weights, evaluate each perturbed policy's fitness, and take a step in the
direction that fitness-weights those perturbations.

`fitness_fn` is deliberately a plug-in: this module doesn't know or care
whether it's scoring hand-specified situations (see `__main__` below, and
`evaluate_situations`) or a live Minecraft rollout through bot-bridge - only
that it maps a flat parameter vector to a scalar reward. Swapping in a real
rollout later means writing a new `fitness_fn`, not touching `run_es`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interface.motor_decoder import ACTIONS, MotorDecoder  # noqa: E402
from interface.sensory_encoder import FEATURE_NAMES, SensoryEncoder  # noqa: E402
from sim.lif import LIFNetwork, LIFParams  # noqa: E402


@dataclass
class ESConfig:
    generations: int = 30
    population_size: int = 20  # mirrored, so 2x this many evaluations per generation
    sigma: float = 0.3  # perturbation stddev
    lr: float = 0.2  # step size
    seed: int = 0


@dataclass
class ESResult:
    theta: np.ndarray
    best_fitness: float
    history: list[float] = field(default_factory=list)


def flatten_params(encoder: SensoryEncoder, decoder: MotorDecoder) -> np.ndarray:
    return np.concatenate([encoder.weights.ravel(), decoder.weights.ravel()])


def unflatten_params(theta: np.ndarray, encoder: SensoryEncoder, decoder: MotorDecoder) -> None:
    """Writes `theta` back into encoder/decoder.weights in place."""
    enc_size = encoder.weights.size
    encoder.weights = theta[:enc_size].reshape(encoder.weights.shape)
    decoder.weights = theta[enc_size:].reshape(decoder.weights.shape)


def run_es(
    theta0: np.ndarray,
    fitness_fn: Callable[[np.ndarray], float] | None,
    config: ESConfig,
    batch_fitness_fn: Callable[[list[np.ndarray]], list[float]] | None = None,
    on_generation: Callable[[int, np.ndarray, float], None] | None = None,
) -> ESResult:
    """`fitness_fn` evaluates one theta at a time (the simple case - see
    the proxy task below). `batch_fitness_fn`, if given, takes the whole
    list of perturbed thetas for a generation at once and returns their
    fitnesses in the same order - this is the hook for real parallelism
    (see training/live_rollout.py + multi_bridge.py): the caller can
    dispatch that list across many live bot-bridge instances concurrently
    instead of evaluating them one at a time. Exactly one of the two must
    be given.

    `on_generation(gen, best_theta_so_far, best_fitness_so_far)`, if given,
    fires after every generation - use it to checkpoint progress to disk.
    A long live-reward run costs real, unrecoverable wall-clock time; only
    saving at the very end means one crash partway through loses all of
    it.
    """
    if (fitness_fn is None) == (batch_fitness_fn is None):
        raise ValueError("exactly one of fitness_fn or batch_fitness_fn must be given")

    def evaluate_many(thetas: list[np.ndarray]) -> list[float]:
        if batch_fitness_fn is not None:
            return batch_fitness_fn(thetas)
        return [fitness_fn(t) for t in thetas]

    rng = np.random.default_rng(config.seed)
    theta = theta0.copy()

    if batch_fitness_fn is not None:
        # A solo baseline call would use exactly 1 of N parallel workers -
        # harmless for the fast in-process proxy task below, but a real
        # live rollout can now take minutes (real dig/craft/goto round
        # trips), during which N-1 bots would sit completely idle waiting
        # for this one call to finish before generation 0 even starts.
        # Found live: reported as "all bots are still besides 1." Gen 0's
        # own candidates (mirrored noise around theta0) establish a real
        # baseline anyway, using full parallelism from the first round.
        best_theta, best_fitness = theta.copy(), -np.inf
        history: list[float] = []
    else:
        best_theta, best_fitness = theta.copy(), evaluate_many([theta])[0]
        history = [best_fitness]

    for gen in range(config.generations):
        noise = rng.standard_normal((config.population_size, theta.size))
        candidates = []
        for i in range(config.population_size):
            candidates.append(theta + config.sigma * noise[i])
            candidates.append(theta - config.sigma * noise[i])
        fitnesses = np.array(evaluate_many(candidates))

        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            sign = 1 if gen_best_idx % 2 == 0 else -1
            best_theta = theta + sign * config.sigma * noise[gen_best_idx // 2]

        std = fitnesses.std()
        centered = (fitnesses - fitnesses.mean()) / std if std > 1e-8 else np.zeros_like(fitnesses)
        # combine the mirrored pair's advantage per noise direction
        pair_advantage = centered[0::2] - centered[1::2]
        gradient_estimate = (pair_advantage[:, None] * noise).mean(axis=0)
        theta = theta + config.lr * gradient_estimate

        history.append(best_fitness)
        print(f"gen {gen:3d}: best_fitness_so_far={best_fitness:.3f} gen_mean={fitnesses.mean():.3f}")
        if on_generation is not None:
            on_generation(gen, best_theta, best_fitness)

    return ESResult(theta=best_theta, best_fitness=best_fitness, history=history)


# --- Proxy validation task: hand-specified situation -> correct-action pairs.
# Real reward (a live Minecraft rollout through bot-bridge) is far too slow
# to evaluate thousands of times per training run, so this validates the
# training *mechanism* against the real connectome first: can ES actually
# learn a better mapping than random init, on this substrate, for *some*
# concrete task - before trusting it with the expensive live version.

def _obs(**overrides) -> dict:
    base = {
        "health": 20, "food": 20, "yaw": 0.0, "onGround": True,
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "nearbyEntities": [], "nearbyBlocks": [],
    }
    base.update(overrides)
    return base


SITUATIONS: list[tuple[dict, dict[str, bool]]] = [
    (
        _obs(),
        {"forward": True, "left": False, "right": False, "jump": False, "attack": False, "mine_ahead": False},
    ),
    (
        _obs(nearbyBlocks=[{"x": 0, "y": 0, "z": 1, "name": "stone"}]),
        {"forward": False, "left": False, "right": False, "jump": False, "attack": False, "mine_ahead": True},
    ),
    (
        _obs(nearbyEntities=[{"name": "zombie", "distance": 3.0}]),
        {"forward": False, "left": False, "right": False, "jump": False, "attack": True, "mine_ahead": False},
    ),
    (
        _obs(onGround=False, velocity={"x": 0.0, "y": -0.5, "z": 0.0}),
        {"forward": False, "left": False, "right": False, "jump": False, "attack": False, "mine_ahead": False},
    ),
    (
        _obs(food=2),
        {"forward": True, "left": False, "right": False, "jump": False, "attack": False, "mine_ahead": False},
    ),
]


def evaluate_situations(
    theta: np.ndarray,
    encoder: SensoryEncoder,
    decoder: MotorDecoder,
    weights,
    lif_params: LIFParams,
    sim_ticks: int = 10,
) -> float:
    """Fitness = fraction of (situation, action) pairs the network gets right."""
    unflatten_params(theta, encoder, decoder)
    correct, total = 0, 0
    for observation, target in SITUATIONS:
        net = LIFNetwork(weights, lif_params)
        ext_current = encoder.encode(observation, net.n)
        spike_window = np.zeros((sim_ticks, net.n))
        for t in range(sim_ticks):
            spike_window[t] = net.step(ext_current)
        predicted = decoder.decode(spike_window)
        # The situations below only specify the movement/physical actions.
        # Task verbs (craft/place/smelt) have no meaningful "correct" answer
        # in a fabricated situation with no world behind it, so they're not
        # scored here - this proxy task exists to validate the ES mechanism,
        # not the progression ladder (that's live_rollout.py's job).
        for action in ACTIONS:
            if action not in target:
                continue
            total += 1
            if predicted[action] == target[action]:
                correct += 1
    return correct / total


if __name__ == "__main__":
    import pandas as pd

    from connectome.graph import build_adjacency, identify_pools

    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "hemibrain"
    neurons_df = pd.read_parquet(data_dir / "neurons.parquet")
    conn_df = pd.read_parquet(data_dir / "connections.parquet")
    weights, neuron_meta = build_adjacency(neurons_df, conn_df)

    in_strength = np.abs(weights).sum(axis=1).A.flatten()
    in_strength[in_strength == 0] = 1.0
    norm_weights = weights.multiply(1.0 / in_strength[:, None]).tocsr()

    input_idx, output_idx = identify_pools(neuron_meta, r"^(LC|LT|LPLC|MeTu)", r"^DN")
    lif_params = LIFParams(syn_scale=1500, b_adapt=0.4, tau_adapt=40.0, depression_frac=0.5, tau_depression=30.0)

    encoder = SensoryEncoder(input_idx, seed=0)
    decoder = MotorDecoder(output_idx, seed=0)
    theta0 = flatten_params(encoder, decoder)

    def fitness_fn(theta: np.ndarray) -> float:
        return evaluate_situations(theta, encoder, decoder, norm_weights, lif_params)

    print(f"random-init fitness: {fitness_fn(theta0):.3f}")

    result = run_es(theta0, fitness_fn, ESConfig(generations=25, population_size=15, sigma=0.5, lr=0.3, seed=0))

    print(f"\nfinal trained fitness: {fitness_fn(result.theta):.3f} (best seen: {result.best_fitness:.3f})")

    unflatten_params(result.theta, encoder, decoder)
    out_path = data_dir.parent / "trained_interface.npz"
    np.savez(
        out_path,
        encoder_weights=encoder.weights,
        decoder_weights=decoder.weights,
        input_idx=input_idx,
        output_idx=output_idx,
    )
    print(f"saved trained weights to {out_path}")
