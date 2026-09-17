"""Evolution-strategy training of the sensory encoder + motor decoder (M5)."""

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
    population_size: int = 20
    sigma: float = 0.3
    lr: float = 0.2
    seed: int = 0


@dataclass
class ESResult:
    theta: np.ndarray
    best_fitness: float
    history: list[float] = field(default_factory=list)


def flatten_params(encoder: SensoryEncoder, decoder: MotorDecoder) -> np.ndarray:
    return np.concatenate([encoder.weights.ravel(), decoder.weights.ravel()])


def unflatten_params(theta: np.ndarray, encoder: SensoryEncoder, decoder: MotorDecoder) -> None:
    enc_size = encoder.weights.size
    encoder.weights = theta[:enc_size].reshape(encoder.weights.shape)
    decoder.weights = theta[enc_size:].reshape(decoder.weights.shape)


def run_es(theta0: np.ndarray, fitness_fn: Callable[[np.ndarray], float] | None, config: ESConfig,
           batch_fitness_fn: Callable[[list[np.ndarray]], list[float]] | None = None,
           on_generation: Callable[[int, np.ndarray, float], None] | None = None) -> ESResult:
    if (fitness_fn is None) == (batch_fitness_fn is None):
        raise ValueError("exactly one of fitness_fn or batch_fitness_fn must be given")

    def evaluate_many(thetas):
        return batch_fitness_fn(thetas) if batch_fitness_fn is not None else [fitness_fn(t) for t in thetas]

    rng = np.random.default_rng(config.seed)
    theta = theta0.copy()
    best_theta, best_fitness = theta.copy(), evaluate_many([theta])[0]
    history = [best_fitness]
    for gen in range(config.generations):
        noise = rng.standard_normal((config.population_size, theta.size))
        candidates = [candidate for i in range(config.population_size)
                      for candidate in (theta + config.sigma * noise[i], theta - config.sigma * noise[i])]
        fitnesses = np.array(evaluate_many(candidates))
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            sign = 1 if gen_best_idx % 2 == 0 else -1
            best_theta = theta + sign * config.sigma * noise[gen_best_idx // 2]
        std = fitnesses.std()
        centered = (fitnesses - fitnesses.mean()) / std if std > 1e-8 else np.zeros_like(fitnesses)
        theta += config.lr * ((centered[0::2] - centered[1::2])[:, None] * noise).mean(axis=0)
        history.append(best_fitness)
        print(f"gen {gen:3d}: best_fitness_so_far={best_fitness:.3f} gen_mean={fitnesses.mean():.3f}")
        if on_generation is not None:
            on_generation(gen, best_theta, best_fitness)
    return ESResult(theta=best_theta, best_fitness=best_fitness, history=history)


def _obs(**overrides) -> dict:
    base = {"health": 20, "food": 20, "yaw": 0.0, "onGround": True,
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "nearbyEntities": [], "nearbyBlocks": []}
    base.update(overrides)
    return base


def _target(**true_actions) -> dict[str, bool]:
    """Make a complete target so newly added speech actions are trained too."""
    return {action: bool(true_actions.get(action, False)) for action in ACTIONS}


SITUATIONS: list[tuple[dict, dict[str, bool]]] = [
    (_obs(), _target(forward=True)),
    (_obs(nearbyBlocks=[{"x": 0, "y": 0, "z": 1, "name": "stone"}], inventory=[{"name": "dirt", "count": 4}]), _target(place_ahead=True)),
    (_obs(nearbyBlocks=[{"x": 0, "y": 0, "z": 1, "name": "stone"}]), _target(mine_ahead=True)),
    (_obs(nearbyEntities=[{"name": "zombie", "kind": "Hostile mobs", "distance": 3.0}]), _target(attack=True, say_help=True)),
    (_obs(onGround=False, velocity={"x": 0.0, "y": -0.5, "z": 0.0}), _target()),
    (_obs(food=2), _target(forward=True, say_hungry=True)),
    (_obs(health=4), _target(say_hurt=True)),
]


def evaluate_situations(theta: np.ndarray, encoder: SensoryEncoder, decoder: MotorDecoder, weights,
                        lif_params: LIFParams, sim_ticks: int = 10) -> float:
    unflatten_params(theta, encoder, decoder)
    correct = total = 0
    for observation, target in SITUATIONS:
        net = LIFNetwork(weights, lif_params)
        ext_current = encoder.encode(observation, net.n)
        spike_window = np.zeros((sim_ticks, net.n))
        for tick in range(sim_ticks):
            spike_window[tick] = net.step(ext_current)
        predicted = decoder.decode(spike_window)
        for action in ACTIONS:
            total += 1
            correct += predicted[action] == target[action]
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
    result = run_es(theta0, lambda theta: evaluate_situations(theta, encoder, decoder, norm_weights, lif_params), ESConfig(generations=25, population_size=15, sigma=0.5, lr=0.3, seed=0))
    unflatten_params(result.theta, encoder, decoder)
    out_path = data_dir.parent / "trained_interface.npz"
    np.savez(out_path, encoder_weights=encoder.weights, decoder_weights=decoder.weights, input_idx=input_idx, output_idx=output_idx)
    print(f"saved trained weights to {out_path}")
