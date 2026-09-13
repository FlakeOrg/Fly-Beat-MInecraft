"""Tests for the ES trainer itself (brain/training/train_interface.py).

These validate `run_es`'s optimization mechanics in isolation with a trivial
fitness function - fast, and independent of the connectome or LIF sim (the
real-connectome behavioral demo in `train_interface.py`'s `__main__` is a
much slower, separate validation - see training/README.md for that result).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import SensoryEncoder  # noqa: E402
from training.train_interface import ESConfig, flatten_params, run_es, unflatten_params  # noqa: E402


def test_batch_mode_never_evaluates_a_lone_candidate():
    """Found live: a solo baseline call before generation 0 used exactly 1 of
    N parallel bridges to evaluate theta0 alone, and with slow real-Minecraft
    episodes that left N-1 bots idle for the whole call - reported as "all
    bots are still besides 1." Every batch_fitness_fn call must request more
    than one candidate at once, so all workers always have something to do
    from the very first round."""
    call_sizes = []

    def batch_fitness_fn(thetas):
        call_sizes.append(len(thetas))
        return [float(np.sum(t)) for t in thetas]

    run_es(
        np.zeros(4),
        fitness_fn=None,
        config=ESConfig(generations=2, population_size=3, sigma=0.1, lr=0.1, seed=0),
        batch_fitness_fn=batch_fitness_fn,
    )
    assert all(size > 1 for size in call_sizes)


def test_es_improves_on_random_init_for_a_simple_target():
    rng = np.random.default_rng(0)
    target = rng.uniform(-1, 1, size=20)
    theta0 = np.zeros(20)

    def fitness_fn(theta: np.ndarray) -> float:
        return -np.sum((theta - target) ** 2)

    initial_fitness = fitness_fn(theta0)
    result = run_es(theta0, fitness_fn, ESConfig(generations=20, population_size=15, sigma=0.5, lr=0.3, seed=0))

    assert result.best_fitness > initial_fitness
    assert result.history[-1] >= result.history[0]


def test_es_history_is_monotonically_non_decreasing():
    rng = np.random.default_rng(1)
    target = rng.uniform(-1, 1, size=10)

    def fitness_fn(theta: np.ndarray) -> float:
        return -np.sum((theta - target) ** 2)

    result = run_es(np.zeros(10), fitness_fn, ESConfig(generations=10, population_size=10, sigma=0.4, lr=0.2, seed=0))
    assert all(b >= a - 1e-9 for a, b in zip(result.history, result.history[1:]))


def test_flatten_unflatten_roundtrip():
    input_idx = np.array([0, 1, 2])
    output_idx = np.array([3, 4, 5, 6])
    encoder = SensoryEncoder(input_idx, seed=0)
    decoder = MotorDecoder(output_idx, seed=0)

    theta = flatten_params(encoder, decoder)
    assert theta.shape == (encoder.weights.size + decoder.weights.size,)

    new_theta = theta + 1.0
    unflatten_params(new_theta, encoder, decoder)
    np.testing.assert_allclose(encoder.weights.ravel(), theta[: encoder.weights.size] + 1.0)
    np.testing.assert_allclose(decoder.weights.ravel(), theta[encoder.weights.size :] + 1.0)


def test_unflatten_preserves_shapes():
    input_idx = np.arange(5)
    output_idx = np.arange(6)
    encoder = SensoryEncoder(input_idx, seed=0)
    decoder = MotorDecoder(output_idx, seed=0)
    enc_shape, dec_shape = encoder.weights.shape, decoder.weights.shape

    theta = flatten_params(encoder, decoder)
    unflatten_params(theta * 2, encoder, decoder)

    assert encoder.weights.shape == enc_shape
    assert decoder.weights.shape == dec_shape
