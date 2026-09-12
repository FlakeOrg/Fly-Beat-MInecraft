"""Sanity tests for the LIF simulator (M3): stability + signal propagation.

These run against the synthetic stand-in graph (`sim/synthetic.py`), not the
real connectome - the point here is to validate the simulator itself, not
any particular biology. The tuned parameters (syn_scale, b_adapt, tau_adapt)
match `sim/demo.py`; see that file's comments for how they were found.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

from lif import LIFNetwork, LIFParams  # noqa: E402
from synthetic import make_synthetic_graph  # noqa: E402


@pytest.fixture
def tuned_network():
    n_neurons = 200
    weights, input_idx, output_idx = make_synthetic_graph(
        n_neurons=n_neurons,
        p_connect=0.08,
        weight_scale=1.0,
        inhibitory_frac=0.2,
        inhibitory_strength=2.0,
        seed=0,
    )
    params = LIFParams(syn_scale=4.0, b_adapt=0.4, tau_adapt=40.0)
    net = LIFNetwork(weights, params)
    return net, n_neurons, input_idx, output_idx


def _drive_trace(t_steps, n_neurons, input_idx, drive_until, amplitude=2.0):
    ext = np.zeros((t_steps, n_neurons))
    ext[:drive_until, input_idx] = amplitude
    return ext


def test_silent_network_stays_silent(tuned_network):
    net, n_neurons, _, _ = tuned_network
    ext = np.zeros((100, n_neurons))
    spikes = net.run(ext)
    assert spikes.sum() == 0


def test_no_nans_or_infs(tuned_network):
    net, n_neurons, input_idx, _ = tuned_network
    ext = _drive_trace(300, n_neurons, input_idx, drive_until=120)
    spikes = net.run(ext)
    assert np.isfinite(spikes).all()
    assert np.isfinite(net.v).all()


def test_does_not_saturate(tuned_network):
    net, n_neurons, input_idx, _ = tuned_network
    ext = _drive_trace(300, n_neurons, input_idx, drive_until=120)
    spikes = net.run(ext)
    assert spikes.mean() < 0.9, "network fired on nearly every tick - runaway/saturated"


def test_output_pool_responds_to_driven_input(tuned_network):
    net, n_neurons, input_idx, output_idx = tuned_network
    ext = _drive_trace(300, n_neurons, input_idx, drive_until=120)
    spikes = net.run(ext)
    driven_rate = spikes[40:120, output_idx].mean()
    assert driven_rate > 0.05, "output pool barely fired while input was driven"


def test_activity_decays_after_drive_removed(tuned_network):
    net, n_neurons, input_idx, output_idx = tuned_network
    ext = _drive_trace(300, n_neurons, input_idx, drive_until=120)
    spikes = net.run(ext)
    after_rate = spikes[220:300, output_idx].mean()
    assert after_rate < 0.02, "output pool kept firing long after drive was cut - self-sustaining runaway"


def test_rejects_non_square_weights():
    with pytest.raises(ValueError):
        LIFNetwork(np.zeros((3, 4)))


def test_rejects_wrong_shaped_current(tuned_network):
    net, n_neurons, _, _ = tuned_network
    with pytest.raises(ValueError):
        net.step(np.zeros(n_neurons + 1))
