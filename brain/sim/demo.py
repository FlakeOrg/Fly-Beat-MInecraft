"""Manual sanity-check script for the LIF simulator (M3).

Run with:  python demo.py   (from brain/sim/, or `python -m sim.demo` from brain/)

Drives a synthetic graph's input pool with constant current for a while,
then cuts the drive, and prints per-tick firing rates for the input pool,
the rest of the network, and the output pool. What "plausible" looks like
here: the input pool fires reliably while driven, the output pool starts
near zero and rises after a lag (signal propagation takes a few ticks), and
nothing blows up (all neurons firing every tick) or flatlines everywhere.
"""

from __future__ import annotations

import numpy as np

from lif import LIFNetwork, LIFParams
from synthetic import make_synthetic_graph


def main() -> None:
    n_neurons = 200
    weights, input_idx, output_idx = make_synthetic_graph(
        n_neurons=n_neurons, p_connect=0.08, weight_scale=1.0, seed=0
    )
    other_idx = np.setdiff1d(np.arange(n_neurons), np.concatenate([input_idx, output_idx]))

    # syn_scale=4.0 was tuned by sweeping against this synthetic graph's
    # weight scale (see git history / demo.py) to land in the regime where
    # firing tracks the input pool instead of either staying silent or
    # locking into permanent self-sustained activity - a different weight
    # scale (e.g. the real hemibrain's raw synapse counts) will need its own
    # sweep, this number isn't a universal constant.
    net = LIFNetwork(weights, LIFParams(syn_scale=4.0, b_adapt=0.4, tau_adapt=40.0))

    t_steps = 300
    drive_until = 120
    ext_current = np.zeros((t_steps, n_neurons))
    ext_current[:drive_until, input_idx] = 2.0

    spikes = net.run(ext_current)

    print(f"{'tick':>5} {'input rate':>11} {'other rate':>11} {'output rate':>12}")
    window = 10
    for start in range(0, t_steps, window):
        end = min(start + window, t_steps)
        chunk = spikes[start:end]
        input_rate = chunk[:, input_idx].mean()
        other_rate = chunk[:, other_idx].mean()
        output_rate = chunk[:, output_idx].mean()
        marker = " <- drive cut" if start == drive_until else ""
        print(f"{start:>5} {input_rate:>11.3f} {other_rate:>11.3f} {output_rate:>12.3f}{marker}")

    driven_rate = spikes[40:drive_until, output_idx].mean()
    after_rate = spikes[drive_until + 100 :, output_idx].mean()

    if np.isnan(spikes).any():
        print("\nFAIL: NaNs in spike trace")
    elif spikes.mean() > 0.9:
        print("\nFAIL: network is saturated (near-constant firing)")
    elif driven_rate == 0.0:
        print("\nFAIL: output pool never fired while input was driven - signal isn't propagating")
    elif after_rate > 0.05:
        print("\nFAIL: output pool still firing well after drive was cut - self-sustaining runaway")
    else:
        print("\nOK: stable, output pool responds to the driven input, and quiets back down after.")


if __name__ == "__main__":
    main()
