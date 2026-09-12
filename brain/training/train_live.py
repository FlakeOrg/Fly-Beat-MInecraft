"""Live-reward ES training (M5, live phase): trains the encoder/decoder
against real Minecraft episodes played through N parallel bot-bridge
instances - genuine trial-and-error learning against the real game, no
admin shortcuts, no handed-out items. Many "fly brains" (the same shared
weights, perturbed differently per ES population member) play
concurrently each generation, and all of their real, earned outcomes
combine into one update - see live_rollout.py for exactly what "earned"
means and multi_bridge.py for how the parallel bots are launched.

Each real episode costs real, un-speed-up-able wall-clock Minecraft ticks,
so this is deliberately much smaller in scale than train_interface.py's
offline proxy-task run (25 generations x 30 evals, seconds total) - this
is a first, honest end-to-end validation that live parallel trial-and-
error training actually works, not a full training run to game-beating
competence. Increase N_PARALLEL_BOTS / ESConfig generations/population
once this is confirmed working, ideally left running unattended for far
longer than one interactive session.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import load_real_graph  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import SensoryEncoder  # noqa: E402
from training.live_rollout import run_episode  # noqa: E402
from training.multi_bridge import launch_bridges, stop_all, wait_until_all_spawned  # noqa: E402
from training.train_interface import ESConfig, flatten_params, run_es, unflatten_params  # noqa: E402

N_PARALLEL_BOTS = 3
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUT_PATH = DATA_DIR / "trained_interface_live.npz"


def main() -> None:
    weights, input_idx, output_idx = load_real_graph()
    seed_encoder = SensoryEncoder(input_idx, seed=0)
    seed_decoder = MotorDecoder(output_idx, seed=0)
    theta0 = flatten_params(seed_encoder, seed_decoder)

    print(f"launching {N_PARALLEL_BOTS} parallel bot-bridge instances...")
    bridges = launch_bridges(N_PARALLEL_BOTS)
    try:
        wait_until_all_spawned(bridges)
        print("all bridges spawned:", [b.username for b in bridges])
        time.sleep(2.0)  # let each bot fully settle into the world before scoring starts

        # One encoder/decoder pair per worker slot so concurrent rollouts
        # (each writing its assigned theta into them via run_episode's
        # unflatten_params call) never clobber each other mid-evaluation.
        worker_encoders = [SensoryEncoder(input_idx, seed=0) for _ in bridges]
        worker_decoders = [MotorDecoder(output_idx, seed=0) for _ in bridges]

        def batch_fitness_fn(thetas: list[np.ndarray]) -> list[float]:
            results = [0.0] * len(thetas)
            with ThreadPoolExecutor(max_workers=len(bridges)) as pool:
                futures = {}
                for i, theta in enumerate(thetas):
                    worker = i % len(bridges)
                    future = pool.submit(
                        run_episode,
                        theta,
                        worker_encoders[worker],
                        worker_decoders[worker],
                        weights,
                        bridges[worker].url,
                    )
                    futures[future] = i
                for future in futures:
                    results[futures[future]] = future.result()
            return results

        result = run_es(
            theta0,
            fitness_fn=None,
            config=ESConfig(generations=5, population_size=4, sigma=0.5, lr=0.3, seed=0),
            batch_fitness_fn=batch_fitness_fn,
        )

        print(f"\nfinal live-trained fitness (best seen): {result.best_fitness:.3f}")
        print(f"fitness history: {[round(f, 3) for f in result.history]}")

        final_encoder = SensoryEncoder(input_idx, seed=0)
        final_decoder = MotorDecoder(output_idx, seed=0)
        unflatten_params(result.theta, final_encoder, final_decoder)
        np.savez(
            OUT_PATH,
            encoder_weights=final_encoder.weights,
            decoder_weights=final_decoder.weights,
            input_idx=input_idx,
            output_idx=output_idx,
        )
        print(f"saved live-trained weights to {OUT_PATH}")

    finally:
        print("stopping bot-bridge instances...")
        stop_all(bridges)


if __name__ == "__main__":
    main()
