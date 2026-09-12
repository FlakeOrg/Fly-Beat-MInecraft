"""Live-reward ES training (M5, live phase): trains the encoder/decoder
against real Minecraft episodes played through N parallel bot-bridge
instances - genuine trial-and-error learning against the real game, no
admin shortcuts, no handed-out items. Many "fly brains" (the same shared
weights, perturbed differently per ES population member) play
concurrently each generation, and all of their real, earned outcomes
combine into one update - see live_rollout.py for exactly what "earned"
means and multi_bridge.py for how the parallel bots are launched.

Each real episode costs real, un-speed-up-able wall-clock Minecraft ticks.
A first small run (3 bots, 5 generations, population 4) validated the
mechanism end to end: fitness 2.413 -> 5.139 (see training/README.md).
Current settings (20 bots, 200 generations, population 10 -> 20 mirrored
candidates per generation, one per bot with no queueing) are a genuine
long unattended run, not something to babysit interactively - checkpoints
every generation via `on_generation` (data/trained_interface_live.npz),
so it's safe to leave running and safe to interrupt.
"""

from __future__ import annotations

import queue
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

N_PARALLEL_BOTS = 20
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUT_PATH = DATA_DIR / "trained_interface_live.npz"


def save_weights(theta: np.ndarray, input_idx: np.ndarray, output_idx: np.ndarray, path: Path = OUT_PATH) -> None:
    encoder = SensoryEncoder(input_idx, seed=0)
    decoder = MotorDecoder(output_idx, seed=0)
    unflatten_params(theta, encoder, decoder)
    np.savez(
        path,
        encoder_weights=encoder.weights,
        decoder_weights=decoder.weights,
        input_idx=input_idx,
        output_idx=output_idx,
    )


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
            # A plain ThreadPoolExecutor with `worker = i % N` does NOT
            # guarantee two candidates assigned the same worker index never
            # run concurrently: as soon as any thread frees up, the pool
            # picks up the next queued task in submission order regardless
            # of which worker index it was meant for, so a fast-finishing
            # episode can free a thread that then picks up the *next*
            # candidate for a worker slot whose *previous* candidate is
            # still running - two threads would then hit the same bridge
            # and clobber the same shared encoder/decoder mid-evaluation.
            # A fixed pool of persistent per-worker loops pulling from one
            # shared queue is what actually guarantees exclusivity.
            results: list[float | None] = [None] * len(thetas)
            work_queue: queue.Queue = queue.Queue()
            for i, theta in enumerate(thetas):
                work_queue.put((i, theta))

            def worker_loop(worker_idx: int) -> None:
                while True:
                    try:
                        i, theta = work_queue.get_nowait()
                    except queue.Empty:
                        return
                    results[i] = run_episode(
                        theta, worker_encoders[worker_idx], worker_decoders[worker_idx], weights, bridges[worker_idx].url
                    )

            with ThreadPoolExecutor(max_workers=len(bridges)) as pool:
                futures = [pool.submit(worker_loop, w) for w in range(len(bridges))]
                for future in futures:
                    future.result()
            return results

        def checkpoint(gen: int, best_theta: np.ndarray, best_fitness: float) -> None:
            # A long unattended run costs real, unrecoverable wall-clock
            # time - only saving at the very end means one crash partway
            # through loses all of it. Cheap to do every generation (a few
            # hundred KB write), so no need to throttle it.
            save_weights(best_theta, input_idx, output_idx)
            print(f"checkpointed generation {gen} (best_fitness={best_fitness:.3f}) to {OUT_PATH}")

        result = run_es(
            theta0,
            fitness_fn=None,
            config=ESConfig(generations=200, population_size=10, sigma=0.5, lr=0.3, seed=0),
            batch_fitness_fn=batch_fitness_fn,
            on_generation=checkpoint,
        )

        print(f"\nfinal live-trained fitness (best seen): {result.best_fitness:.3f}")
        print(f"fitness history: {[round(f, 3) for f in result.history]}")
        save_weights(result.theta, input_idx, output_idx)
        print(f"saved final live-trained weights to {OUT_PATH}")

    finally:
        print("stopping bot-bridge instances...")
        stop_all(bridges)


if __name__ == "__main__":
    main()
