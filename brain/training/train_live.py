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

import argparse
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
from training.multi_bridge import (  # noqa: E402
    Shard,
    assign_fly_skins,
    launch_bridges,
    launch_shards,
    stop_all,
    wait_until_all_spawned,
)
from training.train_interface import ESConfig, flatten_params, run_es, unflatten_params  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUT_PATH = DATA_DIR / "trained_interface_live.npz"
PROXY_WEIGHTS_PATH = DATA_DIR / "trained_interface.npz"
USE_TASK_MANAGER = False


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


def load_initial_theta(input_idx: np.ndarray, output_idx: np.ndarray, source: str) -> np.ndarray:
    encoder = SensoryEncoder(input_idx, seed=0)
    decoder = MotorDecoder(output_idx, seed=0)

    paths = {
        "live": OUT_PATH,
        "proxy": PROXY_WEIGHTS_PATH,
    }
    path = paths.get(source)
    if path is not None and path.exists():
        saved = np.load(path)
        pools_match = np.array_equal(saved["input_idx"], input_idx) and np.array_equal(saved["output_idx"], output_idx)
        # The saved weights also have to match the *current* feature and
        # action counts. Both changed when the bot gained inventory senses
        # and craft verbs, so older checkpoints are the wrong shape - load
        # them blindly and training dies on a matmul instead of starting.
        shapes_match = (
            saved["encoder_weights"].shape == encoder.weights.shape
            and saved["decoder_weights"].shape == decoder.weights.shape
        )
        if pools_match and shapes_match:
            encoder.weights = saved["encoder_weights"]
            decoder.weights = saved["decoder_weights"]
            print(f"starting from {source} weights: {path}")
            return flatten_params(encoder, decoder)
        reason = "neuron pools" if not pools_match else "feature/action layout"
        print(f"ignoring {path}: {reason} does not match current interface - starting fresh")

    if source == "live":
        return load_initial_theta(input_idx, output_idx, "proxy")

    print("starting from default seeded interface weights")
    return flatten_params(encoder, decoder)


def main(
    n_parallel_bots: int = 20,
    generations: int = 200,
    population_size: int = 10,
    episode_steps: int = 150,
    init: str = "live",
    assign_skins: bool = True,
    enable_viewers: bool = False,
    base_viewer_port: int = 3100,
    shard_ports: list[int] | None = None,
) -> None:
    """`shard_ports`, if given, spreads `n_parallel_bots` bots per port
    across that many independent Minecraft server instances (see
    multi_bridge.Shard) instead of putting all of them on one server at
    the default port - see training/README.md (or just ask) for why:
    one server's tick loop is single-threaded, so this is what actually
    lets a training run scale past what one instance alone can sustain."""
    weights, input_idx, output_idx = load_real_graph()
    theta0 = load_initial_theta(input_idx, output_idx, init)

    if shard_ports:
        total_bots = n_parallel_bots * len(shard_ports)
        print(
            f"launching {n_parallel_bots} bots on each of {len(shard_ports)} shards "
            f"({total_bots} total) (generations={generations}, population_size={population_size}, "
            f"episode_steps={episode_steps})..."
        )
        bridges = launch_shards(
            [Shard(port=port, bots=n_parallel_bots) for port in shard_ports],
            base_viewer_port=base_viewer_port if enable_viewers else None,
        )
    else:
        print(
            f"launching {n_parallel_bots} self-training bot-bridge instances "
            f"(generations={generations}, population_size={population_size}, episode_steps={episode_steps})..."
        )
        bridges = launch_bridges(n_parallel_bots, base_viewer_port=base_viewer_port if enable_viewers else None)
    if enable_viewers:
        print(f"3D spectate viewers enabled starting at port {base_viewer_port} - run dashboard/server.py to watch them all in one page")
    try:
        wait_until_all_spawned(bridges)
        print("all bridges spawned:", [b.username for b in bridges])
        time.sleep(2.0)  # let each bot fully settle into the world before scoring starts
        if assign_skins:
            assign_fly_skins(bridges)

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
                        theta,
                        worker_encoders[worker_idx],
                        worker_decoders[worker_idx],
                        weights,
                        bridges[worker_idx].url,
                        use_task_manager=USE_TASK_MANAGER,
                        episode_steps=episode_steps,
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
            config=ESConfig(generations=generations, population_size=population_size, sigma=0.5, lr=0.3, seed=0),
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
    parser = argparse.ArgumentParser(description="Train the fly-brain interface against live Minecraft rollouts.")
    parser.add_argument("--bots", type=int, default=20, help="number of parallel Mineflayer bots to launch")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=10, help="mirrored ES pairs; each generation evaluates 2x this many policies")
    parser.add_argument("--episode-steps", type=int, default=150)
    parser.add_argument("--init", choices=["live", "proxy", "default"], default="live")
    parser.add_argument("--no-skins", action="store_true", help="skip cosmetic skin assignment")
    parser.add_argument(
        "--viewers", action="store_true", help="start a prismarine-viewer 3D spectate server per bot (see dashboard/server.py)"
    )
    parser.add_argument("--base-viewer-port", type=int, default=3100)
    parser.add_argument(
        "--shard-ports",
        type=str,
        default=None,
        help="comma-separated Minecraft server ports, one per shard (e.g. 25565,25566) - "
        "--bots then means bots PER shard, not total. Omit to use a single server as before.",
    )
    args = parser.parse_args()
    shard_ports = [int(p) for p in args.shard_ports.split(",")] if args.shard_ports else None
    main(
        n_parallel_bots=args.bots,
        generations=args.generations,
        population_size=args.population_size,
        episode_steps=args.episode_steps,
        init=args.init,
        assign_skins=not args.no_skins,
        enable_viewers=args.viewers,
        base_viewer_port=args.base_viewer_port,
        shard_ports=shard_ports,
    )
