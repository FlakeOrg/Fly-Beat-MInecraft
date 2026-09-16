"""FlyBrainMain: one standalone bot running the swarm's current best
trained brain continuously - not part of ES training/population sampling,
so nothing here feeds back into trained_interface_live.npz or changes what
the actual training swarm perceives or is scored on.

Same connectome, same neuron pools, same base senses as every other bot
(agent/loop.py). The one deliberate, disclosed difference: its sensory
encoder (interface/swarm_sensory_encoder.py) gets a few extra input
channels carrying a live summary of how the rest of the training swarm
(the FlyBrainN bots launched by training/multi_bridge.py) is doing right
now - a background thread polls their bridges read-only every few seconds
for this. It cannot see individual bots' surroundings or inventories in
detail, only the aggregate: how far the swarm has gotten, whether anyone's
badly hurt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.goals import END_GOAL, TRAINING_STEPS, milestone_score, next_training_step  # noqa: E402
from agent.loop import ATTACKABLE_ENTITY_KINDS, LIF_PARAMS, SIM_TICKS_PER_ACTION, load_real_graph  # noqa: E402
from agent.movement import movement_command_from_actions  # noqa: E402
from agent.self_actions import perform  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.swarm_sensory_encoder import SwarmSensoryEncoder  # noqa: E402
from sim.lif import LIFNetwork  # noqa: E402

BOT_BRIDGE_DIR = Path(__file__).resolve().parent.parent.parent / "bot-bridge"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LIVE_WEIGHTS_PATH = DATA_DIR / "trained_interface_live.npz"
LOG_PATH = DATA_DIR / "fly_brain_main.log"

MAX_MILESTONE_SCORE = sum(step.reward for step in TRAINING_STEPS)
SWARM_POLL_INTERVAL_S = 3.0
SWARM_DANGER_HEALTH = 8.0
ATTACK_RANGE = 4.0
CONTROL_STEP_SLEEP_S = 0.1
STATUS_EVERY = 100


def launch_bridge(port: int, username: str, viewer_port: int | None = None) -> subprocess.Popen:
    """Starts FlyBrainMain's own bot-bridge process, the same way
    training/multi_bridge.py starts the numbered training bots - just a
    single instance with a fixed, human name instead of FlyBrainN."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BRIDGE_PORT"] = str(port)
    env["MC_USERNAME"] = username
    if viewer_port is not None:
        env["VIEWER_PORT"] = str(viewer_port)
    log_file = open(LOG_PATH, "w")
    return subprocess.Popen(
        ["node", "src/index.js"],
        cwd=str(BOT_BRIDGE_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def wait_until_spawned(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if LOG_PATH.exists() and "spawned" in LOG_PATH.read_text(errors="ignore"):
            return
        time.sleep(1.0)
    raise TimeoutError("FlyBrainMain's bot-bridge never reported spawned")


def summarize_swarm(observations: list[dict]) -> dict:
    """Pure aggregation step, kept separate from the polling/connection
    machinery below so it's testable without a live bridge."""
    scores = [milestone_score(obs) / MAX_MILESTONE_SCORE for obs in observations]
    low_health_count = sum(1 for obs in observations if obs["health"] <= SWARM_DANGER_HEALTH)
    connected = len(observations)
    return {
        "progress_avg": sum(scores) / len(scores) if scores else 0.0,
        "progress_best": max(scores) if scores else 0.0,
        "danger_frac": (low_health_count / connected) if connected else 0.0,
        "connected": connected,
    }


def _poll_one_swarm_bot(url: str, index: int, obs_cache: dict, cache_lock: threading.Lock) -> None:
    """One persistent connection per swarm bot, held open for as long as
    it stays reachable - reconnecting from scratch every poll cycle (the
    first version of this) forced a fresh WebSocket handshake across all
    12 bridges every few seconds, adding needless connection churn on top
    of a server that was already CPU-strained running the swarm itself
    (the same class of bug as the dashboard's iframe-reload issue)."""
    while True:
        try:
            with BridgeClient(url, timeout=5.0) as client:
                while True:
                    obs = client.get_observation()
                    with cache_lock:
                        obs_cache[index] = obs
                    time.sleep(SWARM_POLL_INTERVAL_S)
        except (BridgeError, OSError, TimeoutError):
            with cache_lock:
                obs_cache.pop(index, None)
            time.sleep(SWARM_POLL_INTERVAL_S)


def _poll_swarm(bridge_urls: list[str], summary: dict, lock: threading.Lock) -> None:
    """Launches one persistent poller thread per swarm bot, then just
    re-aggregates whatever's in the shared cache every cycle - no network
    calls happen in this loop itself."""
    obs_cache: dict = {}
    cache_lock = threading.Lock()
    for i, url in enumerate(bridge_urls):
        threading.Thread(target=_poll_one_swarm_bot, args=(url, i, obs_cache, cache_lock), daemon=True).start()

    while True:
        with cache_lock:
            observations = list(obs_cache.values())
        with lock:
            summary.update(summarize_swarm(observations))
        time.sleep(SWARM_POLL_INTERVAL_S)


def build_swarm_brain() -> tuple[LIFNetwork, SwarmSensoryEncoder, MotorDecoder]:
    weights, input_idx, output_idx = load_real_graph()
    net = LIFNetwork(weights, LIF_PARAMS)
    encoder = SwarmSensoryEncoder(input_idx, seed=0)
    decoder = MotorDecoder(output_idx, seed=0)

    if not LIVE_WEIGHTS_PATH.exists():
        print("no trained weights found - using hand-built random init")
        return net, encoder, decoder

    saved = np.load(LIVE_WEIGHTS_PATH)
    pools_match = np.array_equal(saved["input_idx"], input_idx) and np.array_equal(saved["output_idx"], output_idx)
    rows_match = pools_match and saved["encoder_weights"].shape[0] == encoder.weights.shape[0]
    base_cols = saved["encoder_weights"].shape[1] if rows_match else None

    if rows_match and base_cols is not None and base_cols <= encoder.weights.shape[1]:
        # Reuse the swarm's real trained senses for every ordinary feature
        # column; the new swarm-summary columns stay at their hand-built
        # random init - there is nothing to have trained them against yet,
        # since this bot alone carries them.
        encoder.weights[:, :base_cols] = saved["encoder_weights"]
        print(f"loaded {base_cols} trained sense columns from {LIVE_WEIGHTS_PATH}; swarm columns hand-initialized")
    else:
        print(f"ignoring {LIVE_WEIGHTS_PATH} for senses: pools/shape mismatch - using hand-built random init")

    if pools_match and saved["decoder_weights"].shape == decoder.weights.shape:
        decoder.weights = saved["decoder_weights"]
    else:
        print(f"ignoring {LIVE_WEIGHTS_PATH} for actions: pools/shape mismatch - using hand-built random init")

    return net, encoder, decoder


def run(bridge_url: str, swarm_bridge_urls: list[str], n_steps: int | None = None) -> None:
    net, encoder, decoder = build_swarm_brain()
    print(f"FlyBrainMain: connectome {net.n} neurons, watching {len(swarm_bridge_urls)} swarm bots, goal={END_GOAL}")

    summary: dict = {"progress_avg": 0.0, "progress_best": 0.0, "danger_frac": 0.0, "connected": 0}
    lock = threading.Lock()
    threading.Thread(target=_poll_swarm, args=(swarm_bridge_urls, summary, lock), daemon=True).start()

    step = 0
    with BridgeClient(bridge_url) as client:
        while n_steps is None or step < n_steps:
            try:
                observation = client.get_observation()
            except BridgeError as e:
                print(f"observation unavailable ({e}), retrying in 2s...")
                time.sleep(2.0)
                continue

            with lock:
                swarm_summary = dict(summary)

            ext_current = encoder.encode(observation, net.n, swarm_summary=swarm_summary)
            spike_window = np.zeros((SIM_TICKS_PER_ACTION, net.n))
            for tick in range(SIM_TICKS_PER_ACTION):
                spike_window[tick] = net.step(ext_current)

            actions = decoder.decode(spike_window)
            move_command = movement_command_from_actions(actions, observation, step)
            try:
                client.do_action(move_command)
            except BridgeError:
                pass

            task_action = decoder.decode_task_action(spike_window)
            if task_action is not None:
                perform(task_action, observation, client)

            if actions.get("attack"):
                attackable = sorted(
                    (e for e in observation.get("nearbyEntities", []) if e.get("kind") in ATTACKABLE_ENTITY_KINDS),
                    key=lambda e: e["distance"],
                )
                if attackable and attackable[0]["distance"] <= ATTACK_RANGE:
                    try:
                        client.do_action({"type": "attack", "entityId": attackable[0]["id"]})
                    except BridgeError:
                        pass

            if step % STATUS_EVERY == 0:
                own_next = next_training_step(observation)
                print(
                    f"[FlyBrainMain] step={step} own_next={own_next.label} "
                    f"swarm(progress_avg={swarm_summary['progress_avg']:.2f}, "
                    f"progress_best={swarm_summary['progress_best']:.2f}, "
                    f"danger_frac={swarm_summary['danger_frac']:.2f}, "
                    f"connected={swarm_summary['connected']})"
                )

            step += 1
            time.sleep(CONTROL_STEP_SLEEP_S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlyBrainMain, the swarm-aware flagship bot.")
    parser.add_argument("--port", type=int, default=8200, help="this bot's own bridge port")
    parser.add_argument("--viewer-port", type=int, default=3200, help="0 to disable its 3D spectate viewer")
    parser.add_argument("--username", type=str, default="FlyBrainMain")
    parser.add_argument("--swarm-base-port", type=int, default=8090, help="first FlyBrainN training bot's bridge port")
    parser.add_argument("--swarm-count", type=int, default=12, help="number of FlyBrainN training bots to watch")
    args = parser.parse_args()

    viewer_port = args.viewer_port or None
    process = launch_bridge(args.port, args.username, viewer_port)
    try:
        print(f"waiting for {args.username}'s bridge to spawn on port {args.port}...")
        wait_until_spawned()
        swarm_urls = [f"ws://localhost:{args.swarm_base_port + i}" for i in range(args.swarm_count)]
        run(f"ws://localhost:{args.port}", swarm_urls)
    finally:
        process.terminate()
