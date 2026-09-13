"""The real-time control loop: observation -> encoder -> sim step(s) ->
decoder -> action, tying bot-bridge to the LIF network.

Runs on the real hemibrain connectome (connectome/fetch_hemibrain.py +
graph.py) with the tuned regime found while wiring it up - see
connectome/README.md for how syn_scale/b_adapt/tau_adapt/depression_frac
were chosen, and for the honest caveat that this network settles into
persistent activity rather than a clean reflex arc (still carries real,
per-input-distinguishable structure - see the participation-ratio finding
there, and training/README.md's proxy-task result).

If `data/trained_interface.npz` exists (written by
training/train_interface.py), the encoder/decoder use those ES-trained
weights instead of the hand-built random init - use whichever the last
training run produced.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.goals import END_GOAL, next_training_step  # noqa: E402
from agent.movement import movement_command_from_actions  # noqa: E402
from agent.self_actions import perform  # noqa: E402
from agent.task_manager import TaskManager  # noqa: E402
from connectome.graph import build_adjacency, identify_pools  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import SensoryEncoder  # noqa: E402
from sim.lif import LIFNetwork, LIFParams  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TRAINED_WEIGHTS_PATH = DATA_DIR / "trained_interface.npz"

# Tuned against the real connectome - see connectome/README.md for how
# these were found and what "tuned" means here (stable + input-sensitive,
# not "behaves like a simple reflex arc" - it doesn't, at this scale).
SIM_TICKS_PER_ACTION = 10
LIF_PARAMS = LIFParams(syn_scale=1500.0, b_adapt=0.4, tau_adapt=40.0, depression_frac=0.5, tau_depression=30.0)
INPUT_TYPE_PATTERN = r"^(LC|LT|LPLC|MeTu)"
OUTPUT_TYPE_PATTERN = r"^DN"

# mineflayer's entity "kind" for things that can actually be attacked -
# excludes dropped items, projectiles, etc. (see the comment where this is
# used: attacking one of those gets the bot kicked for "attacking an
# invalid entity").
ATTACKABLE_ENTITY_KINDS = {"Hostile mobs", "Passive mobs", "Player", "Water creature", "Ambient mobs"}


def load_real_graph():
    """Loads the real, degree-normalized hemibrain adjacency + pools.
    Shared with training/live_rollout.py - the same graph the live control
    loop uses is what gets trained against."""
    neurons_df = pd.read_parquet(DATA_DIR / "hemibrain" / "neurons.parquet")
    conn_df = pd.read_parquet(DATA_DIR / "hemibrain" / "connections.parquet")
    weights, neuron_meta = build_adjacency(neurons_df, conn_df)

    in_strength = np.abs(weights).sum(axis=1).A.flatten()
    in_strength[in_strength == 0] = 1.0
    norm_weights = weights.multiply(1.0 / in_strength[:, None]).tocsr()

    input_idx, output_idx = identify_pools(neuron_meta, INPUT_TYPE_PATTERN, OUTPUT_TYPE_PATTERN)
    return norm_weights, input_idx, output_idx


def build_brain() -> tuple[LIFNetwork, SensoryEncoder, MotorDecoder]:
    weights, input_idx, output_idx = load_real_graph()
    net = LIFNetwork(weights, LIF_PARAMS)

    probe_encoder, probe_decoder = SensoryEncoder(input_idx), MotorDecoder(output_idx)
    saved = np.load(TRAINED_WEIGHTS_PATH) if TRAINED_WEIGHTS_PATH.exists() else None
    usable = saved is not None and (
        np.array_equal(saved["input_idx"], input_idx)
        and np.array_equal(saved["output_idx"], output_idx)
        # Feature and action counts change as the bot gains new senses and
        # verbs; an older checkpoint is then simply the wrong shape.
        and saved["encoder_weights"].shape == probe_encoder.weights.shape
        and saved["decoder_weights"].shape == probe_decoder.weights.shape
    )
    if usable:
        encoder = SensoryEncoder(input_idx, weights=saved["encoder_weights"])
        decoder = MotorDecoder(output_idx, weights=saved["decoder_weights"])
        print(f"loaded trained weights from {TRAINED_WEIGHTS_PATH}")
    elif saved is not None:
        encoder, decoder = probe_encoder, probe_decoder
        print(f"ignoring {TRAINED_WEIGHTS_PATH}: does not match the current interface - using fresh init")
    else:
        encoder = SensoryEncoder(input_idx)
        decoder = MotorDecoder(output_idx)
        print("no trained weights found - using hand-built random init")

    return net, encoder, decoder


def run(
    n_steps: int | None = None,
    bridge_url: str = "ws://localhost:8081",
    status_every: int = 10,
    use_task_manager: bool = True,
) -> None:
    """Runs the control loop. `n_steps=None` runs until interrupted (Ctrl+C).

    Each step, the task manager gets first say (task_manager.py): if it has
    a specific scripted subgoal action (walk to this log and mine it, craft
    this item, flee that threat), that's what runs. Only when it has
    nothing specific to do does control fall through to the fly-brain's
    trained reflexes - that's the intended division of labor, not a
    fallback of convenience.
    """
    net, encoder, decoder = build_brain()
    task_manager = TaskManager() if use_task_manager else None
    mode = "assisted" if task_manager is not None else "self"
    print(f"connectome: {net.n} neurons, {len(encoder.input_idx)} input, {len(decoder.output_idx)} output")
    print(f"mode={mode} goal={END_GOAL}")

    step = 0
    with BridgeClient(bridge_url) as client:
        try:
            while n_steps is None or step < n_steps:
                try:
                    observation = client.get_observation()
                except BridgeError as e:
                    # bot-bridge reconnects on its own after a kick/disconnect
                    # (e.g. a stale entity ID at the moment of an attack) -
                    # wait it out instead of crashing the whole control loop.
                    print(f"observation unavailable ({e}), retrying in 2s...")
                    time.sleep(2.0)
                    continue

                handled = task_manager.step(observation, client) if task_manager is not None else False
                if handled:
                    actions = {"scripted": task_manager.status}
                else:
                    ext_current = encoder.encode(observation, net.n)

                    spike_window = np.zeros((SIM_TICKS_PER_ACTION, net.n))
                    for tick in range(SIM_TICKS_PER_ACTION):
                        spike_window[tick] = net.step(ext_current)

                    actions = decoder.decode(spike_window)
                    move_command = movement_command_from_actions(actions, observation, step)
                    if not any(actions.values()):
                        actions = {**actions, "explore": True}
                    try:
                        client.do_action(move_command)
                    except BridgeError:
                        pass  # bot may have just been kicked/disconnected; next loop's get_observation will wait it out

                    if actions.get("attack"):
                        # Found live: picking the nearest entity regardless
                        # of kind could target a dropped item, which the
                        # server rejects as "attacking an invalid entity"
                        # and kicks the bot for - restrict to attackable kinds.
                        attackable = [e for e in observation["nearbyEntities"] if e.get("kind") in ATTACKABLE_ENTITY_KINDS]
                        entities = sorted(attackable, key=lambda e: e["distance"])
                        if entities and entities[0]["distance"] <= 4.0:
                            try:
                                client.do_action({"type": "attack", "entityId": entities[0]["id"]})
                            except BridgeError:
                                pass  # target may have died/left range between observation and action

                    if actions.get("mine_ahead"):
                        # best-effort: dig whatever's directly ahead at foot level, if anything
                        ahead = [b for b in observation["nearbyBlocks"] if b["y"] == 0 and abs(b["x"]) + abs(b["z"]) == 1]
                        if ahead:
                            pos = observation["position"]
                            b = ahead[0]
                            try:
                                client.do_action({"type": "dig", "x": pos["x"] + b["x"], "y": pos["y"], "z": pos["z"] + b["z"]})
                            except BridgeError:
                                pass  # dig can fail for lots of legitimate reasons (out of reach, unbreakable, etc.)

                    # Whichever craft/place/smelt verb the network asked for
                    # most strongly, if any. Nothing here decides what it
                    # ought to be doing - see agent/self_actions.py.
                    task_action = decoder.decode_task_action(spike_window)
                    if task_action is not None:
                        performed = perform(task_action, observation, client)
                        actions = {**actions, "task": f"{task_action}{'' if performed else ' (failed)'}"}

                if step % status_every == 0:
                    pos = observation["position"]
                    stage = task_manager.status if task_manager is not None else f"self: next={next_training_step(observation).label}"
                    print(
                        f"step {step}: stage={stage} pos=({pos['x']:.1f},{pos['y']:.1f},{pos['z']:.1f}) "
                        f"health={observation['health']} actions={actions}"
                    )
                step += 1
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("interrupted")
        finally:
            client.do_action({"type": "stop"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly-brain Minecraft control loop.")
    parser.add_argument("--bridge-url", default="ws://localhost:8081")
    parser.add_argument("--steps", type=int, default=None, help="number of control steps to run; default runs forever")
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument(
        "--self",
        action="store_true",
        help="turn off scripted TaskManager help; the bot only gets observations, actions, and learned weights",
    )
    args = parser.parse_args()
    run(args.steps, args.bridge_url, args.status_every, use_task_manager=not args.self)
