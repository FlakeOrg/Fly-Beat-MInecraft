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

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.task_manager import TaskManager  # noqa: E402
from connectome.graph import build_adjacency, identify_pools  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import PLACEABLE_ITEM_NAMES, SensoryEncoder, _facing_offset  # noqa: E402
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

    if TRAINED_WEIGHTS_PATH.exists():
        saved = np.load(TRAINED_WEIGHTS_PATH)
        encoder = SensoryEncoder(input_idx)
        decoder = MotorDecoder(output_idx)
        compatible = (
            np.array_equal(saved["input_idx"], input_idx)
            and np.array_equal(saved["output_idx"], output_idx)
            and saved["encoder_weights"].shape == encoder.weights.shape
            and saved["decoder_weights"].shape == decoder.weights.shape
        )
        if compatible:
            encoder.weights = saved["encoder_weights"]
            decoder.weights = saved["decoder_weights"]
            print(f"loaded trained weights from {TRAINED_WEIGHTS_PATH}")
        else:
            print("trained weights are incompatible with the current interface - using fresh init")
    else:
        encoder = SensoryEncoder(input_idx)
        decoder = MotorDecoder(output_idx)
        print("no trained weights found - using hand-built random init")

    return net, encoder, decoder


def run(n_steps: int | None = None, bridge_url: str = "ws://localhost:8081", status_every: int = 10) -> None:
    """Runs the control loop. `n_steps=None` runs until interrupted (Ctrl+C).

    Each step, the task manager gets first say (task_manager.py): it may
    issue a short survival override for imminent danger, but it never
    imposes a scripted objective or game plan. When there is no urgent
    danger, control falls through to the fly-brain and the reward-driven
    policy.
    """
    net, encoder, decoder = build_brain()
    task_manager = TaskManager()
    print(f"connectome: {net.n} neurons, {len(encoder.input_idx)} input, {len(decoder.output_idx)} output")

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

                scripted_actions = task_manager.decide(observation)
                if scripted_actions is not None:
                    actions = {"scripted": task_manager.status()}
                    for command in scripted_actions:
                        try:
                            client.do_action(command)
                        except BridgeError:
                            pass  # e.g. a dig/craft that's no longer valid this tick; try again next step
                else:
                    ext_current = encoder.encode(observation, net.n)

                    spike_window = np.zeros((SIM_TICKS_PER_ACTION, net.n))
                    for tick in range(SIM_TICKS_PER_ACTION):
                        spike_window[tick] = net.step(ext_current)

                    actions = decoder.decode(spike_window)
                    move_command = {"type": "move", **{k: actions.get(k, False) for k in ("forward", "left", "right", "jump")}}
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
                        ahead = [b for b in observation["nearbyBlocks"]
                            if b["x"] == ahead_x and b["z"] == ahead_z]
                        if ahead:
                            pos = observation["position"]
                            b = ahead[0]
                            try:
                                client.do_action({"type": "dig", "x": pos["x"] + b["x"], "y": pos["y"], "z": pos["z"] + b["z"]})
                            except BridgeError:
                                pass  # dig can fail for lots of legitimate reasons (out of reach, unbreakable, etc.)

                    if actions.get("place_ahead"):
                        print("🧱 FLY WANTS TO PLACE")

                        ahead_x, ahead_z = _facing_offset(observation["yaw"])

                        ahead = [
                            b for b in observation["nearbyBlocks"]
                            if b["x"] == ahead_x and b["z"] == ahead_z
                        ]

                        items = [
                            i for i in observation.get("inventory", [])
                            if i.get("name") in PLACEABLE_ITEM_NAMES
                            and i.get("count", 0) > 0
                        ]

                        print(
                            "PLACE CHECK:",
                            "offset=", (ahead_x, ahead_z),
                            "blocks=", [(b["x"], b["y"], b["z"], b.get("name", "?")) for b in ahead],
                            "items=", [(i["name"], i["count"]) for i in items]
                        )

                        if ahead and items:
                            pos = observation["position"]
                            b = min(ahead, key=lambda block: abs(block["y"]))

                            try:
                                client.do_action({
                                    "type": "place",
                                    "x": int(np.floor(pos["x"]) + b["x"]),
                                    "y": int(np.floor(pos["y"]) + b["y"]),
                                    "z": int(np.floor(pos["z"]) + b["z"]),
                                    "face": "up",
                                    "item": items[0]["name"],
                                })
                            except BridgeError:
                                pass

                if step % status_every == 0:
                    pos = observation["position"]
                    print(
                        f"step {step}: mode={task_manager.status()} pos=({pos['x']:.1f},{pos['y']:.1f},{pos['z']:.1f}) "
                        f"health={observation['health']} actions={actions}"
                    )
                step += 1
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("interrupted")
        finally:
            client.do_action({"type": "stop"})


if __name__ == "__main__":
    run()
