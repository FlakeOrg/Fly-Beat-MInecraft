"""The real-time control loop (M4): observation -> encoder -> sim step(s) ->
decoder -> action, tying bot-bridge to the LIF network.

Still runs on the *synthetic* stand-in graph (sim/synthetic.py), not the
real hemibrain connectome - that swap happens once a NEUPRINT_TOKEN is
available (see connectome/fetch_hemibrain.py) and its own syn_scale/
b_adapt/tau_adapt sweep has been done against the real weight distribution
(see connectome/README.md). This milestone validates plumbing end to end
(a real observation can drive the network and the network's output can
drive the bot), not intelligence - the hand-built encoder/decoder weights
have no reason to produce good Minecraft play yet; that's M5 (trained
weights) and M6 (a real task manager on top).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import SensoryEncoder  # noqa: E402
from sim.lif import LIFNetwork, LIFParams  # noqa: E402
from sim.synthetic import make_synthetic_graph  # noqa: E402

# Same tuned regime as sim/demo.py - see its comments for how these were found.
SIM_TICKS_PER_ACTION = 10
SYN_SCALE = 4.0
B_ADAPT = 0.4
TAU_ADAPT = 40.0
EXT_CURRENT_GAIN = 3.0


def build_brain(n_neurons: int = 200, seed: int = 0) -> tuple[LIFNetwork, np.ndarray, np.ndarray]:
    weights, input_idx, output_idx = make_synthetic_graph(
        n_neurons=n_neurons,
        p_connect=0.08,
        weight_scale=1.0,
        inhibitory_frac=0.2,
        inhibitory_strength=2.0,
        seed=seed,
    )
    net = LIFNetwork(weights, LIFParams(syn_scale=SYN_SCALE, b_adapt=B_ADAPT, tau_adapt=TAU_ADAPT))
    return net, input_idx, output_idx


def run(n_steps: int = 50, bridge_url: str = "ws://localhost:8081") -> None:
    net, input_idx, output_idx = build_brain()
    encoder = SensoryEncoder(input_idx, gain=EXT_CURRENT_GAIN)
    decoder = MotorDecoder(output_idx)

    with BridgeClient(bridge_url) as client:
        for step in range(n_steps):
            observation = client.get_observation()
            ext_current = encoder.encode(observation, net.n)

            spike_window = np.zeros((SIM_TICKS_PER_ACTION, net.n))
            for tick in range(SIM_TICKS_PER_ACTION):
                spike_window[tick] = net.step(ext_current)

            actions = decoder.decode(spike_window)
            move_command = {"type": "move", **{k: actions.get(k, False) for k in ("forward", "left", "right", "jump")}}
            client.do_action(move_command)

            if actions.get("attack"):
                entities = sorted(observation["nearbyEntities"], key=lambda e: e["distance"])
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

            print(f"step {step}: actions={actions}")
            time.sleep(0.1)

        client.do_action({"type": "stop"})


if __name__ == "__main__":
    run()
