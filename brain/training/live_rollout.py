"""Live Minecraft reward for ES training (M5, live phase).

Reward is earned purely by what the bot does in a real episode against a
real running server - no admin shortcuts, no handed-out items, no
teleports. An episode starts wherever the bot naturally already is
(including its own respawn point if it died in a previous episode) and is
scored only from observable, self-earned progress: staying alive, keeping
health, moving under its own power, and gathering resources by its own
mining.

This is deliberately a separate, slower path from
train_interface.py's evaluate_situations() proxy task - a real episode
takes real wall-clock Minecraft ticks (no speeding that up), so this
exists for validating the training mechanism against genuine survival
reward, not for the bulk of iteration during algorithm development.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.loop import LIF_PARAMS, SIM_TICKS_PER_ACTION  # noqa: E402
from agent.task_manager import TaskManager  # noqa: E402
from interface.motor_decoder import MotorDecoder  # noqa: E402
from interface.sensory_encoder import SensoryEncoder  # noqa: E402
from sim.lif import LIFNetwork  # noqa: E402
from training.train_interface import unflatten_params  # noqa: E402

EPISODE_STEPS = 150
STEP_SLEEP_S = 0.1


def _total_resources(observation: dict) -> int:
    return sum(i["count"] for i in observation["inventory"])


def run_episode(
    theta: np.ndarray,
    encoder: SensoryEncoder,
    decoder: MotorDecoder,
    weights,
    bridge_url: str,
    use_task_manager: bool = True,
    episode_steps: int = EPISODE_STEPS,
) -> float:
    """Runs one real episode against a live bot-bridge instance and returns
    an earned reward. Mutates `encoder`/`decoder` in place (via
    unflatten_params) to whatever `theta` this evaluation is for.

    Never raises: a long unattended run must survive one worker's bot-
    bridge process crashing or refusing a connection without taking down
    the whole training run - such a rollout just scores 0.0 rather than
    propagating the exception.
    """
    try:
        return _run_episode_inner(theta, encoder, decoder, weights, bridge_url, use_task_manager, episode_steps)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"rollout against {bridge_url} failed ({exc!r}); scoring 0.0")
        return 0.0


def _run_episode_inner(
    theta: np.ndarray,
    encoder: SensoryEncoder,
    decoder: MotorDecoder,
    weights,
    bridge_url: str,
    use_task_manager: bool,
    episode_steps: int,
) -> float:
    unflatten_params(theta, encoder, decoder)
    net = LIFNetwork(weights, LIF_PARAMS)
    task_manager = TaskManager() if use_task_manager else None

    with BridgeClient(bridge_url) as client:
        try:
            start_obs = client.get_observation()
        except BridgeError:
            return 0.0  # bridge not ready this instant - treat as a wasted rollout, not a crash

        start_pos = start_obs["position"]
        start_resources = _total_resources(start_obs)
        max_distance = 0.0
        died = False
        final_health = start_obs["health"]

        for _ in range(episode_steps):
            try:
                observation = client.get_observation()
            except BridgeError:
                break  # kicked/disconnected mid-episode; score what was earned so far

            final_health = observation["health"]
            if final_health <= 0:
                died = True
                break

            dx = observation["position"]["x"] - start_pos["x"]
            dz = observation["position"]["z"] - start_pos["z"]
            max_distance = max(max_distance, (dx**2 + dz**2) ** 0.5)

            scripted = task_manager.decide(observation) if task_manager is not None else None
            if scripted is not None:
                for command in scripted:
                    try:
                        client.do_action(command)
                    except BridgeError:
                        pass
            else:
                ext_current = encoder.encode(observation, net.n)
                spike_window = np.zeros((SIM_TICKS_PER_ACTION, net.n))
                for tick in range(SIM_TICKS_PER_ACTION):
                    spike_window[tick] = net.step(ext_current)
                actions = decoder.decode(spike_window)
                move_command = {
                    "type": "move",
                    **{k: actions.get(k, False) for k in ("forward", "left", "right", "jump")},
                }
                try:
                    client.do_action(move_command)
                except BridgeError:
                    pass

            time.sleep(STEP_SLEEP_S)

        try:
            final_obs = client.get_observation()
            final_health = final_obs["health"]
            final_resources = _total_resources(final_obs)
        except BridgeError:
            final_resources = start_resources

        try:
            client.do_action({"type": "stop"})
        except BridgeError:
            pass

    survival_bonus = 0.0 if died else 1.0
    health_fraction = final_health / 20.0
    resources_gained = max(0, final_resources - start_resources)
    reward = survival_bonus + health_fraction + 0.1 * max_distance + 0.5 * resources_gained

    print(
        f"  episode [{bridge_url}]: reward={reward:.3f} "
        f"(died={died}, health={final_health}/20, distance={max_distance:.1f}, "
        f"resources_gained={resources_gained}, stage={task_manager.stage if task_manager else 'n/a'})"
    )
    return reward
