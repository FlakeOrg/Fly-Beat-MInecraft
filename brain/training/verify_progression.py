"""Live end-to-end check that the task manager can actually drive a single
bot from nothing to an iron pickaxe against a real server.

This is deliberately *not* part of training - it's the honest answer to
"can it do the thing at all", run against one bot with nothing handed to
it: no admin commands, no given items, no teleports. Everything it ends up
holding it mined, crafted or smelted itself.

Usage (from brain/, with a server up and a bridge on the given port):
    .venv/Scripts/python.exe training/verify_progression.py [steps] [ws-url]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.task_manager import TaskManager  # noqa: E402
from training.live_rollout import MILESTONE_REWARDS  # noqa: E402

STEP_SLEEP_S = 0.2


def summarize_inventory(observation: dict) -> str:
    items = sorted(observation["inventory"], key=lambda i: -i["count"])
    return ", ".join(f"{i['name']}x{i['count']}" for i in items[:10]) or "(empty)"


def main(max_steps: int, bridge_url: str) -> None:
    task_manager = TaskManager()
    seen_milestones: set[str] = set()
    started = time.time()

    with BridgeClient(bridge_url) as client:
        for step in range(max_steps):
            try:
                observation = client.get_observation()
            except BridgeError as exc:
                print(f"[{step}] observation unavailable ({exc}); waiting")
                time.sleep(2.0)
                continue

            for item in MILESTONE_REWARDS:
                if item not in seen_milestones and any(i["name"] == item for i in observation["inventory"]):
                    seen_milestones.add(item)
                    elapsed = time.time() - started
                    print(f"*** MILESTONE after {elapsed:.0f}s (step {step}): obtained {item} ***")

            if task_manager.has_goal_item(observation):
                print(f"\nGOAL REACHED: iron pickaxe obtained after {time.time() - started:.0f}s, {step} steps")
                print(f"final inventory: {summarize_inventory(observation)}")
                return

            task_manager.step(observation, client)

            if step % 5 == 0:
                pos = observation["position"]
                print(
                    f"[{step:4d}] y={pos['y']:.0f} hp={observation['health']:.0f} "
                    f"food={observation['food']} | {task_manager.status} | {summarize_inventory(observation)}"
                )
            time.sleep(STEP_SLEEP_S)

    try:
        final = client_observation(bridge_url)
        print(f"\nstopped after {max_steps} steps. milestones reached: {sorted(seen_milestones)}")
        print(f"final inventory: {summarize_inventory(final)}")
    except Exception:
        print(f"\nstopped after {max_steps} steps. milestones reached: {sorted(seen_milestones)}")


def client_observation(bridge_url: str) -> dict:
    with BridgeClient(bridge_url) as client:
        return client.get_observation()


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8199"
    main(steps, url)
