"""Launches N bot-bridge (Node/Mineflayer) processes, each a distinctly-
named bot joining the same Minecraft server, so ES population members can
be evaluated as genuinely concurrent live rollouts instead of one at a
time - many fly brains playing (and being scored) at once, each
contributing to the same shared, evolving weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402

BOT_BRIDGE_DIR = Path(__file__).resolve().parent.parent.parent / "bot-bridge"
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "multi_bridge_logs"

# Registered once via scripts/generate_fly_skins.py + a one-time
# `/sr createcustom` run against each raw.githubusercontent.com URL (see
# assets/fly_skins/) - purely cosmetic, unrelated to training/reward.
FLY_SKIN_VARIANTS = ["housefly", "bluebottle", "greenbottle", "fruitfly", "firefly"]


class BridgeInstance:
    def __init__(self, port: int, username: str, process: subprocess.Popen, log_path: Path):
        self.port = port
        self.username = username
        self.process = process
        self.log_path = log_path
        self.url = f"ws://localhost:{port}"

    def is_spawned(self) -> bool:
        if not self.log_path.exists():
            return False
        return "spawned" in self.log_path.read_text(errors="ignore")

    def stop(self) -> None:
        self.process.terminate()


def launch_bridges(n: int, base_port: int = 8090, base_username: str = "FlyBrain") -> list[BridgeInstance]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    instances = []
    for i in range(n):
        port = base_port + i
        username = f"{base_username}{i}"
        log_path = LOG_DIR / f"bridge_{i}.log"
        env = os.environ.copy()
        env["BRIDGE_PORT"] = str(port)
        env["MC_USERNAME"] = username
        log_file = open(log_path, "w")
        process = subprocess.Popen(
            ["node", "src/index.js"],
            cwd=str(BOT_BRIDGE_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        instances.append(BridgeInstance(port, username, process, log_path))
    return instances


def wait_until_all_spawned(instances: list[BridgeInstance], timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    pending = set(range(len(instances)))
    while pending and time.time() < deadline:
        for i in list(pending):
            if instances[i].is_spawned():
                pending.discard(i)
        if pending:
            time.sleep(1.0)
    if pending:
        names = [instances[i].username for i in pending]
        raise TimeoutError(f"bridges never reported spawned: {names}")


def assign_fly_skins(instances: list[BridgeInstance]) -> None:
    """Gives each bot one of the generated fly skins, cycling through the
    variants so a run of more bots than variants still looks varied.
    Skin identity has zero effect on behavior/reward - purely cosmetic,
    sent once right after spawn via the bot's own `/skin set` chat command
    (self-skin-set needs no special permission, unlike setting another
    player's skin - see scripts/apply_fly_skins.py for why that path was
    used to register the skins in the first place)."""
    for i, instance in enumerate(instances):
        skin = FLY_SKIN_VARIANTS[i % len(FLY_SKIN_VARIANTS)]
        try:
            with BridgeClient(instance.url) as client:
                client.do_action({"type": "chat", "message": f"/skin set {skin}"})
        except BridgeError as exc:
            print(f"failed to set skin for {instance.username}: {exc}")


def stop_all(instances: list[BridgeInstance]) -> None:
    for instance in instances:
        instance.stop()
