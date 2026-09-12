"""Launches N bot-bridge (Node/Mineflayer) processes, each a distinctly-
named bot joining the same Minecraft server, so ES population members can
be evaluated as genuinely concurrent live rollouts instead of one at a
time - many fly brains playing (and being scored) at once, each
contributing to the same shared, evolving weights.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

BOT_BRIDGE_DIR = Path(__file__).resolve().parent.parent.parent / "bot-bridge"
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "multi_bridge_logs"


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


def stop_all(instances: list[BridgeInstance]) -> None:
    for instance in instances:
        instance.stop()
