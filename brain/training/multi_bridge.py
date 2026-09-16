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
from dataclasses import dataclass
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
    def __init__(
        self,
        port: int,
        username: str,
        process: subprocess.Popen,
        log_path: Path,
        viewer_port: int | None = None,
    ):
        self.port = port
        self.username = username
        self.process = process
        self.log_path = log_path
        self.url = f"ws://localhost:{port}"
        self.viewer_port = viewer_port

    def is_spawned(self) -> bool:
        if not self.log_path.exists():
            return False
        return "spawned" in self.log_path.read_text(errors="ignore")

    def stop(self) -> None:
        self.process.terminate()


def launch_bridges(
    n: int,
    base_port: int = 8090,
    base_username: str = "FlyBrain",
    base_viewer_port: int | None = None,
    mc_host: str = "localhost",
    mc_port: int = 25565,
    log_prefix: str = "bridge",
    start_index: int = 0,
) -> list[BridgeInstance]:
    """`base_viewer_port`, if given, starts a prismarine-viewer 3D spectate
    server per bot at base_viewer_port+i (see dashboard/server.py) - left
    unset by default so a plain training run doesn't pay the extra
    per-bot CPU/memory cost of a viewer nobody's watching.

    `mc_host`/`mc_port` point every bot this call launches at one specific
    Minecraft server - see launch_shards() below for spreading bots across
    several independent servers instead of all of them into one.
    `log_prefix` keeps each shard's bridge logs from overwriting each
    other's identically-numbered files in data/multi_bridge_logs/.
    `start_index` offsets the FlyBrainN numbering (both the in-game
    username and this call's own port/viewer-port math), so a later
    shard's bots don't reuse the same names as an earlier one's - harmless
    for connectivity (each shard is a separate server/player namespace) but
    confusing on the shared dashboard otherwise.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    instances = []
    for i in range(n):
        port = base_port + i
        username = f"{base_username}{start_index + i}"
        log_path = LOG_DIR / f"{log_prefix}_{i}.log"
        env = os.environ.copy()
        env["BRIDGE_PORT"] = str(port)
        env["MC_USERNAME"] = username
        env["MC_HOST"] = mc_host
        env["MC_PORT"] = str(mc_port)
        viewer_port = None
        if base_viewer_port is not None:
            viewer_port = base_viewer_port + i
            env["VIEWER_PORT"] = str(viewer_port)
        log_file = open(log_path, "w")
        process = subprocess.Popen(
            ["node", "src/index.js"],
            cwd=str(BOT_BRIDGE_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        instances.append(BridgeInstance(port, username, process, log_path, viewer_port))
    return instances


@dataclass
class Shard:
    """One independent Minecraft server instance and how many bots to run
    against it. A single monolithic server's tick loop is single-threaded,
    which is what actually caps how many bots one instance can sustain
    (found live: 12 bots was already enough to push tick time to hundreds
    of ticks behind) - splitting bots across several separate server
    processes lets each use its own CPU core for ticking, instead of all
    bots competing for one thread's worth of tick budget."""

    host: str = "localhost"
    port: int = 25565
    bots: int = 12


def launch_shards(
    shards: list[Shard],
    base_username: str = "FlyBrain",
    base_bridge_port: int = 8090,
    base_viewer_port: int | None = None,
) -> list[BridgeInstance]:
    """Launches bridges across every shard and returns them as one flat
    list - callers (train_live.py's worker pool) don't need to know or
    care which physical server backs any given bridge; ES just sees N
    workers, same as the single-server case."""
    instances: list[BridgeInstance] = []
    next_bridge_port = base_bridge_port
    bot_index = 0
    for shard_i, shard in enumerate(shards):
        viewer_port = base_viewer_port + bot_index if base_viewer_port is not None else None
        shard_instances = launch_bridges(
            shard.bots,
            base_port=next_bridge_port,
            base_username=base_username,
            base_viewer_port=viewer_port,
            mc_host=shard.host,
            mc_port=shard.port,
            log_prefix=f"shard{shard_i}_bridge",
            start_index=bot_index,
        )
        instances.extend(shard_instances)
        next_bridge_port += shard.bots
        bot_index += shard.bots
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
