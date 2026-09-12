"""One-time setup: registers the generated fly skins (assets/fly_skins/)
as real, signed SkinsRestorer custom skins, so bots can `/skin set <name>`
for themselves with no special permission afterwards (see
training/multi_bridge.py's assign_fly_skins()).

Why this is a whole script and not just an RCON command: SkinsRestorer's
`/skin` and `/sr` command families silently no-op for any non-Player
sender (verified empirically - console/RCON gets no error, no output, no
state change, regardless of the skin name's validity or whether the
target has ever joined). So registering a *new* custom skin (`/sr
createcustom`, which needs elevated permission) has to be run as a real,
temporarily-OPed bot via its own chat, not from the console. Run this
once after the skin PNGs exist at their public
raw.githubusercontent.com URL (i.e. after they're committed and pushed).

Usage: run from brain/, with a Minecraft server (RCON enabled) already up
and bot-bridge dependencies installed:
    .venv/Scripts/python.exe ../scripts/apply_fly_skins.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from mcrcon import MCRcon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain"))

from agent.bridge_client import BridgeClient  # noqa: E402
from training.multi_bridge import FLY_SKIN_VARIANTS  # noqa: E402

REPO_RAW_BASE = "https://raw.githubusercontent.com/guruchamp-vol2/Fly-Beat-MInecraft/main/assets/fly_skins"
BOT_BRIDGE_DIR = Path(__file__).resolve().parent.parent / "bot-bridge"
RCON_HOST = "localhost"
RCON_PORT = 25575
ADMIN_USERNAME = "SkinAdmin"
ADMIN_PORT = 8199


def main(rcon_password: str) -> None:
    log_path = Path(__file__).resolve().parent.parent / "data" / "skin_admin_bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            ["node", "src/index.js"],
            cwd=str(BOT_BRIDGE_DIR),
            env={"BRIDGE_PORT": str(ADMIN_PORT), "MC_USERNAME": ADMIN_USERNAME, **_os_environ()},
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.time() + 30
        while "spawned" not in log_path.read_text(errors="ignore") and time.time() < deadline:
            time.sleep(1.0)

        with MCRcon(RCON_HOST, rcon_password, port=RCON_PORT) as mcr:
            mcr.command(f"op {ADMIN_USERNAME}")

        with BridgeClient(f"ws://localhost:{ADMIN_PORT}") as client:
            for name in FLY_SKIN_VARIANTS:
                cmd = f"/sr createcustom {name} {REPO_RAW_BASE}/{name}.png classic"
                print("sending:", cmd)
                client.do_action({"type": "chat", "message": cmd})
                time.sleep(3.0)  # give each MineSkin/Mojang signing round-trip time to land

        with MCRcon(RCON_HOST, rcon_password, port=RCON_PORT) as mcr:
            mcr.command(f"deop {ADMIN_USERNAME}")
        print("done - skins registered, SkinAdmin de-opped")
    finally:
        process.terminate()


def _os_environ() -> dict:
    import os

    return dict(os.environ)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: apply_fly_skins.py <rcon_password>")
        sys.exit(1)
    main(sys.argv[1])
