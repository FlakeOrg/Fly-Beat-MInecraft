"""Live spectate dashboard: one local web page showing every training bot's
3D view (via each bot-bridge's prismarine-viewer, started with
`train_live.py --viewers`) plus a live text readout of health, food,
position, inventory, and progress toward the next training-ladder step.

Purely observational - this reads bot-bridge's own `getObservation` and the
same milestone ladder used for reward (agent/goals.py), same as any other
client. It cannot issue actions and has no effect on training.

Run alongside a `--viewers`-enabled training run:
    python dashboard/server.py --bots 12
then open http://localhost:8500 in a browser on the same machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.bridge_client import BridgeClient, BridgeError  # noqa: E402
from agent.goals import milestone_score, next_training_step  # noqa: E402

POLL_INTERVAL_S = 1.0
RECONNECT_DELAY_S = 2.0

_state_lock = threading.Lock()
_state: dict[int, dict] = {}


def _aggregate_inventory(observation: dict) -> list[dict]:
    counts: dict[str, int] = {}
    for item in observation.get("inventory", []):
        counts[item["name"]] = counts.get(item["name"], 0) + item["count"]
    return [{"name": name, "count": count} for name, count in sorted(counts.items())]


def _poll_bot(index: int, bridge_port: int, viewer_port: int | None, username: str) -> None:
    url = f"ws://localhost:{bridge_port}"
    with _state_lock:
        _state[index] = {
            "index": index,
            "name": username,
            "bridge_port": bridge_port,
            "viewer_port": viewer_port,
            "connected": False,
        }

    while True:
        try:
            with BridgeClient(url) as client:
                while True:
                    try:
                        observation = client.get_observation()
                    except BridgeError:
                        # The WebSocket to bot-bridge itself is fine here -
                        # this means the bot just isn't connected/spawned
                        # yet (e.g. still joining, or mid-reconnect after a
                        # kick). Found live: tearing down and reopening the
                        # whole socket for this added pointless reconnect
                        # churn on top of a bot that was already struggling
                        # to join during a server overload. Retry on the
                        # same connection instead.
                        with _state_lock:
                            _state[index]["connected"] = False
                        time.sleep(RECONNECT_DELAY_S)
                        continue
                    step = next_training_step(observation)
                    with _state_lock:
                        _state[index].update(
                            {
                                "connected": True,
                                "health": observation.get("health"),
                                "food": observation.get("food"),
                                "position": observation.get("position"),
                                "held_item": observation.get("heldItem"),
                                "inventory": _aggregate_inventory(observation),
                                "milestone_score": milestone_score(observation),
                                "next_step": step.label,
                                "updated_at": time.time(),
                            }
                        )
                    time.sleep(POLL_INTERVAL_S)
        except OSError:
            pass  # bridge process not listening yet, or the connection itself dropped
        with _state_lock:
            _state[index]["connected"] = False
        time.sleep(RECONNECT_DELAY_S)


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Fly-Beat-Minecraft: live spectate</title>
<style>
  body { margin: 0; background: #14161a; color: #e8e8e8; font: 13px/1.4 system-ui, sans-serif; }
  h1 { font-size: 16px; margin: 0; padding: 10px 14px; background: #1c1f26; border-bottom: 1px solid #2a2e37; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 10px; padding: 10px; }
  .bot { background: #1c1f26; border: 1px solid #2a2e37; border-radius: 6px; overflow: hidden; }
  .bot.disconnected { opacity: 0.45; }
  .bot iframe { width: 100%; height: 220px; border: 0; display: block; background: #000; }
  .bot .stats { padding: 8px 10px; }
  .bot .name { font-weight: 600; }
  .bot .hp { color: #ff8a8a; }
  .bot .food { color: #d7c37a; }
  .bot .next { color: #8ec6ff; }
  .bot .inv { color: #9aa0aa; margin-top: 4px; word-break: break-word; }
  .bot .status-badge { float: right; font-size: 11px; padding: 1px 6px; border-radius: 3px; }
  .status-badge.up { background: #234f2f; color: #8ee6a3; }
  .status-badge.down { background: #4f2323; color: #e68e8e; }
  .bot.flagship { border-color: #6b5bd6; box-shadow: 0 0 0 1px #6b5bd6; }
  .bot.flagship .name::after { content: " \2605"; color: #b3a6ff; }
</style>
</head>
<body>
<h1>Fly-Beat-Minecraft &mdash; live spectate</h1>
<div class="grid" id="grid"></div>
<script>
// One panel div (and one iframe) is created per bot ONCE, the first time
// that bot's index is seen, and never rebuilt again. Found live: an
// earlier version rebuilt every bot's whole panel - iframe included - on
// every tick, which reloaded each 3D viewer from scratch every second.
// That tore down and reopened its WebSocket connection nonstop (visible in
// the bridge logs as constant "client connected"/"client disconnected"
// churn and a MaxListenersExceededWarning leak) and never gave the scene a
// chance to actually render - which is exactly what "spectate doesn't
// work" looks like from the outside. Only the text stats are updated now.
const panels = new Map(); // bot.index -> element refs

function ensurePanel(bot) {
  let refs = panels.get(bot.index);
  if (refs) return refs;

  const div = document.createElement('div');
  div.className = 'bot' + (bot.index === -1 ? ' flagship' : '');
  const viewer = bot.viewer_port
    ? `<iframe src="http://localhost:${bot.viewer_port}"></iframe>`
    : `<div style="height:220px;display:flex;align-items:center;justify-content:center;color:#666;">no viewer (start training with --viewers)</div>`;
  div.innerHTML = `
    ${viewer}
    <div class="stats">
      <span class="name">${bot.name}</span>
      <span class="status-badge"></span><br>
      <span class="hp"></span> &nbsp;
      <span class="food"></span> &nbsp;
      <span class="next"></span>
      <div class="inv"></div>
    </div>`;
  document.getElementById('grid').appendChild(div);

  refs = {
    root: div,
    badge: div.querySelector('.status-badge'),
    hp: div.querySelector('.hp'),
    food: div.querySelector('.food'),
    next: div.querySelector('.next'),
    inv: div.querySelector('.inv'),
  };
  panels.set(bot.index, refs);
  return refs;
}

function render(bots) {
  for (const bot of bots) {
    const refs = ensurePanel(bot);
    refs.root.classList.toggle('disconnected', !bot.connected);
    refs.badge.className = 'status-badge ' + (bot.connected ? 'up' : 'down');
    refs.badge.textContent = bot.connected ? 'live' : 'reconnecting';
    refs.hp.textContent = `HP ${bot.health ?? '?'}/20`;
    refs.food.textContent = `food ${bot.food ?? '?'}/20`;
    refs.next.textContent = `next: ${bot.next_step ?? '?'} (score ${bot.milestone_score ?? 0})`;
    refs.inv.textContent = (bot.inventory || []).map(i => `${i.name}x${i.count}`).join(', ') || 'empty';
  }
}

async function tick() {
  try {
    const res = await fetch('/api/state');
    const bots = await res.json();
    render(bots);
  } catch (e) {
    // dashboard server itself unreachable momentarily - next tick retries
  }
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # this dashboard is polled every second; the default access log is pure noise

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with _state_lock:
                bots = [_state[i] for i in sorted(_state)]
            body = json.dumps(bots).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main(
    n_bots: int = 12,
    base_bridge_port: int = 8090,
    base_viewer_port: int | None = 3100,
    base_username: str = "FlyBrain",
    port: int = 8500,
    include_main: bool = True,
    main_bridge_port: int = 8200,
    main_viewer_port: int | None = 3200,
    main_username: str = "FlyBrainMain",
) -> None:
    for i in range(n_bots):
        viewer_port = base_viewer_port + i if base_viewer_port is not None else None
        thread = threading.Thread(
            target=_poll_bot,
            args=(i, base_bridge_port + i, viewer_port, f"{base_username}{i}"),
            daemon=True,
        )
        thread.start()

    if include_main:
        # Index -1 sorts before every numbered training bot, so the
        # flagship bot (agent/fly_brain_main.py) always renders first -
        # and stays a plain int key alongside the others (_state is
        # rendered via `sorted(_state)`, which can't mix str and int keys).
        thread = threading.Thread(
            target=_poll_bot,
            args=(-1, main_bridge_port, main_viewer_port, main_username),
            daemon=True,
        )
        thread.start()

    server = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"dashboard listening on http://localhost:{port} (watching {n_bots} bots)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live spectate dashboard for the training bots.")
    parser.add_argument("--bots", type=int, default=12)
    parser.add_argument("--base-bridge-port", type=int, default=8090)
    parser.add_argument("--base-viewer-port", type=int, default=3100)
    parser.add_argument("--no-viewers", action="store_true", help="hide the 3D panes (bridge data only)")
    parser.add_argument("--base-username", type=str, default="FlyBrain")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--no-main", action="store_true", help="don't show FlyBrainMain (agent/fly_brain_main.py)")
    parser.add_argument("--main-bridge-port", type=int, default=8200)
    parser.add_argument("--main-viewer-port", type=int, default=3200)
    parser.add_argument("--main-username", type=str, default="FlyBrainMain")
    args = parser.parse_args()
    main(
        n_bots=args.bots,
        base_bridge_port=args.base_bridge_port,
        base_viewer_port=None if args.no_viewers else args.base_viewer_port,
        base_username=args.base_username,
        port=args.port,
        include_main=not args.no_main,
        main_bridge_port=args.main_bridge_port,
        main_viewer_port=None if args.no_viewers else args.main_viewer_port,
        main_username=args.main_username,
    )
