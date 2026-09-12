"""WebSocket client for talking to bot-bridge (the Node/Mineflayer process).

Protocol (see bot-bridge/src/index.js): request `{id, method, params}`,
response `{id, result}` or `{id, error}`. Unsolicited `{type: "event", ...}`
messages are lifecycle notifications (spawn/death/kicked/end) and are handed
to an optional callback instead of being matched to a pending request.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

from websockets.sync.client import ClientConnection, connect


class BridgeError(RuntimeError):
    """Raised when bot-bridge returns an {error} response."""


class BridgeClient:
    def __init__(
        self,
        url: str = "ws://localhost:8081",
        on_event: Callable[[dict], None] | None = None,
        timeout: float = 15.0,
    ):
        # A hard backstop on top of bot-bridge's own per-action timeout
        # (see index.js's ACTION_TIMEOUT_MS): if the bridge process itself
        # is dead or wedged, a training loop (M5) that calls this in a tight
        # rollout loop must not hang forever waiting on one response.
        self.url = url
        self.on_event = on_event
        self.timeout = timeout
        self._ws: ClientConnection | None = None
        self._id_counter = itertools.count(1)

    def connect(self) -> None:
        self._ws = connect(self.url)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def __enter__(self) -> "BridgeClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _call(self, method: str, params: dict | None = None) -> Any:
        if self._ws is None:
            raise RuntimeError("not connected - call connect() or use as a context manager")

        request_id = next(self._id_counter)
        self._ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))

        while True:
            try:
                raw = self._ws.recv(timeout=self.timeout)
            except TimeoutError as exc:
                raise BridgeError(f"no response to '{method}' within {self.timeout}s") from exc
            message = json.loads(raw)
            if message.get("type") == "event":
                if self.on_event is not None:
                    self.on_event(message)
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise BridgeError(message["error"])
                return message.get("result")

    def get_observation(self) -> dict:
        return self._call("getObservation")

    def do_action(self, command: dict) -> dict:
        return self._call("doAction", command)

    def ping(self) -> dict:
        return self._call("ping")
