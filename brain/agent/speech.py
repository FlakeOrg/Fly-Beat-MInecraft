"""Small, event-driven speech system for the fly.

Speech is deliberately separate from the connectome motor outputs.  The fly
can choose *when* to speak through simple internal/game-state drives while the
bridge remains responsible only for sending text to Minecraft.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any


DEFAULT_PHRASES = {
    "hello": "hello",
    "hungry": "i hungry",
    "hurt": "i hurt",
    "help": "i need help",
    "found": "i found something",
    "made": "i made something",
}


class SpeechController:
    """Turn important game events into occasional, short chat messages.

    The controller is edge-triggered for persistent conditions such as low
    health/food and has a global cooldown, so a 10 Hz control loop cannot
    flood the server chat.  ``enabled=False`` is useful for headless training.
    """

    def __init__(
        self,
        cooldown_seconds: float = 8.0,
        enabled: bool = True,
        phrases: Mapping[str, str] | None = None,
        clock=time.monotonic,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.phrases = dict(DEFAULT_PHRASES)
        if phrases:
            self.phrases.update(phrases)
        self._clock = clock
        self._last_spoke = float("-inf")
        self._previous: dict[str, bool] = {}
        self._greeted = False

    def _rising(self, key: str, value: bool) -> bool:
        previous = self._previous.get(key, False)
        self._previous[key] = value
        return value and not previous

    def _say(self, client: Any, key: str) -> bool:
        if not self.enabled or key not in self.phrases:
            return False
        now = self._clock()
        if now - self._last_spoke < self.cooldown_seconds:
            return False
        client.do_action({"type": "chat", "message": self.phrases[key]})
        self._last_spoke = now
        return True

    def consider(self, observation: dict, actions: dict | None, client: Any) -> bool:
        """Speak once when a notable condition starts.

        Returns whether a message was sent.  Explicit ``say_*``/``speech``
        action flags are supported for future decoder or task-manager output;
        the current implementation also derives messages from observations.
        """
        actions = actions or {}
        health = float(observation.get("health", 20))
        food = float(observation.get("food", 20))
        entities = observation.get("nearbyEntities", [])

        explicit = actions.get("speech")
        if isinstance(explicit, str) and explicit in self.phrases:
            return self._say(client, explicit)
        for action_key, phrase_key in (
            ("say_hello", "hello"),
            ("say_hungry", "hungry"),
            ("say_hurt", "hurt"),
            ("say_help", "help"),
            ("say_found", "found"),
            ("say_made", "made"),
        ):
            if actions.get(action_key):
                return self._say(client, phrase_key)

        # Greeting is sent once after the first usable observation.
        if not self._greeted:
            self._greeted = True
            return self._say(client, "hello")

        if self._rising("hurt", health <= 6):
            return self._say(client, "hurt")
        if self._rising("hungry", food <= 6):
            return self._say(client, "hungry")

        hostile_near = any(
            entity.get("kind") == "Hostile mobs" and entity.get("distance", float("inf")) <= 6
            for entity in entities
        )
        if self._rising("hostile_near", hostile_near):
            return self._say(client, "help")

        return False
