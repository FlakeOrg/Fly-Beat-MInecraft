"""Tests for the event-driven fly speech controller."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.speech import SpeechController  # noqa: E402


class FakeClient:
    def __init__(self):
        self.commands = []

    def do_action(self, command):
        self.commands.append(command)
        return {"ok": True}


def observation(**overrides):
    value = {"health": 20, "food": 20, "nearbyEntities": []}
    value.update(overrides)
    return value


def test_greets_once():
    client = FakeClient()
    speech = SpeechController(cooldown_seconds=0)

    assert speech.consider(observation(), {}, client)
    assert not speech.consider(observation(), {}, client)
    assert [c["message"] for c in client.commands] == ["hello"]


def test_low_health_and_hunger_are_edge_triggered():
    client = FakeClient()
    speech = SpeechController(cooldown_seconds=0)
    speech.consider(observation(), {}, client)  # greeting

    assert speech.consider(observation(health=5), {}, client)
    assert not speech.consider(observation(health=5), {}, client)
    # Recovering resets the condition and allows a later warning.
    speech.consider(observation(health=20), {}, client)
    assert speech.consider(observation(health=20, food=5), {}, client)
    assert [c["message"] for c in client.commands] == ["hello", "i hurt", "i hungry"]


def test_cooldown_prevents_chat_spam():
    client = FakeClient()
    now = [0.0]
    speech = SpeechController(cooldown_seconds=5, clock=lambda: now[0])
    speech.consider(observation(), {}, client)

    # A new event occurs during the cooldown, but is not sent.
    now[0] = 1.0
    speech.consider(observation(health=5), {}, client)
    assert len(client.commands) == 1

    now[0] = 6.0
    # The condition is already active, so it will not repeat until it resets.
    speech.consider(observation(health=5), {}, client)
    assert len(client.commands) == 1


def test_explicit_decoder_message_is_supported():
    client = FakeClient()
    speech = SpeechController(cooldown_seconds=0)
    speech.consider(observation(), {}, client)
    assert speech.consider(observation(), {"speech": "made"}, client)
    assert client.commands[-1]["message"] == "i made something"


@pytest.mark.parametrize("bad", [-1, -0.1])
def test_negative_cooldown_is_rejected(bad):
    with pytest.raises(ValueError):
        SpeechController(cooldown_seconds=bad)
