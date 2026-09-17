"""Tests for the event-driven fly speech controller."""

import sys
from pathlib import Path

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


def test_brain_speech_intent_is_sent():
    client = FakeClient()
    speech = SpeechController(cooldown_seconds=0)
    speech.consider(observation(), {}, client)
    assert speech.consider(observation(), {"say_hungry": True}, client)
    assert client.commands[-1]["message"] == "i hungry"


def test_decoder_intents_have_distinct_outputs():
    from interface.motor_decoder import ACTIONS, MotorDecoder
    import numpy as np

    decoder = MotorDecoder(np.arange(13), threshold=0.1)
    assert "say_hungry" in ACTIONS
    assert decoder.weights.shape[0] == len(ACTIONS)


def test_hunger_intent_is_not_repeated_without_a_new_intent():
    client = FakeClient()
    speech = SpeechController(cooldown_seconds=0)
    speech.consider(observation(), {}, client)
    assert speech.consider(observation(), {"say_hungry": True}, client)
    assert speech.consider(observation(), {"say_hungry": True}, client)
    assert len(client.commands) == 3
