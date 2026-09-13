"""Helpers for turning decoded actions into Mineflayer movement commands."""

from __future__ import annotations

import math

MOVEMENT_KEYS = ("forward", "left", "right", "jump")


def _facing_offset(yaw: float) -> tuple[int, int]:
    """Rounds Mineflayer yaw to the relative block coordinate ahead."""
    dx = -math.sin(yaw)
    dz = math.cos(yaw)
    return int(round(dx)), int(round(dz))


def _block_directly_ahead(observation: dict) -> bool:
    ahead_x, ahead_z = _facing_offset(float(observation.get("yaw", 0.0)))
    for block in observation.get("nearbyBlocks", []):
        if block.get("x") != ahead_x or block.get("z") != ahead_z:
            continue
        if block.get("y") not in (0, 1):
            continue
        if block.get("boundingBox", "block") == "empty":
            continue
        return True
    return False


def movement_command_from_actions(actions: dict[str, bool], observation: dict, step: int = 0) -> dict:
    """Builds a move command, adding exploration only for a total no-op.

    The motor decoder is allowed to be silent, especially before training.
    The live bot should not translate that silence into standing still
    forever, so an all-false action vector becomes a cautious exploratory
    walk. Purposeful non-movement actions, such as attack or mine_ahead, are
    left alone.
    """
    movement = {key: bool(actions.get(key, False)) for key in MOVEMENT_KEYS}
    if any(actions.values()):
        return {"type": "move", **movement}

    movement["forward"] = True
    if _block_directly_ahead(observation):
        movement["jump"] = bool(observation.get("onGround", True))
        if (step // 10) % 2 == 0:
            movement["right"] = True
        else:
            movement["left"] = True
    return {"type": "move", **movement}
