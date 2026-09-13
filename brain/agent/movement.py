"""Helpers for turning decoded actions into Mineflayer movement commands."""

from __future__ import annotations

import math

MOVEMENT_KEYS = ("forward", "left", "right", "jump")

# Only physical/movement actions count toward "is it doing something" - task
# verbs (craft/place/smelt) are decided and sent completely separately (see
# decode_task_action) and don't imply anything about whether the bot should
# also be walking. A real live bug: once task verbs were added to the action
# space, checking all(actions.values()) made this fire constantly (task
# verbs cross threshold almost every tick), which suppressed the explore
# fallback even when every movement key was false - the bots stood dead
# still, firing craft attempts nonstop, never walking anywhere to find
# material for them.
PHYSICAL_ACTIONS = MOVEMENT_KEYS + ("attack", "mine_ahead")


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
    if any(actions.get(key, False) for key in PHYSICAL_ACTIONS):
        return {"type": "move", **movement}

    movement["forward"] = True
    if _block_directly_ahead(observation):
        movement["jump"] = bool(observation.get("onGround", True))
        if (step // 10) % 2 == 0:
            movement["right"] = True
        else:
            movement["left"] = True
    return {"type": "move", **movement}
