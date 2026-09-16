"""Tests for the spectate dashboard's pure helpers (dashboard/server.py).

The dashboard is otherwise all I/O (bridge polling threads, an HTTP
server) - nothing else here is worth unit testing in isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.server import _aggregate_inventory  # noqa: E402


def test_aggregate_inventory_sums_same_named_items_across_slots():
    observation = {
        "inventory": [
            {"slot": 0, "name": "oak_log", "count": 3},
            {"slot": 5, "name": "oak_log", "count": 2},
            {"slot": 9, "name": "cobblestone", "count": 12},
        ]
    }
    assert _aggregate_inventory(observation) == [
        {"name": "cobblestone", "count": 12},
        {"name": "oak_log", "count": 5},
    ]


def test_aggregate_inventory_empty():
    assert _aggregate_inventory({"inventory": []}) == []


def test_aggregate_inventory_missing_key_defaults_empty():
    assert _aggregate_inventory({}) == []
