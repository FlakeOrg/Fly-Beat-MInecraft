"""Tests for the pure aggregation step in agent/fly_brain_main.py - the
rest of that file is bridge/subprocess I/O, not unit-testable in
isolation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.fly_brain_main import MAX_MILESTONE_SCORE, SWARM_DANGER_HEALTH, summarize_swarm  # noqa: E402


def make_obs(health: float, inventory=None) -> dict:
    return {"health": health, "inventory": inventory or []}


def test_summarize_swarm_with_no_observations():
    summary = summarize_swarm([])
    assert summary == {"progress_avg": 0.0, "progress_best": 0.0, "danger_frac": 0.0, "connected": 0}


def test_summarize_swarm_progress_is_zero_with_no_milestones():
    summary = summarize_swarm([make_obs(20), make_obs(20)])
    assert summary["progress_avg"] == 0.0
    assert summary["progress_best"] == 0.0
    assert summary["connected"] == 2


def test_summarize_swarm_progress_best_reflects_the_leading_bot():
    ahead = make_obs(20, inventory=[{"name": "crafting_table", "count": 1}])
    behind = make_obs(20)
    summary = summarize_swarm([ahead, behind])
    assert summary["progress_best"] > 0.0
    assert summary["progress_best"] > summary["progress_avg"]


def test_summarize_swarm_danger_frac_counts_critical_health_bots():
    critical = make_obs(SWARM_DANGER_HEALTH)
    healthy = make_obs(20)
    summary = summarize_swarm([critical, healthy, healthy, healthy])
    assert summary["danger_frac"] == pytest.approx(0.25)


def test_summarize_swarm_danger_frac_zero_when_everyone_is_healthy():
    summary = summarize_swarm([make_obs(20), make_obs(19)])
    assert summary["danger_frac"] == 0.0


def test_max_milestone_score_is_positive():
    assert MAX_MILESTONE_SCORE > 0
