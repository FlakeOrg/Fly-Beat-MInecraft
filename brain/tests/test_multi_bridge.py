"""Tests for training/multi_bridge.py's bridge-launch bookkeeping - port,
username, and per-shard Minecraft host/port assignment. subprocess.Popen is
mocked throughout: this is pure arithmetic/plumbing, not a live-process test.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.multi_bridge import Shard, launch_bridges, launch_shards  # noqa: E402


def _fake_popen(*args, **kwargs):
    return MagicMock(spec=["terminate"])


@patch("training.multi_bridge.subprocess.Popen", side_effect=_fake_popen)
@patch("training.multi_bridge.open", create=True)
def test_launch_bridges_assigns_sequential_ports_and_usernames(mock_open, mock_popen):
    instances = launch_bridges(3, base_port=8090)
    assert [i.port for i in instances] == [8090, 8091, 8092]
    assert [i.username for i in instances] == ["FlyBrain0", "FlyBrain1", "FlyBrain2"]


@patch("training.multi_bridge.subprocess.Popen", side_effect=_fake_popen)
@patch("training.multi_bridge.open", create=True)
def test_launch_bridges_start_index_offsets_usernames_not_ports(mock_open, mock_popen):
    instances = launch_bridges(2, base_port=8090, start_index=6)
    assert [i.port for i in instances] == [8090, 8091]
    assert [i.username for i in instances] == ["FlyBrain6", "FlyBrain7"]


@patch("training.multi_bridge.subprocess.Popen", side_effect=_fake_popen)
@patch("training.multi_bridge.open", create=True)
def test_launch_bridges_sets_mc_host_and_port_env(mock_open, mock_popen):
    launch_bridges(1, mc_host="localhost", mc_port=25566)
    env = mock_popen.call_args.kwargs["env"]
    assert env["MC_HOST"] == "localhost"
    assert env["MC_PORT"] == "25566"


@patch("training.multi_bridge.subprocess.Popen", side_effect=_fake_popen)
@patch("training.multi_bridge.open", create=True)
def test_launch_shards_gives_every_bot_a_unique_bridge_port(mock_open, mock_popen):
    instances = launch_shards([Shard(port=25565, bots=3), Shard(port=25566, bots=3)])
    ports = [i.port for i in instances]
    assert len(ports) == len(set(ports)) == 6


@patch("training.multi_bridge.subprocess.Popen", side_effect=_fake_popen)
@patch("training.multi_bridge.open", create=True)
def test_launch_shards_gives_every_bot_a_unique_username(mock_open, mock_popen):
    instances = launch_shards([Shard(port=25565, bots=3), Shard(port=25566, bots=3)])
    usernames = [i.username for i in instances]
    assert usernames == ["FlyBrain0", "FlyBrain1", "FlyBrain2", "FlyBrain3", "FlyBrain4", "FlyBrain5"]


@patch("training.multi_bridge.subprocess.Popen", side_effect=_fake_popen)
@patch("training.multi_bridge.open", create=True)
def test_launch_shards_points_each_bots_batch_at_its_own_server(mock_open, mock_popen):
    launch_shards([Shard(port=25565, bots=2), Shard(port=25566, bots=2)])
    envs = [call.kwargs["env"] for call in mock_popen.call_args_list]
    assert [e["MC_PORT"] for e in envs] == ["25565", "25565", "25566", "25566"]


@patch("training.multi_bridge.subprocess.Popen", side_effect=_fake_popen)
@patch("training.multi_bridge.open", create=True)
def test_launch_shards_viewer_ports_continue_across_shards(mock_open, mock_popen):
    instances = launch_shards([Shard(port=25565, bots=2), Shard(port=25566, bots=2)], base_viewer_port=3100)
    assert [i.viewer_port for i in instances] == [3100, 3101, 3102, 3103]
