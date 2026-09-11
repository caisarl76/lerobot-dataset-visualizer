from types import SimpleNamespace

from backend.robot_motion import read_robot_motion
from fastapi import HTTPException
import pandas as pd
import pytest


def _state(names, root_names=None):
    features = {"observation.state": {"names": names}}
    if root_names is not None:
        features["observation.root_orientation"] = {"names": root_names}
    return SimpleNamespace(info={"features": features})


def _write_episode(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_robot_motion_filters_episode_and_preserves_order_and_quaternions(tmp_path):
    path = tmp_path / "episode.parquet"
    _write_episode(
        path,
        [
            {"episode_index": 1, "timestamp": 0.2, "observation.state": [9, 9],
             "observation.root_orientation": [1, 0, 0, 0]},
            {"episode_index": 0, "timestamp": 0.0, "observation.state": [1, 2],
             "observation.root_orientation": [1, 0, 0, 0]},
            {"episode_index": 0, "timestamp": 0.1, "observation.state": [3, 4],
             "observation.root_orientation": [0.9, 0.1, 0.2, 0.3]},
        ],
    )
    result = read_robot_motion(
        _state(["left_joint", "right_joint"], ["base_qw", "base_qx", "base_qy", "base_qz"]), 0, path
    )
    assert result == {
        "joint_names": ["left_joint", "right_joint"],
        "timestamps": [0.0, 0.1],
        "positions": [[1.0, 2.0], [3.0, 4.0]],
        "root_orientations": [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.2, 0.3]],
    }


def test_robot_motion_rejects_nonmonotonic_and_nonfinite_values(tmp_path):
    path = tmp_path / "episode.parquet"
    _write_episode(
        path,
        [
            {"episode_index": 0, "timestamp": 0.2, "observation.state": [1, 2]},
            {"episode_index": 0, "timestamp": 0.1, "observation.state": [3, 4]},
        ],
    )
    with pytest.raises(HTTPException) as error:
        read_robot_motion(_state(["x", "y"]), 0, path)
    assert error.value.status_code == 422

    _write_episode(path, [{"episode_index": 0, "timestamp": 0.0, "observation.state": [float("nan"), 2]}])
    with pytest.raises(HTTPException) as error:
        read_robot_motion(_state(["x", "y"]), 0, path)
    assert error.value.status_code == 422


def test_robot_motion_missing_episode_is_404(tmp_path):
    path = tmp_path / "episode.parquet"
    _write_episode(path, [{"episode_index": 0, "timestamp": 0.0, "observation.state": [1, 2]}])
    with pytest.raises(HTTPException) as error:
        read_robot_motion(_state(["x", "y"]), 4, path)
    assert error.value.status_code == 404


def test_robot_motion_omits_unsupported_root_orientation(tmp_path):
    path = tmp_path / "episode.parquet"
    _write_episode(path, [{"episode_index": 0, "timestamp": 0.0, "observation.state": [1, 2]}])
    result = read_robot_motion(_state(["x", "y"], ["qx", "qy", "qz", "qw"]), 0, path)
    assert result["root_orientations"] is None


@pytest.mark.parametrize("names", [["", "y"], ["x", "x"]])
def test_robot_motion_rejects_invalid_joint_names(tmp_path, names):
    path = tmp_path / "episode.parquet"
    _write_episode(path, [{"episode_index": 0, "timestamp": 0.0, "observation.state": [1, 2]}])
    with pytest.raises(HTTPException) as error:
        read_robot_motion(_state(names), 0, path)
    assert error.value.status_code == 422


def test_robot_motion_rejects_negative_timestamp_and_zero_quaternion(tmp_path):
    path = tmp_path / "episode.parquet"
    _write_episode(path, [{"episode_index": 0, "timestamp": -0.1, "observation.state": [1, 2]}])
    with pytest.raises(HTTPException) as error:
        read_robot_motion(_state(["x", "y"]), 0, path)
    assert error.value.status_code == 422

    _write_episode(
        path,
        [{"episode_index": 0, "timestamp": 0.0, "observation.state": [1, 2],
          "observation.root_orientation": [0, 0, 0, 0]}],
    )
    with pytest.raises(HTTPException) as error:
        read_robot_motion(_state(["x", "y"], ["base_qw", "base_qx", "base_qy", "base_qz"]), 0, path)
    assert error.value.status_code == 422
