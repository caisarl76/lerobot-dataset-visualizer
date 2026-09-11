"""Read-only extraction of measured robot motion from an episode parquet file."""

from __future__ import annotations

import math
from typing import Any

from fastapi import HTTPException
import pandas as pd


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(422, "Robot motion contains a non-numeric value") from exc
    if not math.isfinite(result):
        raise HTTPException(422, "Robot motion contains a non-finite value")
    return result


def read_robot_motion(state, episode_index: int, episode_data_path) -> dict[str, Any]:
    """Extract measured state and optional WXYZ root orientation for one episode."""
    features = state.info.get("features", {})
    state_feature = features.get("observation.state")
    names = state_feature.get("names") if isinstance(state_feature, dict) else None
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise HTTPException(422, "Dataset has no supported observation.state feature names")
    root_feature = features.get("observation.root_orientation")
    root_names = root_feature.get("names") if isinstance(root_feature, dict) else None
    root_enabled = root_names == ["base_qw", "base_qx", "base_qy", "base_qz"]

    columns = ["episode_index", "timestamp", "observation.state"]
    if root_enabled:
        columns.append("observation.root_orientation")
    try:
        frame = pd.read_parquet(episode_data_path, columns=columns)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, "Unsupported or unreadable episode motion data") from exc
    frame = frame.loc[frame["episode_index"] == episode_index]
    if frame.empty:
        raise HTTPException(404, "Episode not found")

    timestamps: list[float] = []
    positions: list[list[float]] = []
    orientations: list[list[float]] | None = [] if root_enabled else None
    previous = None
    for row in frame.itertuples(index=False, name=None):
        timestamp = _finite(row[1])
        if timestamp < 0:
            raise HTTPException(422, "Episode timestamps are invalid or negative")
        if previous is not None and timestamp <= previous:
            raise HTTPException(422, "Episode timestamps are invalid or nonmonotonic")
        previous = timestamp
        try:
            state_values = list(row[2])
        except (TypeError, ValueError):
            state_values = []
        if len(state_values) != len(names):
            raise HTTPException(422, "observation.state does not match its feature names")
        timestamps.append(timestamp)
        positions.append([_finite(value) for value in state_values])
        if root_enabled:
            try:
                root_values = list(row[3])
            except (TypeError, ValueError):
                root_values = []
            if len(root_values) != 4:
                raise HTTPException(422, "observation.root_orientation is malformed")
            orientation = [_finite(value) for value in root_values]
            if sum(value * value for value in orientation) == 0:
                raise HTTPException(422, "observation.root_orientation has zero norm")
            orientations.append(orientation)
    return {
        "joint_names": names,
        "timestamps": timestamps,
        "positions": positions,
        "root_orientations": orientations,
    }
