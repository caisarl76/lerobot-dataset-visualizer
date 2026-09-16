"""Apply optional filters once to a newly prepared review workspace."""

import json
from pathlib import Path

import pyarrow.parquet as pq

try:
    from .annotation_clipping import normalize_exclusions
    from .annotation_transition_filter import (
        LEG_JOINT_NAMES,
        detect_transition_intervals,
    )
except ImportError:
    from annotation_clipping import normalize_exclusions
    from annotation_transition_filter import (
        LEG_JOINT_NAMES,
        detect_transition_intervals,
    )


def apply_transition_filter(run, engine):
    root = Path(run["root"])
    info = json.loads((root / "meta/info.json").read_text())
    features = info.get("features", {})
    columns = ["teleop.stream_mode", "observation.state", "frame_index"]
    summary = {
        "enabled": True,
        "episodes_filtered": 0,
        "intervals": 0,
        "frames_excluded": 0,
        "skipped_episodes": [],
        "window_seconds": 0.2,
        "joint_range_radians": 0.1,
        "consecutive_windows": 3,
        "mode_pairs": [[1, 2], [1, 3], [2, 1], [3, 1]],
        "motion_joints": {"planner_entry": "legs", "pose_entry": "body"},
    }
    names = features.get("observation.state", {}).get("names")
    if (
        not all(key in features for key in columns)
        or not isinstance(names, list)
        or any(name not in names for name in LEG_JOINT_NAMES)
    ):
        summary["skipped_reason"] = (
            "Required mode or named joint-state fields are missing"
        )
        return summary
    records = {}
    if info.get("codebase_version") != "v2.1":
        try:
            records = {
                record.episode_index: record for record in engine.iter_episodes(root)
            }
        except (ValueError, OSError, KeyError) as exc:
            summary["skipped_reason"] = f"Episode metadata unavailable: {exc}"
            return summary
    cached_path, cached_table = None, None
    for ep, episode in run["episodes"].items():
        if episode.get("generation_status") == "failed":
            summary["skipped_episodes"].append(int(ep))
            continue
        try:
            if info.get("codebase_version") == "v2.1":
                path = root / info["data_path"].format(
                    episode_chunk=int(ep) // info["chunks_size"], episode_index=int(ep)
                )
                table = pq.read_table(path, columns=columns)
            else:
                record = records[int(ep)]
                if record.data_path != cached_path:
                    cached_table = pq.read_table(record.data_path, columns=columns)
                    cached_path = record.data_path
                table = cached_table.slice(record.row_offset, record.row_count)
            count = table.num_rows
            if table["frame_index"].to_pylist() != list(range(count)):
                raise ValueError("Frame indices are not contiguous")
            intervals = detect_transition_intervals(
                table["teleop.stream_mode"].to_pylist(),
                table["observation.state"].to_pylist(),
                info.get("fps"),
                names,
            )
            if not intervals:
                continue
            merged = normalize_exclusions(
                episode.get("excluded_intervals", []) + intervals, count
            )
        except (ValueError, OSError, KeyError, TypeError):
            summary["skipped_episodes"].append(int(ep))
            continue
        episode["excluded_intervals"] = merged
        episode["transition_filter_intervals"] = intervals
        summary["episodes_filtered"] += 1
        summary["intervals"] += len(intervals)
        summary["frames_excluded"] += sum(
            item["end_frame"] - item["start_frame"] for item in intervals
        )
    return summary
