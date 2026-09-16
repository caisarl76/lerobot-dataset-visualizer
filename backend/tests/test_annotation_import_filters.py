# ruff: noqa: F811
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from annotation_import_filters import apply_transition_filter


def test_shared_v3_episodes_are_sliced_and_existing_exclusions_merged(tmp_path):
    names = [
        f"{side}_{joint}_joint"
        for side in ["left", "right"]
        for joint in [
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        ]
    ]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 50,
                "features": {
                    "teleop.stream_mode": {},
                    "frame_index": {},
                    "observation.state": {"names": names},
                },
            }
        )
    )
    q = np.zeros((100, 12))
    q[70:, 0] = np.arange(30) * 0.03
    path = tmp_path / "shared.parquet"
    pq.write_table(
        pa.table(
            {
                "teleop.stream_mode": [[1]] * 60 + [[3]] * 40,
                "observation.state": q.tolist(),
                "frame_index": list(range(50)) * 2,
            }
        ),
        path,
    )
    records = [
        SimpleNamespace(
            episode_index=ep, data_path=path, row_offset=ep * 50, row_count=50
        )
        for ep in [0, 1]
    ]
    run = {
        "root": str(tmp_path),
        "episodes": {
            "0": {},
            "1": {"excluded_intervals": [{"start_frame": 8, "end_frame": 12}]},
        },
    }
    before = path.read_bytes()
    result = apply_transition_filter(
        run, SimpleNamespace(iter_episodes=lambda _: records)
    )
    assert result["episodes_filtered"] == 1
    assert "excluded_intervals" not in run["episodes"]["0"]
    detected = run["episodes"]["1"]["transition_filter_intervals"]
    assert detected[0]["start_frame"] == 10
    assert 10 < detected[0]["end_frame"] < 30
    assert run["episodes"]["1"]["excluded_intervals"] == [
        {"start_frame": 8, "end_frame": detected[0]["end_frame"]}
    ]
    assert path.read_bytes() == before


from test_annotation_prepare_recovery import raw_v21  # noqa: E402,F401


@pytest.mark.parametrize("enabled", [None, False])
def test_prepare_defaults_filter_on_and_honors_opt_out(
    raw_v21, tmp_path, monkeypatch, enabled
):  # noqa: F811
    import app
    from test_annotation_run_generation import SynchronousPool

    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(app, "_annotation_pool", SynchronousPool())
    calls = []

    def filtering(run, engine):
        calls.append(run["root"])
        run["episodes"]["0"]["excluded_intervals"] = [
            {"start_frame": 1, "end_frame": 2}
        ]
        return {
            "enabled": True,
            "intervals": 1,
            "episodes_filtered": 1,
            "frames_excluded": 1,
            "skipped_episodes": [],
        }

    monkeypatch.setattr(app, "apply_transition_filter", filtering)
    options = {} if enabled is None else {"exclude_transition_pauses": enabled}
    request = app.PrepareRequest(local_path=str(raw_v21), **options)
    started = app.prepare_annotation_dataset(request)
    job = app.get_annotation_job(started["job_id"])
    assert job["status"] == "completed", job
    result = job["result"]
    _, run = app._workflow(result["repo_id"].split("/")[1])
    assert bool(calls) == (enabled is None)
    assert run["import_filters"]["transition_pauses"]["enabled"] == (enabled is None)
    if enabled is None:
        assert run["episodes"]["0"]["excluded_intervals"] == [
            {"start_frame": 1, "end_frame": 2}
        ]
        assert any(
            "excluded 1 intervals" in s for s in result["validation"]["warnings"]
        )
    else:
        assert "excluded_intervals" not in run["episodes"]["0"]


def test_missing_g1_joint_names_reports_filter_skipped(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 50,
                "features": {
                    "teleop.stream_mode": {},
                    "frame_index": {},
                    "observation.state": {"names": ["gripper"]},
                },
            }
        )
    )
    run = {"root": str(tmp_path), "episodes": {"0": {}}}
    summary = apply_transition_filter(run, SimpleNamespace())
    assert "skipped_reason" in summary
    assert run["episodes"] == {"0": {}}
