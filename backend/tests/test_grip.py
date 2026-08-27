from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from curation.db import CurationDatabase
from curation.grip import (
    HAND_FEATURE_NAMES,
    HAND_STATE_SLICES,
    GripDiagnostic,
    compute_grip_diagnostic,
    grip_advisories,
    select_grip_extrema,
)
from curation.review import ReviewService
from curation.source import SourceRegistry
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _reference(values: np.ndarray, timestamps: np.ndarray, duration_s: float) -> dict[str, Any]:
    q05, q95 = np.percentile(values, [5, 95], axis=0, method="linear")
    spans = q95 - q05
    usable = spans > 1e-6
    normalized = (np.clip(values[:, usable], q05[usable], q95[usable]) - q05[usable]) / spans[usable]
    baseline = np.median(normalized[(timestamps >= 0) & (timestamps < min(0.5, duration_s))], axis=0)
    aperture = np.sqrt(np.mean(np.square(normalized - baseline), axis=1, dtype=np.float64))
    padded = np.pad(aperture.astype(np.float64), (5, 5), mode="edge")
    smoothed = np.convolve(padded, np.ones(11, dtype=np.float64) / 11.0, mode="valid")
    candidates = np.arange(5, len(values) - 5)
    derivatives = smoothed[candidates + 5] - smoothed[candidates - 5]
    grasp = int(candidates[int(np.argmax(derivatives))])
    release_candidates = candidates[candidates > grasp]
    release_derivatives = smoothed[release_candidates + 5] - smoothed[release_candidates - 5]
    release = int(release_candidates[int(np.argmin(release_derivatives))])
    return {
        "grasp_frame": grasp,
        "release_frame": release,
        "grasp_derivative": float(smoothed[grasp + 5] - smoothed[grasp - 5]),
        "release_derivative": float(smoothed[release + 5] - smoothed[release - 5]),
        "aperture": aperture,
        "smoothed": smoothed,
        "usable": tuple(int(index) for index in np.flatnonzero(usable)),
    }


def _signal(frame_count: int = 61) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(frame_count, dtype=np.float64) / 10.0
    envelope = np.zeros(frame_count, dtype=np.float64)
    envelope[12:23] = np.linspace(0.0, 1.0, 11)
    envelope[23:39] = 1.0
    envelope[39:50] = np.linspace(1.0, 0.0, 11)
    values = np.stack(
        [envelope * scale + offset for scale, offset in zip(np.linspace(0.7, 1.3, 7), np.arange(7) * 0.01)],
        axis=1,
    )
    return values, timestamps


def _with_nan(values: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    changed = values.copy()
    changed[20, 2] = np.nan
    return changed, timestamps, 6.1


@pytest.mark.parametrize("side,expected_slice", [("left", slice(22, 29)), ("right", slice(36, 43))])
def test_hand_contract_uses_exact_seven_source_features_in_order(side: str, expected_slice: slice) -> None:
    assert HAND_STATE_SLICES[side] == expected_slice
    assert HAND_FEATURE_NAMES[side] == tuple(
        f"{side}_{suffix}"
        for suffix in (
            "hand_index_0_joint",
            "hand_index_1_joint",
            "hand_middle_0_joint",
            "hand_middle_1_joint",
            "hand_thumb_0_joint",
            "hand_thumb_1_joint",
            "hand_thumb_2_joint",
        )
    )


def test_grip_algorithm_matches_independent_float64_numpy_reference_exactly() -> None:
    values, timestamps = _signal()
    expected = _reference(values, timestamps, duration_s=6.1)

    result = compute_grip_diagnostic(
        hand_values=values,
        timestamps=timestamps,
        duration_s=6.1,
        side="left",
    )

    assert result.status == "available"
    assert result.usable_joint_indices == expected["usable"]
    assert result.grasp_frame == expected["grasp_frame"]
    assert result.release_frame == expected["release_frame"]
    assert result.grasp_timestamp_s == timestamps[expected["grasp_frame"]]
    assert result.release_timestamp_s == timestamps[expected["release_frame"]]
    assert result.grasp_derivative == expected["grasp_derivative"]
    assert result.release_derivative == expected["release_derivative"]
    np.testing.assert_array_equal(result.aperture, expected["aperture"])
    np.testing.assert_array_equal(result.smoothed_aperture, expected["smoothed"])


def test_percentiles_are_linear_baseline_is_half_open_and_ties_use_smallest_index() -> None:
    # The outlier at exactly 0.5 s must not enter the [0, 0.5) median baseline.
    timestamps = np.arange(31, dtype=np.float64) / 10.0
    values = np.zeros((31, 7), dtype=np.float64)
    values[5:18] = 0.37
    values[18:] = 1.0
    values[24:] = 0.0
    expected = _reference(values, timestamps, duration_s=3.1)

    result = compute_grip_diagnostic(
        hand_values=values,
        timestamps=timestamps,
        duration_s=3.1,
        side="right",
    )

    assert result.status == "available"
    assert result.grasp_frame == expected["grasp_frame"]
    assert result.release_frame == expected["release_frame"]
    assert result.grasp_derivative > 0
    assert result.release_derivative < 0


def test_tied_extrema_choose_smallest_indices_and_failure_modes_are_independent() -> None:
    selected = select_grip_extrema(
        candidate_indices=np.array([5, 6, 7, 8, 9], dtype=np.int64),
        derivatives=np.array([0.8, 0.8, -0.4, -0.4, -0.2], dtype=np.float64),
    )
    assert selected == {"grasp_frame": 5, "release_frame": 7, "grasp_derivative": 0.8, "release_derivative": -0.4}

    no_release = select_grip_extrema(
        candidate_indices=np.array([5, 6], dtype=np.int64),
        derivatives=np.array([0.1, 0.9], dtype=np.float64),
    )
    assert no_release == {"unavailable": "release_not_available"}
    grasp_nonpositive = select_grip_extrema(
        candidate_indices=np.array([5, 6, 7], dtype=np.int64),
        derivatives=np.array([0.0, -0.2, -0.4], dtype=np.float64),
    )
    assert grasp_nonpositive == {"unavailable": "invalid_grasp_derivative_sign"}
    release_nonnegative = select_grip_extrema(
        candidate_indices=np.array([5, 6, 7], dtype=np.int64),
        derivatives=np.array([0.5, 0.2, 0.0], dtype=np.float64),
    )
    assert release_nonnegative == {"unavailable": "invalid_release_derivative_sign"}


def test_usable_joint_threshold_is_strictly_greater_than_one_e_minus_six() -> None:
    base, timestamps = _signal()
    base_range = np.percentile(base[:, 0], 95, method="linear") - np.percentile(base[:, 0], 5, method="linear")
    exact = np.float64(1e-6) / base_range
    above = np.nextafter(np.float64(1e-6), np.float64(np.inf)) / base_range
    values = np.column_stack([base[:, 0] * exact] * 3 + [base[:, 0] * above] * 4)

    result = compute_grip_diagnostic(
        hand_values=values,
        timestamps=timestamps,
        duration_s=6.1,
        side="left",
    )

    assert result.status == "available"
    assert result.usable_joint_indices == (3, 4, 5, 6)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda values, timestamps: (values[:10], timestamps[:10], 1.0), "insufficient_frames"),
        (
            lambda values, timestamps: (
                np.column_stack([values[:, :3], np.zeros((len(values), 4))]),
                timestamps,
                6.1,
            ),
            "insufficient_usable_joints",
        ),
        (
            _with_nan,
            "non_finite_values",
        ),
        (lambda values, timestamps: (values, timestamps[::-1], 6.1), "invalid_timestamps"),
        (lambda values, timestamps: (values, timestamps, 0.0), "invalid_duration"),
    ],
)
def test_defined_numeric_failures_return_unavailable(
    mutation: Any,
    reason: str,
) -> None:
    values, timestamps = _signal()
    changed_values, changed_timestamps, duration_s = mutation(values, timestamps)
    result = compute_grip_diagnostic(
        hand_values=changed_values,
        timestamps=changed_timestamps,
        duration_s=duration_s,
        side="left",
    )
    assert result.status == "unavailable"
    assert result.reason == reason
    assert result.grasp_frame is None
    assert result.release_frame is None


def test_no_release_after_grasp_and_wrong_derivative_signs_are_unavailable() -> None:
    timestamps = np.arange(31, dtype=np.float64) / 10.0
    monotonic = np.repeat(np.linspace(0, 1, 31, dtype=np.float64)[:, None], 7, axis=1)
    no_release = compute_grip_diagnostic(
        hand_values=monotonic,
        timestamps=timestamps,
        duration_s=3.1,
        side="left",
    )
    assert no_release.status == "unavailable"
    assert no_release.reason in {"release_not_available", "invalid_derivative_sign"}


def test_advisory_threshold_is_strict_and_never_mutates_decision() -> None:
    diagnostic = GripDiagnostic.available(
        side="left",
        usable_joint_indices=(0, 1, 2, 3),
        grasp_frame=10,
        release_frame=40,
        grasp_timestamp_s=1.0,
        release_timestamp_s=4.0,
        grasp_derivative=0.5,
        release_derivative=-0.5,
        aperture=np.zeros(51, dtype=np.float64),
        smoothed_aperture=np.zeros(51, dtype=np.float64),
    )
    decision = {"review_state": "approved_keep", "transition_frames": [31, 32, 33, 34, 50, 51]}
    before = copy.deepcopy(decision)
    timestamps = np.arange(60, dtype=np.float64) / 10.0

    warnings = grip_advisories(diagnostic, decision=decision, timestamps=timestamps)

    assert warnings == ["grip_grasp_disagrees_with_step_2_start"]
    assert decision == before
    exactly_two = {**decision, "transition_frames": [30, 31, 32, 33, 60 - 1, 60 - 0]}
    # Step 2 at 3.0 s differs by exactly 2.0 s and therefore is not warned.
    assert "grip_grasp_disagrees_with_step_2_start" not in grip_advisories(
        diagnostic,
        decision=exactly_two,
        timestamps=np.arange(61, dtype=np.float64) / 10.0,
    )


def _source_with_state(tmp_path: Path, *, feature_names: list[str] | None = None) -> SourceRegistry:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    values, timestamps = _signal()
    names = [f"body_{index}" for index in range(43)]
    names[22:29] = HAND_FEATURE_NAMES["left"]
    names[36:43] = HAND_FEATURE_NAMES["right"]
    if feature_names is not None:
        names = feature_names
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "total_episodes": 1,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {"observation.state": {"shape": [43], "names": names}},
            }
        )
    )
    (source / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": len(timestamps), "tasks": ["task"]}) + "\n"
    )
    state = np.zeros((len(values), 43), dtype=np.float64)
    state[:, 22:29] = values
    state[:, 36:43] = values
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * len(timestamps),
                "frame_index": list(range(len(timestamps))),
                "timestamp": timestamps,
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(state.reshape(-1), type=pa.float64()), 43
                ),
            }
        ),
        source / "data" / "chunk-000" / "episode_000000.parquet",
    )
    return SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "manifest")


def test_review_service_reads_pinned_state_and_reports_advisory_without_mutation(tmp_path: Path) -> None:
    registry = _source_with_state(tmp_path)
    database = CurationDatabase(tmp_path / "workspace" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)
    service.open_workspace("local/pnp_trash", actor="curator")
    draft = service.save_draft(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=0,
        actor="curator",
        object_name="can",
        pickup_hand="left",
        turn_direction="right",
        transition_frames=[50, 51, 52, 53, 59, 60],
    )
    service.approve_keep(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=draft["revision"],
        actor="curator",
        reviewer="reviewer",
    )
    dataset = database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    before = database.get_episode(dataset_id=dataset["id"], source_episode_index=0)

    payload = service.grip_diagnostic("local/pnp_trash", 0)

    after = database.get_episode(dataset_id=dataset["id"], source_episode_index=0)
    assert payload["status"] == "available"
    assert payload["advisories"]
    assert before == after


def test_missing_or_misordered_registered_feature_names_are_unavailable(tmp_path: Path) -> None:
    valid = [f"body_{index}" for index in range(43)]
    valid[22:29] = HAND_FEATURE_NAMES["left"]
    valid[36:43] = HAND_FEATURE_NAMES["right"]
    valid[22], valid[23] = valid[23], valid[22]
    registry = _source_with_state(tmp_path, feature_names=valid)
    database = CurationDatabase(tmp_path / "workspace" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)
    service.open_workspace("local/pnp_trash", actor="curator")
    service.save_draft(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=0,
        actor="curator",
        pickup_hand="left",
    )

    payload = service.grip_diagnostic("local/pnp_trash", 0)

    assert payload["status"] == "unavailable"
    assert payload["reason"] == "feature_contract_mismatch"


def test_missing_registered_feature_name_is_unavailable(tmp_path: Path) -> None:
    names = [f"body_{index}" for index in range(43)]
    names[22:29] = HAND_FEATURE_NAMES["left"]
    names[36:43] = HAND_FEATURE_NAMES["right"]
    registry = _source_with_state(tmp_path, feature_names=names[:-1])
    database = CurationDatabase(tmp_path / "workspace" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)
    service.open_workspace("local/pnp_trash", actor="curator")
    service.save_draft(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=0,
        actor="curator",
        pickup_hand="right",
    )

    payload = service.grip_diagnostic("local/pnp_trash", 0)

    assert payload["status"] == "unavailable"
    assert payload["reason"] == "feature_contract_mismatch"


@pytest.mark.parametrize(
    ("pickup_hand", "expected_reason"),
    [(None, "pickup_hand_unselected"), ("left", "source_unreadable")],
)
def test_unavailable_grip_short_circuits_missing_timeline_with_null_comparisons(
    tmp_path: Path,
    pickup_hand: str | None,
    expected_reason: str,
) -> None:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    names = [f"body_{index}" for index in range(43)]
    names[22:29] = HAND_FEATURE_NAMES["left"]
    names[36:43] = HAND_FEATURE_NAMES["right"]
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "total_episodes": 1,
                "features": {"observation.state": {"shape": [43], "names": names}},
            }
        )
    )
    (source / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 20, "tasks": ["task"]}) + "\n"
    )
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "manifest")
    database = CurationDatabase(tmp_path / "workspace" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)
    service.open_workspace("local/pnp_trash", actor="curator")
    dataset = database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    with database.open_connection() as connection:
        connection.execute(
            "UPDATE episodes SET review_state='draft', pickup_hand=? "
            "WHERE dataset_id=? AND source_episode_index=0",
            (pickup_hand, dataset["id"]),
        )

    payload = service.grip_diagnostic("local/pnp_trash", 0)

    assert payload["status"] == "unavailable"
    assert payload["reason"] == expected_reason
    assert payload["advisories"] == []
    assert payload["grasp_delta_s"] is None
    assert payload["release_delta_s"] is None
