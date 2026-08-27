"""Deterministic hand-signal diagnostics for pnp-trash review boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .source import SourceRecord

_HAND_SUFFIXES = (
    "hand_index_0_joint",
    "hand_index_1_joint",
    "hand_middle_0_joint",
    "hand_middle_1_joint",
    "hand_thumb_0_joint",
    "hand_thumb_1_joint",
    "hand_thumb_2_joint",
)
HAND_STATE_SLICES = {"left": slice(22, 29), "right": slice(36, 43)}
HAND_FEATURE_NAMES = {side: tuple(f"{side}_{suffix}" for suffix in _HAND_SUFFIXES) for side in ("left", "right")}


@dataclass(frozen=True)
class GripDiagnostic:
    status: str
    side: str | None
    reason: str | None
    usable_joint_indices: tuple[int, ...] = ()
    grasp_frame: int | None = None
    release_frame: int | None = None
    grasp_timestamp_s: float | None = None
    release_timestamp_s: float | None = None
    grasp_derivative: float | None = None
    release_derivative: float | None = None
    aperture: np.ndarray | None = None
    smoothed_aperture: np.ndarray | None = None

    @classmethod
    def unavailable(cls, *, side: str | None, reason: str) -> "GripDiagnostic":
        return cls(status="unavailable", side=side, reason=reason)

    @classmethod
    def available(
        cls,
        *,
        side: str,
        usable_joint_indices: tuple[int, ...],
        grasp_frame: int,
        release_frame: int,
        grasp_timestamp_s: float,
        release_timestamp_s: float,
        grasp_derivative: float,
        release_derivative: float,
        aperture: np.ndarray,
        smoothed_aperture: np.ndarray,
    ) -> "GripDiagnostic":
        return cls(
            status="available",
            side=side,
            reason=None,
            usable_joint_indices=usable_joint_indices,
            grasp_frame=grasp_frame,
            release_frame=release_frame,
            grasp_timestamp_s=grasp_timestamp_s,
            release_timestamp_s=release_timestamp_s,
            grasp_derivative=grasp_derivative,
            release_derivative=release_derivative,
            aperture=np.asarray(aperture, dtype=np.float64),
            smoothed_aperture=np.asarray(smoothed_aperture, dtype=np.float64),
        )

    def as_response(self, *, advisories: Sequence[str] = ()) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "side": self.side,
            "reason": self.reason,
            "usable_joint_indices": list(self.usable_joint_indices),
            "grasp": None,
            "release": None,
            "advisories": list(advisories),
        }
        if self.status == "available":
            result["grasp"] = {
                "frame": self.grasp_frame,
                "timestamp_s": self.grasp_timestamp_s,
                "derivative": self.grasp_derivative,
            }
            result["release"] = {
                "frame": self.release_frame,
                "timestamp_s": self.release_timestamp_s,
                "derivative": self.release_derivative,
            }
        return result


def compute_grip_diagnostic(
    *,
    hand_values: Any,
    timestamps: Any,
    duration_s: float,
    side: str,
) -> GripDiagnostic:
    """Apply the frozen seven-joint NumPy algorithm without heuristic fallbacks."""

    if side not in HAND_STATE_SLICES:
        return GripDiagnostic.unavailable(side=None, reason="invalid_hand")
    try:
        values = np.asarray(hand_values, dtype=np.float64)
        timeline = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return GripDiagnostic.unavailable(side=side, reason="invalid_values")
    if values.ndim != 2 or values.shape[1] != 7:
        return GripDiagnostic.unavailable(side=side, reason="feature_contract_mismatch")
    if timeline.ndim != 1 or len(timeline) != len(values) or not np.isfinite(timeline).all():
        return GripDiagnostic.unavailable(side=side, reason="invalid_timestamps")
    if len(values) < 11:
        return GripDiagnostic.unavailable(side=side, reason="insufficient_frames")
    if not np.isfinite(values).all():
        return GripDiagnostic.unavailable(side=side, reason="non_finite_values")
    if len(timeline) < 1 or timeline[0] < 0 or np.any(timeline[1:] <= timeline[:-1]):
        return GripDiagnostic.unavailable(side=side, reason="invalid_timestamps")
    try:
        duration = float(duration_s)
    except (TypeError, ValueError, OverflowError):
        return GripDiagnostic.unavailable(side=side, reason="invalid_duration")
    if not math.isfinite(duration) or duration <= 0:
        return GripDiagnostic.unavailable(side=side, reason="invalid_duration")

    q05, q95 = np.percentile(values, [5, 95], axis=0, method="linear")
    ranges = q95 - q05
    usable = ranges > np.float64(1e-6)
    usable_indices = tuple(int(index) for index in np.flatnonzero(usable))
    if len(usable_indices) < 4:
        return GripDiagnostic.unavailable(side=side, reason="insufficient_usable_joints")

    clipped = np.clip(values[:, usable], q05[usable], q95[usable])
    scaled = (clipped - q05[usable]) / ranges[usable]
    baseline_mask = (timeline >= np.float64(0.0)) & (
        timeline < np.float64(min(np.float64(0.5), np.float64(duration)))
    )
    if not baseline_mask.any():
        return GripDiagnostic.unavailable(side=side, reason="baseline_unavailable")
    baseline = np.median(scaled[baseline_mask], axis=0)
    aperture = np.sqrt(np.mean(np.square(scaled - baseline), axis=1, dtype=np.float64))
    padded = np.pad(aperture.astype(np.float64), (5, 5), mode="edge")
    kernel = np.ones(11, dtype=np.float64) / np.float64(11.0)
    smoothed = np.convolve(padded, kernel, mode="valid")

    candidate_indices = np.arange(5, len(values) - 5, dtype=np.int64)
    if candidate_indices.size == 0:
        return GripDiagnostic.unavailable(side=side, reason="release_not_available")
    derivatives = smoothed[candidate_indices + 5] - smoothed[candidate_indices - 5]
    extrema = select_grip_extrema(candidate_indices=candidate_indices, derivatives=derivatives)
    unavailable = extrema.get("unavailable")
    if unavailable is not None:
        reason = "invalid_derivative_sign" if unavailable.startswith("invalid_") else unavailable
        return GripDiagnostic.unavailable(side=side, reason=reason)
    grasp_frame = int(extrema["grasp_frame"])
    release_frame = int(extrema["release_frame"])
    grasp_derivative = float(extrema["grasp_derivative"])
    release_derivative = float(extrema["release_derivative"])
    return GripDiagnostic.available(
        side=side,
        usable_joint_indices=usable_indices,
        grasp_frame=grasp_frame,
        release_frame=release_frame,
        grasp_timestamp_s=float(timeline[grasp_frame]),
        release_timestamp_s=float(timeline[release_frame]),
        grasp_derivative=grasp_derivative,
        release_derivative=release_derivative,
        aperture=aperture,
        smoothed_aperture=smoothed,
    )


def select_grip_extrema(
    *,
    candidate_indices: np.ndarray,
    derivatives: np.ndarray,
) -> dict[str, int | float | str]:
    """Select deterministic extrema and expose each unavailable branch independently."""

    candidates = np.asarray(candidate_indices)
    values = np.asarray(derivatives, dtype=np.float64)
    if (
        candidates.ndim != 1
        or values.ndim != 1
        or len(candidates) != len(values)
        or not len(candidates)
        or not np.issubdtype(candidates.dtype, np.integer)
        or not np.isfinite(values).all()
        or np.any(candidates[1:] <= candidates[:-1])
    ):
        raise ValueError("candidate indices and derivatives must be aligned finite vectors")
    grasp_position = int(np.argmax(values))
    grasp_frame = int(candidates[grasp_position])
    eligible_positions = np.flatnonzero(candidates > grasp_frame)
    if eligible_positions.size == 0:
        return {"unavailable": "release_not_available"}
    release_relative = int(np.argmin(values[eligible_positions]))
    release_position = int(eligible_positions[release_relative])
    grasp_derivative = float(values[grasp_position])
    release_derivative = float(values[release_position])
    if grasp_derivative <= 0:
        return {"unavailable": "invalid_grasp_derivative_sign"}
    if release_derivative >= 0:
        return {"unavailable": "invalid_release_derivative_sign"}
    return {
        "grasp_frame": grasp_frame,
        "release_frame": int(candidates[release_position]),
        "grasp_derivative": grasp_derivative,
        "release_derivative": release_derivative,
    }


def grip_advisories(
    diagnostic: GripDiagnostic,
    *,
    decision: Mapping[str, Any],
    timestamps: Sequence[float],
) -> list[str]:
    """Compare candidates to approved human starts without mutating either input."""

    if diagnostic.status != "available" or decision.get("review_state") != "approved_keep":
        return []
    transitions = decision.get("transition_frames")
    if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)) or len(transitions) != 6:
        return []
    timeline = tuple(timestamps)
    comparisons = (
        (0, diagnostic.grasp_timestamp_s, "grip_grasp_disagrees_with_step_2_start"),
        (4, diagnostic.release_timestamp_s, "grip_release_disagrees_with_step_6_start"),
    )
    warnings: list[str] = []
    for position, candidate_s, warning in comparisons:
        frame = transitions[position]
        if type(frame) is not int or frame < 0 or frame >= len(timeline) or candidate_s is None:
            continue
        if abs(float(timeline[frame]) - candidate_s) > 2.0:
            warnings.append(warning)
    return warnings


def approved_boundary_deltas(
    diagnostic: GripDiagnostic,
    *,
    decision: Mapping[str, Any],
    timestamps: Sequence[float],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {"grasp_delta_s": None, "release_delta_s": None}
    if diagnostic.status != "available" or decision.get("review_state") != "approved_keep":
        return result
    transitions = decision.get("transition_frames")
    if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)) or len(transitions) != 6:
        return result
    timeline = tuple(timestamps)
    for key, position, candidate_s in (
        ("grasp_delta_s", 0, diagnostic.grasp_timestamp_s),
        ("release_delta_s", 4, diagnostic.release_timestamp_s),
    ):
        frame = transitions[position]
        if type(frame) is int and 0 <= frame < len(timeline) and candidate_s is not None:
            result[key] = abs(float(timeline[frame]) - candidate_s)
    return result


def read_grip_diagnostic(
    record: SourceRecord,
    *,
    source_episode_index: int,
    side: str,
) -> GripDiagnostic:
    """Read one registered episode through pinned descriptors and apply the contract."""

    if side not in HAND_STATE_SLICES:
        return GripDiagnostic.unavailable(side=None, reason="invalid_hand")
    try:
        info = _read_registered_json(record, "meta/info.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return GripDiagnostic.unavailable(side=side, reason="source_unreadable")
    feature = info.get("features", {}).get("observation.state")
    state_slice = HAND_STATE_SLICES[side]
    if (
        not isinstance(feature, Mapping)
        or feature.get("shape") != [43]
        or not isinstance(feature.get("names"), list)
        or tuple(feature["names"][state_slice]) != HAND_FEATURE_NAMES[side]
    ):
        return GripDiagnostic.unavailable(side=side, reason="feature_contract_mismatch")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or fps <= 0:
        return GripDiagnostic.unavailable(side=side, reason="invalid_duration")
    paths = _episode_parquet_paths(record, source_episode_index)
    if len(paths) != 1:
        return GripDiagnostic.unavailable(side=side, reason="source_unreadable")
    asset = record.open_asset(paths[0])
    if asset is None:
        return GripDiagnostic.unavailable(side=side, reason="source_unreadable")
    try:
        with os.fdopen(os.dup(asset.fd), "rb") as handle:
            table = pq.read_table(
                handle,
                columns=["episode_index", "frame_index", "timestamp", "observation.state"],
            )
        digest = _sha256_fd(asset.fd)
        if not record.verify_pinned_asset(asset, sha256=digest):
            return GripDiagnostic.unavailable(side=side, reason="source_changed")
    except (OSError, pa.ArrowException, TypeError, ValueError):
        return GripDiagnostic.unavailable(side=side, reason="source_unreadable")
    finally:
        asset.close()
    rows: list[tuple[int, float, Any]] = []
    try:
        for episode, frame, timestamp, state in zip(
            table.column("episode_index").to_pylist(),
            table.column("frame_index").to_pylist(),
            table.column("timestamp").to_pylist(),
            table.column("observation.state").to_pylist(),
            strict=True,
        ):
            if int(episode) == source_episode_index:
                rows.append((int(frame), float(timestamp), state))
    except (TypeError, ValueError, OverflowError):
        return GripDiagnostic.unavailable(side=side, reason="source_unreadable")
    rows.sort(key=lambda row: row[0])
    if [frame for frame, _, _ in rows] != list(range(len(rows))):
        return GripDiagnostic.unavailable(side=side, reason="source_unreadable")
    try:
        state_values = np.asarray([state for _, _, state in rows], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return GripDiagnostic.unavailable(side=side, reason="source_unreadable")
    if state_values.ndim != 2 or state_values.shape[1] != 43:
        return GripDiagnostic.unavailable(side=side, reason="feature_contract_mismatch")
    timestamps = np.asarray([timestamp for _, timestamp, _ in rows], dtype=np.float64)
    return compute_grip_diagnostic(
        hand_values=state_values[:, state_slice],
        timestamps=timestamps,
        duration_s=len(rows) / float(fps),
        side=side,
    )


def _episode_parquet_paths(record: SourceRecord, source_episode_index: int) -> list[str]:
    name = f"episode_{source_episode_index:06d}.parquet"
    return sorted(path for path in record.file_hashes if path.startswith("data/") and path.endswith(f"/{name}"))


def _read_registered_json(record: SourceRecord, relative_path: str) -> dict[str, Any]:
    asset = record.open_asset(relative_path)
    if asset is None:
        raise ValueError("registered source asset is unavailable")
    try:
        contents = _read_fd(asset.fd)
        digest = hashlib.sha256(contents).hexdigest()
        if not record.verify_pinned_asset(asset, sha256=digest):
            raise ValueError("registered source asset changed")
        document = json.loads(contents.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("registered JSON asset must be an object")
        return document
    finally:
        asset.close()


def _read_fd(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()
