"""Read-only deterministic audit summaries for pnp-trash curation."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
import threading
from typing import Any, Callable, Mapping, Sequence

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contact_sheets import (
    ContactSheetDatasetIdentity,
    ContactSheetEvidenceInspector,
    ContactSheetUnavailable,
    approval_evidence_snapshot,
    final_contact_sheet_path,
    proposal_contact_sheet_path,
)
from .db import CurationDatabase
from .grip import GripDiagnostic, approved_boundary_deltas, grip_advisories, read_grip_diagnostic
from .models import AttemptState, JobState, ReviewState
from .source import SourceRecord, SourceRegistry

_STEP_COLUMNS = tuple(f"step_{step}_start_frame" for step in range(2, 8))
_TRANSITION_NAMES = tuple(f"step_{step}_start" for step in range(2, 8))
_PHASE_NAMES = tuple(f"step_{step}" for step in range(1, 8))
_SOURCE_EVIDENCE_CACHE_LIMIT = 4


class _PinnedIdentityChanged(RuntimeError):
    pass


@dataclass(frozen=True)
class _SourceAuditEvidence:
    fps: float | None
    timelines: Mapping[int, tuple[float, ...]]
    video_readable: Mapping[int, bool]
    grip: Mapping[tuple[int, str], GripDiagnostic]
    unreadable: tuple[dict[str, Any], ...]
    pinned_identities_match: bool


_SOURCE_EVIDENCE_CACHE: OrderedDict[tuple[object, ...], _SourceAuditEvidence] = OrderedDict()
_SOURCE_EVIDENCE_CACHE_LOCK = threading.Lock()


class AuditNotFound(LookupError):
    pass


class AuditWorkspaceUnavailable(RuntimeError):
    pass


def build_audit_report(
    *,
    database: CurationDatabase,
    source_registry: SourceRegistry,
    dataset_alias: str,
    snapshot_hook: Callable[[], None] | None = None,
    source_scan_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    record = source_registry.records.get(dataset_alias)
    if record is None:
        raise AuditNotFound("dataset alias is not registered")
    dataset, episodes, cosmos_rows, approval_events = _database_snapshot(
        database,
        dataset_alias=dataset_alias,
        snapshot_hook=snapshot_hook,
    )
    if dataset is None:
        raise AuditWorkspaceUnavailable("workspace has not been opened")
    database_identity_matches = (
        dataset["source_path"] == str(record.root) and dataset["source_manifest_sha256"] == record.fingerprint
    )
    initial_inventory_matches = database_identity_matches and record.verify_current_inventory()
    if source_scan_hook is not None:
        source_scan_hook()
    review_counts = Counter(row["review_state"] for row in episodes)
    source_cache_key = _source_evidence_cache_key(record, episodes)
    if initial_inventory_matches:
        evidence = _cached_source_evidence(source_cache_key, record=record, episodes=episodes)
        unreadable = [dict(item) for item in evidence.unreadable]
        timelines = dict(evidence.timelines)
        fps = evidence.fps
    else:
        evidence = _SourceAuditEvidence(
            fps=None,
            timelines={},
            video_readable={},
            grip={},
            unreadable=(),
            pinned_identities_match=False,
        )
        unreadable = [
            {"source_episode_index": None, "asset_kind": "source_inventory", "reason": "fingerprint_mismatch"}
        ]
        timelines = {}
        fps = None

    boundary_errors: list[dict[str, Any]] = []
    transition_values: dict[str, list[float]] = {name: [] for name in _TRANSITION_NAMES}
    duration_values: dict[str, list[float]] = {name: [] for name in _PHASE_NAMES}
    grip_disagreements: list[dict[str, Any]] = []
    grip_unavailable: list[dict[str, Any]] = []
    for row in episodes:
        index = row["source_episode_index"]
        transitions = [row[column] for column in _STEP_COLUMNS]
        boundary_errors.extend(_boundary_errors(index, row["source_length"], row["review_state"], transitions))
        timeline = timelines.get(index)
        if row["review_state"] != ReviewState.APPROVED_KEEP.value or timeline is None or fps is None:
            continue
        if any(type(frame) is not int or not 0 <= frame < len(timeline) for frame in transitions):
            continue
        integer_frames = [int(frame) for frame in transitions]
        transition_times = [timeline[frame] for frame in integer_frames]
        for name, value in zip(_TRANSITION_NAMES, transition_times, strict=True):
            transition_values[name].append(value)
        duration_s = row["source_length"] / fps
        boundaries = [0.0, *transition_times, duration_s]
        for name, start, end in zip(_PHASE_NAMES, boundaries[:-1], boundaries[1:], strict=True):
            duration_values[name].append(end - start)

        hand = row["pickup_hand"]
        if hand not in {"left", "right"}:
            grip_unavailable.append({"source_episode_index": index, "reason": "pickup_hand_unselected"})
            continue
        diagnostic = evidence.grip.get((index, hand))
        if diagnostic is None:
            diagnostic = GripDiagnostic.unavailable(side=hand, reason="source_unreadable")
        decision = {"review_state": row["review_state"], "transition_frames": transitions}
        warnings = grip_advisories(diagnostic, decision=decision, timestamps=timeline)
        if diagnostic.status != "available":
            grip_unavailable.append({"source_episode_index": index, "reason": diagnostic.reason})
        elif warnings:
            grip_disagreements.append(
                {
                    "source_episode_index": index,
                    "warnings": warnings,
                    **approved_boundary_deltas(diagnostic, decision=decision, timestamps=timeline),
                }
            )

    cosmos = _cosmos_summary(*cosmos_rows)
    contact_sheet_issues = _contact_sheet_issues(
        inspector=ContactSheetEvidenceInspector(database.path.parent),
        identity=ContactSheetDatasetIdentity(
            dataset_id=dataset["id"],
            dataset_alias=dataset["alias"],
            source_manifest_sha256=dataset["source_manifest_sha256"],
        ),
        proposals=cosmos_rows[2],
        approval_events=approval_events,
    )
    final_inventory_matches = record.verify_current_inventory()
    identity_matches = (
        database_identity_matches
        and initial_inventory_matches
        and evidence.pinned_identities_match
        and final_inventory_matches
    )
    if not identity_matches and not any(item["asset_kind"] == "source_inventory" for item in unreadable):
        unreadable.append(
            {"source_episode_index": None, "asset_kind": "source_inventory", "reason": "fingerprint_mismatch"}
        )
        with _SOURCE_EVIDENCE_CACHE_LOCK:
            _SOURCE_EVIDENCE_CACHE.pop(source_cache_key, None)
    return {
        "dataset_alias": dataset_alias,
        "review_state_counts": {state.value: review_counts.get(state.value, 0) for state in ReviewState},
        "cosmos": cosmos,
        "transition_time_distributions": {
            name: _distribution(values) for name, values in transition_values.items()
        },
        "phase_duration_distributions": {name: _distribution(values) for name, values in duration_values.items()},
        "boundary_errors": sorted(
            boundary_errors,
            key=lambda item: (item["source_episode_index"], item["code"], item.get("step", 0)),
        ),
        "grip_disagreements": sorted(grip_disagreements, key=lambda item: item["source_episode_index"]),
        "grip_unavailable": sorted(grip_unavailable, key=lambda item: item["source_episode_index"]),
        "unreadable_files": sorted(
            unreadable,
            key=lambda item: (
                -1 if item["source_episode_index"] is None else item["source_episode_index"],
                item["asset_kind"],
            ),
        ),
        "contact_sheet_issues": contact_sheet_issues,
        "source_fingerprint": {
            "expected_sha256": dataset["source_manifest_sha256"],
            "current_matches": identity_matches,
        },
    }


def _source_evidence_cache_key(
    record: SourceRecord,
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[object, ...]:
    identities = tuple(
        (
            path,
            record.file_hashes[path],
            identity.device,
            identity.inode,
            identity.size,
            identity.mtime_ns,
            identity.ctime_ns,
        )
        for path, identity in sorted(record.file_identities.items(), key=lambda item: item[0].encode("utf-8"))
    )
    episode_requirements = tuple(
        (row["source_episode_index"], row["source_length"], row["pickup_hand"]) for row in episodes
    )
    return (record.fingerprint, identities, episode_requirements)


def _cached_source_evidence(
    key: tuple[object, ...],
    *,
    record: SourceRecord,
    episodes: Sequence[Mapping[str, Any]],
) -> _SourceAuditEvidence:
    with _SOURCE_EVIDENCE_CACHE_LOCK:
        cached = _SOURCE_EVIDENCE_CACHE.get(key)
        if cached is not None:
            _SOURCE_EVIDENCE_CACHE.move_to_end(key)
            return cached
    evidence = _build_source_evidence(record, episodes)
    if evidence.pinned_identities_match:
        with _SOURCE_EVIDENCE_CACHE_LOCK:
            _SOURCE_EVIDENCE_CACHE[key] = evidence
            _SOURCE_EVIDENCE_CACHE.move_to_end(key)
            while len(_SOURCE_EVIDENCE_CACHE) > _SOURCE_EVIDENCE_CACHE_LIMIT:
                _SOURCE_EVIDENCE_CACHE.popitem(last=False)
    return evidence


def _build_source_evidence(
    record: SourceRecord,
    episodes: Sequence[Mapping[str, Any]],
) -> _SourceAuditEvidence:
    unreadable: list[dict[str, Any]] = []
    timelines: dict[int, tuple[float, ...]] = {}
    videos: dict[int, bool] = {}
    grip: dict[tuple[int, str], GripDiagnostic] = {}
    pinned_matches = True
    fps: float | None = None
    try:
        info = _read_registered_json(record, "meta/info.json")
        raw_fps = info.get("fps")
        if isinstance(raw_fps, bool) or not isinstance(raw_fps, (int, float)):
            raise ValueError("invalid FPS")
        fps = float(raw_fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("invalid FPS")
    except _PinnedIdentityChanged:
        pinned_matches = False
        unreadable.append({"source_episode_index": None, "asset_kind": "metadata", "reason": "unreadable"})
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        unreadable.append({"source_episode_index": None, "asset_kind": "metadata", "reason": "unreadable"})
    for row in episodes:
        index = row["source_episode_index"]
        try:
            timelines[index] = _read_episode_timeline(record, index, row["source_length"])
        except _PinnedIdentityChanged:
            pinned_matches = False
            unreadable.append({"source_episode_index": index, "asset_kind": "parquet", "reason": "unreadable"})
        except (OSError, pa.ArrowException, TypeError, ValueError):
            unreadable.append({"source_episode_index": index, "asset_kind": "parquet", "reason": "unreadable"})
        try:
            videos[index] = _registered_video_readable(record, index)
        except _PinnedIdentityChanged:
            pinned_matches = False
            videos[index] = False
        if not videos[index]:
            unreadable.append({"source_episode_index": index, "asset_kind": "video", "reason": "unreadable"})
        hand = row["pickup_hand"]
        if hand in {"left", "right"}:
            diagnostic = read_grip_diagnostic(record, source_episode_index=index, side=hand)
            grip[(index, hand)] = diagnostic
            if diagnostic.reason == "source_changed":
                pinned_matches = False
    return _SourceAuditEvidence(
        fps=fps,
        timelines=timelines,
        video_readable=videos,
        grip=grip,
        unreadable=tuple(unreadable),
        pinned_identities_match=pinned_matches,
    )


def _database_snapshot(
    database: CurationDatabase,
    *,
    dataset_alias: str,
    snapshot_hook: Callable[[], None] | None,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    with database._read() as connection:
        connection.execute("BEGIN")
        try:
            dataset_row = connection.execute("SELECT * FROM datasets WHERE alias=?", (dataset_alias,)).fetchone()
            dataset = None if dataset_row is None else dict(dataset_row)
            if dataset is None:
                episodes: list[dict[str, Any]] = []
                jobs: list[dict[str, Any]] = []
                attempts: list[dict[str, Any]] = []
                proposals: list[dict[str, Any]] = []
                approval_events: list[dict[str, Any]] = []
            else:
                episodes = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM episodes WHERE dataset_id=? ORDER BY source_episode_index",
                        (dataset["id"],),
                    )
                ]
                if snapshot_hook is not None:
                    snapshot_hook()
                jobs = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT state FROM cosmos_jobs WHERE dataset_id=?",
                        (dataset["id"],),
                    )
                ]
                attempts = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT attempt.state, attempt.error_class
                        FROM cosmos_attempts AS attempt
                        JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                        WHERE job.dataset_id=?
                        """,
                        (dataset["id"],),
                    )
                ]
                proposals = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT proposal.id, proposal.state, proposal.model_response_json,
                            attempt.source_episode_index,
                            proposal.step_2_start_frame, proposal.step_3_start_frame,
                            proposal.step_4_start_frame, proposal.step_5_start_frame,
                            proposal.step_6_start_frame, proposal.step_7_start_frame
                        FROM cosmos_proposals AS proposal
                        JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                        JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                        WHERE job.dataset_id=?
                        """,
                        (dataset["id"],),
                    )
                ]
                approval_events = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT event.episode_id, event.new_revision, event.details_json,
                            episode.source_episode_index
                        FROM audit_events AS event
                        JOIN episodes AS episode ON episode.id=event.episode_id
                        WHERE event.dataset_id=? AND event.operation='keep_approved'
                        ORDER BY event.created_at, event.id
                        """,
                        (dataset["id"],),
                    )
                ]
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    return dataset, episodes, (jobs, attempts, proposals), approval_events


def _contact_sheet_issues(
    *,
    inspector: ContactSheetEvidenceInspector,
    identity: ContactSheetDatasetIdentity,
    proposals: Sequence[Mapping[str, Any]],
    approval_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for proposal in proposals:
        try:
            relative_path = proposal_contact_sheet_path(identity, proposal["id"])
            binding = {
                "schema_version": 1,
                "kind": "proposal",
                "dataset_id": identity.dataset_id,
                "dataset_alias": identity.dataset_alias,
                "source_manifest_sha256": identity.source_manifest_sha256,
                "source_episode_index": proposal["source_episode_index"],
                "proposal_id": proposal["id"],
                "approval_revision": None,
                "final_transition_frames": None,
                "proposal_transition_frames": [proposal[f"step_{step}_start_frame"] for step in range(2, 8)],
            }
            reason = inspector.status(relative_path, expected_binding=binding)
        except (TypeError, ValueError):
            reason = "conflict"
        if reason != "available":
            issues.append(
                {
                    "kind": "proposal",
                    "proposal_id": proposal["id"],
                    "source_episode_index": proposal["source_episode_index"],
                    "reason": reason,
                }
            )

    for event in approval_events:
        source_episode_index = event["source_episode_index"]
        approval_revision = event["new_revision"]
        try:
            details = json.loads(event["details_json"])
            snapshot = approval_evidence_snapshot(
                details,
                identity=identity,
                source_episode_index=source_episode_index,
                approval_revision=approval_revision,
            )
            relative_path = final_contact_sheet_path(identity, source_episode_index, approval_revision)
            binding = {
                "schema_version": 1,
                "kind": "final",
                "dataset_id": identity.dataset_id,
                "dataset_alias": identity.dataset_alias,
                "source_manifest_sha256": identity.source_manifest_sha256,
                "source_episode_index": source_episode_index,
                "proposal_id": snapshot["proposal_id"],
                "approval_revision": approval_revision,
                "final_transition_frames": snapshot["final_transition_frames"],
                "proposal_transition_frames": snapshot["proposal_transition_frames"],
            }
            reason = inspector.status(relative_path, expected_binding=binding)
        except (ContactSheetUnavailable, TypeError, ValueError, json.JSONDecodeError):
            reason = "conflict"
        if reason != "available":
            issues.append(
                {
                    "kind": "final",
                    "source_episode_index": source_episode_index,
                    "approval_revision": approval_revision,
                    "reason": reason,
                }
            )
    return sorted(
        issues,
        key=lambda item: (
            item["source_episode_index"],
            item["kind"],
            item.get("proposal_id", ""),
            item.get("approval_revision", 0),
        ),
    )


def _cosmos_summary(
    jobs: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    job_counts = Counter(row["state"] for row in jobs)
    attempt_counts = Counter(row["state"] for row in attempts)
    error_counts = Counter(row["error_class"] for row in attempts if row["error_class"])
    proposal_state_counts = Counter(row["state"] for row in proposals)
    results: Counter[str] = Counter()
    for proposal in proposals:
        try:
            response = json.loads(proposal["model_response_json"])
        except (TypeError, json.JSONDecodeError):
            results["invalid"] += 1
            continue
        if isinstance(response, Mapping) and response.get("episode_complete") is True:
            results["complete"] += 1
        elif isinstance(response, Mapping) and response.get("episode_complete") is False:
            results["incomplete"] += 1
        else:
            results["invalid"] += 1
    return {
        "job_state_counts": _all_counts(job_counts, tuple(state.value for state in JobState)),
        "attempt_state_counts": _all_counts(attempt_counts, tuple(state.value for state in AttemptState)),
        "proposal_state_counts": dict(sorted(proposal_state_counts.items())),
        "proposal_result_counts": dict(sorted(results.items())),
        "attempt_error_class_counts": dict(sorted(error_counts.items())),
    }


def _all_counts(counts: Counter[str], vocabulary: Sequence[str]) -> dict[str, int]:
    # Keep empty responses compact while returning every observed frozen enum value.
    return {state: counts[state] for state in vocabulary if counts[state]}


def _boundary_errors(
    source_episode_index: int,
    source_length: int,
    review_state: str,
    transitions: Sequence[Any],
) -> list[dict[str, Any]]:
    if source_length <= 0:
        return [
            {"source_episode_index": source_episode_index, "code": "zero_length_episode"},
            {"source_episode_index": source_episode_index, "code": "incomplete_coverage"},
        ]
    if review_state not in {ReviewState.DRAFT.value, ReviewState.APPROVED_KEEP.value}:
        return []
    errors: list[dict[str, Any]] = []
    if any(frame is None for frame in transitions):
        errors.append({"source_episode_index": source_episode_index, "code": "incomplete_coverage"})
    integers = [frame for frame in transitions if type(frame) is int]
    if len(integers) != len(transitions) or any(frame <= 0 or frame >= source_length for frame in integers):
        errors.append({"source_episode_index": source_episode_index, "code": "out_of_range_transition"})
    for position, (left, right) in enumerate(zip(transitions, transitions[1:]), start=2):
        if type(left) is not int or type(right) is not int:
            continue
        if right <= left:
            errors.append(
                {
                    "source_episode_index": source_episode_index,
                    "code": "non_increasing_transitions",
                    "step": position + 1,
                }
            )
        if right == left:
            errors.append(
                {
                    "source_episode_index": source_episode_index,
                    "code": "zero_length_phase",
                    "step": position + 1,
                }
            )
    if errors and not any(error["code"] == "incomplete_coverage" for error in errors):
        errors.append({"source_episode_index": source_episode_index, "code": "incomplete_coverage"})
    return errors


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min_s": None,
            "p25_s": None,
            "median_s": None,
            "p75_s": None,
            "max_s": None,
        }
    array = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(array, [0, 25, 50, 75, 100], method="linear")
    return {
        "count": int(array.size),
        "min_s": float(percentiles[0]),
        "p25_s": float(percentiles[1]),
        "median_s": float(percentiles[2]),
        "p75_s": float(percentiles[3]),
        "max_s": float(percentiles[4]),
    }


def _episode_paths(record: SourceRecord, source_episode_index: int) -> tuple[list[str], list[str]]:
    name = f"episode_{source_episode_index:06d}"
    parquet = sorted(
        path for path in record.file_hashes if path.startswith("data/") and path.endswith(f"/{name}.parquet")
    )
    videos = sorted(
        path
        for path in record.file_hashes
        if path.startswith("videos/") and "/observation.images.ego_view/" in path and path.endswith(f"/{name}.mp4")
    )
    return parquet, videos


def _read_episode_timeline(
    record: SourceRecord,
    source_episode_index: int,
    expected_length: int,
) -> tuple[float, ...]:
    paths, _ = _episode_paths(record, source_episode_index)
    if len(paths) != 1:
        raise ValueError("episode parquet is unavailable")
    asset = record.open_asset(paths[0])
    if asset is None:
        raise _PinnedIdentityChanged("episode parquet identity changed")
    try:
        with os.fdopen(os.dup(asset.fd), "rb") as handle:
            table = pq.read_table(handle, columns=["episode_index", "frame_index", "timestamp"])
        digest = _sha256_fd(asset.fd)
        if not record.verify_pinned_asset(asset, sha256=digest):
            raise _PinnedIdentityChanged("episode parquet changed")
    finally:
        asset.close()
    rows: list[tuple[int, float]] = []
    for episode, frame, timestamp in zip(
        table.column("episode_index").to_pylist(),
        table.column("frame_index").to_pylist(),
        table.column("timestamp").to_pylist(),
        strict=True,
    ):
        if int(episode) == source_episode_index:
            rows.append((int(frame), float(timestamp)))
    rows.sort(key=lambda row: row[0])
    if len(rows) != expected_length or [frame for frame, _ in rows] != list(range(expected_length)):
        raise ValueError("episode timeline does not match source length")
    timeline = tuple(timestamp for _, timestamp in rows)
    if any(not math.isfinite(value) for value in timeline) or any(
        left >= right for left, right in zip(timeline, timeline[1:])
    ):
        raise ValueError("episode timeline is invalid")
    return timeline


def _registered_video_readable(record: SourceRecord, source_episode_index: int) -> bool:
    _, paths = _episode_paths(record, source_episode_index)
    if len(paths) != 1:
        return False
    asset = record.open_asset(paths[0])
    if asset is None:
        raise _PinnedIdentityChanged("episode video identity changed")
    try:
        with os.fdopen(os.dup(asset.fd), "rb") as handle:
            with av.open(handle, mode="r") as container:
                if len(container.streams.video) != 1:
                    return False
        digest = _sha256_fd(asset.fd)
        if not record.verify_pinned_asset(asset, sha256=digest):
            raise _PinnedIdentityChanged("episode video changed")
        return True
    except (OSError, av.error.FFmpegError, ValueError):
        return False
    finally:
        asset.close()


def _read_registered_json(record: SourceRecord, relative_path: str) -> dict[str, Any]:
    asset = record.open_asset(relative_path)
    if asset is None:
        raise _PinnedIdentityChanged("registered JSON identity changed")
    try:
        contents = _read_fd(asset.fd)
        if not record.verify_pinned_asset(asset, sha256=hashlib.sha256(contents).hexdigest()):
            raise _PinnedIdentityChanged("registered JSON changed")
        document = json.loads(contents.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("registered JSON must be an object")
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
