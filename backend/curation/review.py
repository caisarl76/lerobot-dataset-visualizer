"""Revisioned human review workflow for pnp-trash curation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, Callable

import pyarrow.parquet as pq

from .db import (
    CurationDatabase,
    IllegalStateTransition,
    OptimisticConflict,
    PromptContractConflict,
    PromptMigrationConflict,
    PromptMigrationDowngrade,
    WorkspacePromptContractConflict,
    WorkspaceSourceConflict,
)
from .models import ReviewState
from .prompts import (
    PROMPT_TEMPLATE_SHA256,
    PROMPT_TEMPLATE_VERSION,
    expand_prompts,
    normalize_object_name,
)
from .source import SourceRecord, SourceRegistry

_UNSET = object()
_STEP_COLUMNS = tuple(f"step_{step}_start_frame" for step in range(2, 8))
_REVIEW_STATES = tuple(state.value for state in ReviewState)


class ReviewError(RuntimeError):
    status_code = 400

    def __init__(self, message: str, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(message)


class ReviewNotFound(ReviewError):
    status_code = 404


class ReviewConflict(ReviewError):
    status_code = 409


class ReviewValidation(ReviewError):
    status_code = 422

    def __init__(self, *issues: str) -> None:
        super().__init__("; ".join(issues), {"error": "invalid_review", "issues": list(issues)})


@dataclass
class _SourceDataset:
    record: SourceRecord
    lengths: dict[int, int]
    timestamp_cache: dict[int, list[float]] = field(default_factory=dict)

    @classmethod
    def load(cls, record: SourceRecord) -> "_SourceDataset":
        lengths = _read_episode_lengths(record)
        if not lengths:
            raise ReviewValidation("source dataset contains no episodes")
        return cls(record=record, lengths=lengths)

    def timestamps(self, source_episode_index: int) -> list[float]:
        cached = self.timestamp_cache.get(source_episode_index)
        if cached is not None:
            return list(cached)
        expected_length = self.lengths[source_episode_index]
        rows: list[tuple[int, float]] = []
        for relative_path in _candidate_episode_data_paths(self.record, source_episode_index):
            table = _read_registered_parquet(
                self.record,
                relative_path,
                required_columns={"episode_index", "frame_index", "timestamp"},
            )
            if table is None:
                continue
            episode_indices = table.column("episode_index").to_pylist()
            frame_indices = table.column("frame_index").to_pylist()
            timestamps = table.column("timestamp").to_pylist()
            rows.extend(
                (int(frame_index), float(timestamp))
                for episode, frame_index, timestamp in zip(episode_indices, frame_indices, timestamps, strict=True)
                if int(episode) == source_episode_index
            )
            if len(rows) >= expected_length:
                break
        rows.sort(key=lambda item: item[0])
        if [frame for frame, _ in rows] != list(range(expected_length)):
            raise ReviewValidation(
                f"episode {source_episode_index} timestamps do not match its declared source length"
            )
        values = [timestamp for _, timestamp in rows]
        if any(not math.isfinite(value) for value in values) or any(
            left >= right for left, right in zip(values, values[1:])
        ):
            raise ReviewValidation(f"episode {source_episode_index} timestamps must be finite and increasing")
        self.timestamp_cache[source_episode_index] = values
        return list(values)


def _read_episode_lengths(record: SourceRecord) -> dict[int, int]:
    if "meta/episodes.jsonl" in record.file_hashes:
        lengths: dict[int, int] = {}
        try:
            lines = _read_registered_bytes(record, "meta/episodes.jsonl").decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ReviewValidation("meta/episodes.jsonl must be UTF-8") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                length = int(row["length"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ReviewValidation(f"invalid meta/episodes.jsonl row {line_number}") from error
            if episode_index < 0 or length < 0 or episode_index in lengths:
                raise ReviewValidation(f"invalid meta/episodes.jsonl row {line_number}")
            lengths[episode_index] = length
        return lengths

    lengths: dict[int, int] = {}
    metadata_paths = sorted(
        path for path in record.file_hashes if path.startswith("meta/episodes/") and path.endswith(".parquet")
    )
    for relative_path in metadata_paths:
        table = _read_registered_parquet(
            record,
            relative_path,
            required_columns={"episode_index", "length"},
        )
        if table is None:
            continue
        for episode_index, length in zip(
            table.column("episode_index").to_pylist(), table.column("length").to_pylist(), strict=True
        ):
            index = int(episode_index)
            if index < 0 or int(length) < 0 or index in lengths:
                raise ReviewValidation("invalid or duplicate episode metadata")
            lengths[index] = int(length)
    return lengths


def _candidate_episode_data_paths(record: SourceRecord, source_episode_index: int) -> list[str]:
    exact_name = f"episode_{source_episode_index:06d}.parquet"
    paths = sorted(
        path for path in record.file_hashes if path.startswith("data/") and path.endswith(f"/{exact_name}")
    )
    if paths:
        return paths
    return sorted(path for path in record.file_hashes if path.startswith("data/") and path.endswith(".parquet"))


def _source_identity_conflict(record: SourceRecord) -> ReviewConflict:
    return ReviewConflict(
        "registered source file identity no longer matches the source fingerprint",
        {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
    )


def _persisted_lengths_match(
    database: CurationDatabase,
    dataset_id: int,
    source_lengths: dict[int, int],
) -> bool:
    persisted = {
        row["source_episode_index"]: row["source_length"] for row in database.list_episodes(dataset_id=dataset_id)
    }
    return persisted == source_lengths


def _read_registered_bytes(record: SourceRecord, relative_path: str) -> bytes:
    asset = record.open_asset(relative_path)
    if asset is None:
        raise _source_identity_conflict(record)
    try:
        chunks: list[bytes] = []
        offset = 0
        while chunk := os.pread(asset.fd, 1024 * 1024, offset):
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)
    finally:
        asset.close()


def _read_registered_parquet(
    record: SourceRecord,
    relative_path: str,
    *,
    required_columns: set[str],
) -> Any | None:
    asset = record.open_asset(relative_path)
    if asset is None:
        raise _source_identity_conflict(record)
    duplicate = -1
    try:
        duplicate = os.dup(asset.fd)
        with os.fdopen(duplicate, "rb") as handle:
            duplicate = -1
            parquet = pq.ParquetFile(handle)
            if not required_columns.issubset(parquet.schema_arrow.names):
                return None
            return parquet.read(columns=sorted(required_columns))
    finally:
        if duplicate >= 0:
            os.close(duplicate)
        asset.close()


class ReviewService:
    """Application service that keeps model proposals separate from human finals."""

    def __init__(
        self,
        *,
        database: CurationDatabase,
        source_registry: SourceRegistry,
        prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
        prompt_template_sha256: str = PROMPT_TEMPLATE_SHA256,
        prompt_expander: Callable[..., list[str]] = expand_prompts,
    ) -> None:
        self.database = database
        self.source_registry = source_registry
        self.prompt_template_version = prompt_template_version
        self.prompt_template_sha256 = prompt_template_sha256
        self._prompt_expander = prompt_expander
        self._sources: dict[str, _SourceDataset] = {}

    def open_workspace(self, dataset_alias: str, *, actor: str) -> dict[str, Any]:
        actor = _identity(actor, "actor")
        record = self.source_registry.records.get(dataset_alias)
        if record is None:
            raise ReviewNotFound(
                "dataset alias is not registered",
                {"error": "dataset_alias_not_found", "dataset_alias": dataset_alias},
            )
        if not record.verify_current_inventory():
            raise _source_identity_conflict(record)
        if dataset_alias in self._sources:
            current_dataset = self.database.get_dataset(alias=dataset_alias)
            if current_dataset is not None and (
                current_dataset["prompt_template_version"] != self.prompt_template_version
                or current_dataset["prompt_template_sha256"] != self.prompt_template_sha256
            ):
                raise _stale_service_conflict(dataset_alias, current_dataset)
        source = self._sources.get(dataset_alias) or _SourceDataset.load(record)
        try:
            self.database.open_review_workspace(
                alias=dataset_alias,
                source_path=str(record.root),
                source_manifest_sha256=record.fingerprint,
                episode_lengths=source.lengths,
                prompt_template_version=self.prompt_template_version,
                prompt_template_sha256=self.prompt_template_sha256,
                actor=actor,
            )
        except WorkspaceSourceConflict as error:
            raise ReviewConflict(
                "registered source fingerprint does not match current source",
                {"error": "source_fingerprint_mismatch", "dataset_alias": dataset_alias},
            ) from error
        except WorkspacePromptContractConflict as error:
            raise _stale_service_conflict(dataset_alias, error.current_dataset) from error
        self._sources[dataset_alias] = source
        return {
            "dataset_alias": dataset_alias,
            "source_fingerprint": record.fingerprint,
            "prompt_template_version": self.prompt_template_version,
            "prompt_template_sha256": self.prompt_template_sha256,
            "episode_count": len(source.lengths),
        }

    def migrate_prompt_contract(
        self,
        dataset_alias: str,
        *,
        expected_prompt_template_version: str,
        expected_prompt_template_sha256: str,
        actor: str,
    ) -> dict[str, Any]:
        """Explicitly CAS the configured prompt contract into one workspace."""
        actor = _identity(actor, "actor")
        record = self.source_registry.records.get(dataset_alias)
        if record is None:
            raise ReviewNotFound(
                "dataset alias is not registered",
                {"error": "dataset_alias_not_found", "dataset_alias": dataset_alias},
            )
        if not record.verify_current_inventory():
            raise _source_identity_conflict(record)
        dataset = self.database.get_dataset(alias=dataset_alias)
        if dataset is None:
            raise ReviewConflict(
                "workspace has not been initialized",
                {"error": "workspace_not_open", "dataset_alias": dataset_alias},
            )
        if dataset["source_path"] != str(record.root) or dataset["source_manifest_sha256"] != record.fingerprint:
            raise _source_identity_conflict(record)
        source = self._sources.get(dataset_alias) or _SourceDataset.load(record)
        if not _persisted_lengths_match(self.database, dataset["id"], source.lengths):
            raise _source_identity_conflict(record)
        try:
            invalidated = self.database.migrate_prompt_template(
                dataset_id=dataset["id"],
                expected_prompt_template_version=expected_prompt_template_version,
                expected_prompt_template_sha256=expected_prompt_template_sha256,
                prompt_template_version=self.prompt_template_version,
                prompt_template_sha256=self.prompt_template_sha256,
                actor=actor,
            )
        except (PromptMigrationConflict, PromptMigrationDowngrade) as error:
            raise ReviewConflict(str(error), error.payload) from error
        self._sources[dataset_alias] = source
        return {
            "dataset_alias": dataset_alias,
            "prompt_template_version": self.prompt_template_version,
            "prompt_template_sha256": self.prompt_template_sha256,
            "invalidated_episode_indices": [row["source_episode_index"] for row in invalidated],
        }

    def summary(self, dataset_alias: str) -> dict[str, Any]:
        dataset, source = self._workspace(dataset_alias)
        counts = {state: 0 for state in _REVIEW_STATES}
        for episode in self.database.list_episodes(dataset_id=dataset["id"]):
            counts[episode["review_state"]] += 1
        return {
            "dataset_alias": dataset_alias,
            "episode_count": len(source.lengths),
            "counts": counts,
            "prompt_template_version": self.prompt_template_version,
            "prompt_template_sha256": self.prompt_template_sha256,
        }

    def get_episode(self, dataset_alias: str, source_episode_index: int) -> dict[str, Any]:
        dataset, source = self._workspace(dataset_alias)
        row = self.database.get_episode(dataset_id=dataset["id"], source_episode_index=source_episode_index)
        if row is None or source_episode_index not in source.lengths:
            raise ReviewNotFound(
                "episode not found",
                {"error": "episode_not_found", "source_episode_index": source_episode_index},
            )
        return self._episode_response(dataset_alias, dataset, source, row)

    def save_draft(
        self,
        *,
        dataset_alias: str,
        source_episode_index: int,
        expected_revision: int,
        actor: str,
        object_name: str | None | object = _UNSET,
        pickup_hand: str | None | object = _UNSET,
        turn_direction: str | None | object = _UNSET,
        transition_frames: list[int | None] | None | object = _UNSET,
    ) -> dict[str, Any]:
        dataset, source = self._workspace(dataset_alias)
        actor = _identity(actor, "actor")
        self._preflight(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            frozenset({ReviewState.PENDING, ReviewState.DRAFT}),
            "draft_saved",
        )
        changes: dict[str, Any] = {
            "review_state": ReviewState.DRAFT,
            "approval_revision": None,
            "reviewer": None,
            "approved_at": None,
            "prompt_template_sha256": self.prompt_template_sha256,
        }
        if object_name is not _UNSET:
            if object_name is None:
                changes["object_name"] = None
            else:
                try:
                    changes["object_name"] = normalize_object_name(object_name)
                except (TypeError, ValueError) as error:
                    raise ReviewValidation(str(error)) from error
        if pickup_hand is not _UNSET:
            changes["pickup_hand"] = _optional_enum(pickup_hand, "pickup_hand")
        if turn_direction is not _UNSET:
            changes["turn_direction"] = _optional_enum(turn_direction, "turn_direction")
        if transition_frames is not _UNSET:
            transitions = _validate_transitions(transition_frames, source.lengths[source_episode_index], False)
            changes.update(dict(zip(_STEP_COLUMNS, transitions, strict=True)))
        return self._transition(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            actor,
            frozenset({ReviewState.PENDING, ReviewState.DRAFT}),
            changes,
            "draft_saved",
        )

    def apply_proposal(
        self,
        *,
        dataset_alias: str,
        source_episode_index: int,
        expected_revision: int,
        actor: str,
    ) -> dict[str, Any]:
        dataset, source = self._workspace(dataset_alias)
        actor = _identity(actor, "actor")
        self._preflight(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            frozenset({ReviewState.PENDING, ReviewState.DRAFT}),
            "proposal_applied",
        )
        proposal = self.database.get_active_proposal(
            dataset_id=dataset["id"], source_episode_index=source_episode_index
        )
        if proposal is None:
            raise ReviewNotFound(
                "active proposal not found",
                {"error": "active_proposal_not_found", "source_episode_index": source_episode_index},
            )
        transitions = [proposal[column] for column in _STEP_COLUMNS]
        transitions = _validate_transitions(transitions, source.lengths[source_episode_index], False)
        changes: dict[str, Any] = dict(zip(_STEP_COLUMNS, transitions, strict=True))
        changes.update(
            {
                "review_state": ReviewState.DRAFT,
                "approval_revision": None,
                "reviewer": None,
                "approved_at": None,
                "prompt_template_sha256": self.prompt_template_sha256,
            }
        )
        return self._transition(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            actor,
            frozenset({ReviewState.PENDING, ReviewState.DRAFT}),
            changes,
            "proposal_applied",
        )

    def approve_keep(
        self,
        *,
        dataset_alias: str,
        source_episode_index: int,
        expected_revision: int,
        actor: str,
        reviewer: str,
    ) -> dict[str, Any]:
        dataset, source = self._workspace(dataset_alias)
        actor = _identity(actor, "actor")
        reviewer = _identity(reviewer, "reviewer")
        row = self._preflight(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            frozenset({ReviewState.DRAFT}),
            "keep_approved",
        )
        issues: list[str] = []
        normalized_object: str | None = None
        try:
            normalized_object = normalize_object_name(row["object_name"])
        except (TypeError, ValueError):
            issues.append("object_name must be nonempty")
        else:
            if normalized_object != row["object_name"]:
                issues.append("object_name is not already normalized")
        if row["pickup_hand"] not in {"left", "right"}:
            issues.append("pickup_hand must be left or right")
        if row["turn_direction"] not in {"left", "right"}:
            issues.append("turn_direction must be left or right")
        if (
            normalized_object is not None
            and normalized_object == row["object_name"]
            and row["pickup_hand"] in {"left", "right"}
            and row["turn_direction"] in {"left", "right"}
        ):
            canonical_prompts = expand_prompts(
                object_name=normalized_object,
                hand=row["pickup_hand"],
                turn=row["turn_direction"],
            )
            try:
                prompts = self._prompt_expander(
                    object_name=normalized_object,
                    hand=row["pickup_hand"],
                    turn=row["turn_direction"],
                )
            except Exception:
                prompts = None
            if (
                not isinstance(prompts, list)
                or len(prompts) != 7
                or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
                or prompts != canonical_prompts
            ):
                issues.append(
                    "prompt expansion must yield exactly seven nonempty prompts matching the canonical generator"
                )
        try:
            _validate_transitions(
                [row[column] for column in _STEP_COLUMNS], source.lengths[source_episode_index], True
            )
        except ReviewValidation as error:
            issues.extend(error.payload["issues"])
        if row["prompt_template_sha256"] != dataset["prompt_template_sha256"]:
            issues.append("draft prompt template hash is stale")
        if issues:
            raise ReviewValidation(*issues)
        new_revision = expected_revision + 1
        return self._transition(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            actor,
            frozenset({ReviewState.DRAFT}),
            {
                "review_state": ReviewState.APPROVED_KEEP,
                "approval_revision": new_revision,
                "reviewer": reviewer,
                "approved_at": _utc_now(),
                "rejection_reason": None,
            },
            "keep_approved",
            required_prompt_template_version=self.prompt_template_version,
            required_prompt_template_sha256=self.prompt_template_sha256,
        )

    def approve_reject(
        self,
        *,
        dataset_alias: str,
        source_episode_index: int,
        expected_revision: int,
        actor: str,
        reviewer: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        dataset, source = self._workspace(dataset_alias)
        actor = _identity(actor, "actor")
        reviewer = _identity(reviewer, "reviewer")
        self._preflight(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            frozenset({ReviewState.PENDING, ReviewState.DRAFT}),
            "reject_approved",
        )
        return self._transition(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            actor,
            frozenset({ReviewState.PENDING, ReviewState.DRAFT}),
            {
                "review_state": ReviewState.APPROVED_REJECT,
                "approval_revision": expected_revision + 1,
                "reviewer": reviewer,
                "approved_at": _utc_now(),
                "rejection_reason": reason,
            },
            "reject_approved",
        )

    def reopen(
        self,
        *,
        dataset_alias: str,
        source_episode_index: int,
        expected_revision: int,
        actor: str,
    ) -> dict[str, Any]:
        dataset, source = self._workspace(dataset_alias)
        actor = _identity(actor, "actor")
        self._preflight(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            frozenset({ReviewState.APPROVED_KEEP, ReviewState.APPROVED_REJECT}),
            "approval_reopened",
        )
        return self._transition(
            dataset_alias,
            dataset,
            source,
            source_episode_index,
            expected_revision,
            actor,
            frozenset({ReviewState.APPROVED_KEEP, ReviewState.APPROVED_REJECT}),
            {
                "review_state": ReviewState.DRAFT,
                "approval_revision": None,
                "reviewer": None,
                "approved_at": None,
            },
            "approval_reopened",
        )

    def _workspace(self, dataset_alias: str) -> tuple[dict[str, Any], _SourceDataset]:
        record = self.source_registry.records.get(dataset_alias)
        dataset = self.database.get_dataset(alias=dataset_alias)
        if record is None:
            raise ReviewNotFound(
                "dataset alias is not registered",
                {"error": "dataset_alias_not_found", "dataset_alias": dataset_alias},
            )
        if not record.verify_current_inventory():
            raise _source_identity_conflict(record)
        if dataset is None:
            raise ReviewConflict(
                "workspace has not been opened",
                {"error": "workspace_not_open", "dataset_alias": dataset_alias},
            )
        if dataset["source_path"] != str(record.root) or dataset["source_manifest_sha256"] != record.fingerprint:
            raise ReviewConflict(
                "registered source fingerprint does not match current source",
                {"error": "source_fingerprint_mismatch", "dataset_alias": dataset_alias},
            )
        if (
            dataset["prompt_template_version"] != self.prompt_template_version
            or dataset["prompt_template_sha256"] != self.prompt_template_sha256
        ):
            raise _stale_service_conflict(dataset_alias, dataset)
        source = self._sources.get(dataset_alias)
        if source is None:
            source = _SourceDataset.load(record)
            if not _persisted_lengths_match(self.database, dataset["id"], source.lengths):
                raise _source_identity_conflict(record)
            self._sources[dataset_alias] = source
        return dataset, source

    def _episode_row(self, dataset_id: int, source_episode_index: int) -> dict[str, Any]:
        row = self.database.get_episode(dataset_id=dataset_id, source_episode_index=source_episode_index)
        if row is None:
            raise ReviewNotFound(
                "episode not found",
                {"error": "episode_not_found", "source_episode_index": source_episode_index},
            )
        return row

    def _preflight(
        self,
        dataset_alias: str,
        dataset: dict[str, Any],
        source: _SourceDataset,
        source_episode_index: int,
        expected_revision: int,
        allowed_states: frozenset[ReviewState],
        operation: str,
    ) -> dict[str, Any]:
        row = self._episode_row(dataset["id"], source_episode_index)
        if row["revision"] != expected_revision:
            raise ReviewConflict(
                "episode revision does not match expected_revision",
                {
                    "error": "revision_conflict",
                    "episode": self._episode_response(dataset_alias, dataset, source, row),
                },
            )
        current_state = ReviewState(row["review_state"])
        if current_state not in allowed_states:
            error_name = (
                "approval_locked"
                if current_state in {ReviewState.APPROVED_KEEP, ReviewState.APPROVED_REJECT}
                else "invalid_review_transition"
            )
            raise ReviewConflict(
                "review state does not allow this operation",
                {
                    "error": error_name,
                    "operation": operation,
                    "current_state": current_state.value,
                },
            )
        return row

    def _transition(
        self,
        dataset_alias: str,
        dataset: dict[str, Any],
        source: _SourceDataset,
        source_episode_index: int,
        expected_revision: int,
        actor: str,
        allowed_states: frozenset[ReviewState],
        changes: dict[str, Any],
        operation: str,
        *,
        required_prompt_template_version: str | None = None,
        required_prompt_template_sha256: str | None = None,
    ) -> dict[str, Any]:
        if source_episode_index not in source.lengths:
            raise ReviewNotFound(
                "episode not found",
                {"error": "episode_not_found", "source_episode_index": source_episode_index},
            )
        try:
            row = self.database.transition_review_episode(
                dataset_id=dataset["id"],
                source_episode_index=source_episode_index,
                expected_revision=expected_revision,
                allowed_states=allowed_states,
                changes=changes,
                actor=actor,
                operation=operation,
                required_prompt_template_version=required_prompt_template_version,
                required_prompt_template_sha256=required_prompt_template_sha256,
            )
        except PromptContractConflict as error:
            raise ReviewConflict(str(error), error.payload) from error
        except OptimisticConflict as error:
            raise ReviewConflict(
                "episode revision does not match expected_revision",
                {
                    "error": "revision_conflict",
                    "episode": self._episode_response(dataset_alias, dataset, source, error.current_episode),
                },
            ) from error
        except IllegalStateTransition as error:
            current_state = error.payload["current_state"]
            error_name = (
                "approval_locked" if current_state.startswith("approved_") else "invalid_review_transition"
            )
            raise ReviewConflict(
                "review state does not allow this operation",
                {
                    "error": error_name,
                    "operation": operation,
                    "current_state": current_state,
                },
            ) from error
        return self._episode_response(dataset_alias, dataset, source, row)

    def _episode_response(
        self,
        dataset_alias: str,
        dataset: dict[str, Any],
        source: _SourceDataset,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        source_episode_index = row["source_episode_index"]
        proposal_row = self.database.get_active_proposal(
            dataset_id=dataset["id"], source_episode_index=source_episode_index
        )
        proposal = _proposal_response(proposal_row)
        prompts: list[str] | None = None
        if (
            row["object_name"] is not None
            and row["pickup_hand"] in {"left", "right"}
            and row["turn_direction"] in {"left", "right"}
        ):
            try:
                prompts = self._prompt_expander(
                    object_name=row["object_name"],
                    hand=row["pickup_hand"],
                    turn=row["turn_direction"],
                )
            except Exception:
                prompts = None
        proposal_warnings = [] if proposal is None else proposal["warnings"]
        warnings = sorted(set([*proposal_warnings, *_decision_warnings(row, dataset, source)]))
        return {
            "dataset_alias": dataset_alias,
            "source_episode_index": source_episode_index,
            "source_length": row["source_length"],
            "timestamps": source.timestamps(source_episode_index),
            "decision": {
                "review_state": row["review_state"],
                "object_name": row["object_name"],
                "pickup_hand": row["pickup_hand"],
                "turn_direction": row["turn_direction"],
                "transition_frames": [row[column] for column in _STEP_COLUMNS],
                "rejection_reason": row["rejection_reason"],
                "prompt_template_sha256": row["prompt_template_sha256"],
            },
            "active_proposal": proposal,
            "prompt_preview": prompts,
            "warnings": warnings,
            "revision": row["revision"],
            "approval_locked": row["review_state"]
            in {ReviewState.APPROVED_KEEP.value, ReviewState.APPROVED_REJECT.value},
            "reviewer": row["reviewer"],
            "approval_revision": row["approval_revision"],
            "approved_at": row["approved_at"],
        }


def _proposal_response(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "attempt_id": row["attempt_id"],
        "model_response": json.loads(row["model_response_json"]),
        "transition_frames": [row[column] for column in _STEP_COLUMNS],
        "warnings": json.loads(row["validation_warnings_json"]),
        "created_at": row["created_at"],
    }


def _stale_service_conflict(dataset_alias: str, dataset: dict[str, Any]) -> ReviewConflict:
    return ReviewConflict(
        "review service prompt contract is stale",
        {
            "error": "stale_review_service",
            "dataset_alias": dataset_alias,
            "current_prompt_template_version": dataset["prompt_template_version"],
            "current_prompt_template_sha256": dataset["prompt_template_sha256"],
        },
    )


def _decision_warnings(row: dict[str, Any], dataset: dict[str, Any], source: _SourceDataset) -> list[str]:
    state = ReviewState(row["review_state"])
    if state is ReviewState.PENDING:
        return []
    warnings: list[str] = []
    prefix = "draft" if state is ReviewState.DRAFT else state.value
    if state in {ReviewState.DRAFT, ReviewState.APPROVED_KEEP}:
        try:
            normalized_object = normalize_object_name(row["object_name"])
        except (TypeError, ValueError):
            warnings.append(f"{prefix}_object_name_invalid")
        else:
            if normalized_object != row["object_name"]:
                warnings.append(f"{prefix}_object_name_invalid")
        if row["pickup_hand"] not in {"left", "right"}:
            warnings.append(f"{prefix}_pickup_hand_invalid")
        if row["turn_direction"] not in {"left", "right"}:
            warnings.append(f"{prefix}_turn_direction_invalid")
        try:
            _validate_transitions(
                [row[column] for column in _STEP_COLUMNS],
                source.lengths[row["source_episode_index"]],
                True,
            )
        except ReviewValidation:
            warnings.append(f"{prefix}_transition_frames_invalid")
        if row["prompt_template_sha256"] != dataset["prompt_template_sha256"]:
            warnings.append(f"{prefix}_prompt_template_stale")
    if state in {ReviewState.APPROVED_KEEP, ReviewState.APPROVED_REJECT}:
        if not isinstance(row["reviewer"], str) or not row["reviewer"].strip():
            warnings.append("approval_reviewer_missing")
        if row["approval_revision"] != row["revision"]:
            warnings.append("approval_revision_mismatch")
        if not row["approved_at"]:
            warnings.append("approval_timestamp_missing")
    elif any(row[field] is not None for field in ("reviewer", "approval_revision", "approved_at")):
        warnings.append("draft_approval_fields_stale")
    return warnings


def _identity(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewValidation(f"{field_name} must be nonempty")
    return value.strip()


def _optional_enum(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if value not in {"left", "right"}:
        raise ReviewValidation(f"{field_name} must be left or right")
    return str(value)


def _validate_transitions(values: object, source_length: int, require_complete: bool) -> list[int | None]:
    if not isinstance(values, list) or len(values) != 6:
        raise ReviewValidation("transition_frames must contain exactly six entries")
    normalized: list[int | None] = []
    previous = 0
    for value in values:
        if value is None:
            if require_complete:
                raise ReviewValidation("approved_keep requires six transition frames")
            normalized.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReviewValidation("transition frames must be integers or null")
        if value <= previous or value <= 0 or value >= source_length:
            raise ReviewValidation(
                "transition frames must be strictly increasing, in range, and produce seven nonempty spans"
            )
        normalized.append(value)
        previous = value
    return normalized


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
