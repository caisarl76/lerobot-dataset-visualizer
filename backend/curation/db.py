"""SQLite persistence for the isolated, resumable curation workspace.

This module deliberately opens a fresh configured connection for each public
operation.  Curation has a single authoritative SQLite database, while browser
requests, the worker, and exporter are independent processes; connection-local
PRAGMAs therefore cannot be treated as one-time process setup.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Any
from uuid import uuid4

from .models import (
    EXPORT_STATE_TRANSITIONS,
    JOB_STATE_TRANSITIONS,
    AttemptState,
    ExportState,
    JobState,
    ProposalState,
    ReviewState,
)

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MILLISECONDS = 5_000
ARTIFACT_KIND_COSMOS_REQUEST = "request"
ARTIFACT_KIND_COSMOS_RESPONSE = "response"
ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE = "repair_response"
ARTIFACT_KIND_COSMOS_PARSED = "parsed"
ARTIFACT_KIND_STRUCTURAL_REPORT = "structural_report"
ARTIFACT_KIND_GROOT_STATS_REPORT = "gr00t_stats_report"
ARTIFACT_KIND_GROOT_LOADER_REPORT = "gr00t_loader_report"
ARTIFACT_KIND_FINAL_CONSISTENCY_REPORT = "final_consistency_report"
_STEP_COLUMNS = tuple(f"step_{step}_start_frame" for step in range(2, 8))
_EPISODE_CHANGE_COLUMNS = frozenset(
    {
        "review_state",
        "object_name",
        "pickup_hand",
        "turn_direction",
        *_STEP_COLUMNS,
        "approval_revision",
        "reviewer",
        "approved_at",
        "rejection_reason",
        "prompt_template_sha256",
    }
)


class RetryableDatabaseError(RuntimeError):
    """A lock timeout that callers may safely retry."""

    status_code = 503
    payload = {"error": "database_busy", "retryable": True}


class IncompatibleCurationDatabase(RuntimeError):
    """The worker was pointed at a database other than the exact current schema."""


class OptimisticConflict(RuntimeError):
    """An HTTP-ready revision conflict containing the newest episode record."""

    status_code = 409

    def __init__(self, current_episode: dict[str, Any]) -> None:
        self.current_episode = current_episode
        self.payload = {"error": "revision_conflict", "episode": current_episode}
        super().__init__("episode revision does not match expected_revision")


class StateTransitionConflict(RuntimeError):
    """A delayed worker attempted a state transition from a stale predecessor."""

    status_code = 409

    def __init__(self, *, entity: str, identifier: str, expected_state: str, current_state: str) -> None:
        self.payload = {
            "error": "state_transition_conflict",
            "entity": entity,
            "id": identifier,
            "expected_state": expected_state,
            "current_state": current_state,
        }
        super().__init__(f"{entity} {identifier} is {current_state}, not expected {expected_state}")


class IllegalStateTransition(RuntimeError):
    """A compare-and-swap matched but the requested lifecycle edge is not legal."""

    status_code = 409

    def __init__(self, *, entity: str, identifier: str, current_state: str, target_state: str) -> None:
        self.payload = {
            "error": "illegal_state_transition",
            "entity": entity,
            "id": identifier,
            "current_state": current_state,
            "target_state": target_state,
        }
        super().__init__(f"{entity} {identifier} cannot transition from {current_state} to {target_state}")


class WorkspaceSourceConflict(RuntimeError):
    """A registered alias no longer describes the exact same immutable source."""

    def __init__(self, alias: str) -> None:
        self.alias = alias
        super().__init__(f"registered workspace source does not match alias {alias}")


class WorkspacePromptContractConflict(RuntimeError):
    """A normal workspace open used a different prompt contract."""

    def __init__(self, alias: str, current_dataset: dict[str, Any]) -> None:
        self.alias = alias
        self.current_dataset = current_dataset
        super().__init__(f"workspace prompt contract does not match alias {alias}")


class ArtifactReconciliationConflict(RuntimeError):
    """Persisted artifact evidence or its attempt pointer differs from an exact retry."""

    status_code = 409
    payload = {"error": "artifact_reconciliation_conflict"}

    def __init__(self) -> None:
        super().__init__("persisted artifact evidence does not match the exact reconciliation request")


class PromptMigrationConflict(RuntimeError):
    """An explicit prompt migration lost its compare-and-swap."""

    status_code = 409

    def __init__(
        self,
        *,
        expected_version: str,
        expected_sha256: str,
        current_version: str,
        current_sha256: str,
    ) -> None:
        self.payload = {
            "error": "prompt_migration_conflict",
            "expected_prompt_template_version": expected_version,
            "expected_prompt_template_sha256": expected_sha256,
            "current_prompt_template_version": current_version,
            "current_prompt_template_sha256": current_sha256,
        }
        super().__init__("prompt migration expected contract is no longer current")


class PromptMigrationDowngrade(RuntimeError):
    """An explicit migration attempted a non-increasing prompt version."""

    status_code = 409

    def __init__(self, *, current_version: str, requested_version: str) -> None:
        self.payload = {
            "error": "prompt_migration_downgrade",
            "current_prompt_template_version": current_version,
            "requested_prompt_template_version": requested_version,
        }
        super().__init__("prompt migration must increase the version within the same contract lineage")


class PromptContractConflict(RuntimeError):
    """Approval raced with a dataset prompt-contract update."""

    status_code = 409

    def __init__(
        self,
        *,
        required_version: str,
        required_sha256: str,
        dataset_version: str,
        dataset_sha256: str,
        episode_sha256: str | None,
    ) -> None:
        self.payload = {
            "error": "prompt_contract_conflict",
            "required_prompt_template_version": required_version,
            "required_prompt_template_sha256": required_sha256,
            "dataset_prompt_template_version": dataset_version,
            "dataset_prompt_template_sha256": dataset_sha256,
            "episode_prompt_template_sha256": episode_sha256,
        }
        super().__init__("prompt contract changed before approval could be committed")


def canonical_json(value: Any) -> str:
    """Return the one JSON representation used by every hash-bearing record."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _enum_values(enum: type[Any]) -> str:
    return ", ".join(repr(member.value) for member in enum)


def _prompt_contract_generation(version: str) -> tuple[str, int] | None:
    lineage, separator, generation = version.rpartition("-v")
    if not separator or not lineage or not generation.isdecimal() or generation.startswith("0"):
        return None
    number = int(generation)
    if number < 1:
        return None
    return lineage, number


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else dict(row)


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be a nonempty POSIX-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must not be absolute or contain traversal components")
    return path.as_posix()


def _is_lock_error(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _validate_http_exchange(exchange: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one neutral HTTP-envelope record before canonical append-only storage."""
    required_keys = {"phase", "started_at", "finished_at", "request", "response", "error"}
    if set(exchange) != required_keys:
        raise ValueError("HTTP exchange must contain exactly phase, timing, request, response, and error")
    phase = exchange["phase"]
    if phase not in {"initial", "repair"}:
        raise ValueError("HTTP exchange phase must be initial or repair")
    if not all(isinstance(exchange[key], str) and exchange[key] for key in ("started_at", "finished_at")):
        raise ValueError("HTTP exchange timestamps must be nonempty strings")
    request = exchange["request"]
    if not isinstance(request, Mapping) or set(request) != {"method", "url", "body_sha256"}:
        raise ValueError("HTTP exchange request must contain method, url, and body_sha256")
    if not isinstance(request["method"], str) or not request["method"]:
        raise ValueError("HTTP exchange request method must be a nonempty string")
    if not isinstance(request["url"], str) or not request["url"]:
        raise ValueError("HTTP exchange request URL must be a nonempty string")
    body_sha256 = request["body_sha256"]
    if (
        not isinstance(body_sha256, str)
        or len(body_sha256) != 64
        or any(char not in "0123456789abcdef" for char in body_sha256)
    ):
        raise ValueError("HTTP exchange request body_sha256 must be lowercase SHA-256")

    response = exchange["response"]
    if response is not None:
        response_keys = {"status_code", "id", "model", "created", "usage", "finish_reason"}
        if not isinstance(response, Mapping) or set(response) != response_keys:
            raise ValueError("HTTP exchange response has an invalid envelope shape")
        if not isinstance(response["status_code"], int) or not 100 <= response["status_code"] <= 599:
            raise ValueError("HTTP exchange response status_code must be an HTTP status")
        if any(
            response[key] is not None and not isinstance(response[key], str)
            for key in ("id", "model", "finish_reason")
        ):
            raise ValueError("HTTP exchange response identifiers must be strings or null")
        if response["created"] is not None and not isinstance(response["created"], int):
            raise ValueError("HTTP exchange response created must be an integer or null")
        if response["usage"] is not None and not isinstance(response["usage"], Mapping):
            raise ValueError("HTTP exchange response usage must be an object or null")

    error = exchange["error"]
    if error is not None:
        if not isinstance(error, Mapping) or set(error) != {"class", "summary"}:
            raise ValueError("HTTP exchange error has an invalid envelope shape")
        if not all(isinstance(error[key], str) and error[key] for key in ("class", "summary")):
            raise ValueError("HTTP exchange error values must be nonempty strings")
    if response is None and error is None:
        raise ValueError("HTTP exchange must contain a response envelope or transport error")
    return json.loads(canonical_json(dict(exchange)))


class CurationDatabase:
    """Versioned SQLite repository with short serialized write transactions."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def open_connection(self) -> sqlite3.Connection:
        """Open a fresh connection with all required connection-local settings."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=BUSY_TIMEOUT_MILLISECONDS / 1_000)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MILLISECONDS}")
        except Exception:
            connection.close()
            raise
        return connection

    def validate_worker_compatibility(self) -> None:
        """Read-only gate for the exact schema the standalone worker may mutate."""

        if not self.path.is_file():
            raise IncompatibleCurationDatabase("curation database is not a regular file")
        connection: sqlite3.Connection | None = None
        try:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True, isolation_level=None)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            version_row = connection.execute("PRAGMA user_version").fetchone()
            if version_row is None or type(version_row[0]) is not int or version_row[0] != SCHEMA_VERSION:
                raise IncompatibleCurationDatabase("curation database schema version is incompatible")
            observed = _schema_signature(connection)
            if observed != _expected_v1_schema_signature():
                raise IncompatibleCurationDatabase("curation database schema manifest is incompatible")
            check_row = connection.execute("PRAGMA quick_check(1)").fetchone()
            if check_row is None or tuple(check_row) != ("ok",):
                raise IncompatibleCurationDatabase("curation database integrity check failed")
        except IncompatibleCurationDatabase:
            raise
        except (IndexError, TypeError, ValueError, sqlite3.DatabaseError):
            raise IncompatibleCurationDatabase("curation database could not be validated") from None
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = self.open_connection()
        except sqlite3.OperationalError as error:
            if _is_lock_error(error):
                raise RetryableDatabaseError("curation database is busy; retry the operation") from error
            raise
        try:
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                if _is_lock_error(error):
                    raise RetryableDatabaseError("curation database is busy; retry the operation") from error
                raise
            try:
                yield connection
            except sqlite3.OperationalError as error:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                if _is_lock_error(error):
                    raise RetryableDatabaseError("curation database is busy; retry the operation") from error
                raise
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            else:
                try:
                    connection.execute("COMMIT")
                except sqlite3.OperationalError as error:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    if _is_lock_error(error):
                        raise RetryableDatabaseError("curation database is busy; retry the operation") from error
                    raise
        finally:
            connection.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = self.open_connection()
        except sqlite3.OperationalError as error:
            if _is_lock_error(error):
                raise RetryableDatabaseError("curation database is busy; retry the operation") from error
            raise
        try:
            try:
                yield connection
            except sqlite3.OperationalError as error:
                if _is_lock_error(error):
                    raise RetryableDatabaseError("curation database is busy; retry the operation") from error
                raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Apply the v1 schema exactly once in a single write transaction."""
        with self._write() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"database schema {version} is newer than supported {SCHEMA_VERSION}")
            if version == SCHEMA_VERSION:
                return
            if version != 0:
                raise RuntimeError(f"unsupported curation schema version {version}")
            for statement in _migration_v1_statements():
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def register_dataset(
        self,
        *,
        alias: str,
        source_path: str,
        source_manifest_sha256: str,
        prompt_template_version: str,
        prompt_template_sha256: str,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._write() as connection:
            cursor = connection.execute(
                """
                INSERT INTO datasets(
                    alias, source_path, source_manifest_sha256, prompt_template_version,
                    prompt_template_sha256, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alias,
                    source_path,
                    source_manifest_sha256,
                    prompt_template_version,
                    prompt_template_sha256,
                    now,
                    now,
                ),
            )
            return _require_row(
                connection.execute("SELECT * FROM datasets WHERE id=?", (cursor.lastrowid,)).fetchone()
            )

    def open_review_workspace(
        self,
        *,
        alias: str,
        source_path: str,
        source_manifest_sha256: str,
        episode_lengths: Mapping[int, int],
        prompt_template_version: str,
        prompt_template_sha256: str,
        actor: str,
    ) -> dict[str, Any]:
        """Atomically register or validate one exact source and its prompt contract."""
        if not actor:
            raise ValueError("actor is required")
        normalized_lengths = dict(episode_lengths)
        if not normalized_lengths or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
            for index, length in normalized_lengths.items()
        ):
            raise ValueError("episode_lengths must be a nonempty mapping of nonnegative integers")
        now = _utc_now()
        with self._write() as connection:
            dataset = _row(connection.execute("SELECT * FROM datasets WHERE alias=?", (alias,)).fetchone())
            if dataset is None:
                cursor = connection.execute(
                    """
                    INSERT INTO datasets(
                        alias, source_path, source_manifest_sha256, prompt_template_version,
                        prompt_template_sha256, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alias,
                        source_path,
                        source_manifest_sha256,
                        prompt_template_version,
                        prompt_template_sha256,
                        now,
                        now,
                    ),
                )
                dataset = _require_row(
                    connection.execute("SELECT * FROM datasets WHERE id=?", (cursor.lastrowid,)).fetchone()
                )
                for source_episode_index, source_length in sorted(normalized_lengths.items()):
                    _insert_workspace_episode(
                        connection,
                        dataset_id=dataset["id"],
                        source_episode_index=source_episode_index,
                        source_length=source_length,
                        now=now,
                    )
            else:
                if (
                    dataset["source_path"] != source_path
                    or dataset["source_manifest_sha256"] != source_manifest_sha256
                ):
                    raise WorkspaceSourceConflict(alias)
                persisted_lengths = {
                    row["source_episode_index"]: row["source_length"]
                    for row in connection.execute(
                        """
                        SELECT source_episode_index, source_length
                        FROM episodes WHERE dataset_id=? ORDER BY source_episode_index
                        """,
                        (dataset["id"],),
                    )
                }
                if persisted_lengths != normalized_lengths:
                    raise WorkspaceSourceConflict(alias)
                if (
                    dataset["prompt_template_version"] != prompt_template_version
                    or dataset["prompt_template_sha256"] != prompt_template_sha256
                ):
                    raise WorkspacePromptContractConflict(alias, dataset)
            dataset = _require_row(connection.execute("SELECT * FROM datasets WHERE alias=?", (alias,)).fetchone())
            episodes = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM episodes WHERE dataset_id=? ORDER BY source_episode_index",
                    (dataset["id"],),
                )
            ]
            return {"dataset": dataset, "episodes": episodes, "invalidated": []}

    def get_dataset(self, *, alias: str) -> dict[str, Any] | None:
        with self._read() as connection:
            return _row(connection.execute("SELECT * FROM datasets WHERE alias=?", (alias,)).fetchone())

    def create_episode(self, *, dataset_id: int, source_episode_index: int, source_length: int) -> dict[str, Any]:
        now = _utc_now()
        with self._write() as connection:
            cursor = connection.execute(
                """
                INSERT INTO episodes(
                    dataset_id, source_episode_index, source_length, review_state,
                    revision, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (dataset_id, source_episode_index, source_length, ReviewState.PENDING.value, now, now),
            )
            return _require_row(
                connection.execute("SELECT * FROM episodes WHERE id=?", (cursor.lastrowid,)).fetchone()
            )

    def get_episode(self, *, dataset_id: int, source_episode_index: int) -> dict[str, Any] | None:
        with self._read() as connection:
            return _row(
                connection.execute(
                    "SELECT * FROM episodes WHERE dataset_id=? AND source_episode_index=?",
                    (dataset_id, source_episode_index),
                ).fetchone()
            )

    def list_episodes(self, *, dataset_id: int) -> list[dict[str, Any]]:
        with self._read() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM episodes WHERE dataset_id=? ORDER BY source_episode_index",
                    (dataset_id,),
                )
            ]

    def get_active_proposal(self, *, dataset_id: int, source_episode_index: int) -> dict[str, Any] | None:
        """Return the one active model proposal for an exact dataset episode."""
        with self._read() as connection:
            return _row(
                connection.execute(
                    """
                    SELECT proposal.*
                    FROM cosmos_proposals AS proposal
                    JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                    JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                    WHERE job.dataset_id=? AND attempt.source_episode_index=?
                        AND proposal.state=?
                    """,
                    (dataset_id, source_episode_index, ProposalState.ACTIVE.value),
                ).fetchone()
            )

    def update_episode(
        self,
        *,
        dataset_id: int,
        source_episode_index: int,
        expected_revision: int,
        changes: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        unknown = set(changes).difference(_EPISODE_CHANGE_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported episode fields: {', '.join(sorted(unknown))}")
        if not actor:
            raise ValueError("actor is required")
        with self._write() as connection:
            current = _require_row(
                connection.execute(
                    "SELECT * FROM episodes WHERE dataset_id=? AND source_episode_index=?",
                    (dataset_id, source_episode_index),
                ).fetchone(),
                "episode not found",
            )
            if current["revision"] != expected_revision:
                raise OptimisticConflict(current)
            fields = {key: (value.value if hasattr(value, "value") else value) for key, value in changes.items()}
            fields["revision"] = expected_revision + 1
            fields["updated_at"] = _utc_now()
            assignments = ", ".join(f"{name}=?" for name in fields)
            connection.execute(
                f"UPDATE episodes SET {assignments} WHERE id=?",
                (*fields.values(), current["id"]),
            )
            updated = _require_row(
                connection.execute("SELECT * FROM episodes WHERE id=?", (current["id"],)).fetchone()
            )
            self._append_audit_event(
                connection,
                dataset_id=dataset_id,
                actor=actor,
                operation="episode_updated",
                episode_id=current["id"],
                previous_revision=expected_revision,
                new_revision=updated["revision"],
            )
            return updated

    def transition_review_episode(
        self,
        *,
        dataset_id: int,
        source_episode_index: int,
        expected_revision: int,
        allowed_states: frozenset[ReviewState],
        changes: Mapping[str, Any],
        actor: str,
        operation: str,
        required_prompt_template_version: str | None = None,
        required_prompt_template_sha256: str | None = None,
        audit_details: Mapping[str, Any] | None = None,
        snapshot_active_proposal_for_contact_sheet: bool = False,
    ) -> dict[str, Any]:
        """Apply one revisioned human-review edge in the same write lock."""
        unknown = set(changes).difference(_EPISODE_CHANGE_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported episode fields: {', '.join(sorted(unknown))}")
        if not actor:
            raise ValueError("actor is required")
        if not operation:
            raise ValueError("operation is required")
        if (required_prompt_template_version is None) != (required_prompt_template_sha256 is None):
            raise ValueError("required prompt template version and SHA256 must be provided together")
        if snapshot_active_proposal_for_contact_sheet and operation != "keep_approved":
            raise ValueError("only keep_approved may snapshot a contact-sheet proposal")
        with self._write() as connection:
            current = _require_row(
                connection.execute(
                    "SELECT * FROM episodes WHERE dataset_id=? AND source_episode_index=?",
                    (dataset_id, source_episode_index),
                ).fetchone(),
                "episode not found",
            )
            if current["revision"] != expected_revision:
                raise OptimisticConflict(current)
            current_state = ReviewState(current["review_state"])
            if current_state not in allowed_states:
                target = changes.get("review_state", current_state)
                if hasattr(target, "value"):
                    target = target.value
                raise IllegalStateTransition(
                    entity="episode",
                    identifier=str(source_episode_index),
                    current_state=current_state.value,
                    target_state=str(target),
                )
            if required_prompt_template_version is not None:
                dataset = _require_row(
                    connection.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone(),
                    "dataset not found",
                )
                if (
                    dataset["prompt_template_version"] != required_prompt_template_version
                    or dataset["prompt_template_sha256"] != required_prompt_template_sha256
                    or current["prompt_template_sha256"] != required_prompt_template_sha256
                ):
                    raise PromptContractConflict(
                        required_version=required_prompt_template_version,
                        required_sha256=required_prompt_template_sha256,
                        dataset_version=dataset["prompt_template_version"],
                        dataset_sha256=dataset["prompt_template_sha256"],
                        episode_sha256=current["prompt_template_sha256"],
                    )
            details = dict(audit_details or {})
            if snapshot_active_proposal_for_contact_sheet:
                dataset_identity = _require_row(
                    connection.execute(
                        "SELECT id, alias, source_manifest_sha256 FROM datasets WHERE id=?",
                        (dataset_id,),
                    ).fetchone(),
                    "dataset not found",
                )
                proposal = connection.execute(
                    """
                    SELECT proposal.id,
                        proposal.step_2_start_frame, proposal.step_3_start_frame,
                        proposal.step_4_start_frame, proposal.step_5_start_frame,
                        proposal.step_6_start_frame, proposal.step_7_start_frame
                    FROM cosmos_proposals AS proposal
                    JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                    JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                    WHERE job.dataset_id=? AND attempt.source_episode_index=?
                        AND proposal.state='active'
                    """,
                    (dataset_id, source_episode_index),
                ).fetchall()
                if len(proposal) > 1:
                    raise RuntimeError("multiple active proposals violate the contact-sheet snapshot contract")
                proposal_row = None if not proposal else proposal[0]
                proposal_id = None if proposal_row is None else proposal_row["id"]
                approval_revision = changes.get("approval_revision")
                final_frames = [
                    changes.get(f"step_{step}_start_frame", current[f"step_{step}_start_frame"])
                    for step in range(2, 8)
                ]
                proposal_frames = (
                    [None] * 6
                    if proposal_row is None
                    else [proposal_row[f"step_{step}_start_frame"] for step in range(2, 8)]
                )
                if (
                    type(approval_revision) is not int
                    or approval_revision < 1
                    or any(type(frame) is not int for frame in final_frames)
                ):
                    raise ValueError("keep approval contact-sheet evidence is incomplete")
                details["contact_sheet_proposal_id"] = proposal_id
                details["contact_sheet_evidence"] = {
                    "schema_version": 1,
                    "dataset_id": dataset_identity["id"],
                    "dataset_alias": dataset_identity["alias"],
                    "source_manifest_sha256": dataset_identity["source_manifest_sha256"],
                    "source_episode_index": source_episode_index,
                    "approval_revision": approval_revision,
                    "final_transition_frames": final_frames,
                    "proposal_id": proposal_id,
                    "proposal_transition_frames": proposal_frames,
                }
            fields = {key: (value.value if hasattr(value, "value") else value) for key, value in changes.items()}
            fields["revision"] = expected_revision + 1
            fields["updated_at"] = _utc_now()
            assignments = ", ".join(f"{name}=?" for name in fields)
            connection.execute(
                f"UPDATE episodes SET {assignments} WHERE id=?",
                (*fields.values(), current["id"]),
            )
            updated = _require_row(
                connection.execute("SELECT * FROM episodes WHERE id=?", (current["id"],)).fetchone()
            )
            self._append_audit_event(
                connection,
                dataset_id=dataset_id,
                actor=actor,
                operation=operation,
                episode_id=current["id"],
                previous_revision=expected_revision,
                new_revision=updated["revision"],
                details=details,
            )
            return updated

    def migrate_prompt_template(
        self,
        *,
        dataset_id: int,
        expected_prompt_template_version: str,
        expected_prompt_template_sha256: str,
        prompt_template_version: str,
        prompt_template_sha256: str,
        actor: str,
    ) -> list[dict[str, Any]]:
        """CAS a prompt upgrade and atomically invalidate only approved keeps."""
        if not actor:
            raise ValueError("actor is required")
        with self._write() as connection:
            dataset = _require_row(
                connection.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone(),
                "dataset not found",
            )
            if (
                dataset["prompt_template_version"] != expected_prompt_template_version
                or dataset["prompt_template_sha256"] != expected_prompt_template_sha256
            ):
                raise PromptMigrationConflict(
                    expected_version=expected_prompt_template_version,
                    expected_sha256=expected_prompt_template_sha256,
                    current_version=dataset["prompt_template_version"],
                    current_sha256=dataset["prompt_template_sha256"],
                )
            if (
                dataset["prompt_template_version"] == prompt_template_version
                and dataset["prompt_template_sha256"] == prompt_template_sha256
            ):
                return []
            current_generation = _prompt_contract_generation(dataset["prompt_template_version"])
            requested_generation = _prompt_contract_generation(prompt_template_version)
            if (
                current_generation is None
                or requested_generation is None
                or requested_generation[0] != current_generation[0]
                or requested_generation[1] <= current_generation[1]
            ):
                raise PromptMigrationDowngrade(
                    current_version=dataset["prompt_template_version"],
                    requested_version=prompt_template_version,
                )
            return self._migrate_prompt_template_in_transaction(
                connection,
                dataset=dataset,
                prompt_template_version=prompt_template_version,
                prompt_template_sha256=prompt_template_sha256,
                actor=actor,
            )

    def _migrate_prompt_template_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        dataset: Mapping[str, Any],
        prompt_template_version: str,
        prompt_template_sha256: str,
        actor: str,
    ) -> list[dict[str, Any]]:
        now = _utc_now()
        hash_changed = dataset["prompt_template_sha256"] != prompt_template_sha256
        if not hash_changed and dataset["prompt_template_version"] == prompt_template_version:
            return []
        connection.execute(
            """
            UPDATE datasets
            SET prompt_template_version=?, prompt_template_sha256=?, updated_at=?
            WHERE id=?
            """,
            (prompt_template_version, prompt_template_sha256, now, dataset["id"]),
        )
        if not hash_changed:
            return []
        approved_keeps = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM episodes WHERE dataset_id=? AND review_state=? ORDER BY source_episode_index",
                (dataset["id"], ReviewState.APPROVED_KEEP.value),
            )
        ]
        invalidated: list[dict[str, Any]] = []
        for episode in approved_keeps:
            new_revision = episode["revision"] + 1
            connection.execute(
                """
                UPDATE episodes
                SET review_state=?, revision=?, approval_revision=NULL, reviewer=NULL,
                    approved_at=NULL, prompt_template_sha256=?, updated_at=?
                WHERE id=?
                """,
                (
                    ReviewState.DRAFT.value,
                    new_revision,
                    prompt_template_sha256,
                    now,
                    episode["id"],
                ),
            )
            self._append_audit_event(
                connection,
                dataset_id=dataset["id"],
                actor=actor,
                operation="prompt_template_invalidated",
                episode_id=episode["id"],
                previous_revision=episode["revision"],
                new_revision=new_revision,
                details={
                    "previous_prompt_template_sha256": dataset["prompt_template_sha256"],
                    "prompt_template_sha256": prompt_template_sha256,
                },
            )
            invalidated.append(
                _require_row(connection.execute("SELECT * FROM episodes WHERE id=?", (episode["id"],)).fetchone())
            )
        return invalidated

    def create_cosmos_job(
        self, *, dataset_id: int, configuration: Mapping[str, Any], parent_job_id: str | None = None
    ) -> dict[str, Any]:
        identifier = str(uuid4())
        now = _utc_now()
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO cosmos_jobs(
                    id, dataset_id, parent_job_id, configuration_json, state, total_attempts,
                    succeeded_attempts, manual_only_attempts, cancel_requested, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, ?, ?)
                """,
                (
                    identifier,
                    dataset_id,
                    parent_job_id,
                    canonical_json(dict(configuration)),
                    JobState.QUEUED.value,
                    now,
                    now,
                ),
            )
            return _require_row(
                connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (identifier,)).fetchone()
            )

    def set_cosmos_job_state(self, *, job_id: str, expected_state: JobState, state: JobState) -> dict[str, Any]:
        with self._write() as connection:
            current = _require_row(
                connection.execute(
                    "SELECT * FROM cosmos_jobs WHERE id=?",
                    (job_id,),
                ).fetchone(),
                "Cosmos job not found",
            )
            if current["state"] != expected_state.value:
                raise StateTransitionConflict(
                    entity="cosmos_job",
                    identifier=job_id,
                    expected_state=expected_state.value,
                    current_state=current["state"],
                )
            if state is expected_state:
                return current
            if state not in JOB_STATE_TRANSITIONS[expected_state]:
                raise IllegalStateTransition(
                    entity="cosmos_job",
                    identifier=job_id,
                    current_state=current["state"],
                    target_state=state.value,
                )
            connection.execute(
                "UPDATE cosmos_jobs SET state=?, updated_at=? WHERE id=?", (state.value, _utc_now(), current["id"])
            )
            return _require_row(
                connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (current["id"],)).fetchone()
            )

    def create_export_snapshot(self, *, dataset_id: int, staging_path: str, final_path: str) -> dict[str, Any]:
        """Freeze approved rows and their hash before any exporter work begins."""
        identifier = str(uuid4())
        now = _utc_now()
        with self._write() as connection:
            document, rows = _approval_snapshot_document(connection, dataset_id)
            snapshot_sha256 = canonical_json_sha256(document)
            connection.execute(
                """
                INSERT INTO exports(
                    id, dataset_id, state, approval_snapshot_sha256, staging_path,
                    final_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    dataset_id,
                    ExportState.QUEUED.value,
                    snapshot_sha256,
                    staging_path,
                    final_path,
                    now,
                    now,
                ),
            )
            for row in rows:
                connection.execute(
                    """
                    INSERT INTO export_episodes(
                        export_id, source_episode_index, source_length, review_state, object_name, pickup_hand,
                        turn_direction, step_2_start_frame, step_3_start_frame, step_4_start_frame,
                        step_5_start_frame, step_6_start_frame, step_7_start_frame, revision,
                        approval_revision, reviewer, approved_at, rejection_reason, prompt_template_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        row["source_episode_index"],
                        row["source_length"],
                        row["review_state"],
                        row["object_name"],
                        row["pickup_hand"],
                        row["turn_direction"],
                        row["step_2_start_frame"],
                        row["step_3_start_frame"],
                        row["step_4_start_frame"],
                        row["step_5_start_frame"],
                        row["step_6_start_frame"],
                        row["step_7_start_frame"],
                        row["revision"],
                        row["approval_revision"],
                        row["reviewer"],
                        row["approved_at"],
                        row["rejection_reason"],
                        row["prompt_template_sha256"],
                    ),
                )
            return _require_row(connection.execute("SELECT * FROM exports WHERE id=?", (identifier,)).fetchone())

    def set_export_state(
        self, *, export_id: str, expected_state: ExportState, state: ExportState
    ) -> dict[str, Any]:
        with self._write() as connection:
            current = _require_row(
                connection.execute(
                    "SELECT * FROM exports WHERE id=?",
                    (export_id,),
                ).fetchone(),
                "export not found",
            )
            if current["state"] != expected_state.value:
                raise StateTransitionConflict(
                    entity="export",
                    identifier=export_id,
                    expected_state=expected_state.value,
                    current_state=current["state"],
                )
            if state is expected_state:
                return current
            if state not in EXPORT_STATE_TRANSITIONS[expected_state]:
                raise IllegalStateTransition(
                    entity="export",
                    identifier=export_id,
                    current_state=current["state"],
                    target_state=state.value,
                )
            connection.execute(
                "UPDATE exports SET state=?, updated_at=? WHERE id=?", (state.value, _utc_now(), current["id"])
            )
            return _require_row(
                connection.execute("SELECT * FROM exports WHERE id=?", (current["id"],)).fetchone()
            )

    def append_http_exchange_history(
        self,
        *,
        attempt_id: str,
        exchange: Mapping[str, Any],
        _connection: sqlite3.Connection | None = None,
        _deduplicate_exact_replay: bool = False,
    ) -> dict[str, Any]:
        """Append one canonical HTTP envelope without rewriting prior exchange evidence."""
        normalized_exchange = _validate_http_exchange(exchange)
        transaction = self._write() if _connection is None else nullcontext(_connection)
        with transaction as connection:
            current = _require_row(
                connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone(),
                "Cosmos attempt not found",
            )
            history = json.loads(current["http_exchange_history_json"])
            if not isinstance(history, list):  # Defensive: the v1 CHECK normally makes this unreachable.
                raise RuntimeError("attempt HTTP exchange history is not an array")
            if not (_deduplicate_exact_replay and normalized_exchange in history):
                history.append(normalized_exchange)
                connection.execute(
                    "UPDATE cosmos_attempts SET http_exchange_history_json=?, updated_at=? WHERE id=?",
                    (canonical_json(history), _utc_now(), attempt_id),
                )
            return _require_row(
                connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone()
            )

    def append_audit_event(
        self,
        *,
        dataset_id: int,
        actor: str,
        operation: str,
        episode_id: int | None = None,
        job_id: str | None = None,
        export_id: str | None = None,
        previous_revision: int | None = None,
        new_revision: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._write() as connection:
            return self._append_audit_event(
                connection,
                dataset_id=dataset_id,
                actor=actor,
                operation=operation,
                episode_id=episode_id,
                job_id=job_id,
                export_id=export_id,
                previous_revision=previous_revision,
                new_revision=new_revision,
                details=details,
            )

    def _append_audit_event(self, connection: sqlite3.Connection, **event: Any) -> dict[str, Any]:
        identifier = str(uuid4())
        connection.execute(
            """
            INSERT INTO audit_events(
                id, dataset_id, actor, operation, episode_id, job_id, export_id,
                previous_revision, new_revision, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                event["dataset_id"],
                event["actor"],
                event["operation"],
                event.get("episode_id"),
                event.get("job_id"),
                event.get("export_id"),
                event.get("previous_revision"),
                event.get("new_revision"),
                canonical_json(dict(event.get("details") or {})),
                _utc_now(),
            ),
        )
        return _require_row(connection.execute("SELECT * FROM audit_events WHERE id=?", (identifier,)).fetchone())

    def create_artifact(
        self,
        *,
        kind: str,
        relative_path: str,
        media_type: str,
        byte_size: int,
        sha256: str,
        attempt_id: str | None = None,
        export_id: str | None = None,
    ) -> dict[str, Any]:
        path = _safe_relative_path(relative_path)
        if (attempt_id is None) == (export_id is None):
            raise ValueError("artifact must belong to exactly one attempt or export")
        identifier = str(uuid4())
        with self._write() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(
                    id, attempt_id, export_id, kind, relative_path, media_type,
                    byte_size, sha256, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (identifier, attempt_id, export_id, kind, path, media_type, byte_size, sha256, _utc_now()),
            )
            return _require_row(connection.execute("SELECT * FROM artifacts WHERE id=?", (identifier,)).fetchone())

    def reconcile_attempt_artifact(
        self,
        *,
        dataset_id: int,
        attempt_id: str,
        kind: str,
        relative_path: str,
        media_type: str,
        byte_size: int,
        sha256: str,
        update_attempt_pointer: bool = False,
        _connection: sqlite3.Connection | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Atomically create-or-return one exact attempt artifact and optional pointer."""

        if type(dataset_id) is not int or dataset_id <= 0:
            raise ValueError("dataset_id must be a positive built-in integer")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt_id is required")
        allowed_kinds = {
            ARTIFACT_KIND_COSMOS_REQUEST,
            ARTIFACT_KIND_COSMOS_RESPONSE,
            ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE,
            ARTIFACT_KIND_COSMOS_PARSED,
        }
        if kind not in allowed_kinds:
            raise ValueError("attempt artifact kind is invalid")
        if type(update_attempt_pointer) is not bool:
            raise ValueError("update_attempt_pointer must be a built-in boolean")
        pointer_columns = {
            ARTIFACT_KIND_COSMOS_REQUEST: "request_artifact_id",
            ARTIFACT_KIND_COSMOS_RESPONSE: "response_artifact_id",
        }
        pointer_column = pointer_columns.get(kind)
        if update_attempt_pointer and pointer_column is None:
            raise ValueError("only a canonical request or response artifact can update an attempt pointer")
        path = _safe_relative_path(relative_path)
        if not isinstance(media_type, str) or not media_type:
            raise ValueError("artifact media_type is required")
        try:
            attempt_id.encode("utf-8")
            media_type.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("artifact identifiers must be valid UTF-8") from error
        if type(byte_size) is not int or byte_size < 0:
            raise ValueError("artifact byte_size must be a nonnegative built-in integer")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("artifact sha256 must be lowercase hexadecimal")

        expected = {
            "attempt_id": attempt_id,
            "export_id": None,
            "kind": kind,
            "relative_path": path,
            "media_type": media_type,
            "byte_size": byte_size,
            "sha256": sha256,
        }

        def matches(artifact: Mapping[str, Any]) -> bool:
            return all(artifact.get(key) == value for key, value in expected.items())

        transaction = self._write() if _connection is None else nullcontext(_connection)
        with transaction as connection:
            attempt_owner = _row(
                connection.execute(
                    """
                    SELECT attempt.*, job.dataset_id AS artifact_dataset_id
                    FROM cosmos_attempts AS attempt
                    JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                    WHERE attempt.id=?
                    """,
                    (attempt_id,),
                ).fetchone()
            )
            if attempt_owner is None:
                raise LookupError("Cosmos attempt not found")
            if attempt_owner["artifact_dataset_id"] != dataset_id:
                raise ArtifactReconciliationConflict()

            artifact = _row(
                connection.execute(
                    "SELECT * FROM artifacts WHERE relative_path=?",
                    (path,),
                ).fetchone()
            )
            artifacts_for_kind = [
                _require_row(row)
                for row in connection.execute(
                    "SELECT * FROM artifacts WHERE attempt_id=? AND kind=? ORDER BY id",
                    (attempt_id, kind),
                ).fetchall()
            ]
            if len(artifacts_for_kind) > 1:
                raise ArtifactReconciliationConflict()
            if artifacts_for_kind and (artifact is None or artifact["id"] != artifacts_for_kind[0]["id"]):
                raise ArtifactReconciliationConflict()

            pointer_id = None if pointer_column is None else attempt_owner[pointer_column]
            if pointer_id is not None:
                pointed = _require_row(
                    connection.execute("SELECT * FROM artifacts WHERE id=?", (pointer_id,)).fetchone(),
                    "attempt artifact pointer is broken",
                )
                if artifact is None or artifact["id"] != pointer_id or not matches(pointed):
                    raise ArtifactReconciliationConflict()
                return {
                    "artifact": pointed,
                    "attempt": _require_row(
                        connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone()
                    ),
                }

            if artifact is None:
                artifact_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        id, attempt_id, export_id, kind, relative_path, media_type,
                        byte_size, sha256, created_at
                    ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        attempt_id,
                        kind,
                        path,
                        media_type,
                        byte_size,
                        sha256,
                        _utc_now(),
                    ),
                )
                artifact = _require_row(
                    connection.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
                )
            elif not matches(artifact):
                raise ArtifactReconciliationConflict()

            if update_attempt_pointer:
                connection.execute(
                    f"UPDATE cosmos_attempts SET {pointer_column}=?, updated_at=? WHERE id=?",
                    (artifact["id"], _utc_now(), attempt_id),
                )
            return {
                "artifact": artifact,
                "attempt": _require_row(
                    connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone()
                ),
            }

    def approval_snapshot(self, *, dataset_id: int) -> dict[str, Any]:
        """Return a stable approval snapshot independent of SQLite row insertion order."""
        with self._read() as connection:
            document, _ = _approval_snapshot_document(connection, dataset_id)
        return {"document": document, "sha256": canonical_json_sha256(document)}


def _require_row(row: sqlite3.Row | None, message: str = "row not found") -> dict[str, Any]:
    value = _row(row)
    if value is None:
        raise LookupError(message)
    return value


def _insert_workspace_episode(
    connection: sqlite3.Connection,
    *,
    dataset_id: int,
    source_episode_index: int,
    source_length: int,
    now: str,
) -> dict[str, Any]:
    cursor = connection.execute(
        """
        INSERT INTO episodes(
            dataset_id, source_episode_index, source_length, review_state,
            revision, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 0, ?, ?)
        """,
        (
            dataset_id,
            source_episode_index,
            source_length,
            ReviewState.PENDING.value,
            now,
            now,
        ),
    )
    return _require_row(connection.execute("SELECT * FROM episodes WHERE id=?", (cursor.lastrowid,)).fetchone())


def _approval_snapshot_document(
    connection: sqlite3.Connection, dataset_id: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = _require_row(connection.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone())
    episodes = [
        dict(row)
        for row in connection.execute(
            """
            SELECT source_episode_index, source_length, review_state, object_name, pickup_hand, turn_direction,
                   step_2_start_frame, step_3_start_frame, step_4_start_frame, step_5_start_frame,
                   step_6_start_frame, step_7_start_frame, revision, approval_revision, reviewer,
                   approved_at, rejection_reason, prompt_template_sha256
            FROM episodes
            WHERE dataset_id=? AND review_state IN (?, ?)
            ORDER BY source_episode_index ASC
            """,
            (dataset_id, ReviewState.APPROVED_KEEP.value, ReviewState.APPROVED_REJECT.value),
        )
    ]
    return (
        {
            "source_manifest_sha256": dataset["source_manifest_sha256"],
            "prompt_template_sha256": dataset["prompt_template_sha256"],
            "episodes": episodes,
        },
        episodes,
    )


_SCHEMA_SIGNATURE_QUERY = """
    SELECT type, name, tbl_name, sql
    FROM sqlite_schema
    WHERE name NOT LIKE 'sqlite_%'
    ORDER BY type, name
"""


def _schema_signature(connection: sqlite3.Connection) -> tuple[tuple[str, str, str, str], ...]:
    rows = connection.execute(_SCHEMA_SIGNATURE_QUERY).fetchall()
    signature: list[tuple[str, str, str, str]] = []
    for row in rows:
        if len(row) != 4 or any(not isinstance(value, str) or not value for value in row):
            raise IncompatibleCurationDatabase("curation database schema manifest is malformed")
        signature.append(tuple(row))
    return tuple(signature)


@lru_cache(maxsize=1)
def _expected_v1_schema_signature() -> tuple[tuple[str, str, str, str], ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for statement in _migration_v1_statements():
            connection.execute(statement)
        return _schema_signature(connection)
    finally:
        connection.close()


def _migration_v1_statements() -> tuple[str, ...]:
    review_states = _enum_values(ReviewState)
    job_states = _enum_values(JobState)
    attempt_states = _enum_values(AttemptState)
    export_states = _enum_values(ExportState)
    proposal_states = _enum_values(ProposalState)
    return (
        """
        CREATE TABLE datasets (
            id INTEGER PRIMARY KEY,
            alias TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            source_manifest_sha256 TEXT NOT NULL CHECK(
                length(source_manifest_sha256)=64 AND
                source_manifest_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            prompt_template_version TEXT NOT NULL,
            prompt_template_sha256 TEXT NOT NULL CHECK(
                length(prompt_template_sha256)=64 AND
                prompt_template_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
            source_episode_index INTEGER NOT NULL CHECK(source_episode_index >= 0),
            source_length INTEGER NOT NULL CHECK(source_length >= 0),
            review_state TEXT NOT NULL CHECK(review_state IN ({review_states})),
            object_name TEXT,
            pickup_hand TEXT CHECK(pickup_hand IN ('left', 'right') OR pickup_hand IS NULL),
            turn_direction TEXT CHECK(turn_direction IN ('left', 'right') OR turn_direction IS NULL),
            step_2_start_frame INTEGER CHECK(step_2_start_frame >= 0 OR step_2_start_frame IS NULL),
            step_3_start_frame INTEGER CHECK(step_3_start_frame >= 0 OR step_3_start_frame IS NULL),
            step_4_start_frame INTEGER CHECK(step_4_start_frame >= 0 OR step_4_start_frame IS NULL),
            step_5_start_frame INTEGER CHECK(step_5_start_frame >= 0 OR step_5_start_frame IS NULL),
            step_6_start_frame INTEGER CHECK(step_6_start_frame >= 0 OR step_6_start_frame IS NULL),
            step_7_start_frame INTEGER CHECK(step_7_start_frame >= 0 OR step_7_start_frame IS NULL),
            revision INTEGER NOT NULL CHECK(revision >= 0),
            approval_revision INTEGER CHECK(approval_revision >= 0 OR approval_revision IS NULL),
            reviewer TEXT,
            approved_at TEXT,
            rejection_reason TEXT,
            prompt_template_sha256 TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(dataset_id, source_episode_index)
        )
        """,
        f"""
        CREATE TABLE cosmos_jobs (
            id TEXT PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
            parent_job_id TEXT REFERENCES cosmos_jobs(id) ON DELETE RESTRICT,
            configuration_json TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ({job_states})),
            total_attempts INTEGER NOT NULL CHECK(total_attempts >= 0),
            succeeded_attempts INTEGER NOT NULL CHECK(succeeded_attempts >= 0),
            manual_only_attempts INTEGER NOT NULL CHECK(manual_only_attempts >= 0),
            cancel_requested INTEGER NOT NULL CHECK(cancel_requested IN (0, 1)),
            owner TEXT,
            lease_expires_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE cosmos_attempts (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES cosmos_jobs(id) ON DELETE RESTRICT,
            source_episode_index INTEGER NOT NULL CHECK(source_episode_index >= 0),
            attempt_number INTEGER NOT NULL CHECK(attempt_number >= 0),
            state TEXT NOT NULL CHECK(state IN ({attempt_states})),
            lease_owner TEXT,
            lease_expires_at TEXT,
            request_artifact_id TEXT REFERENCES artifacts(id) ON DELETE RESTRICT,
            response_artifact_id TEXT REFERENCES artifacts(id) ON DELETE RESTRICT,
            http_exchange_history_json TEXT NOT NULL DEFAULT '[]' CHECK(
                json_valid(http_exchange_history_json) AND json_type(http_exchange_history_json)='array'
            ),
            error_class TEXT,
            error_summary TEXT,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            UNIQUE(job_id, source_episode_index, attempt_number)
        )
        """,
        f"""
        CREATE TABLE cosmos_proposals (
            id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL REFERENCES cosmos_attempts(id) ON DELETE RESTRICT,
            model_response_json TEXT NOT NULL,
            step_2_start_frame INTEGER CHECK(step_2_start_frame >= 0 OR step_2_start_frame IS NULL),
            step_3_start_frame INTEGER CHECK(step_3_start_frame >= 0 OR step_3_start_frame IS NULL),
            step_4_start_frame INTEGER CHECK(step_4_start_frame >= 0 OR step_4_start_frame IS NULL),
            step_5_start_frame INTEGER CHECK(step_5_start_frame >= 0 OR step_5_start_frame IS NULL),
            step_6_start_frame INTEGER CHECK(step_6_start_frame >= 0 OR step_6_start_frame IS NULL),
            step_7_start_frame INTEGER CHECK(step_7_start_frame >= 0 OR step_7_start_frame IS NULL),
            validation_warnings_json TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ({proposal_states})),
            created_at TEXT NOT NULL,
            UNIQUE(attempt_id)
        )
        """,
        f"""
        CREATE TABLE exports (
            id TEXT PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
            state TEXT NOT NULL CHECK(state IN ({export_states})),
            approval_snapshot_sha256 TEXT NOT NULL CHECK(
                length(approval_snapshot_sha256)=64 AND
                approval_snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            staging_path TEXT NOT NULL,
            final_path TEXT NOT NULL,
            structural_artifact_id TEXT REFERENCES artifacts(id) ON DELETE RESTRICT,
            gr00t_stats_artifact_id TEXT REFERENCES artifacts(id) ON DELETE RESTRICT,
            gr00t_loader_artifact_id TEXT REFERENCES artifacts(id) ON DELETE RESTRICT,
            final_consistency_artifact_id TEXT REFERENCES artifacts(id) ON DELETE RESTRICT,
            failure_summary TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE artifacts (
            id TEXT PRIMARY KEY,
            attempt_id TEXT REFERENCES cosmos_attempts(id) ON DELETE RESTRICT,
            export_id TEXT REFERENCES exports(id) ON DELETE RESTRICT,
            kind TEXT NOT NULL,
            relative_path TEXT NOT NULL CHECK(
                length(relative_path) > 0 AND substr(relative_path, 1, 1) != '/' AND
                instr('/' || relative_path || '/', '/../') = 0
            ),
            media_type TEXT NOT NULL,
            byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
            sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
            created_at TEXT NOT NULL,
            CHECK((attempt_id IS NULL) != (export_id IS NULL)),
            UNIQUE(relative_path)
        )
        """,
        """
        CREATE TABLE audit_events (
            id TEXT PRIMARY KEY,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
            actor TEXT NOT NULL CHECK(length(actor) > 0),
            operation TEXT NOT NULL CHECK(length(operation) > 0),
            episode_id INTEGER REFERENCES episodes(id) ON DELETE RESTRICT,
            job_id TEXT REFERENCES cosmos_jobs(id) ON DELETE RESTRICT,
            export_id TEXT REFERENCES exports(id) ON DELETE RESTRICT,
            previous_revision INTEGER CHECK(previous_revision >= 0 OR previous_revision IS NULL),
            new_revision INTEGER CHECK(new_revision >= 0 OR new_revision IS NULL),
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE export_episodes (
            export_id TEXT NOT NULL REFERENCES exports(id) ON DELETE RESTRICT,
            source_episode_index INTEGER NOT NULL CHECK(source_episode_index >= 0),
            source_length INTEGER NOT NULL CHECK(source_length >= 0),
            review_state TEXT NOT NULL CHECK(review_state IN ('approved_keep', 'approved_reject')),
            object_name TEXT,
            pickup_hand TEXT CHECK(pickup_hand IN ('left', 'right') OR pickup_hand IS NULL),
            turn_direction TEXT CHECK(turn_direction IN ('left', 'right') OR turn_direction IS NULL),
            step_2_start_frame INTEGER CHECK(step_2_start_frame >= 0 OR step_2_start_frame IS NULL),
            step_3_start_frame INTEGER CHECK(step_3_start_frame >= 0 OR step_3_start_frame IS NULL),
            step_4_start_frame INTEGER CHECK(step_4_start_frame >= 0 OR step_4_start_frame IS NULL),
            step_5_start_frame INTEGER CHECK(step_5_start_frame >= 0 OR step_5_start_frame IS NULL),
            step_6_start_frame INTEGER CHECK(step_6_start_frame >= 0 OR step_6_start_frame IS NULL),
            step_7_start_frame INTEGER CHECK(step_7_start_frame >= 0 OR step_7_start_frame IS NULL),
            revision INTEGER NOT NULL CHECK(revision >= 0),
            approval_revision INTEGER NOT NULL CHECK(approval_revision >= 0),
            reviewer TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            rejection_reason TEXT,
            prompt_template_sha256 TEXT,
            PRIMARY KEY(export_id, source_episode_index)
        )
        """,
        """
        CREATE UNIQUE INDEX one_active_cosmos_job_per_dataset
        ON cosmos_jobs(dataset_id)
        WHERE state IN ('queued', 'running', 'cancel_requested')
        """,
        """
        CREATE UNIQUE INDEX one_active_export_per_dataset
        ON exports(dataset_id)
        WHERE state IN (
            'queued', 'building', 'core_structural_validated', 'gr00t_stats_validated',
            'gr00t_loader_validated', 'provenance_written', 'final_consistency_validated', 'publishing'
        )
        """,
        """
        CREATE TRIGGER audit_events_no_update
        BEFORE UPDATE ON audit_events
        BEGIN SELECT RAISE(ABORT, 'audit_events are append-only'); END
        """,
        """
        CREATE TRIGGER audit_events_no_delete
        BEFORE DELETE ON audit_events
        BEGIN SELECT RAISE(ABORT, 'audit_events are append-only'); END
        """,
        """
        CREATE TRIGGER cosmos_jobs_parent_dataset_insert
        BEFORE INSERT ON cosmos_jobs
        BEGIN
            SELECT RAISE(ABORT, 'parent job must belong to the same dataset')
            WHERE NEW.parent_job_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM cosmos_jobs AS parent
                WHERE parent.id=NEW.parent_job_id AND parent.dataset_id=NEW.dataset_id
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_jobs_parent_dataset_update
        BEFORE UPDATE OF parent_job_id, dataset_id ON cosmos_jobs
        BEGIN
            SELECT RAISE(ABORT, 'parent job must belong to the same dataset')
            WHERE NEW.parent_job_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM cosmos_jobs AS parent
                WHERE parent.id=NEW.parent_job_id AND parent.dataset_id=NEW.dataset_id
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_jobs_dataset_immutable
        BEFORE UPDATE OF dataset_id ON cosmos_jobs
        BEGIN SELECT RAISE(ABORT, 'job dataset is immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_jobs_configuration_immutable
        BEFORE UPDATE OF configuration_json ON cosmos_jobs
        BEGIN SELECT RAISE(ABORT, 'job configuration is immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_jobs_parent_immutable
        BEFORE UPDATE OF parent_job_id ON cosmos_jobs
        BEGIN SELECT RAISE(ABORT, 'job parent is immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_jobs_terminal_immutable
        BEFORE UPDATE ON cosmos_jobs
        WHEN OLD.state IN ('completed', 'completed_with_failures', 'cancelled', 'failed')
        BEGIN SELECT RAISE(ABORT, 'terminal Cosmos jobs are immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_jobs_no_delete
        BEFORE DELETE ON cosmos_jobs
        BEGIN SELECT RAISE(ABORT, 'Cosmos jobs are append-only'); END
        """,
        """
        CREATE TRIGGER episodes_dataset_immutable
        BEFORE UPDATE OF dataset_id ON episodes
        BEGIN SELECT RAISE(ABORT, 'episode dataset is immutable'); END
        """,
        """
        CREATE TRIGGER episodes_source_index_immutable
        BEFORE UPDATE OF source_episode_index ON episodes
        BEGIN SELECT RAISE(ABORT, 'episode source index is immutable'); END
        """,
        """
        CREATE TRIGGER episodes_attempt_referenced_no_delete
        BEFORE DELETE ON episodes
        BEGIN
            SELECT RAISE(ABORT, 'episode referenced by a Cosmos attempt')
            WHERE EXISTS (
                SELECT 1 FROM cosmos_attempts AS attempt
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE job.dataset_id=OLD.dataset_id AND attempt.source_episode_index=OLD.source_episode_index
            );
        END
        """,
        """
        CREATE TRIGGER exports_dataset_immutable
        BEFORE UPDATE OF dataset_id ON exports
        BEGIN SELECT RAISE(ABORT, 'export dataset is immutable'); END
        """,
        """
        CREATE TRIGGER exports_snapshot_identity_immutable
        BEFORE UPDATE OF approval_snapshot_sha256, staging_path, final_path ON exports
        BEGIN SELECT RAISE(ABORT, 'export approval snapshot and paths are immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_attempts_episode_dataset_insert
        BEFORE INSERT ON cosmos_attempts
        BEGIN
            SELECT RAISE(ABORT, 'attempt episode must belong to the job dataset')
            WHERE NOT EXISTS (
                SELECT 1 FROM cosmos_jobs AS job
                JOIN episodes AS episode ON episode.dataset_id=job.dataset_id
                WHERE job.id=NEW.job_id AND episode.source_episode_index=NEW.source_episode_index
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_attempts_terminal_parent_no_insert
        BEFORE INSERT ON cosmos_attempts
        BEGIN
            SELECT RAISE(ABORT, 'cannot create an attempt for a terminal Cosmos job')
            WHERE EXISTS (
                SELECT 1 FROM cosmos_jobs
                WHERE id=NEW.job_id
                    AND state IN ('completed', 'completed_with_failures', 'cancelled', 'failed')
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_attempts_episode_dataset_update
        BEFORE UPDATE OF job_id, source_episode_index ON cosmos_attempts
        BEGIN
            SELECT RAISE(ABORT, 'attempt episode must belong to the job dataset')
            WHERE NOT EXISTS (
                SELECT 1 FROM cosmos_jobs AS job
                JOIN episodes AS episode ON episode.dataset_id=job.dataset_id
                WHERE job.id=NEW.job_id AND episode.source_episode_index=NEW.source_episode_index
            );
        END
        """,
        f"""
        CREATE TRIGGER cosmos_attempts_artifact_ownership_insert
        BEFORE INSERT ON cosmos_attempts
        BEGIN
            SELECT RAISE(ABORT, 'request artifact must belong to this attempt and be request')
            WHERE NEW.request_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.request_artifact_id AND attempt_id=NEW.id AND kind='{ARTIFACT_KIND_COSMOS_REQUEST}'
            );
            SELECT RAISE(ABORT, 'response artifact must belong to this attempt and be response')
            WHERE NEW.response_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.response_artifact_id AND attempt_id=NEW.id AND kind='{ARTIFACT_KIND_COSMOS_RESPONSE}'
            );
        END
        """,
        f"""
        CREATE TRIGGER cosmos_attempts_artifact_ownership_update
        BEFORE UPDATE OF request_artifact_id, response_artifact_id ON cosmos_attempts
        BEGIN
            SELECT RAISE(ABORT, 'request artifact link is write-once')
            WHERE OLD.request_artifact_id IS NOT NULL AND NEW.request_artifact_id IS NOT OLD.request_artifact_id;
            SELECT RAISE(ABORT, 'response artifact link is write-once')
            WHERE OLD.response_artifact_id IS NOT NULL
                AND NEW.response_artifact_id IS NOT OLD.response_artifact_id;
            SELECT RAISE(ABORT, 'request artifact must belong to this attempt and be request')
            WHERE NEW.request_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.request_artifact_id AND attempt_id=NEW.id AND kind='{ARTIFACT_KIND_COSMOS_REQUEST}'
            );
            SELECT RAISE(ABORT, 'response artifact must belong to this attempt and be response')
            WHERE NEW.response_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.response_artifact_id AND attempt_id=NEW.id AND kind='{ARTIFACT_KIND_COSMOS_RESPONSE}'
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_attempts_identity_immutable
        BEFORE UPDATE OF id, job_id, source_episode_index, attempt_number ON cosmos_attempts
        BEGIN SELECT RAISE(ABORT, 'attempt identity is immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_attempts_terminal_immutable
        BEFORE UPDATE ON cosmos_attempts
        WHEN OLD.state IN ('succeeded', 'manual_only', 'cancelled')
            OR EXISTS (
                SELECT 1 FROM cosmos_jobs
                WHERE id=OLD.job_id
                    AND state IN ('completed', 'completed_with_failures', 'cancelled', 'failed')
            )
        BEGIN SELECT RAISE(ABORT, 'terminal Cosmos attempt or job is immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_attempts_no_delete
        BEFORE DELETE ON cosmos_attempts
        BEGIN SELECT RAISE(ABORT, 'attempts are immutable evidence records'); END
        """,
        """
        CREATE TRIGGER cosmos_attempts_history_append_only
        BEFORE UPDATE OF http_exchange_history_json ON cosmos_attempts
        BEGIN
            SELECT RAISE(ABORT, 'HTTP exchange history must be a JSON array')
            WHERE json_valid(NEW.http_exchange_history_json)=0
                OR json_type(NEW.http_exchange_history_json) != 'array';
            SELECT RAISE(ABORT, 'HTTP exchange history may append exactly one entry')
            WHERE json_array_length(NEW.http_exchange_history_json)
                != json_array_length(OLD.http_exchange_history_json) + 1;
            SELECT RAISE(ABORT, 'HTTP exchange history may not rewrite prior entries')
            WHERE EXISTS (
                SELECT 1
                FROM json_each(OLD.http_exchange_history_json) AS old_entry
                LEFT JOIN json_each(NEW.http_exchange_history_json) AS new_entry ON new_entry.key=old_entry.key
                WHERE new_entry.value IS NULL OR new_entry.value != old_entry.value
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_proposals_no_delete
        BEFORE DELETE ON cosmos_proposals
        BEGIN SELECT RAISE(ABORT, 'proposals are immutable evidence records'); END
        """,
        """
        CREATE TRIGGER cosmos_proposals_terminal_attempt_no_insert
        BEFORE INSERT ON cosmos_proposals
        BEGIN
            SELECT RAISE(ABORT, 'cannot add a proposal to a terminal Cosmos attempt or job')
            WHERE EXISTS (
                SELECT 1
                FROM cosmos_attempts AS attempt
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE attempt.id=NEW.attempt_id
                    AND (
                        attempt.state IN ('succeeded', 'manual_only', 'cancelled')
                        OR job.state IN ('completed', 'completed_with_failures', 'cancelled', 'failed')
                    )
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_proposals_payload_immutable
        BEFORE UPDATE OF id, attempt_id, model_response_json, step_2_start_frame, step_3_start_frame,
            step_4_start_frame, step_5_start_frame, step_6_start_frame, step_7_start_frame,
            validation_warnings_json, created_at ON cosmos_proposals
        BEGIN SELECT RAISE(ABORT, 'proposal payload is immutable'); END
        """,
        """
        CREATE TRIGGER cosmos_proposals_state_transition
        BEFORE UPDATE OF state ON cosmos_proposals
        BEGIN
            SELECT RAISE(ABORT, 'proposal may only transition active to superseded')
            WHERE OLD.state != 'active' OR NEW.state != 'superseded';
        END
        """,
        """
        CREATE TRIGGER cosmos_proposals_one_active_source_insert
        BEFORE INSERT ON cosmos_proposals
        WHEN NEW.state='active'
        BEGIN
            SELECT RAISE(ABORT, 'only one active proposal may exist per dataset episode')
            WHERE EXISTS (
                SELECT 1
                FROM cosmos_proposals AS proposal
                JOIN cosmos_attempts AS existing_attempt ON existing_attempt.id=proposal.attempt_id
                JOIN cosmos_jobs AS existing_job ON existing_job.id=existing_attempt.job_id
                JOIN cosmos_attempts AS new_attempt ON new_attempt.id=NEW.attempt_id
                JOIN cosmos_jobs AS new_job ON new_job.id=new_attempt.job_id
                WHERE proposal.state='active'
                    AND existing_job.dataset_id=new_job.dataset_id
                    AND existing_attempt.source_episode_index=new_attempt.source_episode_index
            );
        END
        """,
        """
        CREATE TRIGGER cosmos_proposals_one_active_source_update
        BEFORE UPDATE OF state, attempt_id ON cosmos_proposals
        WHEN NEW.state='active'
        BEGIN
            SELECT RAISE(ABORT, 'only one active proposal may exist per dataset episode')
            WHERE EXISTS (
                SELECT 1
                FROM cosmos_proposals AS proposal
                JOIN cosmos_attempts AS existing_attempt ON existing_attempt.id=proposal.attempt_id
                JOIN cosmos_jobs AS existing_job ON existing_job.id=existing_attempt.job_id
                JOIN cosmos_attempts AS new_attempt ON new_attempt.id=NEW.attempt_id
                JOIN cosmos_jobs AS new_job ON new_job.id=new_attempt.job_id
                WHERE proposal.id != NEW.id AND proposal.state='active'
                    AND existing_job.dataset_id=new_job.dataset_id
                    AND existing_attempt.source_episode_index=new_attempt.source_episode_index
            );
        END
        """,
        f"""
        CREATE TRIGGER exports_artifact_ownership_insert
        BEFORE INSERT ON exports
        BEGIN
            SELECT RAISE(ABORT, 'structural artifact must belong to this export and be structural_report')
            WHERE NEW.structural_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.structural_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_STRUCTURAL_REPORT}'
            );
            SELECT RAISE(ABORT, 'stats artifact must belong to this export and be gr00t_stats_report')
            WHERE NEW.gr00t_stats_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.gr00t_stats_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_GROOT_STATS_REPORT}'
            );
            SELECT RAISE(ABORT, 'loader artifact must belong to this export and be gr00t_loader_report')
            WHERE NEW.gr00t_loader_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.gr00t_loader_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_GROOT_LOADER_REPORT}'
            );
            SELECT RAISE(ABORT, 'final artifact must belong to this export and be final_consistency_report')
            WHERE NEW.final_consistency_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.final_consistency_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_FINAL_CONSISTENCY_REPORT}'
            );
        END
        """,
        f"""
        CREATE TRIGGER exports_artifact_ownership_update
        BEFORE UPDATE OF id, structural_artifact_id, gr00t_stats_artifact_id, gr00t_loader_artifact_id,
            final_consistency_artifact_id ON exports
        BEGIN
            SELECT RAISE(ABORT, 'structural artifact must belong to this export and be structural_report')
            WHERE NEW.structural_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.structural_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_STRUCTURAL_REPORT}'
            );
            SELECT RAISE(ABORT, 'stats artifact must belong to this export and be gr00t_stats_report')
            WHERE NEW.gr00t_stats_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.gr00t_stats_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_GROOT_STATS_REPORT}'
            );
            SELECT RAISE(ABORT, 'loader artifact must belong to this export and be gr00t_loader_report')
            WHERE NEW.gr00t_loader_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.gr00t_loader_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_GROOT_LOADER_REPORT}'
            );
            SELECT RAISE(ABORT, 'final artifact must belong to this export and be final_consistency_report')
            WHERE NEW.final_consistency_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifacts
                WHERE id=NEW.final_consistency_artifact_id AND export_id=NEW.id
                    AND kind='{ARTIFACT_KIND_FINAL_CONSISTENCY_REPORT}'
            );
        END
        """,
        """
        CREATE TRIGGER audit_events_dataset_ownership_insert
        BEFORE INSERT ON audit_events
        BEGIN
            SELECT RAISE(ABORT, 'audit episode must belong to the audit dataset')
            WHERE NEW.episode_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM episodes WHERE id=NEW.episode_id AND dataset_id=NEW.dataset_id
            );
            SELECT RAISE(ABORT, 'audit job must belong to the audit dataset')
            WHERE NEW.job_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM cosmos_jobs WHERE id=NEW.job_id AND dataset_id=NEW.dataset_id
            );
            SELECT RAISE(ABORT, 'audit export must belong to the audit dataset')
            WHERE NEW.export_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM exports WHERE id=NEW.export_id AND dataset_id=NEW.dataset_id
            );
        END
        """,
        """
        CREATE TRIGGER artifacts_terminal_attempt_no_insert
        BEFORE INSERT ON artifacts
        WHEN NEW.attempt_id IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'cannot add an artifact to a terminal Cosmos attempt or job')
            WHERE EXISTS (
                SELECT 1
                FROM cosmos_attempts AS attempt
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE attempt.id=NEW.attempt_id
                    AND (
                        attempt.state IN ('succeeded', 'manual_only', 'cancelled')
                        OR job.state IN ('completed', 'completed_with_failures', 'cancelled', 'failed')
                    )
            );
        END
        """,
        """
        CREATE TRIGGER artifacts_no_update
        BEFORE UPDATE ON artifacts
        BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END
        """,
        """
        CREATE TRIGGER artifacts_no_delete
        BEFORE DELETE ON artifacts
        BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END
        """,
        """
        CREATE TRIGGER export_episodes_no_update
        BEFORE UPDATE ON export_episodes
        BEGIN SELECT RAISE(ABORT, 'export_episodes are immutable'); END
        """,
        """
        CREATE TRIGGER export_episodes_no_delete
        BEFORE DELETE ON export_episodes
        BEGIN SELECT RAISE(ABORT, 'export_episodes are immutable'); END
        """,
    )
