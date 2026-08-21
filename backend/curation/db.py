"""SQLite persistence for the isolated, resumable curation workspace.

This module deliberately opens a fresh configured connection for each public
operation.  Curation has a single authoritative SQLite database, while browser
requests, the worker, and exporter are independent processes; connection-local
PRAGMAs therefore cannot be treated as one-time process setup.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Any
from uuid import uuid4

from .models import AttemptState, ExportState, JobState, ProposalState, ReviewState

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MILLISECONDS = 5_000
ARTIFACT_KIND_COSMOS_REQUEST = "request"
ARTIFACT_KIND_COSMOS_RESPONSE = "response"
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


def canonical_json(value: Any) -> str:
    """Return the one JSON representation used by every hash-bearing record."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _enum_values(enum: type[Any]) -> str:
    return ", ".join(repr(member.value) for member in enum)


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
            connection.execute(
                "UPDATE exports SET state=?, updated_at=? WHERE id=?", (state.value, _utc_now(), current["id"])
            )
            return _require_row(
                connection.execute("SELECT * FROM exports WHERE id=?", (current["id"],)).fetchone()
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
            created_at TEXT NOT NULL
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
        BEFORE UPDATE OF id, request_artifact_id, response_artifact_id ON cosmos_attempts
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
