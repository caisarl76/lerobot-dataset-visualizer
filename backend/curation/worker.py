"""Persisted batch API and the separate resumable Cosmos worker.

FastAPI only calls :class:`BatchService`; it never constructs or starts a
worker.  The CLI at ``backend/curation_worker.py`` is the sole execution
entrypoint and uses the lease operations in this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

from .config import CurationConfigurationError, CurationSettings
from .contact_sheets import ContactSheetCoordinator, ContactSheetUnavailable
from .cosmos_contract import CosmosContractError, CosmosProposal, build_cosmos_proposal
from .cosmos_transport import (
    MAX_DURATION_SECONDS,
    MAX_PAYLOAD_BYTES,
    MAX_SAMPLED_FRAMES,
    REPAIR_INVALID_RESPONSE_MAX_BYTES,
    TARGET_SAMPLING_FPS,
    ArtifactConflict,
    ArtifactRecord,
    ArtifactSecurityError,
    AtomicArtifactStore,
    CosmosCallObservation,
    CosmosTransport,
    SamplingLimits,
    SamplingOutcome,
    build_canonical_prompt,
    build_parsed_artifact,
    build_request_artifact,
    prepare_episode_samples,
    prepare_initial_request,
)
from .db import (
    ARTIFACT_KIND_COSMOS_PARSED,
    ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE,
    ARTIFACT_KIND_COSMOS_REQUEST,
    ARTIFACT_KIND_COSMOS_RESPONSE,
    CurationDatabase,
    IncompatibleCurationDatabase,
    RetryableDatabaseError,
    _validate_http_exchange,
    canonical_json,
)
from .models import AttemptState, JobState
from .prompts import PROMPT_TEMPLATE_SHA256, PROMPT_TEMPLATE_VERSION
from .source import SourceRecord, SourceRegistry

LEASE_SECONDS = 180
HEARTBEAT_SECONDS = 15
_TERMINAL_JOBS = frozenset(
    {
        JobState.COMPLETED.value,
        JobState.COMPLETED_WITH_FAILURES.value,
        JobState.CANCELLED.value,
        JobState.FAILED.value,
    }
)
_TERMINAL_ATTEMPTS = frozenset(
    {AttemptState.SUCCEEDED.value, AttemptState.MANUAL_ONLY.value, AttemptState.CANCELLED.value}
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class BatchError(RuntimeError):
    status_code = 400

    def __init__(self, payload: Mapping[str, Any], *, status_code: int | None = None) -> None:
        self.payload = dict(payload)
        if status_code is not None:
            self.status_code = status_code
        super().__init__(str(self.payload.get("error", "batch_error")))


class WorkerStateError(RuntimeError):
    """The named command cannot operate on the persisted job state."""


class InvalidPersistedConfiguration(WorkerStateError):
    """Persisted job configuration or attempt history is not trustworthy JSON."""


class ContactSheetReconciliationConflict(RuntimeError):
    """Immutable contact-sheet evidence conflicts with its lifecycle snapshot."""


def _raise_for_contact_sheet_conflicts(statuses: Sequence[Any]) -> None:
    if any(getattr(status, "status", None) == "conflict" for status in statuses):
        raise ContactSheetReconciliationConflict


@dataclass(frozen=True)
class FrozenJobConfiguration:
    dataset_alias: str
    dataset_id: int
    source_path: Path
    source_manifest_sha256: str
    source_fps: float
    episode_indices: tuple[int, ...]
    parent_job_id: str | None
    cosmos_base_url: str
    cosmos_model: str
    cosmos_api_key_env: str
    cosmos_endpoint_identity: str


@dataclass(frozen=True)
class TrustedWorkerAuthority:
    """Secret-free runtime authority used to fence persisted worker snapshots."""

    workspace: Path
    dataset_aliases: tuple[tuple[str, Path], ...]
    cosmos_base_url: str
    cosmos_model: str
    cosmos_api_key_env: str
    cosmos_endpoint_identity: str
    worker_concurrency: int
    http_timeout_seconds: int
    transport_attempts: int
    repair_attempts: int
    target_sampling_fps: int
    maximum_duration_seconds: int
    maximum_sampled_frames: int
    maximum_payload_bytes: int

    @classmethod
    def from_settings(
        cls,
        settings: CurationSettings,
        *,
        cli_workspace: Path,
    ) -> TrustedWorkerAuthority:
        if cli_workspace != settings.workspace:
            raise InvalidPersistedConfiguration("CLI workspace is not the configured curation workspace")
        return cls.from_components(
            workspace=settings.workspace,
            dataset_aliases=settings.dataset_aliases,
            cosmos_base_url=settings.cosmos_base_url,
            cosmos_model=settings.cosmos_model,
            cosmos_api_key_env=settings.cosmos_api_key_env,
            cosmos_endpoint_identity=settings.cosmos_endpoint_identity,
            worker_concurrency=settings.worker_concurrency,
            http_timeout_seconds=settings.http_timeout_seconds,
            transport_attempts=settings.transport_attempts,
            repair_attempts=settings.repair_attempts,
            target_sampling_fps=settings.target_sampling_fps,
            maximum_duration_seconds=settings.maximum_duration_seconds,
            maximum_sampled_frames=settings.maximum_sampled_frames,
            maximum_payload_bytes=settings.maximum_payload_bytes,
        )

    @classmethod
    def from_components(
        cls,
        *,
        workspace: Path,
        dataset_aliases: Mapping[str, Path],
        cosmos_base_url: str,
        cosmos_model: str,
        cosmos_api_key_env: str,
        cosmos_endpoint_identity: str,
        worker_concurrency: int = 1,
        http_timeout_seconds: int = 120,
        transport_attempts: int = 2,
        repair_attempts: int = 1,
        target_sampling_fps: int = 2,
        maximum_duration_seconds: int = 120,
        maximum_sampled_frames: int = 240,
        maximum_payload_bytes: int = 67_108_864,
    ) -> TrustedWorkerAuthority:
        try:
            trusted_workspace = Path(workspace)
            if not trusted_workspace.is_absolute():
                raise ValueError
            aliases = tuple(sorted((alias, Path(path)) for alias, path in dataset_aliases.items()))
            if (
                not aliases
                or any(not alias or not source.is_absolute() for alias, source in aliases)
                or not cosmos_model
                or not cosmos_api_key_env
                or not cosmos_endpoint_identity
            ):
                raise ValueError
            base_url = _validate_base_url(cosmos_base_url)
            frozen_integers = (
                worker_concurrency,
                http_timeout_seconds,
                transport_attempts,
                repair_attempts,
                target_sampling_fps,
                maximum_duration_seconds,
                maximum_sampled_frames,
                maximum_payload_bytes,
            )
            if any(type(value) is not int for value in frozen_integers):
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            raise InvalidPersistedConfiguration("trusted curation worker authority is invalid") from None
        return cls(
            workspace=trusted_workspace,
            dataset_aliases=aliases,
            cosmos_base_url=base_url,
            cosmos_model=cosmos_model,
            cosmos_api_key_env=cosmos_api_key_env,
            cosmos_endpoint_identity=cosmos_endpoint_identity,
            worker_concurrency=worker_concurrency,
            http_timeout_seconds=http_timeout_seconds,
            transport_attempts=transport_attempts,
            repair_attempts=repair_attempts,
            target_sampling_fps=target_sampling_fps,
            maximum_duration_seconds=maximum_duration_seconds,
            maximum_sampled_frames=maximum_sampled_frames,
            maximum_payload_bytes=maximum_payload_bytes,
        )


@dataclass(frozen=True)
class BoundJobAuthority:
    configuration: FrozenJobConfiguration
    workspace: Path
    source_path: Path
    cosmos_base_url: str
    cosmos_model: str
    cosmos_api_key_env: str
    cosmos_endpoint_identity: str


def validate_frozen_job_configuration(raw: str | bytes | bytearray) -> FrozenJobConfiguration:
    """Parse the closed, immutable v1 job configuration without external I/O."""

    top_level_keys = {
        "schema_version",
        "dataset_alias",
        "dataset_id",
        "source_path",
        "source_manifest_sha256",
        "source_fps",
        "episode_indices",
        "prompt",
        "cosmos",
        "sampling",
        "worker",
        "transport",
        "limits",
    }

    def exact_object(value: Any, keys: set[str]) -> dict[str, Any]:
        if type(value) is not dict or set(value) != keys:
            raise ValueError
        return value

    def nonempty_string(value: Any) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError
        return value

    def lowercase_sha256(value: Any) -> str:
        text = nonempty_string(value)
        if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise ValueError
        return text

    try:
        document = json.loads(raw)
        if type(document) is not dict:
            raise ValueError
        observed_top_level = set(document)
        parent_present = "parent_job_id" in observed_top_level
        expected_top_level = top_level_keys | ({"parent_job_id"} if parent_present else set())
        if observed_top_level != expected_top_level:
            raise ValueError
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError
        dataset_alias = nonempty_string(document["dataset_alias"])
        dataset_id = document["dataset_id"]
        if type(dataset_id) is not int or dataset_id <= 0:
            raise ValueError
        source_path_text = nonempty_string(document["source_path"])
        source_path = Path(source_path_text)
        if not source_path.is_absolute():
            raise ValueError
        source_manifest_sha256 = lowercase_sha256(document["source_manifest_sha256"])
        source_fps = document["source_fps"]
        if type(source_fps) is not float or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError
        episode_indices_value = document["episode_indices"]
        if (
            type(episode_indices_value) is not list
            or not episode_indices_value
            or any(type(index) is not int or index < 0 for index in episode_indices_value)
            or episode_indices_value != sorted(set(episode_indices_value))
        ):
            raise ValueError
        episode_indices = tuple(episode_indices_value)
        parent_job_id = document.get("parent_job_id")
        if parent_present:
            parent_job_id = str(UUID(nonempty_string(parent_job_id)))

        prompt = exact_object(document["prompt"], {"version", "sha256"})
        if prompt != {"version": PROMPT_TEMPLATE_VERSION, "sha256": PROMPT_TEMPLATE_SHA256}:
            raise ValueError

        cosmos = exact_object(
            document["cosmos"],
            {"base_url", "model", "api_key_env", "endpoint_identity"},
        )
        cosmos_base_url_raw = nonempty_string(cosmos["base_url"])
        cosmos_base_url = _validate_base_url(cosmos_base_url_raw)
        if cosmos_base_url != cosmos_base_url_raw:
            raise ValueError
        cosmos_model = nonempty_string(cosmos["model"])
        cosmos_api_key_env = nonempty_string(cosmos["api_key_env"])
        cosmos_endpoint_identity = nonempty_string(cosmos["endpoint_identity"])

        sampling = exact_object(document["sampling"], {"target_fps"})
        if type(sampling["target_fps"]) is not int or sampling["target_fps"] != TARGET_SAMPLING_FPS:
            raise ValueError
        worker = exact_object(document["worker"], {"concurrency", "lease_seconds", "heartbeat_seconds"})
        if worker != {
            "concurrency": 1,
            "lease_seconds": 180,
            "heartbeat_seconds": 15,
        } or any(type(value) is not int for value in worker.values()):
            raise ValueError
        transport = exact_object(
            document["transport"],
            {"timeout_seconds", "initial_attempts", "repair_attempts"},
        )
        if transport != {"timeout_seconds": 120, "initial_attempts": 2, "repair_attempts": 1} or any(
            type(value) is not int for value in transport.values()
        ):
            raise ValueError
        limits = exact_object(
            document["limits"],
            {"maximum_duration_seconds", "maximum_sampled_frames", "maximum_payload_bytes"},
        )
        if limits != {
            "maximum_duration_seconds": MAX_DURATION_SECONDS,
            "maximum_sampled_frames": MAX_SAMPLED_FRAMES,
            "maximum_payload_bytes": MAX_PAYLOAD_BYTES,
        } or any(type(value) is not int for value in limits.values()):
            raise ValueError
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        raise InvalidPersistedConfiguration("persisted curation job configuration is invalid") from None

    return FrozenJobConfiguration(
        dataset_alias=dataset_alias,
        dataset_id=dataset_id,
        source_path=source_path,
        source_manifest_sha256=source_manifest_sha256,
        source_fps=source_fps,
        episode_indices=episode_indices,
        parent_job_id=parent_job_id,
        cosmos_base_url=cosmos_base_url,
        cosmos_model=cosmos_model,
        cosmos_api_key_env=cosmos_api_key_env,
        cosmos_endpoint_identity=cosmos_endpoint_identity,
    )


def bind_frozen_job_authority(
    configuration: FrozenJobConfiguration,
    *,
    dataset: Mapping[str, Any],
    authority: TrustedWorkerAuthority,
) -> BoundJobAuthority:
    """Bind an untrusted persisted snapshot to one secret-free runtime authority."""

    trusted_sources = dict(authority.dataset_aliases)
    trusted_source = trusted_sources.get(configuration.dataset_alias)
    expected_frozen_settings = (
        authority.worker_concurrency,
        authority.http_timeout_seconds,
        authority.transport_attempts,
        authority.repair_attempts,
        authority.target_sampling_fps,
        authority.maximum_duration_seconds,
        authority.maximum_sampled_frames,
        authority.maximum_payload_bytes,
    )
    if (
        trusted_source is None
        or configuration.dataset_id != dataset.get("id")
        or configuration.dataset_alias != dataset.get("alias")
        or configuration.source_path != trusted_source
        or dataset.get("source_path") != str(trusted_source)
        or configuration.source_manifest_sha256 != dataset.get("source_manifest_sha256")
        or dataset.get("prompt_template_version") != PROMPT_TEMPLATE_VERSION
        or dataset.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256
        or configuration.cosmos_base_url != authority.cosmos_base_url
        or configuration.cosmos_model != authority.cosmos_model
        or configuration.cosmos_api_key_env != authority.cosmos_api_key_env
        or configuration.cosmos_endpoint_identity != authority.cosmos_endpoint_identity
        or expected_frozen_settings
        != (
            1,
            120,
            2,
            1,
            2,
            MAX_DURATION_SECONDS,
            MAX_SAMPLED_FRAMES,
            MAX_PAYLOAD_BYTES,
        )
    ):
        raise InvalidPersistedConfiguration("persisted curation authority does not match runtime settings")
    return BoundJobAuthority(
        configuration=configuration,
        workspace=authority.workspace,
        source_path=trusted_source,
        cosmos_base_url=authority.cosmos_base_url,
        cosmos_model=authority.cosmos_model,
        cosmos_api_key_env=authority.cosmos_api_key_env,
        cosmos_endpoint_identity=authority.cosmos_endpoint_identity,
    )


class LiveLeaseConflict(RuntimeError):
    """A different worker still owns an unexpired job or attempt lease."""


class AttemptPersistenceGuard:
    """Hold one owner/live-lease transaction across artifact install and registration."""

    def __init__(self, repository: BatchRepository, *, attempt_id: str, owner: str) -> None:
        self.repository = repository
        self.attempt_id = attempt_id
        self.owner = owner
        self.connection: sqlite3.Connection | None = None

    @contextmanager
    def __call__(self) -> Iterator[None]:
        if self.connection is not None:
            raise RuntimeError("attempt persistence guard cannot be nested")
        with self.repository.database._write() as connection:
            self.repository._require_owned_live_attempt(connection, self.attempt_id, owner=self.owner)
            self.connection = connection
            try:
                yield
            finally:
                self.connection = None


class BatchRepository:
    """Short SQLite transactions for batch/attempt lifecycle state."""

    def __init__(
        self,
        database: CurationDatabase,
        *,
        clock: Callable[[], datetime] = _utc_now,
        trusted_authority: TrustedWorkerAuthority | None = None,
    ) -> None:
        self.database = database
        self.clock = clock
        self.trusted_authority = trusted_authority

    def _now(self) -> datetime:
        return _as_utc(self.clock())

    def create_job_with_attempts(
        self,
        *,
        dataset_id: int,
        configuration: Mapping[str, Any],
        episode_indices: Sequence[int],
        parent_job_id: str | None = None,
        attempt_numbers: Mapping[int, int] | None = None,
    ) -> dict[str, Any]:
        job_id = str(uuid4())
        numbers = dict(attempt_numbers or {})
        with self.database._write() as connection:
            now = _timestamp(self._now())
            try:
                connection.execute(
                    """
                    INSERT INTO cosmos_jobs(
                        id, dataset_id, parent_job_id, configuration_json, state, total_attempts,
                        succeeded_attempts, manual_only_attempts, cancel_requested, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?, 0, 0, 0, ?, ?)
                    """,
                    (
                        job_id,
                        dataset_id,
                        parent_job_id,
                        canonical_json(dict(configuration)),
                        len(episode_indices),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                active = connection.execute(
                    """
                    SELECT id FROM cosmos_jobs
                    WHERE dataset_id=? AND state IN ('queued', 'running', 'cancel_requested')
                    """,
                    (dataset_id,),
                ).fetchone()
                if active is not None:
                    raise BatchError(
                        {"error": "active_batch_exists", "job_id": active["id"]}, status_code=409
                    ) from error
                raise
            for episode_index in episode_indices:
                connection.execute(
                    """
                    INSERT INTO cosmos_attempts(
                        id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                    """,
                    (str(uuid4()), job_id, episode_index, numbers.get(episode_index, 0), now, now),
                )
            return self._status_in_transaction(connection, job_id)

    def get_job_row(self, job_id: str) -> dict[str, Any] | None:
        with self.database._read() as connection:
            row = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            return None if row is None else dict(row)

    def cancel_is_requested(self, job_id: str) -> bool:
        with self.database._read() as connection:
            row = connection.execute(
                "SELECT state, cancel_requested FROM cosmos_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise WorkerStateError("job not found")
            return row["state"] == JobState.CANCEL_REQUESTED.value or bool(row["cancel_requested"])

    def find_active_job(self, dataset_id: int) -> dict[str, Any] | None:
        with self.database._read() as connection:
            row = connection.execute(
                """
                SELECT * FROM cosmos_jobs
                WHERE dataset_id=? AND state IN ('queued', 'running', 'cancel_requested')
                LIMIT 1
                """,
                (dataset_id,),
            ).fetchone()
            return None if row is None else dict(row)

    def command_preflight(self, command: LiteralCommand, job_id: str) -> None:
        now = self._now()
        with self.database._read() as connection:
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise WorkerStateError("job not found")
            if command == "run":
                if job["state"] != JobState.QUEUED.value:
                    raise WorkerStateError("run requires a queued job")
                return
            if job["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}:
                raise WorkerStateError("resume requires a running or cancel_requested job")
            if _lease_is_live(job["lease_expires_at"], now):
                raise LiveLeaseConflict("job lease is still live")
            attempt = connection.execute(
                """
                SELECT 1 FROM cosmos_attempts
                WHERE job_id=? AND state IN ('leased', 'requesting') AND lease_expires_at>?
                LIMIT 1
                """,
                (job_id, _timestamp(now)),
            ).fetchone()
            if attempt is not None:
                raise LiveLeaseConflict("attempt lease is still live")

    def status(self, job_id: str) -> dict[str, Any]:
        with self.database._read() as connection:
            return self._status_in_transaction(connection, job_id)

    def _status_in_transaction(self, connection: sqlite3.Connection, job_id: str) -> dict[str, Any]:
        job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise BatchError({"error": "batch_not_found", "job_id": job_id}, status_code=404)
        attempts = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM cosmos_attempts
                WHERE job_id=? ORDER BY source_episode_index, attempt_number, id
                """,
                (job_id,),
            )
        ]
        counts = Counter(row["state"] for row in attempts)
        episodes = [
            {
                "attempt_id": row["id"],
                "attempt_number": row["attempt_number"],
                "source_episode_index": row["source_episode_index"],
                "state": row["state"],
            }
            for row in attempts
        ]
        error_rows = connection.execute(
            """
            SELECT * FROM cosmos_attempts
            WHERE job_id=? AND (error_class IS NOT NULL OR error_summary IS NOT NULL)
            ORDER BY updated_at DESC, id DESC
            LIMIT 20
            """,
            (job_id,),
        ).fetchall()
        errors = [
            {
                "attempt_id": row["id"],
                "source_episode_index": row["source_episode_index"],
                "error_class": row["error_class"],
                "error_summary": row["error_summary"],
            }
            for row in error_rows
        ]
        coverage = connection.execute(
            """
            SELECT count(DISTINCT attempt.source_episode_index)
            FROM cosmos_proposals AS proposal
            JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
            JOIN cosmos_jobs AS proposal_job ON proposal_job.id=attempt.job_id
            WHERE proposal_job.dataset_id=? AND proposal.state='active'
            """,
            (job["dataset_id"],),
        ).fetchone()[0]
        current = next(
            (row["source_episode_index"] for row in attempts if row["state"] in {"leased", "requesting"}),
            None,
        )
        return {
            "job_id": job["id"],
            "parent_job_id": job["parent_job_id"],
            "state": job["state"],
            "configuration": json.loads(job["configuration_json"]),
            "counts": dict(sorted(counts.items())),
            "episodes": episodes,
            "lease": None
            if job["owner"] is None
            else {"owner": job["owner"], "expires_at": job["lease_expires_at"]},
            "current_episode": current,
            "cancel_requested": bool(job["cancel_requested"]),
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "errors": errors,
            "active_proposal_coverage": coverage,
        }

    def cancel(self, job_id: str) -> tuple[int, dict[str, Any]]:
        with self.database._write() as connection:
            now = _timestamp(self._now())
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                return 404, {"error": "batch_not_found", "job_id": job_id}
            state = job["state"]
            if state == JobState.QUEUED.value:
                connection.execute(
                    """
                    UPDATE cosmos_attempts
                    SET state='cancelled', lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE job_id=? AND state IN ('queued', 'retryable', 'leased', 'requesting')
                    """,
                    (now, job_id),
                )
                connection.execute(
                    """
                    UPDATE cosmos_jobs SET state='cancelled', cancel_requested=1,
                        owner=NULL, lease_expires_at=NULL, updated_at=? WHERE id=?
                    """,
                    (now, job_id),
                )
                self.database._append_audit_event(
                    connection,
                    dataset_id=job["dataset_id"],
                    actor="api",
                    operation="batch_cancelled",
                    job_id=job_id,
                    details={"previous_state": state},
                )
                return 200, {"job_id": job_id, "state": "cancelled", "changed": True}
            if state == JobState.RUNNING.value:
                connection.execute(
                    "UPDATE cosmos_jobs SET state='cancel_requested', cancel_requested=1, updated_at=? WHERE id=?",
                    (now, job_id),
                )
                self.database._append_audit_event(
                    connection,
                    dataset_id=job["dataset_id"],
                    actor="api",
                    operation="batch_cancel_requested",
                    job_id=job_id,
                    details={"previous_state": state},
                )
                return 202, {"job_id": job_id, "state": "cancel_requested", "changed": True}
            if state == JobState.CANCEL_REQUESTED.value:
                return 202, {"job_id": job_id, "state": state, "changed": False}
            if state == JobState.CANCELLED.value:
                return 200, {"job_id": job_id, "state": state, "changed": False}
            return 409, {"error": "batch_terminal", "job_id": job_id, "state": state}

    def start_job(self, job_id: str, *, owner: str) -> dict[str, Any]:
        _require_owner_uuid(owner)
        authority = self._require_trusted_authority()
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            expiry = _timestamp(now_dt + timedelta(seconds=LEASE_SECONDS))
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise WorkerStateError("job not found")
            if job["state"] != JobState.QUEUED.value:
                raise WorkerStateError("run requires a queued job")
            self._validate_persisted_job_json(connection, job, authority=authority)
            connection.execute(
                "UPDATE cosmos_jobs SET state='running', owner=?, lease_expires_at=?, updated_at=? WHERE id=?",
                (owner, expiry, now, job_id),
            )
            self.database._append_audit_event(
                connection,
                dataset_id=job["dataset_id"],
                actor=owner,
                operation="batch_started",
                job_id=job_id,
                details={},
            )
            return self._status_in_transaction(connection, job_id)

    def resume_job(self, job_id: str, *, owner: str) -> dict[str, Any]:
        _require_owner_uuid(owner)
        authority = self._require_trusted_authority()
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            expiry = _timestamp(now_dt + timedelta(seconds=LEASE_SECONDS))
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise WorkerStateError("job not found")
            if job["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}:
                raise WorkerStateError("resume requires a running or cancel_requested job")
            if _lease_is_live(job["lease_expires_at"], now_dt):
                raise LiveLeaseConflict("job lease is still live")
            live_attempt = connection.execute(
                """
                SELECT 1 FROM cosmos_attempts
                WHERE job_id=? AND state IN ('leased', 'requesting') AND lease_expires_at>?
                LIMIT 1
                """,
                (job_id, now),
            ).fetchone()
            if live_attempt is not None:
                raise LiveLeaseConflict("attempt lease is still live")
            self._validate_persisted_job_json(connection, job, authority=authority)
            expired = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM cosmos_attempts
                    WHERE job_id=? AND state IN ('leased', 'requesting')
                    ORDER BY source_episode_index
                    """,
                    (job_id,),
                )
            ]
            if job["state"] == JobState.RUNNING.value:
                for attempt in expired:
                    connection.execute(
                        """
                        UPDATE cosmos_attempts SET state='retryable', lease_owner=NULL,
                            lease_expires_at=NULL, error_class='WorkerCrash',
                            error_summary='expired worker lease reclaimed', updated_at=? WHERE id=?
                        """,
                        (now, attempt["id"]),
                    )
                    self.database._append_audit_event(
                        connection,
                        dataset_id=job["dataset_id"],
                        actor=owner,
                        operation="attempt_lease_reclaimed",
                        job_id=job_id,
                        details={
                            "attempt_id": attempt["id"],
                            "source_episode_index": attempt["source_episode_index"],
                        },
                    )
            connection.execute(
                "UPDATE cosmos_jobs SET owner=?, lease_expires_at=?, updated_at=? WHERE id=?",
                (owner, expiry, now, job_id),
            )
            return self._status_in_transaction(connection, job_id)

    def _validate_persisted_job_json(
        self,
        connection: sqlite3.Connection,
        job: sqlite3.Row,
        *,
        authority: TrustedWorkerAuthority,
    ) -> BoundJobAuthority:
        try:
            configuration = validate_frozen_job_configuration(job["configuration_json"])
            if (
                configuration.dataset_id != job["dataset_id"]
                or configuration.parent_job_id != job["parent_job_id"]
                or len(configuration.episode_indices) != job["total_attempts"]
            ):
                raise ValueError
            attempt_rows = connection.execute(
                """
                SELECT source_episode_index, http_exchange_history_json FROM cosmos_attempts
                WHERE job_id=? ORDER BY source_episode_index, attempt_number, id
                """,
                (job["id"],),
            ).fetchall()
            if tuple(row["source_episode_index"] for row in attempt_rows) != configuration.episode_indices:
                raise ValueError
            for row in attempt_rows:
                history = json.loads(row["http_exchange_history_json"])
                if not isinstance(history, list):
                    raise ValueError
                for exchange in history:
                    if not isinstance(exchange, Mapping):
                        raise ValueError
                    _validate_http_exchange(exchange)
            dataset_row = connection.execute(
                "SELECT * FROM datasets WHERE id=?",
                (job["dataset_id"],),
            ).fetchone()
            if dataset_row is None:
                raise ValueError
            return bind_frozen_job_authority(
                configuration,
                dataset=dict(dataset_row),
                authority=authority,
            )
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError, InvalidPersistedConfiguration):
            raise InvalidPersistedConfiguration("persisted curation job JSON is invalid") from None

    def _require_trusted_authority(self) -> TrustedWorkerAuthority:
        if self.trusted_authority is None:
            raise InvalidPersistedConfiguration("trusted curation worker authority is required")
        return self.trusted_authority

    def bind_job_authority(self, job_id: str) -> BoundJobAuthority:
        authority = self._require_trusted_authority()
        with self.database._read() as connection:
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise WorkerStateError("job not found")
            return self._validate_persisted_job_json(connection, job, authority=authority)

    def heartbeat(self, job_id: str, *, owner: str, attempt_id: str | None = None) -> None:
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            expiry = _timestamp(now_dt + timedelta(seconds=LEASE_SECONDS))
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if (
                job is None
                or job["owner"] != owner
                or job["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}
                or not _lease_is_live(job["lease_expires_at"], now_dt)
            ):
                raise LiveLeaseConflict("job lease ownership was lost")
            if attempt_id is not None:
                attempt = connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone()
                if attempt is None or attempt["job_id"] != job_id:
                    raise LiveLeaseConflict("attempt lease ownership was lost")
                if attempt["state"] not in _TERMINAL_ATTEMPTS:
                    if (
                        attempt["state"] not in {AttemptState.LEASED.value, AttemptState.REQUESTING.value}
                        or attempt["lease_owner"] != owner
                        or not _lease_is_live(attempt["lease_expires_at"], now_dt)
                    ):
                        raise LiveLeaseConflict("attempt lease ownership was lost")
                    connection.execute(
                        "UPDATE cosmos_attempts SET lease_expires_at=?, updated_at=? WHERE id=?",
                        (expiry, now, attempt_id),
                    )
            connection.execute(
                "UPDATE cosmos_jobs SET lease_expires_at=?, updated_at=? WHERE id=?",
                (expiry, now, job_id),
            )

    def release_job_after_configuration_error(self, job_id: str, *, owner: str) -> None:
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if (
                job is None
                or job["owner"] != owner
                or job["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}
                or not _lease_is_live(job["lease_expires_at"], now_dt)
            ):
                raise LiveLeaseConflict("job lease ownership was lost")
            connection.execute(
                "UPDATE cosmos_jobs SET owner=NULL, lease_expires_at=?, updated_at=? WHERE id=?",
                (now, now, job_id),
            )
            self.database._append_audit_event(
                connection,
                dataset_id=job["dataset_id"],
                actor=owner,
                operation="worker_configuration_error",
                job_id=job_id,
                details={},
            )

    def claim(self, job_id: str, *, owner: str) -> dict[str, Any] | None:
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            expiry = _timestamp(now_dt + timedelta(seconds=LEASE_SECONDS))
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job["owner"] != owner or not _lease_is_live(job["lease_expires_at"], now_dt):
                raise LiveLeaseConflict("worker does not own a live job lease")
            if job["state"] != JobState.RUNNING.value or job["cancel_requested"]:
                return None
            expired = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM cosmos_attempts
                    WHERE job_id=? AND state IN ('leased', 'requesting')
                        AND lease_expires_at IS NOT NULL AND lease_expires_at<=?
                    ORDER BY source_episode_index, attempt_number, id
                    """,
                    (job_id, now),
                )
            ]
            for stale in expired:
                connection.execute(
                    """
                    UPDATE cosmos_attempts SET state='retryable', lease_owner=NULL,
                        lease_expires_at=NULL, error_class='WorkerCrash',
                        error_summary='expired worker lease reclaimed', updated_at=? WHERE id=?
                    """,
                    (now, stale["id"]),
                )
                self.database._append_audit_event(
                    connection,
                    dataset_id=job["dataset_id"],
                    actor=owner,
                    operation="attempt_lease_reclaimed",
                    job_id=job_id,
                    details={
                        "attempt_id": stale["id"],
                        "source_episode_index": stale["source_episode_index"],
                    },
                )
            if (
                connection.execute(
                    """
                SELECT 1 FROM cosmos_attempts
                WHERE job_id=? AND state IN ('leased', 'requesting') LIMIT 1
                """,
                    (job_id,),
                ).fetchone()
                is not None
            ):
                return None
            attempt = connection.execute(
                """
                SELECT * FROM cosmos_attempts
                WHERE job_id=? AND state IN ('queued', 'retryable')
                ORDER BY source_episode_index, attempt_number, id LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if attempt is None:
                return None
            connection.execute(
                """
                UPDATE cosmos_attempts SET state='leased', lease_owner=?, lease_expires_at=?,
                    error_class=NULL, error_summary=NULL, updated_at=? WHERE id=?
                """,
                (owner, expiry, now, attempt["id"]),
            )
            return dict(
                connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt["id"],)).fetchone()
            )

    def mark_requesting(self, attempt_id: str, *, owner: str) -> dict[str, Any]:
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            attempt = connection.execute(
                """
                SELECT attempt.*, job.state AS job_state, job.owner AS job_owner,
                    job.lease_expires_at AS job_lease_expires_at
                FROM cosmos_attempts AS attempt
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE attempt.id=?
                """,
                (attempt_id,),
            ).fetchone()
            if (
                attempt is None
                or attempt["state"] != AttemptState.LEASED.value
                or attempt["lease_owner"] != owner
                or not _lease_is_live(attempt["lease_expires_at"], now_dt)
                or attempt["job_state"] != JobState.RUNNING.value
                or attempt["job_owner"] != owner
                or not _lease_is_live(attempt["job_lease_expires_at"], now_dt)
            ):
                raise LiveLeaseConflict("attempt lease ownership was lost")
            connection.execute(
                "UPDATE cosmos_attempts SET state='requesting', updated_at=? WHERE id=?", (now, attempt_id)
            )
            return dict(connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone())

    def finish_cancel_requested(self, job_id: str, *, owner: str) -> dict[str, Any]:
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise WorkerStateError("job is not owned cancel_requested work")
            if job["owner"] != owner or not _lease_is_live(job["lease_expires_at"], now_dt):
                raise LiveLeaseConflict("job lease ownership was lost")
            if job["state"] != JobState.CANCEL_REQUESTED.value:
                raise WorkerStateError("job is not cancel_requested work")
            connection.execute(
                """
                UPDATE cosmos_attempts SET state='cancelled', lease_owner=NULL,
                    lease_expires_at=NULL, updated_at=?
                WHERE job_id=? AND state IN ('queued', 'retryable', 'leased', 'requesting')
                """,
                (now, job_id),
            )
            connection.execute(
                """
                UPDATE cosmos_jobs SET state='cancelled', owner=NULL, lease_expires_at=NULL,
                    cancel_requested=1, updated_at=? WHERE id=?
                """,
                (now, job_id),
            )
            self.database._append_audit_event(
                connection,
                dataset_id=job["dataset_id"],
                actor=owner,
                operation="batch_cancelled",
                job_id=job_id,
                details={},
            )
            return self._status_in_transaction(connection, job_id)

    def finish_attempt_manual_only(self, attempt_id: str, *, owner: str, reason: str) -> dict[str, Any]:
        summary = str(reason).encode("utf-8", "backslashreplace").decode("utf-8")[:512]
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            attempt = connection.execute(
                """
                SELECT attempt.*, job.state AS job_state, job.owner AS job_owner,
                    job.lease_expires_at AS job_lease_expires_at
                FROM cosmos_attempts AS attempt
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE attempt.id=?
                """,
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise WorkerStateError("attempt not found")
            if attempt["state"] == AttemptState.MANUAL_ONLY.value:
                return dict(
                    connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone()
                )
            if (
                attempt["state"] not in {AttemptState.LEASED.value, AttemptState.REQUESTING.value}
                or attempt["lease_owner"] != owner
                or not _lease_is_live(attempt["lease_expires_at"], now_dt)
                or attempt["job_state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}
                or attempt["job_owner"] != owner
                or not _lease_is_live(attempt["job_lease_expires_at"], now_dt)
            ):
                raise LiveLeaseConflict("attempt lease ownership was lost")
            connection.execute(
                """
                UPDATE cosmos_attempts SET state='manual_only', lease_owner=NULL,
                    lease_expires_at=NULL, error_class='ManualOnly', error_summary=?, updated_at=?
                WHERE id=?
                """,
                (summary, now, attempt_id),
            )
            return dict(connection.execute("SELECT * FROM cosmos_attempts WHERE id=?", (attempt_id,)).fetchone())

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        with self.database._read() as connection:
            row = connection.execute(
                """
                SELECT attempt.*, job.dataset_id, job.configuration_json
                FROM cosmos_attempts AS attempt
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE attempt.id=?
                """,
                (attempt_id,),
            ).fetchone()
            return None if row is None else dict(row)

    def get_attempt_artifact(self, attempt_id: str, kind: str) -> dict[str, Any] | None:
        with self.database._read() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE attempt_id=? AND kind=? ORDER BY created_at DESC, id DESC LIMIT 1",
                (attempt_id, kind),
            ).fetchone()
            return None if row is None else dict(row)

    def persistence_guard(self, attempt_id: str, *, owner: str) -> AttemptPersistenceGuard:
        return AttemptPersistenceGuard(self, attempt_id=attempt_id, owner=owner)

    def _require_owned_live_attempt(
        self,
        connection: sqlite3.Connection,
        attempt_id: str,
        *,
        owner: str,
    ) -> sqlite3.Row:
        now = self._now()
        attempt = connection.execute(
            """
            SELECT attempt.*, job.dataset_id, job.state AS job_state,
                job.cancel_requested AS job_cancel_requested, job.owner AS job_owner,
                job.lease_expires_at AS job_lease_expires_at
            FROM cosmos_attempts AS attempt
            JOIN cosmos_jobs AS job ON job.id=attempt.job_id
            WHERE attempt.id=?
            """,
            (attempt_id,),
        ).fetchone()
        if (
            attempt is None
            or attempt["job_state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}
            or attempt["job_owner"] != owner
            or not _lease_is_live(attempt["job_lease_expires_at"], now)
            or attempt["state"] != AttemptState.REQUESTING.value
            or attempt["lease_owner"] != owner
            or not _lease_is_live(attempt["lease_expires_at"], now)
        ):
            raise LiveLeaseConflict("attempt persistence lease ownership was lost")
        return attempt

    def register_artifact(
        self,
        *,
        dataset_id: int,
        attempt_id: str,
        kind: str,
        record: ArtifactRecord,
        owner: str,
        update_attempt_pointer: bool = False,
        guard: AttemptPersistenceGuard | None = None,
    ) -> dict[str, Any]:
        if guard is None:
            guard = self.persistence_guard(attempt_id, owner=owner)
            with guard():
                return self.register_artifact(
                    dataset_id=dataset_id,
                    attempt_id=attempt_id,
                    kind=kind,
                    record=record,
                    owner=owner,
                    update_attempt_pointer=update_attempt_pointer,
                    guard=guard,
                )
        connection = guard.connection
        if connection is None or guard.attempt_id != attempt_id or guard.owner != owner:
            raise RuntimeError("artifact registration requires its active persistence guard")
        return self._register_artifact_in_transaction(
            connection,
            dataset_id=dataset_id,
            attempt_id=attempt_id,
            kind=kind,
            record=record,
            update_attempt_pointer=update_attempt_pointer,
        )

    def _register_artifact_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        dataset_id: int,
        attempt_id: str,
        kind: str,
        record: ArtifactRecord,
        update_attempt_pointer: bool,
    ) -> dict[str, Any]:
        reconciled = self.database.reconcile_attempt_artifact(
            dataset_id=dataset_id,
            attempt_id=attempt_id,
            kind=kind,
            relative_path=record.relative_path,
            media_type=record.media_type,
            byte_size=record.byte_size,
            sha256=record.sha256,
            update_attempt_pointer=update_attempt_pointer,
            _connection=connection,
        )
        return reconciled["artifact"]

    def append_observation(
        self, attempt_id: str, observation: CosmosCallObservation, *, owner: str
    ) -> dict[str, Any]:
        normalized = _validate_http_exchange(observation.exchange)
        guard = self.persistence_guard(attempt_id, owner=owner)
        with guard():
            assert guard.connection is not None
            return self._append_observation_in_transaction(
                guard.connection,
                attempt_id=attempt_id,
                normalized_exchange=normalized,
                deduplicate_exact_replay=False,
            )

    def _append_observation_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        attempt_id: str,
        normalized_exchange: Mapping[str, Any],
        deduplicate_exact_replay: bool,
    ) -> dict[str, Any]:
        return self.database.append_http_exchange_history(
            attempt_id=attempt_id,
            exchange=normalized_exchange,
            _connection=connection,
            _deduplicate_exact_replay=deduplicate_exact_replay,
        )

    def register_observed_artifact(
        self,
        *,
        dataset_id: int,
        attempt_id: str,
        kind: str,
        record: ArtifactRecord,
        observation: CosmosCallObservation,
        owner: str,
        update_attempt_pointer: bool,
        guard: AttemptPersistenceGuard | None = None,
    ) -> dict[str, Any]:
        """Atomically register response bytes, pointer, and its HTTP envelope."""

        contracts = {
            ARTIFACT_KIND_COSMOS_RESPONSE: ("initial", "response.txt", True),
            ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE: ("repair", "repair-response.txt", False),
        }
        if kind not in contracts:
            raise ValueError("observed artifact kind must be response or repair_response")
        expected_phase, expected_filename, expected_pointer = contracts[kind]
        if observation.phase != expected_phase:
            raise ValueError("observation phase does not match artifact kind")
        if update_attempt_pointer is not expected_pointer:
            raise ValueError("observation artifact pointer contract does not match its kind")
        visible_content = observation.content or observation.observed_content
        if visible_content is None:
            raise ValueError("observed artifact requires response content")
        encoded_content = visible_content.encode("utf-8")
        if (
            record.relative_path != f"artifacts/cosmos/{attempt_id}/{expected_filename}"
            or record.media_type != "text/plain; charset=utf-8"
            or record.byte_size != len(encoded_content)
            or record.sha256 != hashlib.sha256(encoded_content).hexdigest()
        ):
            raise ValueError("observed artifact record does not match its exact response content")
        normalized_exchange = _validate_http_exchange(observation.exchange)
        if guard is None:
            guard = self.persistence_guard(attempt_id, owner=owner)
            with guard():
                return self.register_observed_artifact(
                    dataset_id=dataset_id,
                    attempt_id=attempt_id,
                    kind=kind,
                    record=record,
                    observation=observation,
                    owner=owner,
                    update_attempt_pointer=update_attempt_pointer,
                    guard=guard,
                )
        connection = guard.connection
        if connection is None or guard.attempt_id != attempt_id or guard.owner != owner:
            raise RuntimeError("observed artifact registration requires its active persistence guard")
        self._register_artifact_in_transaction(
            connection,
            dataset_id=dataset_id,
            attempt_id=attempt_id,
            kind=kind,
            record=record,
            update_attempt_pointer=update_attempt_pointer,
        )
        return self._append_observation_in_transaction(
            connection,
            attempt_id=attempt_id,
            normalized_exchange=normalized_exchange,
            deduplicate_exact_replay=True,
        )

    def complete_proposal(
        self,
        attempt_id: str,
        *,
        owner: str,
        proposal: CosmosProposal,
        validation_warnings: Sequence[str] = (),
    ) -> dict[str, Any] | None:
        warnings_json = canonical_json(list(validation_warnings))
        response_json = canonical_json(proposal.model_response)
        transitions = tuple(proposal.snapped_transition_frames)
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            attempt = connection.execute(
                """
                SELECT attempt.*, job.dataset_id, job.state AS job_state,
                    job.cancel_requested AS job_cancel_requested, job.owner AS job_owner,
                    job.lease_expires_at AS job_lease_expires_at
                FROM cosmos_attempts AS attempt JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE attempt.id=?
                """,
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise WorkerStateError("attempt not found")
            existing = connection.execute(
                "SELECT * FROM cosmos_proposals WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if existing is not None:
                expected = (
                    response_json,
                    *transitions,
                    warnings_json,
                )
                observed = (
                    existing["model_response_json"],
                    *(existing[f"step_{step}_start_frame"] for step in range(2, 8)),
                    existing["validation_warnings_json"],
                )
                if observed != expected or attempt["state"] != AttemptState.SUCCEEDED.value:
                    raise WorkerStateError("persisted proposal does not match recovered proposal")
                return dict(existing)
            if (
                attempt["job_owner"] != owner
                or not _lease_is_live(attempt["job_lease_expires_at"], now_dt)
                or attempt["state"] != AttemptState.REQUESTING.value
                or attempt["lease_owner"] != owner
                or not _lease_is_live(attempt["lease_expires_at"], now_dt)
            ):
                raise LiveLeaseConflict("attempt lease ownership was lost")
            if attempt["job_state"] == JobState.CANCEL_REQUESTED.value or bool(attempt["job_cancel_requested"]):
                return None
            if attempt["job_state"] != JobState.RUNNING.value:
                raise LiveLeaseConflict("job is no longer running")
            connection.execute(
                """
                UPDATE cosmos_proposals SET state='superseded'
                WHERE state='active' AND attempt_id IN (
                    SELECT prior_attempt.id
                    FROM cosmos_attempts AS prior_attempt
                    JOIN cosmos_jobs AS prior_job ON prior_job.id=prior_attempt.job_id
                    WHERE prior_job.dataset_id=? AND prior_attempt.source_episode_index=?
                )
                """,
                (attempt["dataset_id"], attempt["source_episode_index"]),
            )
            proposal_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO cosmos_proposals(
                    id, attempt_id, model_response_json,
                    step_2_start_frame, step_3_start_frame, step_4_start_frame,
                    step_5_start_frame, step_6_start_frame, step_7_start_frame,
                    validation_warnings_json, state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (proposal_id, attempt_id, response_json, *transitions, warnings_json, now),
            )
            connection.execute(
                """
                UPDATE cosmos_attempts SET state='succeeded', lease_owner=NULL,
                    lease_expires_at=NULL, error_class=NULL, error_summary=NULL, updated_at=?
                WHERE id=?
                """,
                (now, attempt_id),
            )
            self.database._append_audit_event(
                connection,
                dataset_id=attempt["dataset_id"],
                actor=owner,
                operation="cosmos_proposal_activated",
                job_id=attempt["job_id"],
                details={
                    "attempt_id": attempt_id,
                    "proposal_id": proposal_id,
                    "source_episode_index": attempt["source_episode_index"],
                },
            )
            return dict(connection.execute("SELECT * FROM cosmos_proposals WHERE id=?", (proposal_id,)).fetchone())

    def list_attempts(self, job_id: str) -> list[dict[str, Any]]:
        with self.database._read() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM cosmos_attempts WHERE job_id=? ORDER BY source_episode_index", (job_id,)
                )
            ]

    def list_artifact_paths(self) -> list[str]:
        with self.database._read() as connection:
            return [row["relative_path"] for row in connection.execute("SELECT relative_path FROM artifacts")]

    def fail_job(self, job_id: str, *, owner: str, summary: str) -> dict[str, Any]:
        bounded = str(summary).encode("utf-8", "backslashreplace").decode("utf-8")[:512]
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if (
                job is None
                or job["owner"] != owner
                or job["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}
                or not _lease_is_live(job["lease_expires_at"], now_dt)
            ):
                raise LiveLeaseConflict("cannot fail a job without its live lease")
            active_attempts = connection.execute(
                """
                SELECT * FROM cosmos_attempts
                WHERE job_id=? AND state IN ('leased', 'requesting')
                """,
                (job_id,),
            ).fetchall()
            if any(
                attempt["lease_owner"] != owner or not _lease_is_live(attempt["lease_expires_at"], now_dt)
                for attempt in active_attempts
            ):
                raise LiveLeaseConflict("cannot fail a job with an unowned or expired attempt lease")
            connection.execute(
                """
                UPDATE cosmos_attempts SET state='retryable', lease_owner=NULL,
                    lease_expires_at=NULL, error_class='WorkerFailure', error_summary=?, updated_at=?
                WHERE job_id=? AND state IN ('queued', 'leased', 'requesting')
                """,
                (bounded, now, job_id),
            )
            connection.execute(
                """
                UPDATE cosmos_jobs SET state='failed', owner=NULL, lease_expires_at=NULL,
                    updated_at=? WHERE id=?
                """,
                (now, job_id),
            )
            self.database._append_audit_event(
                connection,
                dataset_id=job["dataset_id"],
                actor=owner,
                operation="batch_failed",
                job_id=job_id,
                details={"summary": bounded},
            )
            return self._status_in_transaction(connection, job_id)

    def resolve_contact_sheet_conflict(self, job_id: str, *, owner: str) -> dict[str, Any]:
        """Atomically give an already-requested cancellation precedence over failure."""

        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job["owner"] != owner or not _lease_is_live(job["lease_expires_at"], now_dt):
                raise LiveLeaseConflict("cannot resolve a contact-sheet conflict without its live lease")
            if job["state"] == JobState.CANCEL_REQUESTED.value or bool(job["cancel_requested"]):
                connection.execute(
                    """
                    UPDATE cosmos_attempts SET state='cancelled', lease_owner=NULL,
                        lease_expires_at=NULL, updated_at=?
                    WHERE job_id=? AND state IN ('queued', 'retryable', 'leased', 'requesting')
                    """,
                    (now, job_id),
                )
                connection.execute(
                    """
                    UPDATE cosmos_jobs SET state='cancelled', owner=NULL, lease_expires_at=NULL,
                        cancel_requested=1, updated_at=? WHERE id=?
                    """,
                    (now, job_id),
                )
                self.database._append_audit_event(
                    connection,
                    dataset_id=job["dataset_id"],
                    actor=owner,
                    operation="batch_cancelled",
                    job_id=job_id,
                    details={"reason": "contact_sheet_conflict_after_cancel_requested"},
                )
                return self._status_in_transaction(connection, job_id)
            if job["state"] != JobState.RUNNING.value:
                raise LiveLeaseConflict("contact-sheet conflict job is no longer running")
            active_attempts = connection.execute(
                """
                SELECT * FROM cosmos_attempts
                WHERE job_id=? AND state IN ('leased', 'requesting')
                """,
                (job_id,),
            ).fetchall()
            if any(
                attempt["lease_owner"] != owner or not _lease_is_live(attempt["lease_expires_at"], now_dt)
                for attempt in active_attempts
            ):
                raise LiveLeaseConflict("cannot fail a job with an unowned or expired attempt lease")
            summary = "ContactSheetReconciliationConflict"
            connection.execute(
                """
                UPDATE cosmos_attempts SET state='retryable', lease_owner=NULL,
                    lease_expires_at=NULL, error_class='WorkerFailure', error_summary=?, updated_at=?
                WHERE job_id=? AND state IN ('queued', 'leased', 'requesting')
                """,
                (summary, now, job_id),
            )
            connection.execute(
                """
                UPDATE cosmos_jobs SET state='failed', owner=NULL, lease_expires_at=NULL,
                    updated_at=? WHERE id=?
                """,
                (now, job_id),
            )
            self.database._append_audit_event(
                connection,
                dataset_id=job["dataset_id"],
                actor=owner,
                operation="batch_failed",
                job_id=job_id,
                details={"summary": summary},
            )
            return self._status_in_transaction(connection, job_id)

    def finalize_if_done(self, job_id: str, *, owner: str) -> dict[str, Any] | None:
        with self.database._write() as connection:
            now_dt = self._now()
            now = _timestamp(now_dt)
            job = connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()
            if (
                job is None
                or job["owner"] != owner
                or job["state"] not in {JobState.RUNNING.value, JobState.CANCEL_REQUESTED.value}
                or not _lease_is_live(job["lease_expires_at"], now_dt)
            ):
                raise LiveLeaseConflict("job ownership was lost")
            if job["state"] == "cancel_requested":
                return None
            counts = Counter(
                row["state"]
                for row in connection.execute("SELECT state FROM cosmos_attempts WHERE job_id=?", (job_id,))
            )
            if any(counts[state] for state in ("queued", "retryable", "leased", "requesting")):
                return None
            target = "completed_with_failures" if counts["manual_only"] else "completed"
            connection.execute(
                """
                UPDATE cosmos_jobs SET state=?, succeeded_attempts=?, manual_only_attempts=?,
                    owner=NULL, lease_expires_at=NULL, updated_at=? WHERE id=?
                """,
                (target, counts["succeeded"], counts["manual_only"], now, job_id),
            )
            return self._status_in_transaction(connection, job_id)


class BatchService:
    """Authenticated API service that validates and queues immutable jobs."""

    def __init__(
        self,
        *,
        database: CurationDatabase,
        source_registry: SourceRegistry,
        workspace: Path,
        cosmos_base_url: str,
        cosmos_model: str,
        cosmos_api_key_env: str,
        cosmos_endpoint_identity: str,
        capability_client: httpx.Client | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.database = database
        self.source_registry = source_registry
        self.workspace = Path(workspace)
        self.cosmos_base_url = _validate_base_url(cosmos_base_url)
        if not cosmos_model or not cosmos_api_key_env or not cosmos_endpoint_identity:
            raise ValueError("Cosmos model, key environment name, and endpoint identity are required")
        self.cosmos_model = cosmos_model
        self.cosmos_api_key_env = cosmos_api_key_env
        self.cosmos_endpoint_identity = cosmos_endpoint_identity
        self.capability_client = capability_client or httpx.Client(trust_env=False, follow_redirects=False)
        trusted_authority = TrustedWorkerAuthority.from_components(
            workspace=self.workspace,
            dataset_aliases={alias: record.root for alias, record in source_registry.records.items()},
            cosmos_base_url=self.cosmos_base_url,
            cosmos_model=self.cosmos_model,
            cosmos_api_key_env=self.cosmos_api_key_env,
            cosmos_endpoint_identity=self.cosmos_endpoint_identity,
        )
        self.repository = BatchRepository(database, clock=clock, trusted_authority=trusted_authority)

    def start(self, dataset_alias: str, episode_indices: Sequence[int] | None = None) -> dict[str, Any]:
        dataset, selected, configuration = self._freeze(dataset_alias, episode_indices)
        active = self.repository.find_active_job(dataset["id"])
        if active is not None:
            raise BatchError({"error": "active_batch_exists", "job_id": active["id"]}, status_code=409)
        self._validate_capability()
        return self.repository.create_job_with_attempts(
            dataset_id=dataset["id"], configuration=configuration, episode_indices=selected
        )

    def status(self, job_id: str) -> dict[str, Any]:
        return self.repository.status(job_id)

    def cancel(self, job_id: str) -> tuple[int, dict[str, Any]]:
        return self.repository.cancel(job_id)

    def retry(
        self,
        job_id: str,
        *,
        episode_indices: Sequence[int] | None = None,
        failure_states: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        parent = self.repository.get_job_row(job_id)
        if parent is None:
            raise BatchError({"error": "batch_not_found", "job_id": job_id}, status_code=404)
        if parent["state"] not in _TERMINAL_JOBS:
            raise BatchError(
                {"error": "batch_not_terminal", "job_id": job_id, "state": parent["state"]}, status_code=409
            )
        parent_config = json.loads(parent["configuration_json"])
        attempts = self.repository.list_attempts(job_id)
        parent_targets = {row["source_episode_index"] for row in attempts}
        if (episode_indices is None) == (failure_states is None):
            raise BatchError({"error": "invalid_retry_selection"}, status_code=422)
        if episode_indices is not None:
            selected = _episode_selection(episode_indices, parent_targets)
        else:
            allowed = set(failure_states or ())
            if not allowed or not allowed <= {"manual_only", "retryable", "cancelled"}:
                raise BatchError({"error": "invalid_retry_selection"}, status_code=422)
            selected = sorted({row["source_episode_index"] for row in attempts if row["state"] in allowed})
            if not selected:
                raise BatchError({"error": "empty_retry_selection"}, status_code=422)
        configuration = json.loads(canonical_json(parent_config))
        configuration["episode_indices"] = selected
        configuration["parent_job_id"] = job_id
        max_numbers: dict[int, int] = {}
        with self.database._read() as connection:
            for episode_index in selected:
                row = connection.execute(
                    """
                    SELECT max(attempt.attempt_number) AS maximum
                    FROM cosmos_attempts AS attempt
                    JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                    WHERE job.dataset_id=? AND attempt.source_episode_index=?
                    """,
                    (parent["dataset_id"], episode_index),
                ).fetchone()
                max_numbers[episode_index] = int(row["maximum"] or 0) + 1
        return self.repository.create_job_with_attempts(
            dataset_id=parent["dataset_id"],
            configuration=configuration,
            episode_indices=selected,
            parent_job_id=job_id,
            attempt_numbers=max_numbers,
        )

    def _freeze(
        self, dataset_alias: str, episode_indices: Sequence[int] | None
    ) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
        record = self.source_registry.records.get(dataset_alias)
        dataset = self.database.get_dataset(alias=dataset_alias)
        if record is None or dataset is None:
            raise BatchError({"error": "dataset_alias_not_found", "dataset_alias": dataset_alias}, status_code=404)
        if (
            dataset["source_path"] != str(record.root)
            or dataset["source_manifest_sha256"] != record.fingerprint
            or not record.verify_current_inventory()
        ):
            raise BatchError(
                {"error": "source_fingerprint_mismatch", "dataset_alias": dataset_alias}, status_code=409
            )
        episodes = self.database.list_episodes(dataset_id=dataset["id"])
        available = {row["source_episode_index"] for row in episodes}
        selected = sorted(available) if episode_indices is None else _episode_selection(episode_indices, available)
        if not selected:
            raise BatchError({"error": "empty_episode_selection"}, status_code=422)
        info = _read_source_info(record)
        fps = info.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or fps <= 0:
            raise BatchError({"error": "invalid_source_fps"}, status_code=422)
        configuration = {
            "schema_version": 1,
            "dataset_alias": dataset_alias,
            "dataset_id": dataset["id"],
            "source_path": str(record.root),
            "source_manifest_sha256": record.fingerprint,
            "source_fps": float(fps),
            "episode_indices": selected,
            "prompt": {"version": PROMPT_TEMPLATE_VERSION, "sha256": PROMPT_TEMPLATE_SHA256},
            "cosmos": {
                "base_url": self.cosmos_base_url,
                "model": self.cosmos_model,
                "api_key_env": self.cosmos_api_key_env,
                "endpoint_identity": self.cosmos_endpoint_identity,
            },
            "sampling": {"target_fps": TARGET_SAMPLING_FPS},
            "worker": {
                "concurrency": 1,
                "lease_seconds": LEASE_SECONDS,
                "heartbeat_seconds": HEARTBEAT_SECONDS,
            },
            "transport": {"timeout_seconds": 120, "initial_attempts": 2, "repair_attempts": 1},
            "limits": {
                "maximum_duration_seconds": MAX_DURATION_SECONDS,
                "maximum_sampled_frames": MAX_SAMPLED_FRAMES,
                "maximum_payload_bytes": MAX_PAYLOAD_BYTES,
            },
        }
        return dataset, selected, configuration

    def _validate_capability(self) -> None:
        api_key = os.environ.get(self.cosmos_api_key_env)
        if not api_key:
            raise BatchError(
                {"error": "cosmos_api_key_unavailable", "environment": self.cosmos_api_key_env}, status_code=422
            )
        endpoint = self.cosmos_base_url.rstrip("/") + "/models"
        try:
            response = self.capability_client.get(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120.0,
                follow_redirects=False,
            )
            response.raise_for_status()
            document = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            raise BatchError(
                {"error": "cosmos_endpoint_unavailable", "endpoint_identity": self.cosmos_endpoint_identity},
                status_code=422,
            ) from None
        models = document.get("data") if isinstance(document, dict) else None
        identifiers = (
            {item.get("id") for item in models if isinstance(item, dict)} if isinstance(models, list) else set()
        )
        if self.cosmos_model not in identifiers:
            raise BatchError({"error": "cosmos_model_unavailable", "model": self.cosmos_model}, status_code=422)


class CosmosAttemptProcessor:
    """Crash-safe phase orchestrator for one already-leased attempt."""

    def __init__(
        self,
        *,
        repository: BatchRepository,
        source_record: SourceRecord,
        workspace: Path,
        source_fps: float,
        base_url: str,
        model: str,
        api_key: str,
        sampler: Callable[..., SamplingOutcome] = prepare_episode_samples,
        transport_factory: Callable[[], Any] | None = None,
        artifact_store: AtomicArtifactStore | None = None,
        sleep: Callable[[float], None] = time.sleep,
        event_hook: Callable[[str, str], None] | None = None,
        stop_requested: threading.Event | None = None,
        contact_sheet_coordinator: Any | None = None,
    ) -> None:
        self.repository = repository
        self.source_record = source_record
        self.workspace = Path(workspace)
        self.source_fps = float(source_fps)
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.sampler = sampler
        self.transport_factory = transport_factory or (
            lambda: CosmosTransport(base_url=base_url, model=model, api_key=api_key)
        )
        self.artifact_store = artifact_store or AtomicArtifactStore(self.workspace)
        self.sleep = sleep
        self.event_hook = event_hook or (lambda event, attempt_id: None)
        self.stop_requested = stop_requested or threading.Event()
        if contact_sheet_coordinator is None:
            contact_dataset = repository.database.get_dataset(alias=source_record.alias)
            if contact_dataset is None:
                raise WorkerStateError("contact-sheet dataset authority is unavailable")
            contact_sheet_coordinator = ContactSheetCoordinator(
                database=repository.database,
                dataset_id=contact_dataset["id"],
                source_record=source_record,
                workspace=self.workspace,
            )
        self.contact_sheet_coordinator = contact_sheet_coordinator

    def __call__(self, attempt: dict[str, Any], owner: str) -> None:
        attempt_id = attempt["id"]
        episode_index = attempt["source_episode_index"]
        current = self.repository.get_attempt(attempt_id)
        if current is None:
            raise WorkerStateError("attempt not found")
        if current["state"] == AttemptState.SUCCEEDED.value:
            try:
                statuses = self.contact_sheet_coordinator.reconcile_all()
                _raise_for_contact_sheet_conflicts(statuses)
            except ContactSheetUnavailable:
                pass
            except (ArtifactConflict, ArtifactSecurityError) as error:
                raise ContactSheetReconciliationConflict from error
            return
        if self.stop_requested.is_set():
            return
        if self.repository.cancel_is_requested(current["job_id"]):
            return
        parquet_path, video_path = _episode_asset_paths(self.source_record, episode_index)
        if parquet_path is None or video_path is None:
            self.repository.finish_attempt_manual_only(attempt_id, owner=owner, reason="alignment_unproven")
            return
        configuration = json.loads(current["configuration_json"])
        limits = configuration.get("limits", {})
        sampling = self.sampler(
            source_record=self.source_record,
            parquet_asset_path=parquet_path,
            video_asset_path=video_path,
            source_fps=self.source_fps,
            limits=SamplingLimits(
                max_duration_seconds=limits.get("maximum_duration_seconds", MAX_DURATION_SECONDS),
                max_sampled_frames=limits.get("maximum_sampled_frames", MAX_SAMPLED_FRAMES),
                max_payload_bytes=limits.get("maximum_payload_bytes", MAX_PAYLOAD_BYTES),
            ),
        )
        if self.stop_requested.is_set():
            return
        if sampling.status == "manual_only" or sampling.sample is None:
            self.repository.finish_attempt_manual_only(
                attempt_id, owner=owner, reason=sampling.reason or "alignment_unproven"
            )
            return
        sample = sampling.sample
        prepared = prepare_initial_request(model=self.model, prompt=build_canonical_prompt(), sample=sample)
        request_document = build_request_artifact(
            attempt_id=attempt_id,
            source_episode_index=episode_index,
            prepared_request=prepared,
        )
        self._write_request(attempt_id, current["dataset_id"], request_document, owner=owner)
        self.event_hook("request_artifact", attempt_id)

        history = json.loads(self.repository.get_attempt(attempt_id)["http_exchange_history_json"])
        initial_raw = self._read_or_adopt_text(
            attempt_id=attempt_id,
            dataset_id=current["dataset_id"],
            kind=ARTIFACT_KIND_COSMOS_RESPONSE,
            filename="response.txt",
            update_pointer=True,
            owner=owner,
        )
        initial_forced_invalid = initial_raw is not None and not _phase_finished_with_stop(history, "initial")
        transport: Any | None = None
        if initial_raw is None:
            initial_count = sum(item.get("phase") == "initial" for item in history)
            if initial_count and not _exchange_retryable(history[-1]):
                self.repository.finish_attempt_manual_only(attempt_id, owner=owner, reason="transport_failure")
                return
            while initial_count < 2:
                if self.stop_requested.is_set() or self.repository.cancel_is_requested(current["job_id"]):
                    return
                transport = transport or self.transport_factory()
                observation = transport.observe_initial(prepared)
                if _worker_stop_reason(self.stop_requested) in {"lease_lost", "heartbeat_failure"}:
                    return
                self._persist_observation(
                    attempt_id=attempt_id,
                    dataset_id=current["dataset_id"],
                    observation=observation,
                    owner=owner,
                )
                if self.stop_requested.is_set() or self.repository.cancel_is_requested(current["job_id"]):
                    return
                initial_count += 1
                initial_raw = observation.content or observation.observed_content
                if initial_raw is not None:
                    initial_forced_invalid = observation.content is None
                    break
                if not observation.retryable:
                    self.repository.finish_attempt_manual_only(
                        attempt_id, owner=owner, reason=observation.reason or "transport_failure"
                    )
                    return
                if initial_count < 2:
                    if self.stop_requested.is_set():
                        return
                    self.sleep(1.0)
                    if self.stop_requested.is_set():
                        return
            if initial_raw is None:
                self.repository.finish_attempt_manual_only(attempt_id, owner=owner, reason="transport_failure")
                return

        raw_for_proposal = initial_raw
        try:
            if initial_forced_invalid:
                raise CosmosContractError("choices[0].finish_reason must be exactly stop")
            proposal = build_cosmos_proposal(
                initial_raw,
                duration_s=sample.duration_s,
                parquet_timestamps=sample.all_parquet_timestamps,
            )
        except CosmosContractError as initial_error:
            if self.stop_requested.is_set():
                return
            invalid_errors = tuple(initial_error.errors)
            if len(initial_raw.encode("utf-8")) > REPAIR_INVALID_RESPONSE_MAX_BYTES:
                self.repository.finish_attempt_manual_only(
                    attempt_id, owner=owner, reason="repair_input_too_large"
                )
                return
            history = json.loads(self.repository.get_attempt(attempt_id)["http_exchange_history_json"])
            repair_raw = self._read_or_adopt_text(
                attempt_id=attempt_id,
                dataset_id=current["dataset_id"],
                kind=ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE,
                filename="repair-response.txt",
                update_pointer=False,
                owner=owner,
            )
            if repair_raw is not None and not _phase_finished_with_stop(history, "repair"):
                self.repository.finish_attempt_manual_only(attempt_id, owner=owner, reason="repair_transport")
                return
            if repair_raw is None:
                if any(item.get("phase") == "repair" for item in history):
                    self.repository.finish_attempt_manual_only(attempt_id, owner=owner, reason="repair_transport")
                    return
                transport = transport or self.transport_factory()
                if self.stop_requested.is_set() or self.repository.cancel_is_requested(current["job_id"]):
                    return
                repair = transport.observe_repair(
                    invalid_response=initial_raw,
                    validation_errors=invalid_errors,
                )
                if _worker_stop_reason(self.stop_requested) in {"lease_lost", "heartbeat_failure"}:
                    return
                self._persist_observation(
                    attempt_id=attempt_id,
                    dataset_id=current["dataset_id"],
                    observation=repair,
                    owner=owner,
                )
                if self.stop_requested.is_set() or self.repository.cancel_is_requested(current["job_id"]):
                    return
                repair_raw = repair.content or repair.observed_content
                if repair.content is None:
                    self.repository.finish_attempt_manual_only(attempt_id, owner=owner, reason="repair_transport")
                    return
            try:
                proposal = build_cosmos_proposal(
                    repair_raw,
                    duration_s=sample.duration_s,
                    parquet_timestamps=sample.all_parquet_timestamps,
                )
            except CosmosContractError:
                self.repository.finish_attempt_manual_only(attempt_id, owner=owner, reason="repair_invalid")
                return
            raw_for_proposal = repair_raw

        if self.stop_requested.is_set():
            return
        parsed = build_parsed_artifact(
            raw_response=raw_for_proposal,
            duration_s=sample.duration_s,
            parquet_timestamps=sample.all_parquet_timestamps,
            validation_warnings=(),
        )
        parsed_path = f"artifacts/cosmos/{attempt_id}/parsed.json"
        parsed_guard = self.repository.persistence_guard(attempt_id, owner=owner)
        self.artifact_store.write_json(
            parsed_path,
            parsed,
            register=lambda record: self.repository.register_artifact(
                dataset_id=current["dataset_id"],
                attempt_id=attempt_id,
                kind=ARTIFACT_KIND_COSMOS_PARSED,
                record=record,
                owner=owner,
                guard=parsed_guard,
            ),
            authorization_guard=parsed_guard,
        )
        self.event_hook("parsed_artifact", attempt_id)
        if self.stop_requested.is_set():
            return
        committed = self.repository.complete_proposal(attempt_id, owner=owner, proposal=proposal)
        if committed is not None:
            self.event_hook("proposal_commit", attempt_id)
            try:
                self.contact_sheet_coordinator.ensure_proposal(committed["id"])
            except ContactSheetUnavailable:
                self.event_hook("proposal_contact_sheet_pending", attempt_id)
            except (ArtifactConflict, ArtifactSecurityError) as error:
                raise ContactSheetReconciliationConflict from error

    def _write_request(self, attempt_id: str, dataset_id: int, document: Mapping[str, Any], *, owner: str) -> None:
        relative = f"artifacts/cosmos/{attempt_id}/request.json"
        guard = self.repository.persistence_guard(attempt_id, owner=owner)
        self.artifact_store.write_json(
            relative,
            document,
            register=lambda record: self.repository.register_artifact(
                dataset_id=dataset_id,
                attempt_id=attempt_id,
                kind=ARTIFACT_KIND_COSMOS_REQUEST,
                record=record,
                owner=owner,
                update_attempt_pointer=True,
                guard=guard,
            ),
            authorization_guard=guard,
        )

    def _persist_observation(
        self,
        *,
        attempt_id: str,
        dataset_id: int,
        observation: CosmosCallObservation,
        owner: str,
    ) -> None:
        visible_content = observation.content or observation.observed_content
        if visible_content is not None:
            if observation.phase == "initial":
                kind = ARTIFACT_KIND_COSMOS_RESPONSE
                filename = "response.txt"
                update_pointer = True
            else:
                kind = ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE
                filename = "repair-response.txt"
                update_pointer = False
            relative = f"artifacts/cosmos/{attempt_id}/{filename}"
            guard = self.repository.persistence_guard(attempt_id, owner=owner)
            self.artifact_store.write_text(
                relative,
                visible_content,
                register=lambda record: self.repository.register_observed_artifact(
                    dataset_id=dataset_id,
                    attempt_id=attempt_id,
                    kind=kind,
                    record=record,
                    observation=observation,
                    owner=owner,
                    update_attempt_pointer=update_pointer,
                    guard=guard,
                ),
                authorization_guard=guard,
            )
            self.event_hook("response_artifact", attempt_id)
        else:
            self.repository.append_observation(attempt_id, observation, owner=owner)
        self.event_hook("observation_persisted", attempt_id)

    def _read_or_adopt_text(
        self,
        *,
        attempt_id: str,
        dataset_id: int,
        kind: str,
        filename: str,
        update_pointer: bool,
        owner: str,
    ) -> str | None:
        relative = f"artifacts/cosmos/{attempt_id}/{filename}"
        artifact = self.repository.get_attempt_artifact(attempt_id, kind)
        try:
            if artifact is None:
                guard = self.repository.persistence_guard(attempt_id, owner=owner)
                adopted = self.artifact_store.adopt_existing(
                    relative,
                    media_type="text/plain; charset=utf-8",
                    register=lambda record: self.repository.register_artifact(
                        dataset_id=dataset_id,
                        attempt_id=attempt_id,
                        kind=kind,
                        record=record,
                        owner=owner,
                        update_attempt_pointer=update_pointer,
                        guard=guard,
                    ),
                    authorization_guard=guard,
                )
                record = adopted.record
            else:
                record = ArtifactRecord(
                    relative_path=artifact["relative_path"],
                    media_type=artifact["media_type"],
                    byte_size=artifact["byte_size"],
                    sha256=artifact["sha256"],
                )
            recovered = self.artifact_store.read_existing(
                record.relative_path,
                media_type=record.media_type,
                expected_sha256=record.sha256,
                expected_byte_size=record.byte_size,
            )
        except (FileNotFoundError, ArtifactConflict) as error:
            if isinstance(error, ArtifactConflict) and str(error) != "artifact does not exist":
                raise
            return None
        try:
            return recovered.contents.decode("utf-8")
        except UnicodeDecodeError:
            raise WorkerStateError("persisted Cosmos response is not UTF-8") from None


def _episode_selection(values: Sequence[int], available: Iterable[int]) -> list[int]:
    allowed = set(available)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise BatchError({"error": "invalid_episode_selection"}, status_code=422)
    if any(type(value) is not int or value < 0 for value in values):
        raise BatchError({"error": "invalid_episode_selection"}, status_code=422)
    selected = sorted(set(values))
    if not selected or not set(selected) <= allowed:
        raise BatchError({"error": "invalid_episode_selection"}, status_code=422)
    return selected


def _episode_asset_paths(record: SourceRecord, episode_index: int) -> tuple[str | None, str | None]:
    name = f"episode_{episode_index:06d}"
    parquet = sorted(
        path for path in record.file_hashes if path.startswith("data/") and path.endswith(f"/{name}.parquet")
    )
    ego_videos = sorted(
        path
        for path in record.file_hashes
        if path.startswith("videos/") and "/observation.images.ego_view/" in path and path.endswith(f"/{name}.mp4")
    )
    return (
        parquet[0] if len(parquet) == 1 else None,
        ego_videos[0] if len(ego_videos) == 1 else None,
    )


def _phase_finished_with_stop(history: Sequence[Mapping[str, Any]], phase: str) -> bool:
    matching = [exchange for exchange in history if exchange.get("phase") == phase]
    if not matching:
        return False
    response = matching[-1].get("response")
    return isinstance(response, Mapping) and response.get("finish_reason") == "stop"


def _exchange_retryable(exchange: Mapping[str, Any]) -> bool:
    response = exchange.get("response")
    if isinstance(response, Mapping):
        status = response.get("status_code")
        return type(status) is int and (status in {408, 429} or 500 <= status <= 599)
    error = exchange.get("error")
    if not isinstance(error, Mapping):
        return False
    error_class = error.get("class")
    return error_class in {
        "ConnectError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
        "TransportError",
    }


def _read_source_info(record: Any) -> dict[str, Any]:
    asset = record.open_asset("meta/info.json")
    if asset is None:
        raise BatchError({"error": "invalid_source_metadata"}, status_code=422)
    try:
        chunks: list[bytes] = []
        offset = 0
        while chunk := os.pread(asset.fd, 1024 * 1024, offset):
            chunks.append(chunk)
            offset += len(chunk)
        document = json.loads(b"".join(chunks))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BatchError({"error": "invalid_source_metadata"}, status_code=422) from None
    finally:
        asset.close()
    if not isinstance(document, dict):
        raise BatchError({"error": "invalid_source_metadata"}, status_code=422)
    return document


def _validate_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value)
            or parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "@" in parsed.netloc
            or parsed.query
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65_535)
        ):
            raise ValueError
        httpx.URL(value)
    except (TypeError, ValueError, httpx.InvalidURL):
        raise ValueError("Cosmos base URL must be an absolute HTTP(S) URL") from None
    return value.rstrip("/")


def _lease_is_live(value: str | None, now: datetime) -> bool:
    expiry = _parse_timestamp(value)
    return expiry is not None and expiry > now


def _require_owner_uuid(value: str) -> None:
    try:
        UUID(value)
    except (TypeError, ValueError, AttributeError):
        raise ValueError("worker owner must be a UUID") from None


ProcessAttempt = Callable[[dict[str, Any], str], None]


class WorkerStopState:
    """Thread-safe stop event that preserves why work must stop."""

    _PRIORITY = {"signal": 1, "heartbeat_failure": 2, "lease_lost": 3}

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def is_set(self) -> bool:
        return self._event.is_set()

    def request(self, reason: str) -> None:
        if reason not in self._PRIORITY:
            raise ValueError("worker stop reason is invalid")
        with self._lock:
            if self._reason is None or self._PRIORITY[reason] > self._PRIORITY[self._reason]:
                self._reason = reason
            self._event.set()


def _request_worker_stop(stop_requested: threading.Event | WorkerStopState, reason: str) -> None:
    if isinstance(stop_requested, WorkerStopState):
        stop_requested.request(reason)
    else:
        stop_requested.set()


def _worker_stop_reason(stop_requested: threading.Event | WorkerStopState) -> str | None:
    if isinstance(stop_requested, WorkerStopState):
        return stop_requested.reason
    return "signal" if stop_requested.is_set() else None


class JobLeaseHeartbeat:
    """Renew one acquired job lease from preflight through worker shutdown."""

    def __init__(
        self,
        *,
        repository: BatchRepository,
        job_id: str,
        owner: str,
        stop_requested: threading.Event | WorkerStopState,
        interval_seconds: float | None = None,
        max_retryable_failures: int = 3,
        status_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.repository = repository
        self.job_id = job_id
        self.owner = owner
        self.stop_requested = stop_requested
        self.interval_seconds = float(HEARTBEAT_SECONDS) if interval_seconds is None else float(interval_seconds)
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("heartbeat interval must be finite and positive")
        if type(max_retryable_failures) is not int or max_retryable_failures <= 0:
            raise ValueError("heartbeat retryable failure bound must be a positive integer")
        self.max_retryable_failures = max_retryable_failures
        self.status_sink = status_sink or (lambda status: None)
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._active_attempt_id: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def thread(self) -> threading.Thread | None:
        with self._lock:
            return self._thread

    def start(self) -> bool:
        with self._lock:
            if self._started:
                return False
            self._started = True
            thread = threading.Thread(
                target=self._run,
                name=f"curation-heartbeat-{self.job_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def set_active_attempt(self, attempt_id: str | None) -> None:
        with self._lock:
            self._active_attempt_id = attempt_id

    def stop(self) -> None:
        self._shutdown.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.interval_seconds + 1)

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._shutdown.wait(self.interval_seconds):
            with self._lock:
                attempt_id = self._active_attempt_id
            try:
                self.repository.heartbeat(self.job_id, owner=self.owner, attempt_id=attempt_id)
                consecutive_failures = 0
            except RetryableDatabaseError:
                consecutive_failures += 1
                if consecutive_failures >= self.max_retryable_failures:
                    _request_worker_stop(self.stop_requested, "heartbeat_failure")
                emitted = self._emit_status(
                    {
                        "event": "heartbeat_retryable_error",
                        "job_id": self.job_id,
                        "consecutive_failures": consecutive_failures,
                    }
                )
                if not emitted or consecutive_failures >= self.max_retryable_failures:
                    return
            except (LiveLeaseConflict, WorkerStateError):
                try:
                    job = self.repository.get_job_row(self.job_id)
                except RetryableDatabaseError:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_retryable_failures:
                        _request_worker_stop(self.stop_requested, "heartbeat_failure")
                    emitted = self._emit_status(
                        {
                            "event": "heartbeat_retryable_error",
                            "job_id": self.job_id,
                            "consecutive_failures": consecutive_failures,
                        }
                    )
                    if not emitted or consecutive_failures >= self.max_retryable_failures:
                        return
                    continue
                except Exception:
                    self._stop_for_fatal_heartbeat()
                    return
                if job is not None and job["state"] in _TERMINAL_JOBS:
                    return
                _request_worker_stop(self.stop_requested, "lease_lost")
                self._emit_status({"event": "heartbeat_lease_lost", "job_id": self.job_id})
                return
            except Exception:
                self._stop_for_fatal_heartbeat()
                return

    def _stop_for_fatal_heartbeat(self) -> None:
        _request_worker_stop(self.stop_requested, "heartbeat_failure")
        self._emit_status({"event": "heartbeat_fatal_error", "job_id": self.job_id})

    def _emit_status(self, status: Mapping[str, Any]) -> bool:
        try:
            self.status_sink(status)
        except Exception:
            _request_worker_stop(self.stop_requested, "heartbeat_failure")
            return False
        return True


class CurationWorker:
    """Single-concurrency executor over one explicitly named persisted job."""

    def __init__(
        self,
        *,
        repository: BatchRepository,
        process_attempt: ProcessAttempt,
        owner: str | None = None,
        heartbeat: bool = True,
        heartbeat_controller: JobLeaseHeartbeat | None = None,
        status_sink: Callable[[Mapping[str, Any]], None] | None = None,
        stop_requested: threading.Event | WorkerStopState | None = None,
    ) -> None:
        self.repository = repository
        self.process_attempt = process_attempt
        self.owner = owner or str(uuid4())
        self.heartbeat_enabled = heartbeat
        self.heartbeat_controller = heartbeat_controller
        self.status_sink = status_sink or (lambda status: None)
        self.stop_requested = stop_requested or threading.Event()
        self.active_attempt_id: str | None = None

    def execute(
        self,
        command: LiteralCommand,
        job_id: str,
        *,
        acquired_status: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        status = (
            dict(acquired_status)
            if acquired_status is not None
            else (
                self.repository.start_job(job_id, owner=self.owner)
                if command == "run"
                else self.repository.resume_job(job_id, owner=self.owner)
            )
        )
        self.status_sink({"event": "worker_started", "command": command, "job_id": job_id, "owner": self.owner})
        if status["state"] == JobState.CANCEL_REQUESTED.value:
            final = self._finish_cancel_requested(job_id)
            if _worker_stop_reason(self.stop_requested) == "lease_lost":
                return final
            self.status_sink({"event": "worker_terminal", "job_id": job_id, "state": final["state"]})
            return final

        heartbeat_controller = self.heartbeat_controller
        if self.heartbeat_enabled:
            if heartbeat_controller is None:
                heartbeat_controller = JobLeaseHeartbeat(
                    repository=self.repository,
                    job_id=job_id,
                    owner=self.owner,
                    stop_requested=self.stop_requested,
                )
            elif heartbeat_controller.job_id != job_id or heartbeat_controller.owner != self.owner:
                raise ValueError("heartbeat controller does not match worker ownership")
            heartbeat_controller.start()
        try:
            while True:
                if self.stop_requested.is_set():
                    return self.repository.status(job_id)
                current = self.repository.status(job_id)
                if current["state"] == JobState.CANCEL_REQUESTED.value:
                    final = self._finish_cancel_requested(job_id)
                    if _worker_stop_reason(self.stop_requested) == "lease_lost":
                        return final
                    self.status_sink({"event": "worker_terminal", "job_id": job_id, "state": final["state"]})
                    return final
                attempt = self.repository.claim(job_id, owner=self.owner)
                if attempt is None:
                    final = self.repository.finalize_if_done(job_id, owner=self.owner)
                    if final is not None:
                        self.status_sink({"event": "worker_terminal", "job_id": job_id, "state": final["state"]})
                        return final
                    continue
                self.active_attempt_id = attempt["id"]
                if heartbeat_controller is not None:
                    heartbeat_controller.set_active_attempt(attempt["id"])
                attempt = self.repository.mark_requesting(attempt["id"], owner=self.owner)
                self.status_sink(
                    {
                        "event": "attempt_started",
                        "job_id": job_id,
                        "attempt_id": attempt["id"],
                        "source_episode_index": attempt["source_episode_index"],
                    }
                )
                try:
                    self.process_attempt(attempt, self.owner)
                except LiveLeaseConflict:
                    _request_worker_stop(self.stop_requested, "lease_lost")
                    return self.repository.status(job_id)
                finally:
                    self.active_attempt_id = None
                    if heartbeat_controller is not None:
                        heartbeat_controller.set_active_attempt(None)
                self.status_sink(
                    {
                        "event": "attempt_finished",
                        "job_id": job_id,
                        "attempt_id": attempt["id"],
                    }
                )
        finally:
            if heartbeat_controller is not None:
                heartbeat_controller.set_active_attempt(None)
                heartbeat_controller.stop()

    def _finish_cancel_requested(self, job_id: str) -> dict[str, Any]:
        try:
            return self.repository.finish_cancel_requested(job_id, owner=self.owner)
        except LiveLeaseConflict:
            _request_worker_stop(self.stop_requested, "lease_lost")
            return self.repository.status(job_id)


LiteralCommand = Literal["run", "resume"]


def _uuid_argument(value: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError):
        raise argparse.ArgumentTypeError("job ID must be a UUID") from None


def _absolute_workspace(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("workspace must be an absolute path")
    return path.resolve()


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="curation_worker.py")
    parser.add_argument("--workspace", required=True, type=_absolute_workspace)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "resume"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--job-id", required=True, type=_uuid_argument)
    return parser


def _print_status(document: Mapping[str, Any]) -> None:
    print(canonical_json(dict(document)), flush=True)


def _fail_job_for_contact_sheet_conflict(
    repository: BatchRepository,
    *,
    job_id: str,
    owner: str,
    stop_requested: threading.Event | WorkerStopState,
) -> int:
    try:
        final = repository.resolve_contact_sheet_conflict(job_id, owner=owner)
    except LiveLeaseConflict:
        _request_worker_stop(stop_requested, "lease_lost")
        _print_status({"error": "worker_lease_lost", "job_id": job_id})
        return 3
    except RetryableDatabaseError:
        _print_status({"error": "database_busy", "job_id": job_id, "retryable": True})
        return 2
    if final["state"] == JobState.CANCELLED.value:
        _print_status({"event": "worker_terminal", "job_id": job_id, "state": final["state"]})
        return 0
    _print_status(
        {
            "error": "contact_sheet_conflict",
            "job_id": job_id,
            "state": final["state"],
        }
    )
    return 1


def build_attempt_processor(
    *,
    repository: BatchRepository,
    job_id: str,
    workspace: Path,
    stop_requested: threading.Event | WorkerStopState | None = None,
) -> CosmosAttemptProcessor:
    binding = repository.bind_job_authority(job_id)
    if workspace != binding.workspace:
        raise InvalidPersistedConfiguration("processor workspace is not trusted")
    configuration = binding.configuration
    registry = SourceRegistry.from_paths(
        {configuration.dataset_alias: binding.source_path},
        workspace=binding.workspace,
    )
    record = registry.records[configuration.dataset_alias]
    if record.fingerprint != configuration.source_manifest_sha256 or not record.verify_current_inventory():
        raise WorkerStateError("source fingerprint no longer matches the job snapshot")
    api_key = os.environ.get(binding.cosmos_api_key_env)
    if not api_key:
        raise WorkerStateError("configured Cosmos API key environment variable is unavailable")
    artifact_store = AtomicArtifactStore(workspace)
    artifact_store.cleanup_temporary_files(referenced_relative_paths=repository.list_artifact_paths())
    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=record,
        workspace=binding.workspace,
        source_fps=configuration.source_fps,
        base_url=binding.cosmos_base_url,
        model=binding.cosmos_model,
        api_key=api_key,
        artifact_store=artifact_store,
        stop_requested=stop_requested,
    )
    try:
        statuses = processor.contact_sheet_coordinator.reconcile_all()
        _raise_for_contact_sheet_conflicts(statuses)
    except ContactSheetUnavailable:
        pass
    except (ArtifactConflict, ArtifactSecurityError) as error:
        raise ContactSheetReconciliationConflict from error
    return processor


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        settings = CurationSettings.from_env()
        trusted_authority = TrustedWorkerAuthority.from_settings(
            settings,
            cli_workspace=args.workspace,
        )
    except (CurationConfigurationError, InvalidPersistedConfiguration):
        _print_status({"error": "invalid_configuration", "job_id": args.job_id})
        return 2
    database_path = args.workspace / "curation.sqlite3"
    if not database_path.is_file():
        _print_status({"error": "invalid_workspace", "workspace": str(args.workspace)})
        return 2
    database = CurationDatabase(database_path)
    try:
        database.validate_worker_compatibility()
    except (IncompatibleCurationDatabase, sqlite3.DatabaseError, IndexError, KeyError, TypeError):
        _print_status({"error": "invalid_database", "job_id": args.job_id})
        return 2
    repository = BatchRepository(database, trusted_authority=trusted_authority)
    owner = str(uuid4())
    stop_requested = WorkerStopState()
    heartbeat_controller: JobLeaseHeartbeat | None = None

    def request_stop(signum: int, frame: Any) -> None:
        stop_requested.request("signal")

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        try:
            acquired = (
                repository.start_job(args.job_id, owner=owner)
                if args.command == "run"
                else repository.resume_job(args.job_id, owner=owner)
            )
        except LiveLeaseConflict:
            _print_status({"error": "live_lease_conflict", "job_id": args.job_id})
            return 3
        except RetryableDatabaseError:
            _print_status({"error": "database_busy", "job_id": args.job_id, "retryable": True})
            return 2
        except sqlite3.DatabaseError:
            _print_status({"error": "invalid_database", "job_id": args.job_id})
            return 2
        except (IndexError, KeyError, TypeError):
            _print_status({"error": "invalid_database", "job_id": args.job_id})
            return 2
        except InvalidPersistedConfiguration:
            _print_status({"error": "invalid_configuration", "job_id": args.job_id})
            return 2
        except json.JSONDecodeError:
            _print_status({"error": "invalid_configuration", "job_id": args.job_id})
            return 2
        except WorkerStateError:
            _print_status({"error": "invalid_job_state", "job_id": args.job_id})
            return 2

        if acquired["state"] == JobState.CANCEL_REQUESTED.value:
            worker = CurationWorker(
                repository=repository,
                process_attempt=lambda attempt, worker_owner: None,
                owner=owner,
                heartbeat=False,
                status_sink=_print_status,
                stop_requested=stop_requested,
            )
        else:
            heartbeat_controller = JobLeaseHeartbeat(
                repository=repository,
                job_id=args.job_id,
                owner=owner,
                stop_requested=stop_requested,
                status_sink=_print_status,
            )
            heartbeat_controller.start()
            try:
                processor = build_attempt_processor(
                    repository=repository,
                    job_id=args.job_id,
                    workspace=args.workspace,
                    stop_requested=stop_requested,
                )
            except ContactSheetReconciliationConflict:
                heartbeat_controller.stop()
                return _fail_job_for_contact_sheet_conflict(
                    repository,
                    job_id=args.job_id,
                    owner=owner,
                    stop_requested=stop_requested,
                )
            except RetryableDatabaseError:
                heartbeat_controller.stop()
                try:
                    repository.release_job_after_configuration_error(args.job_id, owner=owner)
                except (LiveLeaseConflict, RetryableDatabaseError):
                    pass
                _print_status({"error": "database_busy", "job_id": args.job_id, "retryable": True})
                return 2
            except (ArtifactSecurityError, OSError, ValueError, WorkerStateError):
                heartbeat_controller.stop()
                try:
                    repository.release_job_after_configuration_error(args.job_id, owner=owner)
                except (LiveLeaseConflict, RetryableDatabaseError):
                    pass
                _print_status({"error": "invalid_configuration", "job_id": args.job_id})
                return 2
            worker = CurationWorker(
                repository=repository,
                process_attempt=processor,
                owner=owner,
                heartbeat=True,
                heartbeat_controller=heartbeat_controller,
                status_sink=_print_status,
                stop_requested=stop_requested,
            )
        try:
            final = worker.execute(args.command, args.job_id, acquired_status=acquired)
        except ContactSheetReconciliationConflict:
            return _fail_job_for_contact_sheet_conflict(
                repository,
                job_id=args.job_id,
                owner=owner,
                stop_requested=stop_requested,
            )
        except LiveLeaseConflict:
            stop_requested.request("lease_lost")
            _print_status({"error": "worker_lease_lost", "job_id": args.job_id})
            return 3
        except RetryableDatabaseError:
            _print_status({"error": "database_busy", "job_id": args.job_id, "retryable": True})
            return 2
        except json.JSONDecodeError:
            _print_status({"error": "invalid_configuration", "job_id": args.job_id})
            return 2
        except Exception as error:
            try:
                final = repository.fail_job(
                    args.job_id,
                    owner=owner,
                    summary=type(error).__name__,
                )
            except Exception:
                final = {"state": "failed"}
            _print_status({"event": "worker_failed", "job_id": args.job_id, "state": final["state"]})
            return 1
    finally:
        if heartbeat_controller is not None:
            heartbeat_controller.stop()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    stop_reason = _worker_stop_reason(stop_requested)
    if stop_reason == "signal":
        _print_status({"event": "worker_interrupted", "job_id": args.job_id, "state": final["state"]})
        return 130
    if stop_reason == "lease_lost":
        _print_status({"error": "worker_lease_lost", "job_id": args.job_id, "state": final["state"]})
        return 3
    if stop_reason == "heartbeat_failure":
        _print_status({"error": "worker_heartbeat_failed", "job_id": args.job_id, "state": final["state"]})
        return 1
    return 1 if final["state"] == JobState.FAILED.value else 0
