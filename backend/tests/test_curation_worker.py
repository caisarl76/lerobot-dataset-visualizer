from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import threading
import time
from uuid import uuid4

import av
from curation.config import WorkerSettings
from curation.contact_sheets import (
    ContactSheetDatasetIdentity,
    proposal_contact_sheet_path,
    receipt_path,
)
from curation.cosmos_transport import (
    ArtifactRecord,
    ArtifactSecurityError,
    AtomicArtifactStore,
    CosmosCallObservation,
    PreparedSample,
    SamplingOutcome,
)
from curation.db import (
    ArtifactReconciliationConflict,
    CurationDatabase,
    IncompatibleCurationDatabase,
    _migration_v1_statements,
)
from curation.review import ReviewService
from curation.router import build_curation_router
from curation.source import SourceRegistry
import curation.worker as worker_module
from curation.worker import (
    BatchRepository,
    BatchService,
    CosmosAttemptProcessor,
    CurationWorker,
    LiveLeaseConflict,
    TrustedWorkerAuthority,
    WorkerStateError,
    build_cli_parser,
    cli_main,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _valid_job_configuration(
    *,
    dataset_id: int,
    source_path: Path,
    episode_indices: list[int],
    source_manifest_sha256: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_alias": "local/pnp_trash",
        "dataset_id": dataset_id,
        "source_path": str(source_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "source_fps": 10.0,
        "episode_indices": episode_indices,
        "prompt": {
            "version": worker_module.PROMPT_TEMPLATE_VERSION,
            "sha256": worker_module.PROMPT_TEMPLATE_SHA256,
        },
        "cosmos": {
            "base_url": "http://cosmos/v1",
            "model": "cosmos3-nano",
            "api_key_env": "TEST_COSMOS_KEY",
            "endpoint_identity": "test-h100",
        },
        "sampling": {"target_fps": worker_module.TARGET_SAMPLING_FPS},
        "worker": {
            "concurrency": 1,
            "lease_seconds": worker_module.LEASE_SECONDS,
            "heartbeat_seconds": worker_module.HEARTBEAT_SECONDS,
        },
        "transport": {"timeout_seconds": 120, "initial_attempts": 2, "repair_attempts": 1},
        "limits": {
            "maximum_duration_seconds": worker_module.MAX_DURATION_SECONDS,
            "maximum_sampled_frames": worker_module.MAX_SAMPLED_FRAMES,
            "maximum_payload_bytes": worker_module.MAX_PAYLOAD_BYTES,
        },
    }


def _invalid_job_configuration(case: str, valid: dict[str, object]) -> dict[str, object]:
    document = json.loads(json.dumps(valid))
    if case == "empty":
        return {}
    if case == "wrong_type":
        document["source_fps"] = "10.0"
    elif case == "unknown_field":
        document["unexpected"] = True
    elif case == "missing_nested_field":
        del document["cosmos"]["model"]
    elif case == "wrong_frozen_limit":
        document["limits"]["maximum_sampled_frames"] = worker_module.MAX_SAMPLED_FRAMES + 1
    else:
        raise AssertionError(f"unknown invalid configuration case: {case}")
    return document


def _trusted_runtime_environment(
    *,
    workspace: Path,
    source: Path,
    cosmos_base_url: str = "http://cosmos/v1",
    cosmos_model: str = "cosmos3-nano",
    cosmos_api_key_env: str = "TEST_COSMOS_KEY",
    cosmos_endpoint_identity: str = "test-h100",
) -> dict[str, str]:
    return {
        "CURATION_DATASET_ALIASES_JSON": json.dumps({"local/pnp_trash": str(source.resolve())}),
        "CURATION_WORKSPACE": str(workspace.resolve()),
        "CURATION_OUTPUT": str((workspace.parent / "curation-output").resolve()),
        "CURATION_BROWSER_ORIGIN": "http://127.0.0.1:3000",
        "CURATION_BEARER_TOKEN": "test-curation-token",
        "COSMOS_BASE_URL": cosmos_base_url,
        "COSMOS_MODEL": cosmos_model,
        "COSMOS_API_KEY_ENV": cosmos_api_key_env,
        "COSMOS_ENDPOINT_IDENTITY": cosmos_endpoint_identity,
        "ISAAC_GROOT_ROOT": str((workspace.parent / "isaac-groot").resolve()),
    }


def _set_trusted_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: object,
) -> None:
    values = _trusted_runtime_environment(**kwargs)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _trusted_worker_authority(
    *,
    workspace: Path,
    source: Path,
    cosmos_base_url: str = "http://cosmos/v1",
    cosmos_endpoint_identity: str = "test-h100",
) -> TrustedWorkerAuthority:
    return TrustedWorkerAuthority.from_components(
        workspace=workspace.resolve(),
        dataset_aliases={"local/pnp_trash": source.resolve()},
        cosmos_base_url=cosmos_base_url,
        cosmos_model="cosmos3-nano",
        cosmos_api_key_env="TEST_COSMOS_KEY",
        cosmos_endpoint_identity=cosmos_endpoint_identity,
    )


def _workspace_tree_bytes(root: Path) -> tuple[tuple[str, bool, bytes | None], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.is_dir(),
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    )


def _workspace_tree_shape(root: Path) -> tuple[tuple[str, bool, int | None], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.is_dir(),
            None if path.is_dir() else path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
    )


def _durable_workspace_bytes(root: Path) -> tuple[tuple[str, bool, bytes | None], ...]:
    return tuple(
        entry
        for entry in _workspace_tree_bytes(root)
        if not entry[0].endswith(("curation.sqlite3-wal", "curation.sqlite3-shm"))
    )


def _create_incompatible_worker_database(path: Path, case: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if case in {"partial_newer", "partial_unknown_v1"}:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE cosmos_jobs(id TEXT PRIMARY KEY, state TEXT)")
            connection.execute(f"PRAGMA user_version={999 if case == 'partial_newer' else 1}")
        return
    database = CurationDatabase(path)
    database.initialize()
    with sqlite3.connect(path) as connection:
        if case == "older":
            connection.execute("PRAGMA user_version=0")
        elif case == "newer":
            connection.execute("PRAGMA user_version=2")
        elif case == "superficially_compatible":
            connection.execute("DROP TRIGGER cosmos_jobs_configuration_immutable")
        else:
            raise AssertionError(f"unknown incompatible database case: {case}")


def _database_file_snapshot(path: Path) -> dict[str, bytes]:
    return {
        candidate.name: candidate.read_bytes()
        for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"))
        if candidate.exists()
    }


@contextmanager
def _live_wal_schema(path: Path, *, delete_invariant_trigger: bool) -> Iterator[sqlite3.Connection]:
    if delete_invariant_trigger:
        CurationDatabase(path).initialize()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("BEGIN IMMEDIATE")
        if delete_invariant_trigger:
            connection.execute("DROP TRIGGER cosmos_jobs_configuration_immutable")
        else:
            for statement in _migration_v1_statements():
                connection.execute(statement)
            connection.execute("PRAGMA user_version=1")
        connection.execute("COMMIT")
        assert Path(str(path) + "-wal").stat().st_size > 0
        yield connection
    finally:
        connection.close()


@contextmanager
def _disabled_trigger(connection: sqlite3.Connection, name: str) -> Iterator[None]:
    row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type='trigger' AND name=?",
        (name,),
    ).fetchone()
    assert row is not None and isinstance(row[0], str)
    trigger_sql = row[0]
    connection.execute(f'DROP TRIGGER "{name}"')
    try:
        yield
    finally:
        connection.execute(trigger_sql)


@pytest.fixture
def lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CurationDatabase, BatchRepository, MutableClock, str]:
    database = CurationDatabase(tmp_path / "workspace" / "curation.sqlite3")
    database.initialize()
    dataset = database.register_dataset(
        alias="local/pnp_trash",
        source_path=str(tmp_path / "source"),
        source_manifest_sha256="a" * 64,
        prompt_template_version=worker_module.PROMPT_TEMPLATE_VERSION,
        prompt_template_sha256=worker_module.PROMPT_TEMPLATE_SHA256,
    )
    for index in range(3):
        database.create_episode(dataset_id=dataset["id"], source_episode_index=index, source_length=20)
    clock = MutableClock(datetime.now(timezone.utc))
    repository = BatchRepository(
        database,
        clock=clock,
        trusted_authority=_trusted_worker_authority(
            workspace=database.path.parent,
            source=tmp_path / "source",
        ),
    )
    job = repository.create_job_with_attempts(
        dataset_id=dataset["id"],
        configuration=_valid_job_configuration(
            dataset_id=dataset["id"],
            source_path=tmp_path / "source",
            episode_indices=[0, 1, 2],
        ),
        episode_indices=[0, 1, 2],
    )
    _set_trusted_runtime_environment(
        monkeypatch,
        workspace=database.path.parent,
        source=tmp_path / "source",
    )
    return database, repository, clock, job["job_id"]


def test_atomic_claim_concurrency_one_and_heartbeat(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    claimed = repository.claim(job_id, owner=owner)
    assert claimed is not None
    assert claimed["source_episode_index"] == 0
    assert repository.claim(job_id, owner=owner) is None
    old_expiry = claimed["lease_expires_at"]
    repository.mark_requesting(claimed["id"], owner=owner)
    clock.advance(15)
    repository.heartbeat(job_id, owner=owner, attempt_id=claimed["id"])
    refreshed = repository.list_attempts(job_id)[0]
    assert refreshed["lease_expires_at"] > old_expiry


def test_mutating_acquisition_fails_closed_without_trusted_runtime_authority(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    database, _, _, job_id = lifecycle
    untrusted_repository = BatchRepository(database)
    before = _job_evidence_snapshot(database, job_id)

    with pytest.raises(worker_module.InvalidPersistedConfiguration):
        untrusted_repository.start_job(job_id, owner=str(uuid4()))

    assert _job_evidence_snapshot(database, job_id) == before


def test_heartbeat_tolerates_attempt_that_committed_terminal_before_active_id_clears(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    claimed = repository.claim(job_id, owner=owner)
    assert claimed is not None
    repository.mark_requesting(claimed["id"], owner=owner)
    repository.finish_attempt_manual_only(claimed["id"], owner=owner, reason="done")

    clock.advance(15)
    repository.heartbeat(job_id, owner=owner, attempt_id=claimed["id"])

    assert repository.status(job_id)["lease"]["owner"] == owner


def test_heartbeat_cannot_revive_expired_job_or_attempt_leases(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    attempt = repository.claim(job_id, owner=owner)
    assert attempt is not None
    repository.mark_requesting(str(attempt["id"]), owner=owner)
    job_expiry = repository.get_job_row(job_id)["lease_expires_at"]
    attempt_expiry = repository.get_attempt(str(attempt["id"]))["lease_expires_at"]
    clock.advance(181)

    with pytest.raises(LiveLeaseConflict):
        repository.heartbeat(job_id, owner=owner, attempt_id=str(attempt["id"]))

    assert repository.get_job_row(job_id)["lease_expires_at"] == job_expiry
    assert repository.get_attempt(str(attempt["id"]))["lease_expires_at"] == attempt_expiry


def test_heartbeat_samples_lease_time_only_after_waiting_for_the_write_lock(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    attempt = repository.claim(job_id, owner=owner)
    assert attempt is not None
    repository.mark_requesting(str(attempt["id"]), owner=owner)
    job_expiry = repository.get_job_row(job_id)["lease_expires_at"]
    attempt_expiry = repository.get_attempt(str(attempt["id"]))["lease_expires_at"]
    original_write = database._write
    waiting_for_lock = threading.Event()
    failures: list[BaseException] = []

    @contextmanager
    def observed_write() -> Iterator[object]:
        waiting_for_lock.set()
        with original_write() as connection:
            yield connection

    def blocked_heartbeat() -> None:
        try:
            repository.heartbeat(job_id, owner=owner, attempt_id=str(attempt["id"]))
        except BaseException as error:
            failures.append(error)

    with original_write():
        monkeypatch.setattr(database, "_write", observed_write)
        thread = threading.Thread(target=blocked_heartbeat)
        thread.start()
        assert waiting_for_lock.wait(timeout=1)
        clock.advance(181)

    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(failures) == 1 and isinstance(failures[0], LiveLeaseConflict)
    assert repository.get_job_row(job_id)["lease_expires_at"] == job_expiry
    assert repository.get_attempt(str(attempt["id"]))["lease_expires_at"] == attempt_expiry


def test_mark_requesting_rejects_an_expired_owner_without_replacement(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    attempt = repository.claim(job_id, owner=owner)
    assert attempt is not None
    clock.advance(181)

    with pytest.raises(LiveLeaseConflict):
        repository.mark_requesting(str(attempt["id"]), owner=owner)

    assert repository.get_attempt(str(attempt["id"]))["state"] == "leased"


def test_configuration_release_rejects_an_expired_owner_without_replacement(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    clock.advance(181)

    with pytest.raises(LiveLeaseConflict):
        repository.release_job_after_configuration_error(job_id, owner=owner)

    assert repository.get_job_row(job_id)["owner"] == owner


def test_fail_job_rejects_an_expired_owner_without_replacement(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    attempt = repository.claim(job_id, owner=owner)
    assert attempt is not None
    repository.mark_requesting(str(attempt["id"]), owner=owner)
    clock.advance(181)

    with pytest.raises(LiveLeaseConflict):
        repository.fail_job(job_id, owner=owner, summary="stale failure")

    assert repository.get_job_row(job_id)["state"] == "running"
    assert repository.get_attempt(str(attempt["id"]))["state"] == "requesting"


def test_finalize_rejects_an_expired_owner_without_replacement(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    database, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    with database._write() as connection:
        connection.execute(
            "UPDATE cosmos_attempts SET state='manual_only' WHERE job_id=?",
            (job_id,),
        )
    clock.advance(181)

    with pytest.raises(LiveLeaseConflict):
        repository.finalize_if_done(job_id, owner=owner)

    assert repository.get_job_row(job_id)["state"] == "running"


def test_resume_requires_expired_job_and_attempt_leases_and_reclaims_once(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    database, repository, clock, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    attempt = repository.claim(job_id, owner=owner)
    assert attempt is not None
    repository.mark_requesting(attempt["id"], owner=owner)
    with pytest.raises(LiveLeaseConflict):
        repository.resume_job(job_id, owner=str(uuid4()))
    clock.advance(181)
    resumed_owner = str(uuid4())
    repository.resume_job(job_id, owner=resumed_owner)
    recovered = repository.list_attempts(job_id)[0]
    assert recovered["state"] == "retryable"
    assert recovered["lease_owner"] is None
    with pytest.raises(LiveLeaseConflict):
        repository.finish_attempt_manual_only(
            str(attempt["id"]), owner=owner, reason="stale worker must be fenced"
        )
    with database.open_connection() as connection:
        events = connection.execute(
            "SELECT count(*) FROM audit_events WHERE job_id=? AND operation='attempt_lease_reclaimed'",
            (job_id,),
        ).fetchone()[0]
    assert events == 1


def test_status_returns_only_latest_twenty_errors_by_time_with_stable_ties(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    database, repository, _, job_id = lifecycle
    with database.open_connection() as connection:
        dataset_id = connection.execute("SELECT dataset_id FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone()[
            "dataset_id"
        ]
    for episode_index in range(3, 25):
        database.create_episode(
            dataset_id=dataset_id,
            source_episode_index=episode_index,
            source_length=20,
        )
    with database._write() as connection:
        for episode_index in range(3, 25):
            connection.execute(
                """
                INSERT INTO cosmos_attempts(
                    id, job_id, source_episode_index, attempt_number, state,
                    error_class, error_summary, created_at, updated_at
                ) VALUES (?, ?, ?, 0, 'manual_only', 'Injected', ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    job_id,
                    episode_index,
                    f"episode-{episode_index}",
                    "2026-08-24T00:00:00.000000Z",
                    f"2026-08-24T00:00:{24 - episode_index:02d}.000000Z",
                ),
            )
        original = connection.execute(
            """
            SELECT id FROM cosmos_attempts
            WHERE job_id=? AND source_episode_index<3 ORDER BY source_episode_index
            """,
            (job_id,),
        ).fetchall()
        for episode_index, row in enumerate(original):
            connection.execute(
                """
                UPDATE cosmos_attempts
                SET state='manual_only', error_class='Injected', error_summary=?, updated_at=?
                WHERE id=?
                """,
                (
                    f"episode-{episode_index}",
                    "2026-08-24T00:00:30.000000Z" if episode_index < 2 else "2026-08-24T00:00:29.000000Z",
                    row["id"],
                ),
            )

    status = repository.status(job_id)
    expected = sorted(
        [(row["updated_at"], row["id"], row["source_episode_index"]) for row in repository.list_attempts(job_id)],
        reverse=True,
    )[:20]
    assert len(status["errors"]) == 20
    assert [error["source_episode_index"] for error in status["errors"]] == [item[2] for item in expected]


def test_cancel_requested_worker_finishes_inflight_then_cancels_remainder(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, _, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    active = repository.claim(job_id, owner=owner)
    assert active is not None
    repository.mark_requesting(active["id"], owner=owner)
    status, payload = repository.cancel(job_id)
    assert (status, payload["state"]) == (202, "cancel_requested")
    assert repository.claim(job_id, owner=owner) is None
    repository.finish_attempt_manual_only(active["id"], owner=owner, reason="cancelled_after_response")
    final = repository.finish_cancel_requested(job_id, owner=owner)
    assert final["state"] == "cancelled"
    assert final["counts"] == {"cancelled": 2, "manual_only": 1}


def test_cancel_only_resume_fences_the_expired_previous_owner(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    old_owner = str(uuid4())
    repository.start_job(job_id, owner=old_owner)
    status, _ = repository.cancel(job_id)
    assert status == 202
    old_status = repository.status(job_id)
    clock.advance(181)
    new_owner = str(uuid4())
    repository.resume_job(job_id, owner=new_owner)

    with pytest.raises(LiveLeaseConflict):
        repository.finish_cancel_requested(job_id, owner=old_owner)
    stop = worker_module.WorkerStopState()
    stale_worker = CurationWorker(
        repository=repository,
        process_attempt=lambda attempt, owner: None,
        owner=old_owner,
        heartbeat=False,
        stop_requested=stop,
    )
    assert stale_worker.execute("resume", job_id, acquired_status=old_status)["state"] == "cancel_requested"
    assert stop.reason == "lease_lost"
    assert repository.finish_cancel_requested(job_id, owner=new_owner)["state"] == "cancelled"


def test_worker_processes_one_attempt_at_a_time_and_run_never_attaches(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, _, job_id = lifecycle
    calls: list[int] = []

    def processor(attempt: dict[str, object], owner: str) -> None:
        calls.append(int(attempt["source_episode_index"]))
        repository.finish_attempt_manual_only(str(attempt["id"]), owner=owner, reason="test")

    worker = CurationWorker(repository=repository, process_attempt=processor, owner=str(uuid4()), heartbeat=False)
    result = worker.execute("run", job_id)
    assert result["state"] == "completed_with_failures"
    assert calls == [0, 1, 2]
    with pytest.raises(WorkerStateError):
        worker.execute("run", job_id)


def test_worker_recovery_after_terminal_attempt_makes_no_duplicate_processor_call(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, clock, job_id = lifecycle
    crashed_owner = str(uuid4())
    repository.start_job(job_id, owner=crashed_owner)
    attempt = repository.claim(job_id, owner=crashed_owner)
    assert attempt is not None
    repository.mark_requesting(attempt["id"], owner=crashed_owner)
    repository.finish_attempt_manual_only(attempt["id"], owner=crashed_owner, reason="persisted-before-crash")
    clock.advance(181)
    calls: list[int] = []

    def processor(next_attempt: dict[str, object], owner: str) -> None:
        calls.append(int(next_attempt["source_episode_index"]))
        repository.finish_attempt_manual_only(str(next_attempt["id"]), owner=owner, reason="test")

    resumed = CurationWorker(repository=repository, process_attempt=processor, owner=str(uuid4()), heartbeat=False)
    result = resumed.execute("resume", job_id)
    assert result["state"] == "completed_with_failures"
    assert calls == [1, 2]


def test_cli_parser_freezes_exact_run_and_resume_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parser = build_cli_parser()
    all_zero = "00000000-0000-0000-0000-000000000000"
    run = parser.parse_args(["--workspace", str(workspace.resolve()), "run", "--job-id", all_zero])
    resume = parser.parse_args(["--workspace", str(workspace.resolve()), "resume", "--job-id", all_zero])
    assert (run.command, run.job_id) == ("run", all_zero)
    assert (resume.command, resume.job_id) == ("resume", all_zero)
    with pytest.raises(SystemExit):
        parser.parse_args(["run", all_zero])
    with pytest.raises(SystemExit):
        parser.parse_args(["--workspace", str(workspace), "run", "--job-id", "not-a-uuid"])


def test_cli_workspace_canonicalizes_a_symlinked_ancestor_to_the_runtime_authority(
    tmp_path: Path,
) -> None:
    real_outputs = tmp_path / "real-outputs"
    real_workspace = real_outputs / "curation-workspace"
    real_workspace.mkdir(parents=True)
    linked_outputs = tmp_path / "linked-outputs"
    linked_outputs.symlink_to(real_outputs, target_is_directory=True)
    lexical_workspace = linked_outputs / real_workspace.name
    source = tmp_path / "source"
    source.mkdir()

    environment = _trusted_runtime_environment(workspace=real_workspace, source=source)
    environment["CURATION_WORKSPACE"] = str(lexical_workspace)
    settings = WorkerSettings.from_env(environment)
    parser = build_cli_parser()
    job_id = "00000000-0000-0000-0000-000000000000"

    arguments = parser.parse_args(["--workspace", str(lexical_workspace), "run", "--job-id", job_id])
    authority = TrustedWorkerAuthority.from_settings(
        settings,
        cli_workspace=arguments.workspace,
    )

    assert arguments.workspace == real_workspace.resolve()
    assert authority.workspace == settings.workspace
    with pytest.raises(SystemExit):
        parser.parse_args(["--workspace", "relative/workspace", "resume", "--job-id", job_id])


def test_worker_cli_runs_without_fastapi_or_exporter_only_environment(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, repository, _, job_id = lifecycle
    for name in (
        "CURATION_OUTPUT",
        "CURATION_BROWSER_ORIGIN",
        "CURATION_BEARER_TOKEN",
        "ISAAC_GROOT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    def process_attempt(attempt: dict[str, object], owner: str) -> None:
        repository.finish_attempt_manual_only(str(attempt["id"]), owner=owner, reason="test-only")

    monkeypatch.setattr(worker_module, "build_attempt_processor", lambda **kwargs: process_attempt)

    assert cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "event": "worker_terminal",
        "job_id": job_id,
        "state": "completed_with_failures",
    }


def test_cli_entrypoint_exact_state_and_conflict_exit_codes(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, repository, clock, job_id = lifecycle
    workspace = database.path.parent
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    assert cli_main(["--workspace", str(workspace), "run", "--job-id", job_id]) == 2
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["error"] == "invalid_job_state"
    assert cli_main(["--workspace", str(workspace), "resume", "--job-id", job_id]) == 3
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["error"] == "live_lease_conflict"
    clock.advance(181)


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (worker_module.RetryableDatabaseError("busy"), "database_busy"),
        (ArtifactSecurityError("secret filesystem detail"), "invalid_configuration"),
    ],
)
def test_cli_expected_processor_preflight_failures_are_json_exit_two(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    expected_error: str,
) -> None:
    database, _, _, job_id = lifecycle

    def fail_preflight(**kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(worker_module, "build_attempt_processor", fail_preflight)
    assert cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id]) == 2
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line["error"] == expected_error
    assert "secret filesystem detail" not in json.dumps(line)


def test_cli_contact_sheet_reconciliation_conflict_fails_job_and_clears_all_leases(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, repository, _clock, job_id = lifecycle

    def fail_reconciliation(**kwargs: object) -> object:
        raise worker_module.ContactSheetReconciliationConflict

    monkeypatch.setattr(worker_module, "build_attempt_processor", fail_reconciliation)

    exit_code = cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "error": "contact_sheet_conflict",
        "job_id": job_id,
        "state": "failed",
    }
    status = repository.status(job_id)
    assert status["state"] == "failed"
    with database.open_connection() as connection:
        job = connection.execute(
            "SELECT owner, lease_expires_at FROM cosmos_jobs WHERE id=?", (job_id,)
        ).fetchone()
        attempts = connection.execute(
            "SELECT lease_owner, lease_expires_at FROM cosmos_attempts WHERE job_id=?", (job_id,)
        ).fetchall()
    assert job["owner"] is None
    assert job["lease_expires_at"] is None
    assert all(row["lease_owner"] is None and row["lease_expires_at"] is None for row in attempts)


@pytest.mark.parametrize("failure_point", ["startup", "runtime"])
def test_cli_contact_sheet_conflict_honors_cancel_that_commits_before_resolver_lock(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_point: str,
) -> None:
    database, repository, _clock, job_id = lifecycle

    def cancel_then_conflict() -> None:
        status_code, payload = repository.cancel(job_id)
        assert (status_code, payload["state"]) == (202, "cancel_requested")
        raise worker_module.ContactSheetReconciliationConflict

    if failure_point == "startup":

        def build_failure(**kwargs: object) -> object:
            cancel_then_conflict()

        monkeypatch.setattr(worker_module, "build_attempt_processor", build_failure)
    else:

        def runtime_failure(attempt: object, owner: str) -> None:
            cancel_then_conflict()

        monkeypatch.setattr(
            worker_module,
            "build_attempt_processor",
            lambda **kwargs: runtime_failure,
        )

    exit_code = cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "event": "worker_terminal",
        "job_id": job_id,
        "state": "cancelled",
    }
    persisted = repository.status(job_id)
    assert persisted["state"] == "cancelled"
    assert persisted["counts"] == {"cancelled": 3}
    with database.open_connection() as connection:
        operations = [
            row["operation"]
            for row in connection.execute(
                "SELECT operation FROM audit_events WHERE job_id=? ORDER BY id", (job_id,)
            )
        ]
    assert operations.count("batch_cancel_requested") == 1
    assert operations.count("batch_cancelled") == 1
    assert "batch_failed" not in operations


@pytest.mark.parametrize("failure_point", ["startup", "runtime"])
@pytest.mark.parametrize(
    ("failure", "expected_exit", "expected_status"),
    [
        (
            LiveLeaseConflict("stale owner detail"),
            3,
            {"error": "worker_lease_lost"},
        ),
        (
            worker_module.RetryableDatabaseError("database path and secret detail"),
            2,
            {"error": "database_busy", "retryable": True},
        ),
    ],
)
def test_cli_contact_sheet_conflict_fail_job_preserves_exact_task8_error_mapping(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_point: str,
    failure: BaseException,
    expected_exit: int,
    expected_status: dict[str, object],
) -> None:
    database, repository, _clock, job_id = lifecycle

    if failure_point == "startup":

        def build_failure(**kwargs: object) -> object:
            raise worker_module.ContactSheetReconciliationConflict

        monkeypatch.setattr(worker_module, "build_attempt_processor", build_failure)
    else:

        def runtime_failure(attempt: object, owner: str) -> None:
            raise worker_module.ContactSheetReconciliationConflict

        monkeypatch.setattr(
            worker_module,
            "build_attempt_processor",
            lambda **kwargs: runtime_failure,
        )

    def resolve_contact_sheet_conflict(self: BatchRepository, job_id: str, *, owner: str) -> object:
        raise failure

    monkeypatch.setattr(BatchRepository, "resolve_contact_sheet_conflict", resolve_contact_sheet_conflict)

    exit_code = cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id])

    assert exit_code == expected_exit
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line == {**expected_status, "job_id": job_id}
    persisted = repository.status(job_id)
    assert persisted["state"] == "running"
    with database.open_connection() as connection:
        job = connection.execute(
            "SELECT state, owner, lease_expires_at FROM cosmos_jobs WHERE id=?", (job_id,)
        ).fetchone()
        attempts = connection.execute(
            "SELECT state, lease_owner, lease_expires_at FROM cosmos_attempts WHERE job_id=?",
            (job_id,),
        ).fetchall()
        failed_events = connection.execute(
            "SELECT count(*) FROM audit_events WHERE job_id=? AND operation='batch_failed'",
            (job_id,),
        ).fetchone()[0]
    assert job["state"] == "running"
    assert job["owner"] is not None
    assert job["lease_expires_at"] is not None
    expected_attempt_states = (
        ["queued", "queued", "queued"] if failure_point == "startup" else ["requesting", "queued", "queued"]
    )
    assert [row["state"] for row in attempts] == expected_attempt_states
    if failure_point == "startup":
        assert all(row["lease_owner"] is None for row in attempts)
        assert all(row["lease_expires_at"] is None for row in attempts)
    else:
        assert attempts[0]["lease_owner"] is not None
        assert attempts[0]["lease_expires_at"] is not None
        assert all(row["lease_owner"] is None for row in attempts[1:])
        assert all(row["lease_expires_at"] is None for row in attempts[1:])
    assert failed_events == 0


@pytest.mark.parametrize("failure_point", ["startup", "runtime"])
@pytest.mark.parametrize(
    ("failure_name", "expected_exit", "expected_status"),
    [
        ("LiveLeaseConflict", 3, {"error": "worker_lease_lost"}),
        ("RetryableDatabaseError", 2, {"error": "database_busy", "retryable": True}),
    ],
)
def test_subprocess_contact_sheet_conflict_fail_job_preserves_exact_task8_error_mapping(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    failure_point: str,
    failure_name: str,
    expected_exit: int,
    expected_status: dict[str, object],
) -> None:
    database, _repository, _clock, job_id = lifecycle
    repository_root = Path(__file__).parents[2]
    python = repository_root / "backend" / ".venv" / "bin" / "python"
    if failure_point == "startup":
        processor_patch = (
            "def build_attempt_processor(**kwargs):\n    raise worker.ContactSheetReconciliationConflict\n"
        )
    else:
        processor_patch = (
            "def processor(attempt, owner):\n"
            "    raise worker.ContactSheetReconciliationConflict\n"
            "def build_attempt_processor(**kwargs):\n"
            "    return processor\n"
        )
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(repository_root / 'backend')!r})\n"
        "import curation.worker as worker\n"
        f"{processor_patch}"
        "worker.build_attempt_processor = build_attempt_processor\n"
        "def resolve_contact_sheet_conflict(self, job_id, *, owner):\n"
        f"    raise worker.{failure_name}('sensitive child-process detail')\n"
        "worker.BatchRepository.resolve_contact_sheet_conflict = resolve_contact_sheet_conflict\n"
        "raise SystemExit(worker.cli_main([\n"
        f"    '--workspace', {str(database.path.parent)!r},\n"
        f"    'run', '--job-id', {job_id!r},\n"
        "]))\n"
    )

    process = subprocess.run(
        [str(python), "-c", script],
        cwd=repository_root,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert process.returncode == expected_exit
    assert process.stderr == ""
    assert _status_lines(process)[-1] == {**expected_status, "job_id": job_id}
    assert "sensitive child-process detail" not in process.stdout
    with database.open_connection() as connection:
        job = connection.execute(
            "SELECT state, owner, lease_expires_at FROM cosmos_jobs WHERE id=?", (job_id,)
        ).fetchone()
        attempts = connection.execute(
            "SELECT state, lease_owner, lease_expires_at FROM cosmos_attempts WHERE job_id=? "
            "ORDER BY source_episode_index",
            (job_id,),
        ).fetchall()
        failed_events = connection.execute(
            "SELECT count(*) FROM audit_events WHERE job_id=? AND operation='batch_failed'",
            (job_id,),
        ).fetchone()[0]
    assert job["state"] == "running"
    assert job["owner"] is not None
    assert job["lease_expires_at"] is not None
    expected_states = (
        ["queued", "queued", "queued"] if failure_point == "startup" else ["requesting", "queued", "queued"]
    )
    assert [attempt["state"] for attempt in attempts] == expected_states
    if failure_point == "startup":
        assert all(attempt["lease_owner"] is None for attempt in attempts)
        assert all(attempt["lease_expires_at"] is None for attempt in attempts)
    else:
        assert attempts[0]["lease_owner"] is not None
        assert attempts[0]["lease_expires_at"] is not None
        assert all(attempt["lease_owner"] is None for attempt in attempts[1:])
        assert all(attempt["lease_expires_at"] is None for attempt in attempts[1:])
    assert failed_events == 0


def test_cli_retryable_acquisition_failure_is_json_exit_two(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, _, _, job_id = lifecycle

    class BusyRepository:
        def start_job(self, job_id: str, *, owner: str) -> object:
            raise worker_module.RetryableDatabaseError("database path and secret detail")

    monkeypatch.setattr(worker_module, "BatchRepository", lambda database, **kwargs: BusyRepository())
    assert cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id]) == 2
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line == {"error": "database_busy", "job_id": job_id, "retryable": True}


def test_cli_malformed_database_is_json_exit_two_without_mutating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database_path = workspace / "curation.sqlite3"
    malformed = b"this is a regular file but not a sqlite database\n"
    database_path.write_bytes(malformed)
    job_id = str(uuid4())
    _set_trusted_runtime_environment(
        monkeypatch,
        workspace=workspace,
        source=tmp_path / "source",
    )

    assert cli_main(["--workspace", str(workspace), "run", "--job-id", job_id]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1]) == {
        "error": "invalid_database",
        "job_id": job_id,
    }
    assert captured.err == ""
    assert database_path.read_bytes() == malformed
    assert list(workspace.iterdir()) == [database_path]


@pytest.mark.parametrize(
    "database_case",
    ["partial_newer", "partial_unknown_v1", "older", "newer", "superficially_compatible"],
)
def test_cli_read_only_rejects_incompatible_schema_before_repository_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    database_case: str,
) -> None:
    workspace = tmp_path / "workspace"
    database_path = workspace / "curation.sqlite3"
    _create_incompatible_worker_database(database_path, database_case)
    artifact = workspace / "artifacts" / "sentinel.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"immutable-artifact")
    _set_trusted_runtime_environment(
        monkeypatch,
        workspace=workspace,
        source=tmp_path / "source",
    )
    before = _durable_workspace_bytes(workspace)
    job_id = str(uuid4())

    assert cli_main(["--workspace", str(workspace), "run", "--job-id", job_id]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1]) == {
        "error": "invalid_database",
        "job_id": job_id,
    }
    assert captured.err == ""
    assert _durable_workspace_bytes(workspace) == before


def test_worker_schema_gate_accepts_valid_v1_committed_only_in_live_wal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database_path = workspace / "curation.sqlite3"
    artifact = workspace / "artifacts" / "sentinel.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"immutable-artifact")

    with _live_wal_schema(database_path, delete_invariant_trigger=False):
        immutable_uri = database_path.resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(immutable_uri, uri=True) as stale:
            assert stale.execute("PRAGMA user_version").fetchone()[0] == 0
        before_files = _database_file_snapshot(database_path)
        before_tree = _workspace_tree_shape(workspace)

        CurationDatabase(database_path).validate_worker_compatibility()

        after_files = _database_file_snapshot(database_path)
        assert after_files.keys() == before_files.keys()
        assert {name: value for name, value in after_files.items() if not name.endswith("-shm")} == {
            name: value for name, value in before_files.items() if not name.endswith("-shm")
        }
        assert _workspace_tree_shape(workspace) == before_tree
        assert artifact.read_bytes() == b"immutable-artifact"


def test_worker_schema_gate_rejects_invariant_deleted_only_in_live_wal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database_path = workspace / "curation.sqlite3"
    artifact = workspace / "artifacts" / "sentinel.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"immutable-artifact")

    with _live_wal_schema(database_path, delete_invariant_trigger=True):
        immutable_uri = database_path.resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(immutable_uri, uri=True) as stale:
            assert (
                stale.execute(
                    "SELECT count(*) FROM sqlite_schema WHERE type='trigger' AND name=?",
                    ("cosmos_jobs_configuration_immutable",),
                ).fetchone()[0]
                == 1
            )
        before_files = _database_file_snapshot(database_path)
        before_tree = _workspace_tree_shape(workspace)

        with pytest.raises(IncompatibleCurationDatabase):
            CurationDatabase(database_path).validate_worker_compatibility()

        after_files = _database_file_snapshot(database_path)
        assert after_files.keys() == before_files.keys()
        assert {name: value for name, value in after_files.items() if not name.endswith("-shm")} == {
            name: value for name, value in before_files.items() if not name.endswith("-shm")
        }
        assert _workspace_tree_shape(workspace) == before_tree
        assert artifact.read_bytes() == b"immutable-artifact"


def test_worker_schema_gate_fails_closed_when_committed_wal_is_unreadable(tmp_path: Path) -> None:
    database_path = tmp_path / "workspace" / "curation.sqlite3"
    with _live_wal_schema(database_path, delete_invariant_trigger=False):
        wal_path = Path(str(database_path) + "-wal")
        wal_bytes = wal_path.read_bytes()
        original_mode = wal_path.stat().st_mode
        wal_path.chmod(0)
        try:
            with pytest.raises(IncompatibleCurationDatabase):
                CurationDatabase(database_path).validate_worker_compatibility()
        finally:
            wal_path.chmod(original_mode)
        assert wal_path.read_bytes() == wal_bytes


def test_cli_malformed_persisted_job_json_is_sanitized_and_rolls_back_acquisition(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, repository, _, job_id = lifecycle
    malformed = "{not-json"
    with database._write() as connection:
        with _disabled_trigger(connection, "cosmos_jobs_configuration_immutable"):
            connection.execute(
                "UPDATE cosmos_jobs SET configuration_json=? WHERE id=?",
                (malformed, job_id),
            )

    assert cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1]) == {
        "error": "invalid_configuration",
        "job_id": job_id,
    }
    assert captured.err == ""
    job = repository.get_job_row(job_id)
    assert job is not None
    assert (job["state"], job["owner"], job["configuration_json"]) == ("queued", None, malformed)


def test_cli_malformed_persisted_configuration_is_json_exit_two(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, _, _, job_id = lifecycle
    assert cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id]) == 2
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "error": "invalid_configuration",
        "job_id": job_id,
    }


def test_cli_definitive_processor_lease_loss_is_exit_three_not_signal_or_failure(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, repository, _, job_id = lifecycle

    def build(**kwargs: object) -> object:
        def lose_lease(attempt: dict[str, object], owner: str) -> None:
            raise LiveLeaseConflict("stale worker detail")

        return lose_lease

    monkeypatch.setattr(worker_module, "BatchRepository", lambda database, **kwargs: repository)
    monkeypatch.setattr(worker_module, "build_attempt_processor", build)
    assert cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id]) == 3
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line["error"] == "worker_lease_lost"
    assert repository.status(job_id)["state"] == "running"


@pytest.mark.parametrize("operation", ["status", "claim", "mark_requesting", "finalize_if_done"])
@pytest.mark.parametrize(
    ("failure", "expected_exit", "expected_error"),
    [
        (LiveLeaseConflict("stale owner detail"), 3, "worker_lease_lost"),
        (worker_module.RetryableDatabaseError("database path and secret detail"), 2, "database_busy"),
    ],
)
def test_cli_maps_outer_worker_lease_and_database_failures_without_falsely_failing_job(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operation: str,
    failure: BaseException,
    expected_exit: int,
    expected_error: str,
) -> None:
    database, repository, _, job_id = lifecycle

    class InjectingRepository:
        def __getattr__(self, name: str) -> object:
            return getattr(repository, name)

        def status(self, current_job_id: str) -> dict[str, object]:
            if operation == "status":
                raise failure
            return repository.status(current_job_id)

        def claim(self, current_job_id: str, *, owner: str) -> dict[str, object] | None:
            if operation == "claim":
                raise failure
            if operation == "finalize_if_done":
                return None
            return repository.claim(current_job_id, owner=owner)

        def mark_requesting(self, attempt_id: str, *, owner: str) -> dict[str, object]:
            if operation == "mark_requesting":
                raise failure
            return repository.mark_requesting(attempt_id, owner=owner)

        def finalize_if_done(self, current_job_id: str, *, owner: str) -> dict[str, object] | None:
            if operation == "finalize_if_done":
                raise failure
            return repository.finalize_if_done(current_job_id, owner=owner)

    proxy = InjectingRepository()

    def build(**kwargs: object) -> object:
        def process(attempt: dict[str, object], owner: str) -> None:
            repository.finish_attempt_manual_only(str(attempt["id"]), owner=owner, reason="test")

        return process

    monkeypatch.setattr(worker_module, "BatchRepository", lambda database, **kwargs: proxy)
    monkeypatch.setattr(worker_module, "build_attempt_processor", build)

    assert cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id]) == expected_exit

    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line["error"] == expected_error
    assert "secret" not in json.dumps(line)
    assert repository.status(job_id)["state"] == "running"


def test_cancel_requested_resume_needs_no_processor_source_or_secret(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, repository, _, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    status, _ = repository.cancel(job_id)
    assert status == 202
    with database._write() as connection:
        connection.execute(
            "UPDATE cosmos_jobs SET lease_expires_at='2000-01-01T00:00:00.000000Z' WHERE id=?",
            (job_id,),
        )

    def forbidden_processor(**kwargs: object) -> object:
        raise AssertionError("cancel-only resume must not construct a processor")

    def forbidden_heartbeat(**kwargs: object) -> object:
        raise AssertionError("cancel-only resume must not construct a heartbeat")

    monkeypatch.setattr(worker_module, "build_attempt_processor", forbidden_processor)
    monkeypatch.setattr(worker_module, "JobLeaseHeartbeat", forbidden_heartbeat)
    assert cli_main(["--workspace", str(database.path.parent), "resume", "--job-id", job_id]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[-1]["state"] == "cancelled"
    assert repository.status(job_id)["state"] == "cancelled"


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _job_evidence_snapshot(database: CurationDatabase, job_id: str) -> dict[str, object]:
    with database.open_connection() as connection:
        job = dict(connection.execute("SELECT * FROM cosmos_jobs WHERE id=?", (job_id,)).fetchone())
        attempts = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM cosmos_attempts WHERE job_id=? ORDER BY id",
                (job_id,),
            )
        ]
        audits = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM audit_events WHERE job_id=? ORDER BY id",
                (job_id,),
            )
        ]
        artifacts = [
            dict(row)
            for row in connection.execute(
                """
                SELECT artifact.* FROM artifacts AS artifact
                JOIN cosmos_attempts AS attempt ON attempt.id=artifact.attempt_id
                WHERE attempt.job_id=? ORDER BY artifact.id
                """,
                (job_id,),
            )
        ]
        episodes = [
            dict(row)
            for row in connection.execute(
                """
                SELECT episode.* FROM episodes AS episode
                JOIN cosmos_jobs AS job ON job.dataset_id=episode.dataset_id
                WHERE job.id=? ORDER BY episode.source_episode_index
                """,
                (job_id,),
            )
        ]
    artifact_root = database.path.parent / "artifacts"
    artifact_tree = (
        tuple(
            (
                path.relative_to(artifact_root).as_posix(),
                path.is_dir(),
                None if path.is_dir() else path.read_bytes(),
            )
            for path in sorted(artifact_root.rglob("*"))
        )
        if artifact_root.exists()
        else ()
    )
    return {
        "job": job,
        "attempts": attempts,
        "audits": audits,
        "artifacts": artifacts,
        "episodes": episodes,
        "artifact_tree": artifact_tree,
    }


def _no_signal_install(signum: int, handler: object, installed: dict[int, object] | None = None) -> object:
    previous = signal.SIG_DFL if installed is None else installed.get(signum, signal.SIG_DFL)
    if installed is not None:
        installed[signum] = handler
    return previous


def test_cli_renews_lease_during_blocked_processor_build_and_joins_on_config_error(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, repository, clock, job_id = lifecycle
    build_started = threading.Event()
    release_build = threading.Event()
    result: list[int] = []
    monkeypatch.setattr(worker_module, "HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "BatchRepository", lambda database, **kwargs: repository)
    monkeypatch.setattr(worker_module.signal, "signal", _no_signal_install)

    def blocked_build(**kwargs: object) -> object:
        build_started.set()
        assert release_build.wait(timeout=2)
        raise WorkerStateError("injected configuration error")

    monkeypatch.setattr(worker_module, "build_attempt_processor", blocked_build)
    thread = threading.Thread(
        target=lambda: result.append(
            cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id])
        )
    )
    thread.start()
    try:
        assert build_started.wait(timeout=1)
        for _ in range(2):
            prior_expiry = repository.get_job_row(job_id)["lease_expires_at"]
            clock.advance(170)
            assert _wait_until(
                lambda: repository.get_job_row(job_id)["lease_expires_at"] != prior_expiry,
                timeout=0.5,
            )
        with pytest.raises(LiveLeaseConflict):
            repository.resume_job(job_id, owner=str(uuid4()))
    finally:
        release_build.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert result == [2]
    assert not any(
        candidate.name == f"curation-heartbeat-{job_id}" and candidate.is_alive()
        for candidate in threading.enumerate()
    )


def test_cli_signal_during_blocked_processor_build_joins_heartbeat_without_attempt_work(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, repository, _, job_id = lifecycle
    installed: dict[int, object] = {}
    build_started = threading.Event()
    release_build = threading.Event()
    processor_calls: list[str] = []
    result: list[int] = []
    monkeypatch.setattr(worker_module, "HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(
        worker_module.signal,
        "signal",
        lambda signum, handler: _no_signal_install(signum, handler, installed),
    )

    def blocked_build(**kwargs: object) -> object:
        build_started.set()
        assert release_build.wait(timeout=2)

        def processor(attempt: dict[str, object], owner: str) -> None:
            processor_calls.append(str(attempt["id"]))

        return processor

    monkeypatch.setattr(worker_module, "build_attempt_processor", blocked_build)
    thread = threading.Thread(
        target=lambda: result.append(
            cli_main(["--workspace", str(database.path.parent), "run", "--job-id", job_id])
        )
    )
    thread.start()
    try:
        assert build_started.wait(timeout=1)
        acquired_expiry = repository.get_job_row(job_id)["lease_expires_at"]
        assert _wait_until(
            lambda: repository.get_job_row(job_id)["lease_expires_at"] != acquired_expiry,
            timeout=0.5,
        )
        handler = installed[signal.SIGTERM]
        handler(signal.SIGTERM, None)
    finally:
        release_build.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert result == [130]
    assert processor_calls == []
    assert repository.status(job_id)["state"] == "running"
    assert not any(
        candidate.name == f"curation-heartbeat-{job_id}" and candidate.is_alive()
        for candidate in threading.enumerate()
    )


def test_job_heartbeat_start_is_idempotent_and_never_spawns_a_second_thread(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
) -> None:
    _, repository, _, job_id = lifecycle
    owner = str(uuid4())
    repository.start_job(job_id, owner=owner)
    heartbeat = worker_module.JobLeaseHeartbeat(
        repository=repository,
        job_id=job_id,
        owner=owner,
        stop_requested=threading.Event(),
        interval_seconds=60,
    )
    try:
        assert heartbeat.start() is True
        first_thread = heartbeat.thread
        assert first_thread is not None
        assert heartbeat.start() is False
        assert heartbeat.thread is first_thread
    finally:
        heartbeat.stop()
    assert not first_thread.is_alive()


def test_retryable_heartbeat_failures_are_observable_bounded_and_not_a_signal() -> None:
    class BusyRepository:
        calls = 0

        def heartbeat(self, job_id: str, *, owner: str, attempt_id: str | None) -> None:
            self.calls += 1
            raise worker_module.RetryableDatabaseError("busy")

    repository = BusyRepository()
    stop = worker_module.WorkerStopState()
    statuses: list[dict[str, object]] = []
    heartbeat = worker_module.JobLeaseHeartbeat(
        repository=repository,
        job_id=str(uuid4()),
        owner=str(uuid4()),
        stop_requested=stop,
        interval_seconds=0.005,
        max_retryable_failures=3,
        status_sink=lambda status: statuses.append(dict(status)),
    )
    heartbeat.start()
    assert _wait_until(stop.is_set, timeout=0.5)
    heartbeat.stop()

    assert repository.calls == 3
    assert stop.reason == "heartbeat_failure"
    assert [status["consecutive_failures"] for status in statuses] == [1, 2, 3]
    assert not heartbeat.thread.is_alive()


def test_terminal_job_heartbeat_shutdown_does_not_set_a_stop_reason() -> None:
    class TerminalRepository:
        def heartbeat(self, job_id: str, *, owner: str, attempt_id: str | None) -> None:
            raise LiveLeaseConflict("already terminal")

        def get_job_row(self, job_id: str) -> dict[str, object]:
            return {"state": "completed"}

    stop = worker_module.WorkerStopState()
    heartbeat = worker_module.JobLeaseHeartbeat(
        repository=TerminalRepository(),
        job_id=str(uuid4()),
        owner=str(uuid4()),
        stop_requested=stop,
        interval_seconds=0.005,
    )
    heartbeat.start()
    assert _wait_until(lambda: not heartbeat.thread.is_alive(), timeout=0.5)
    heartbeat.stop()
    assert stop.reason is None


def test_unexpected_heartbeat_failure_is_observable_and_stops_as_fatal_not_signal() -> None:
    class FatalRepository:
        def heartbeat(self, job_id: str, *, owner: str, attempt_id: str | None) -> None:
            raise RuntimeError("secret database detail")

    stop = worker_module.WorkerStopState()
    statuses: list[dict[str, object]] = []
    heartbeat = worker_module.JobLeaseHeartbeat(
        repository=FatalRepository(),
        job_id=str(uuid4()),
        owner=str(uuid4()),
        stop_requested=stop,
        interval_seconds=0.005,
        status_sink=lambda status: statuses.append(dict(status)),
    )
    heartbeat.start()
    assert _wait_until(stop.is_set, timeout=0.5)
    heartbeat.stop()

    assert stop.reason == "heartbeat_failure"
    assert statuses == [{"event": "heartbeat_fatal_error", "job_id": heartbeat.job_id}]
    assert "secret database detail" not in json.dumps(statuses)


@pytest.mark.parametrize(
    ("heartbeat_failure", "expected_reason"),
    [
        (worker_module.RetryableDatabaseError("busy"), "heartbeat_failure"),
        (LiveLeaseConflict("lost"), "lease_lost"),
        (RuntimeError("fatal secret"), "heartbeat_failure"),
    ],
)
def test_heartbeat_status_sink_failure_cannot_prevent_fatal_stop_state(
    heartbeat_failure: BaseException,
    expected_reason: str,
) -> None:
    class FailingRepository:
        calls = 0

        def heartbeat(self, job_id: str, *, owner: str, attempt_id: str | None) -> None:
            self.calls += 1
            raise heartbeat_failure

        def get_job_row(self, job_id: str) -> dict[str, object]:
            return {"state": "running"}

    sink_calls = 0

    def broken_sink(status: object) -> None:
        nonlocal sink_calls
        sink_calls += 1
        raise BrokenPipeError("consumer closed stdout")

    repository = FailingRepository()
    stop = worker_module.WorkerStopState()
    heartbeat = worker_module.JobLeaseHeartbeat(
        repository=repository,
        job_id=str(uuid4()),
        owner=str(uuid4()),
        stop_requested=stop,
        interval_seconds=0.005,
        max_retryable_failures=3,
        status_sink=broken_sink,
    )
    heartbeat.start()
    assert _wait_until(lambda: not heartbeat.thread.is_alive(), timeout=0.5)
    heartbeat.stop()

    assert repository.calls == 1
    assert sink_calls == 1
    assert stop.reason == expected_reason


def _complete_response() -> str:
    phases = [
        "approach_brown_table",
        "pick_up_object",
        "turn_to_find_black_trash_bin",
        "approach_black_trash_bin",
        "lean_down_to_black_trash_bin",
        "drop_object_into_black_trash_bin",
        "stand_straight",
    ]
    return json.dumps(
        {
            "schema_version": 2,
            "episode_complete": True,
            "segments": [
                {
                    "step": index + 1,
                    "phase": phase,
                    "status": "completed",
                    "start_s": float(index),
                    "end_s": float(index + 1),
                    "caption": phase,
                    "confidence": 1.0,
                    "evidence": "visible",
                }
                for index, phase in enumerate(phases)
            ],
            "missing_steps": [],
            "uncertainties": [],
        },
        separators=(",", ":"),
    )


def _prepared_sample() -> PreparedSample:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(buffer, format="JPEG")
    jpeg = buffer.getvalue()
    all_timestamps = tuple(frame / 10 for frame in range(70))
    selected = tuple(range(0, 70, 5))
    return PreparedSample(
        source_fps=10.0,
        total_num_frames=70,
        duration_s=7.0,
        frame_indices=selected,
        parquet_timestamps=tuple(all_timestamps[index] for index in selected),
        all_parquet_timestamps=all_timestamps,
        jpeg_frames=tuple(jpeg for _ in selected),
        decoder_name="test",
        decoder_version="1",
        source_video_sha256="c" * 64,
    )


class FakeTransport:
    def __init__(self) -> None:
        self.initial_calls = 0
        self.repair_calls = 0

    def observe_initial(self, prepared: object) -> CosmosCallObservation:
        self.initial_calls += 1
        exchange = {
            "phase": "initial",
            "started_at": "2026-08-24T00:00:00Z",
            "finished_at": "2026-08-24T00:00:01Z",
            "request": {"method": "POST", "url": "http://cosmos/v1/chat/completions", "body_sha256": "d" * 64},
            "response": {
                "status_code": 200,
                "id": "response",
                "model": "cosmos3-nano",
                "created": 1,
                "usage": {},
                "finish_reason": "stop",
            },
            "error": None,
        }
        return CosmosCallObservation(
            phase="initial",
            content=_complete_response(),
            observed_content=_complete_response(),
            reason=None,
            retryable=False,
            _exchange_json=json.dumps(exchange, sort_keys=True, separators=(",", ":")).encode(),
        )

    def observe_repair(self, **kwargs: object) -> CosmosCallObservation:
        self.repair_calls += 1
        raise AssertionError("repair was not expected")


def _failed_observation(
    *, content: str | None, retryable: bool, finish_reason: str | None
) -> CosmosCallObservation:
    exchange = {
        "phase": "initial",
        "started_at": "2026-08-24T00:00:00Z",
        "finished_at": "2026-08-24T00:00:01Z",
        "request": {"method": "POST", "url": "http://cosmos/v1/chat/completions", "body_sha256": "d" * 64},
        "response": {
            "status_code": 500 if retryable else 200,
            "id": None,
            "model": "cosmos3-nano",
            "created": 1,
            "usage": {},
            "finish_reason": finish_reason,
        },
        "error": None,
    }
    return CosmosCallObservation(
        phase="initial",
        content=content,
        observed_content=content,
        reason="http_status" if retryable else None,
        retryable=retryable,
        _exchange_json=json.dumps(exchange, sort_keys=True, separators=(",", ":")).encode(),
    )


@pytest.fixture
def processor_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True)
    (source / "meta" / "info.json").write_text(json.dumps({"fps": 10}))
    (source / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 70}) + "\n")
    (source / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"registered parquet")
    (source / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4").write_bytes(
        b"registered video"
    )
    workspace = tmp_path / "workspace"
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
    record = registry.records["local/pnp_trash"]
    database = CurationDatabase(workspace / "curation.sqlite3")
    database.initialize()
    dataset = database.register_dataset(
        alias="local/pnp_trash",
        source_path=str(record.root),
        source_manifest_sha256=record.fingerprint,
        prompt_template_version=worker_module.PROMPT_TEMPLATE_VERSION,
        prompt_template_sha256=worker_module.PROMPT_TEMPLATE_SHA256,
    )
    database.create_episode(dataset_id=dataset["id"], source_episode_index=0, source_length=70)
    repository = BatchRepository(
        database,
        trusted_authority=_trusted_worker_authority(workspace=workspace, source=source),
    )
    job = repository.create_job_with_attempts(
        dataset_id=dataset["id"],
        configuration=_valid_job_configuration(
            dataset_id=dataset["id"],
            source_path=record.root,
            source_manifest_sha256=record.fingerprint,
            episode_indices=[0],
        ),
        episode_indices=[0],
    )
    owner = str(uuid4())
    repository.start_job(job["job_id"], owner=owner)
    attempt = repository.claim(job["job_id"], owner=owner)
    assert attempt is not None
    attempt = repository.mark_requesting(attempt["id"], owner=owner)
    _set_trusted_runtime_environment(monkeypatch, workspace=workspace, source=source)
    return database, repository, attempt, owner, registry


@pytest.mark.parametrize(
    "crash_event, expected_calls_after_crash", [("request_artifact", 0), ("response_artifact", 1)]
)
def test_processor_recovers_durable_phase_artifacts_without_duplicate_model_calls(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
    crash_event: str,
    expected_calls_after_crash: int,
) -> None:
    database, repository, attempt, owner, registry = processor_case
    transport = FakeTransport()
    crashed = False

    def event(event_name: str, attempt_id: str) -> None:
        nonlocal crashed
        if event_name == crash_event and not crashed:
            crashed = True
            raise RuntimeError("injected crash")

    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=lambda: transport,
        event_hook=event,
        sleep=lambda seconds: None,
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        processor(attempt, owner)
    assert transport.initial_calls == expected_calls_after_crash
    processor(attempt, owner)
    assert transport.initial_calls == 1
    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM cosmos_proposals").fetchone()[0] == 1
        episode = connection.execute("SELECT * FROM episodes WHERE source_episode_index=0").fetchone()
        assert episode["review_state"] == "pending"
        assert episode["revision"] == 0


def test_expired_old_owner_cannot_install_or_register_response_after_resume(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    database, repository, attempt, old_owner, registry = processor_case
    clock = MutableClock(datetime.now(timezone.utc))
    repository.clock = clock
    request_started = threading.Event()
    release_response = threading.Event()
    failures: list[BaseException] = []
    heartbeat_statuses: list[dict[str, object]] = []
    stop = worker_module.WorkerStopState()

    def database_busy_heartbeat(job_id: str, *, owner: str, attempt_id: str | None) -> None:
        raise worker_module.RetryableDatabaseError("injected heartbeat lock")

    repository.heartbeat = database_busy_heartbeat  # type: ignore[method-assign]
    heartbeat = worker_module.JobLeaseHeartbeat(
        repository=repository,
        job_id=str(attempt["job_id"]),
        owner=old_owner,
        stop_requested=stop,
        interval_seconds=0.005,
        max_retryable_failures=100,
        status_sink=lambda status: heartbeat_statuses.append(dict(status)),
    )
    heartbeat.set_active_attempt(str(attempt["id"]))
    heartbeat.start()

    class BlockingTransport(FakeTransport):
        def observe_initial(self, prepared: object) -> CosmosCallObservation:
            request_started.set()
            assert release_response.wait(timeout=2)
            return super().observe_initial(prepared)

    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=BlockingTransport,
        sleep=lambda seconds: None,
    )

    def run_old_worker() -> None:
        try:
            processor(attempt, old_owner)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run_old_worker)
    thread.start()
    try:
        assert request_started.wait(timeout=1)
        assert _wait_until(lambda: bool(heartbeat_statuses), timeout=0.5)
        clock.advance(181)
        repository.resume_job(str(attempt["job_id"]), owner=str(uuid4()))
    finally:
        release_response.set()
        thread.join(timeout=2)
        heartbeat.stop()

    assert not thread.is_alive()
    assert len(failures) == 1 and isinstance(failures[0], LiveLeaseConflict)
    response_path = database.path.parent / "artifacts" / "cosmos" / str(attempt["id"]) / "response.txt"
    assert not response_path.exists()
    assert repository.get_attempt_artifact(str(attempt["id"]), "response") is None
    recovered = repository.get_attempt(str(attempt["id"]))
    assert recovered is not None
    assert json.loads(recovered["http_exchange_history_json"]) == []
    assert heartbeat_statuses[0]["event"] == "heartbeat_retryable_error"
    assert stop.reason is None


def test_artifact_authorization_guard_enters_after_flock_and_covers_registration(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = AtomicArtifactStore(workspace)
    guard_active = False
    observed: list[str] = []

    @contextmanager
    def authorization_guard() -> Iterator[None]:
        nonlocal guard_active
        lock_fd = os.open(workspace / ".artifact-store.lock", os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guard_active = True
            observed.append("guard_enter")
            yield
        finally:
            guard_active = False
            os.close(lock_fd)
            observed.append("guard_exit")

    def register(record: ArtifactRecord) -> str:
        assert guard_active
        observed.append("register")
        return record.sha256

    result = store.write_text(
        "artifacts/cosmos/attempt/response.txt",
        "evidence",
        register=register,
        authorization_guard=authorization_guard,
    )

    assert result.database_reference == hashlib.sha256(b"evidence").hexdigest()
    assert observed == ["guard_enter", "register", "guard_exit"]


def test_crash_after_proposal_commit_is_exactly_once_and_never_reprocesses(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    database, repository, attempt, owner, registry = processor_case
    transport = FakeTransport()
    crashed = False

    def event(event_name: str, attempt_id: str) -> None:
        nonlocal crashed
        if event_name == "proposal_commit" and not crashed:
            crashed = True
            raise RuntimeError("injected crash")

    class RecordingCoordinator:
        reconcile_calls = 0
        proposal_ids: list[str] = []

        def reconcile_all(self) -> list[object]:
            self.reconcile_calls += 1
            return []

        def ensure_proposal(self, proposal_id: str) -> object:
            self.proposal_ids.append(proposal_id)
            return object()

    coordinator = RecordingCoordinator()

    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=lambda: transport,
        event_hook=event,
        contact_sheet_coordinator=coordinator,
        sleep=lambda seconds: None,
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        processor(attempt, owner)
    processor(attempt, owner)
    assert transport.initial_calls == 1
    assert coordinator.reconcile_calls == 1
    assert coordinator.proposal_ids == []
    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM cosmos_proposals").fetchone()[0] == 1
        assert (
            connection.execute("SELECT state FROM cosmos_attempts WHERE id=?", (attempt["id"],)).fetchone()[0]
            == "succeeded"
        )


def test_successful_proposal_activation_produces_its_contact_sheet(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    database, repository, attempt, owner, registry = processor_case

    class RecordingCoordinator:
        proposal_ids: list[str] = []

        def reconcile_all(self) -> list[object]:
            raise AssertionError("new proposal must be produced directly")

        def ensure_proposal(self, proposal_id: str) -> object:
            self.proposal_ids.append(proposal_id)
            return object()

    coordinator = RecordingCoordinator()
    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=FakeTransport,
        contact_sheet_coordinator=coordinator,
        sleep=lambda seconds: None,
    )

    processor(attempt, owner)

    with database.open_connection() as connection:
        proposal_id = connection.execute("SELECT id FROM cosmos_proposals").fetchone()["id"]
    assert coordinator.proposal_ids == [proposal_id]


def test_cancel_after_parsed_artifact_never_activates_or_supersedes_a_proposal(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    database, repository, attempt, prior_owner, registry = processor_case
    owned_attempt = repository.get_attempt(str(attempt["id"]))
    assert owned_attempt is not None
    prior_processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=FakeTransport,
        sleep=lambda seconds: None,
    )
    prior_processor(attempt, prior_owner)
    prior_job_id = str(attempt["job_id"])
    prior_final = repository.finalize_if_done(prior_job_id, owner=prior_owner)
    assert prior_final is not None and prior_final["state"] == "completed"
    with database.open_connection() as connection:
        prior_proposal_id = connection.execute(
            "SELECT id FROM cosmos_proposals WHERE attempt_id=?", (attempt["id"],)
        ).fetchone()["id"]

    child_configuration = _valid_job_configuration(
        dataset_id=int(owned_attempt["dataset_id"]),
        source_path=registry.records["local/pnp_trash"].root,
        source_manifest_sha256=registry.records["local/pnp_trash"].fingerprint,
        episode_indices=[0],
    )
    child_configuration["parent_job_id"] = prior_job_id
    current_job = repository.create_job_with_attempts(
        dataset_id=int(owned_attempt["dataset_id"]),
        configuration=child_configuration,
        episode_indices=[0],
        parent_job_id=prior_job_id,
        attempt_numbers={0: 1},
    )
    job_id = str(current_job["job_id"])

    def event(event_name: str, attempt_id: str) -> None:
        if event_name == "parsed_artifact":
            status, payload = repository.cancel(job_id)
            assert (status, payload["state"]) == (202, "cancel_requested")

    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=FakeTransport,
        event_hook=event,
        sleep=lambda seconds: None,
    )
    worker = CurationWorker(
        repository=repository,
        process_attempt=processor,
        owner=str(uuid4()),
        heartbeat=False,
    )
    final = worker.execute("run", job_id)

    assert final["state"] == "cancelled"
    assert repository.list_attempts(job_id)[0]["state"] == "cancelled"
    with database.open_connection() as connection:
        proposals = connection.execute("SELECT id, state FROM cosmos_proposals ORDER BY id").fetchall()
    assert [(row["id"], row["state"]) for row in proposals] == [(prior_proposal_id, "active")]


@pytest.mark.parametrize("invalid_content", [None, "{invalid-json"])
def test_stop_after_first_call_prevents_retry_and_repair(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
    invalid_content: str | None,
) -> None:
    database, repository, attempt, owner, registry = processor_case
    stopped = worker_module.threading.Event()

    class StopTransport:
        initial_calls = 0
        repair_calls = 0

        def observe_initial(self, prepared: object) -> CosmosCallObservation:
            self.initial_calls += 1
            stopped.set()
            if invalid_content is None:
                return _failed_observation(content=None, retryable=True, finish_reason=None)
            return _failed_observation(content=invalid_content, retryable=False, finish_reason="stop")

        def observe_repair(self, **kwargs: object) -> CosmosCallObservation:
            self.repair_calls += 1
            raise AssertionError("stop must prevent repair")

    transport = StopTransport()
    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=lambda: transport,
        sleep=lambda seconds: None,
        stop_requested=stopped,
    )
    processor(attempt, owner)
    assert transport.initial_calls == 1
    assert transport.repair_calls == 0
    persisted = repository.get_attempt(str(attempt["id"]))
    assert persisted is not None
    assert len(json.loads(persisted["http_exchange_history_json"])) == 1
    assert persisted["state"] == "requesting"


def test_heartbeat_fatal_during_sampling_stops_before_any_artifact_or_model_call(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    database, repository, attempt, owner, registry = processor_case
    stopped = worker_module.WorkerStopState()
    transport = FakeTransport()

    def stop_during_sampling(**kwargs: object) -> SamplingOutcome:
        stopped.request("heartbeat_failure")
        return SamplingOutcome.ready(_prepared_sample())

    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=stop_during_sampling,
        transport_factory=lambda: transport,
        sleep=lambda seconds: None,
        stop_requested=stopped,
    )
    processor(attempt, owner)

    assert transport.initial_calls == 0
    assert repository.get_attempt_artifact(str(attempt["id"]), "request") is None
    assert not (database.path.parent / "artifacts" / "cosmos" / str(attempt["id"])).exists()


def test_observed_artifact_exact_replay_deduplicates_history_and_rejects_bad_envelopes(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    _, repository, attempt, owner, _ = processor_case
    record = ArtifactRecord(
        relative_path=f"artifacts/cosmos/{attempt['id']}/response.txt",
        media_type="text/plain; charset=utf-8",
        byte_size=3,
        sha256=hashlib.sha256(b"bad").hexdigest(),
    )
    observation = _failed_observation(content="bad", retryable=False, finish_reason="stop")
    kwargs = {
        "dataset_id": attempt["dataset_id"]
        if "dataset_id" in attempt
        else repository.get_attempt(str(attempt["id"]))["dataset_id"],
        "attempt_id": str(attempt["id"]),
        "kind": "response",
        "record": record,
        "observation": observation,
        "owner": owner,
        "update_attempt_pointer": True,
    }
    repository.register_observed_artifact(**kwargs)
    repository.register_observed_artifact(**kwargs)
    persisted = repository.get_attempt(str(attempt["id"]))
    assert persisted is not None
    assert len(json.loads(persisted["http_exchange_history_json"])) == 1

    with pytest.raises(ValueError, match="content"):
        repository.register_observed_artifact(
            **{
                **kwargs,
                "record": ArtifactRecord(
                    relative_path=f"artifacts/cosmos/{attempt['id']}/response.txt",
                    media_type="text/plain; charset=utf-8",
                    byte_size=4,
                    sha256=hashlib.sha256(b"nope").hexdigest(),
                ),
            }
        )

    malformed = CosmosCallObservation(
        phase="repair",
        content="bad",
        observed_content="bad",
        reason=None,
        retryable=False,
        _exchange_json=b'{"phase":"repair"}',
    )
    with pytest.raises(ValueError, match="HTTP exchange"):
        repository.register_observed_artifact(
            dataset_id=kwargs["dataset_id"],
            attempt_id=str(attempt["id"]),
            kind="repair_response",
            record=ArtifactRecord(
                relative_path=f"artifacts/cosmos/{attempt['id']}/repair-response.txt",
                media_type="text/plain; charset=utf-8",
                byte_size=3,
                sha256=hashlib.sha256(b"bad").hexdigest(),
            ),
            observation=malformed,
            owner=owner,
            update_attempt_pointer=False,
        )


def test_identical_contentless_http_calls_are_recorded_as_distinct_observations(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    _, repository, attempt, owner, _ = processor_case
    observation = _failed_observation(content=None, retryable=True, finish_reason=None)

    repository.append_observation(str(attempt["id"]), observation, owner=owner)
    repository.append_observation(str(attempt["id"]), observation, owner=owner)

    persisted = repository.get_attempt(str(attempt["id"]))
    assert persisted is not None
    history = json.loads(persisted["http_exchange_history_json"])
    assert history == [observation.exchange, observation.exchange]


def test_crash_after_two_identical_contentless_calls_never_allows_a_third_call(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    database, repository, attempt, owner, registry = processor_case

    class IdenticalFailureTransport:
        calls = 0

        def observe_initial(self, prepared: object) -> CosmosCallObservation:
            self.calls += 1
            return _failed_observation(content=None, retryable=True, finish_reason=None)

    transport = IdenticalFailureTransport()
    persisted_events = 0

    def crash_after_second_observation(event_name: str, attempt_id: str) -> None:
        nonlocal persisted_events
        if event_name == "observation_persisted":
            persisted_events += 1
            if persisted_events == 2:
                raise RuntimeError("crash after second persisted call")

    crashed = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=lambda: transport,
        event_hook=crash_after_second_observation,
        sleep=lambda seconds: None,
    )
    with pytest.raises(RuntimeError, match="crash after second persisted call"):
        crashed(attempt, owner)

    resumed = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=lambda: transport,
        sleep=lambda seconds: None,
    )
    resumed(attempt, owner)

    assert transport.calls == 2
    persisted = repository.get_attempt(str(attempt["id"]))
    assert persisted is not None
    assert len(json.loads(persisted["http_exchange_history_json"])) == 2
    assert persisted["state"] == "manual_only"


@pytest.mark.parametrize(
    "invalid_case",
    ["empty", "wrong_type", "unknown_field", "missing_nested_field", "wrong_frozen_limit"],
)
@pytest.mark.parametrize(
    ("command", "persisted_job_state"),
    [("run", "queued"), ("resume", "running"), ("resume", "cancel_requested")],
)
def test_cli_semantically_validates_configuration_before_any_acquisition_mutation(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    invalid_case: str,
    command: str,
    persisted_job_state: str,
) -> None:
    database, repository, attempt, old_owner, _ = processor_case
    job_id = str(attempt["job_id"])
    queued = persisted_job_state == "queued"
    current = repository.get_job_row(job_id)
    assert current is not None
    valid = json.loads(current["configuration_json"])
    invalid = _invalid_job_configuration(invalid_case, valid)
    with database._write() as connection:
        with _disabled_trigger(connection, "cosmos_jobs_configuration_immutable"):
            connection.execute(
                """
                UPDATE cosmos_jobs SET state=?, cancel_requested=?, owner=?, lease_expires_at=?,
                    configuration_json=? WHERE id=?
                """,
                (
                    persisted_job_state,
                    int(persisted_job_state == "cancel_requested"),
                    None if queued else old_owner,
                    None if queued else "2000-01-01T00:00:00.000000Z",
                    json.dumps(invalid, sort_keys=True, separators=(",", ":")),
                    job_id,
                ),
            )
        connection.execute(
            """
            UPDATE cosmos_attempts SET state=?, lease_owner=?, lease_expires_at=? WHERE id=?
            """,
            (
                "queued" if queued else "requesting",
                None if queued else old_owner,
                None if queued else "2000-01-01T00:00:00.000000Z",
                attempt["id"],
            ),
        )
    before = _job_evidence_snapshot(database, job_id)
    processor_builds: list[str] = []

    def forbidden_processor_build(**kwargs: object) -> object:
        processor_builds.append(job_id)
        raise AssertionError("invalid persisted configuration must fail before processor/source work")

    monkeypatch.setattr(worker_module, "BatchRepository", lambda database, **kwargs: repository)
    monkeypatch.setattr(worker_module, "build_attempt_processor", forbidden_processor_build)

    assert cli_main(["--workspace", str(database.path.parent), command, "--job-id", job_id]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1]) == {
        "error": "invalid_configuration",
        "job_id": job_id,
    }
    assert captured.err == ""
    assert _job_evidence_snapshot(database, job_id) == before
    assert processor_builds == []


def _tamper_authority_configuration(
    field: str,
    configuration: dict[str, object],
    *,
    attacker_source: Path,
    attacker_base_url: str = "http://127.0.0.1:9/v1",
) -> dict[str, object]:
    document = json.loads(json.dumps(configuration))
    if field == "source_path":
        document["source_path"] = str(attacker_source.resolve())
    elif field == "base_url":
        document["cosmos"]["base_url"] = attacker_base_url
    elif field == "model":
        document["cosmos"]["model"] = "attacker-model"
    elif field == "api_key_env":
        document["cosmos"]["api_key_env"] = "EXFILTRATE_SENTINEL_SECRET"
    elif field == "endpoint_identity":
        document["cosmos"]["endpoint_identity"] = "attacker-endpoint"
    else:
        raise AssertionError(f"unknown authority field: {field}")
    return document


@pytest.mark.parametrize(
    "authority_field",
    ["source_path", "base_url", "model", "api_key_env", "endpoint_identity"],
)
@pytest.mark.parametrize(
    ("command", "persisted_job_state"),
    [("run", "queued"), ("resume", "running"), ("resume", "cancel_requested")],
)
def test_cli_binds_persisted_authority_before_acquisition_or_any_side_effect(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    authority_field: str,
    command: str,
    persisted_job_state: str,
) -> None:
    database, repository, attempt, old_owner, _ = processor_case
    job_id = str(attempt["job_id"])
    queued = persisted_job_state == "queued"
    current = repository.get_job_row(job_id)
    assert current is not None
    tampered = _tamper_authority_configuration(
        authority_field,
        json.loads(current["configuration_json"]),
        attacker_source=database.path.parent.parent / "attacker-source",
    )
    with database._write() as connection:
        with _disabled_trigger(connection, "cosmos_jobs_configuration_immutable"):
            connection.execute(
                """
                UPDATE cosmos_jobs SET state=?, cancel_requested=?, owner=?, lease_expires_at=?,
                    configuration_json=? WHERE id=?
                """,
                (
                    persisted_job_state,
                    int(persisted_job_state == "cancel_requested"),
                    None if queued else old_owner,
                    None if queued else "2000-01-01T00:00:00.000000Z",
                    json.dumps(tampered, sort_keys=True, separators=(",", ":")),
                    job_id,
                ),
            )
        connection.execute(
            """
            UPDATE cosmos_attempts SET state=?, lease_owner=?, lease_expires_at=? WHERE id=?
            """,
            (
                "queued" if queued else "requesting",
                None if queued else old_owner,
                None if queued else "2000-01-01T00:00:00.000000Z",
                attempt["id"],
            ),
        )
    before = _job_evidence_snapshot(database, job_id)
    processor_builds: list[str] = []
    secret_reads: list[str] = []
    sensitive_names = {"TEST_COSMOS_KEY", "EXFILTRATE_SENTINEL_SECRET"}

    class TrackingEnvironment(dict[str, str]):
        def get(self, key: str, default: str | None = None) -> str | None:
            if key in sensitive_names:
                secret_reads.append(key)
            return super().get(key, default)

        def __getitem__(self, key: str) -> str:
            if key in sensitive_names:
                secret_reads.append(key)
            return super().__getitem__(key)

    environment = TrackingEnvironment(os.environ)
    environment["TEST_COSMOS_KEY"] = "trusted-sentinel-must-not-be-read"
    environment["EXFILTRATE_SENTINEL_SECRET"] = "must-never-be-read-or-disclosed"
    monkeypatch.setattr(worker_module.os, "environ", environment)

    def forbidden_processor_build(**kwargs: object) -> object:
        processor_builds.append(job_id)
        raise AssertionError("authority mismatch must fail before processor/source/model work")

    monkeypatch.setattr(worker_module, "BatchRepository", lambda database, **kwargs: repository)
    monkeypatch.setattr(worker_module, "build_attempt_processor", forbidden_processor_build)

    assert cli_main(["--workspace", str(database.path.parent), command, "--job-id", job_id]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1]) == {
        "error": "invalid_configuration",
        "job_id": job_id,
    }
    assert captured.err == ""
    assert _job_evidence_snapshot(database, job_id) == before
    assert processor_builds == []
    assert secret_reads == []


def test_cli_rejects_workspace_outside_trusted_settings_before_database_access_or_mutation(
    lifecycle: tuple[CurationDatabase, BatchRepository, MutableClock, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, _, _, job_id = lifecycle
    before = _job_evidence_snapshot(database, job_id)
    untrusted_target = tmp_path / "separate-workspace"
    untrusted_target.mkdir()
    untrusted_workspace = tmp_path / "untrusted-workspace"
    untrusted_workspace.symlink_to(untrusted_target, target_is_directory=True)
    assert untrusted_workspace.resolve() != database.path.parent.resolve()

    assert cli_main(["--workspace", str(untrusted_workspace), "run", "--job-id", job_id]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1]) == {
        "error": "invalid_configuration",
        "job_id": job_id,
    }
    assert captured.err == ""
    assert _job_evidence_snapshot(database, job_id) == before


@pytest.mark.parametrize(
    ("command", "persisted_job_state"),
    [("run", "queued"), ("resume", "running"), ("resume", "cancel_requested")],
)
def test_cli_validates_all_persisted_histories_before_any_acquisition_mutation(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    persisted_job_state: str,
) -> None:
    database, repository, attempt, old_owner, registry = processor_case
    job_id = str(attempt["job_id"])
    malformed = "[not-json"
    queued = persisted_job_state == "queued"
    with database._write() as connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            """
            UPDATE cosmos_jobs SET state=?, cancel_requested=?, owner=?, lease_expires_at=?
            WHERE id=?
            """,
            (
                persisted_job_state,
                int(persisted_job_state == "cancel_requested"),
                None if queued else old_owner,
                None if queued else "2000-01-01T00:00:00.000000Z",
                job_id,
            ),
        )
        with _disabled_trigger(connection, "cosmos_attempts_history_append_only"):
            connection.execute(
                """
                UPDATE cosmos_attempts SET state=?, lease_owner=?, lease_expires_at=?,
                    http_exchange_history_json=? WHERE id=?
                """,
                (
                    "queued" if queued else "requesting",
                    None if queued else old_owner,
                    None if queued else "2000-01-01T00:00:00.000000Z",
                    malformed,
                    attempt["id"],
                ),
            )
    before = _job_evidence_snapshot(database, job_id)
    transport = FakeTransport()
    processor = CosmosAttemptProcessor(
        repository=repository,
        source_record=registry.records["local/pnp_trash"],
        workspace=database.path.parent,
        source_fps=10,
        base_url="http://cosmos/v1",
        model="cosmos3-nano",
        api_key="test-key",
        sampler=lambda **kwargs: SamplingOutcome.ready(_prepared_sample()),
        transport_factory=lambda: transport,
        sleep=lambda seconds: None,
    )
    monkeypatch.setattr(worker_module, "BatchRepository", lambda database, **kwargs: repository)
    monkeypatch.setattr(worker_module, "build_attempt_processor", lambda **kwargs: processor)

    assert cli_main(["--workspace", str(database.path.parent), command, "--job-id", job_id]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out.splitlines()[-1]) == {
        "error": "invalid_configuration",
        "job_id": job_id,
    }
    assert captured.err == ""
    assert _job_evidence_snapshot(database, job_id) == before
    assert transport.initial_calls == 0


def test_worker_artifact_replay_uses_shared_task7_reconciliation_conflicts(
    processor_case: tuple[CurationDatabase, BatchRepository, dict[str, object], str, SourceRegistry],
) -> None:
    _, repository, attempt, owner, _ = processor_case
    persisted = repository.get_attempt(str(attempt["id"]))
    assert persisted is not None
    dataset_id = int(persisted["dataset_id"])
    relative_path = f"artifacts/cosmos/{attempt['id']}/request.json"
    exact = ArtifactRecord(
        relative_path=relative_path,
        media_type="application/json",
        byte_size=2,
        sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    repository.register_artifact(
        dataset_id=dataset_id,
        attempt_id=str(attempt["id"]),
        kind="request",
        record=exact,
        owner=owner,
        update_attempt_pointer=True,
    )

    with pytest.raises(ArtifactReconciliationConflict):
        repository.register_artifact(
            dataset_id=dataset_id,
            attempt_id=str(attempt["id"]),
            kind="request",
            record=ArtifactRecord(
                relative_path=relative_path,
                media_type="application/json",
                byte_size=3,
                sha256=hashlib.sha256(b"bad").hexdigest(),
            ),
            owner=owner,
            update_attempt_pointer=True,
        )


@dataclass
class _FakeReply:
    status: int = 200
    content: str | None = None
    block: bool = False
    before_response: Callable[[], None] | None = None


class _FakeCosmosServer:
    def __init__(self) -> None:
        self.replies: list[_FakeReply] = []
        self.get_count = 0
        self.post_count = 0
        self.authorization_headers: list[str | None] = []
        self.redirect_url: str | None = None
        self.request_received = threading.Event()
        self.release_response = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner.get_count += 1
                owner.authorization_headers.append(self.headers.get("Authorization"))
                if owner.redirect_url is not None:
                    self.send_response(307)
                    self.send_header("Location", owner.redirect_url)
                    self.end_headers()
                    return
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                self._json(200, {"data": [{"id": "cosmos3-nano"}]})

            def do_POST(self) -> None:
                owner.authorization_headers.append(self.headers.get("Authorization"))
                if owner.redirect_url is not None:
                    self.send_response(307)
                    self.send_header("Location", owner.redirect_url)
                    self.end_headers()
                    return
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                owner.post_count += 1
                reply = owner.replies.pop(0) if owner.replies else _FakeReply(content=_complete_response())
                owner.request_received.set()
                if reply.block:
                    assert owner.release_response.wait(timeout=10)
                if reply.before_response is not None:
                    reply.before_response()
                if reply.status != 200:
                    self._json(reply.status, {"error": "injected"})
                    return
                self._json(
                    200,
                    {
                        "id": f"response-{owner.post_count}",
                        "model": "cosmos3-nano",
                        "created": owner.post_count,
                        "usage": {},
                        "choices": [
                            {
                                "message": {"content": reply.content or _complete_response()},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )

            def log_message(self, format: str, *args: object) -> None:
                return

            def _json(self, status: int, document: object) -> None:
                payload = json.dumps(document, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.release_response.set()
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


def _write_tiny_video(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(70):
            pixels = np.full((16, 16, 3), index % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@dataclass
class _EntrypointCase:
    client: TestClient
    service: BatchService
    database: CurationDatabase
    source: Path
    workspace: Path
    server: _FakeCosmosServer


@pytest.fixture
def entrypoint_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _EntrypointCase:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    video = source / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "total_episodes": 1,
                "total_frames": 70,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {"observation.images.ego_view": {"dtype": "video"}},
            }
        )
    )
    (source / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 70}) + "\n")
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * 70,
                "frame_index": list(range(70)),
                "timestamp": [index / 10 for index in range(70)],
            }
        ),
        source / "data" / "chunk-000" / "episode_000000.parquet",
    )
    _write_tiny_video(video)
    workspace = tmp_path / "workspace"
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
    database = CurationDatabase(workspace / "curation.sqlite3")
    database.initialize()
    review = ReviewService(database=database, source_registry=registry)
    review.open_workspace("local/pnp_trash", actor="integration")
    server = _FakeCosmosServer()
    server.start()
    monkeypatch.setenv("TEST_COSMOS_KEY", "integration-secret")
    _set_trusted_runtime_environment(
        monkeypatch,
        workspace=workspace,
        source=source,
        cosmos_base_url=server.base_url,
        cosmos_endpoint_identity="fake-h100",
    )
    service = BatchService(
        database=database,
        source_registry=registry,
        workspace=workspace,
        cosmos_base_url=server.base_url,
        cosmos_model="cosmos3-nano",
        cosmos_api_key_env="TEST_COSMOS_KEY",
        cosmos_endpoint_identity="fake-h100",
    )
    app = FastAPI()
    app.include_router(build_curation_router(review_service=review, batch_service=service, bearer_token="token"))
    case = _EntrypointCase(
        client=TestClient(app, base_url="http://127.0.0.1"),
        service=service,
        database=database,
        source=source,
        workspace=workspace,
        server=server,
    )
    yield case
    case.client.close()
    server.close()


def _cli_command(workspace: Path, command: str, job_id: str) -> list[str]:
    repository_root = Path(__file__).parents[2]
    return [
        str(repository_root / "backend" / ".venv" / "bin" / "python"),
        str(repository_root / "backend" / "curation_worker.py"),
        "--workspace",
        str(workspace),
        command,
        "--job-id",
        job_id,
    ]


def test_exact_entrypoint_malformed_database_is_json_exit_two_without_traceback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database_path = workspace / "curation.sqlite3"
    malformed = b"not a sqlite database\n"
    database_path.write_bytes(malformed)
    job_id = str(uuid4())

    environment = dict(os.environ)
    environment.update(
        _trusted_runtime_environment(
            workspace=workspace,
            source=tmp_path / "source",
        )
    )
    process = subprocess.run(
        _cli_command(workspace, "run", job_id),
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert process.returncode == 2
    assert process.stderr == ""
    assert process.stdout.splitlines() == [
        json.dumps(
            {"error": "invalid_database", "job_id": job_id},
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    assert database_path.read_bytes() == malformed
    assert list(workspace.iterdir()) == [database_path]


@pytest.mark.parametrize(
    "database_case",
    ["partial_newer", "partial_unknown_v1", "older", "newer", "superficially_compatible"],
)
def test_exact_entrypoint_incompatible_schema_is_read_only_invalid_database(
    tmp_path: Path,
    database_case: str,
) -> None:
    workspace = tmp_path / "workspace"
    database_path = workspace / "curation.sqlite3"
    _create_incompatible_worker_database(database_path, database_case)
    artifact = workspace / "artifacts" / "sentinel.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"immutable-artifact")
    before = _durable_workspace_bytes(workspace)
    job_id = str(uuid4())
    environment = dict(os.environ)
    environment.update(
        _trusted_runtime_environment(
            workspace=workspace,
            source=tmp_path / "source",
        )
    )

    process = subprocess.run(
        _cli_command(workspace, "run", job_id),
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert process.returncode == 2
    assert process.stderr == ""
    assert _status_lines(process) == [{"error": "invalid_database", "job_id": job_id}]
    assert _durable_workspace_bytes(workspace) == before


@pytest.mark.parametrize(
    ("delete_invariant_trigger", "expected_error"),
    [(False, "invalid_job_state"), (True, "invalid_database")],
)
def test_exact_entrypoint_reads_the_committed_live_wal_schema(
    tmp_path: Path,
    delete_invariant_trigger: bool,
    expected_error: str,
) -> None:
    workspace = tmp_path / "workspace"
    database_path = workspace / "curation.sqlite3"
    artifact = workspace / "artifacts" / "sentinel.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"immutable-artifact")
    job_id = str(uuid4())
    environment = dict(os.environ)
    environment.update(
        _trusted_runtime_environment(
            workspace=workspace,
            source=tmp_path / "source",
        )
    )

    with _live_wal_schema(database_path, delete_invariant_trigger=delete_invariant_trigger):
        before_files = _database_file_snapshot(database_path)
        before_tree = _workspace_tree_shape(workspace)
        process = subprocess.run(
            _cli_command(workspace, "run", job_id),
            cwd=Path(__file__).parents[2],
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

        assert process.returncode == 2
        assert process.stderr == ""
        assert _status_lines(process) == [{"error": expected_error, "job_id": job_id}]
        after_files = _database_file_snapshot(database_path)
        if delete_invariant_trigger:
            assert after_files.keys() == before_files.keys()
            assert {name: value for name, value in after_files.items() if not name.endswith("-shm")} == {
                name: value for name, value in before_files.items() if not name.endswith("-shm")
            }
            assert _workspace_tree_shape(workspace) == before_tree
        assert artifact.read_bytes() == b"immutable-artifact"


def _run_entrypoint(
    case: _EntrypointCase,
    command: str,
    job_id: str,
    *,
    include_key: bool = True,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    if include_key:
        environment["TEST_COSMOS_KEY"] = "integration-secret"
    else:
        environment.pop("TEST_COSMOS_KEY", None)
    environment.update(extra_environment or {})
    return subprocess.run(
        _cli_command(case.workspace, command, job_id),
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _status_lines(process: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]


def _start_via_api(case: _EntrypointCase) -> dict[str, object]:
    response = case.client.post(
        "/api/curation/batches",
        headers={"Authorization": "Bearer token"},
        json={"dataset_alias": "local/pnp_trash", "episode_indices": [0]},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize(
    "corruption",
    [
        "job_configuration_run",
        "job_configuration_empty_run",
        "job_configuration_wrong_type_resume_running",
        "job_configuration_unknown_field_resume_cancel_requested",
        "job_configuration_missing_nested_field_run",
        "attempt_history_run",
        "attempt_history_resume_running",
        "attempt_history_resume_cancel_requested",
    ],
)
def test_exact_entrypoint_persisted_json_corruption_is_sanitized_exit_two(
    entrypoint_case: _EntrypointCase,
    corruption: str,
) -> None:
    case = entrypoint_case
    job = _start_via_api(case)
    job_id = str(job["job_id"])
    malformed = "{legacy-corruption"
    command = "resume" if "_resume_" in corruption else "run"
    with case.database._write() as connection:
        if command == "resume":
            owner = str(uuid4())
            job_state = "cancel_requested" if corruption.endswith("cancel_requested") else "running"
            connection.execute(
                """
                UPDATE cosmos_jobs SET state=?, cancel_requested=?, owner=?, lease_expires_at=?
                WHERE id=?
                """,
                (
                    job_state,
                    int(job_state == "cancel_requested"),
                    owner,
                    "2000-01-01T00:00:00.000000Z",
                    job_id,
                ),
            )
            connection.execute(
                """
                UPDATE cosmos_attempts SET state='requesting', lease_owner=?, lease_expires_at=?
                WHERE job_id=?
                """,
                (owner, "2000-01-01T00:00:00.000000Z", job_id),
            )
        if corruption.startswith("job_configuration"):
            if corruption == "job_configuration_run":
                configuration_json = malformed
            else:
                invalid_case = next(
                    candidate
                    for candidate in (
                        "empty",
                        "wrong_type",
                        "unknown_field",
                        "missing_nested_field",
                    )
                    if candidate in corruption
                )
                invalid = _invalid_job_configuration(invalid_case, dict(job["configuration"]))
                configuration_json = json.dumps(invalid, sort_keys=True, separators=(",", ":"))
            with _disabled_trigger(connection, "cosmos_jobs_configuration_immutable"):
                connection.execute(
                    "UPDATE cosmos_jobs SET configuration_json=? WHERE id=?",
                    (configuration_json, job_id),
                )
        else:
            connection.execute("PRAGMA ignore_check_constraints=ON")
            with _disabled_trigger(connection, "cosmos_attempts_history_append_only"):
                connection.execute(
                    "UPDATE cosmos_attempts SET http_exchange_history_json=? WHERE job_id=?",
                    (malformed, job_id),
                )
    before = _job_evidence_snapshot(case.database, job_id)

    process = _run_entrypoint(case, command, job_id)

    assert process.returncode == 2
    assert process.stderr == ""
    assert _status_lines(process)[-1] == {"error": "invalid_configuration", "job_id": job_id}
    assert case.server.post_count == 0
    assert _job_evidence_snapshot(case.database, job_id) == before


@pytest.mark.parametrize(
    "authority_field",
    ["source_path", "base_url", "model", "api_key_env", "endpoint_identity"],
)
@pytest.mark.parametrize(
    ("command", "persisted_job_state"),
    [("run", "queued"), ("resume", "running"), ("resume", "cancel_requested")],
)
def test_exact_entrypoint_rejects_persisted_authority_mismatch_before_side_effects(
    entrypoint_case: _EntrypointCase,
    authority_field: str,
    command: str,
    persisted_job_state: str,
) -> None:
    case = entrypoint_case
    job = _start_via_api(case)
    job_id = str(job["job_id"])
    queued = persisted_job_state == "queued"
    tampered = _tamper_authority_configuration(
        authority_field,
        dict(job["configuration"]),
        attacker_source=case.workspace.parent / "attacker-source",
    )
    with case.database._write() as connection:
        with _disabled_trigger(connection, "cosmos_jobs_configuration_immutable"):
            connection.execute(
                """
                UPDATE cosmos_jobs SET state=?, cancel_requested=?, owner=?, lease_expires_at=?,
                    configuration_json=? WHERE id=?
                """,
                (
                    persisted_job_state,
                    int(persisted_job_state == "cancel_requested"),
                    None if queued else str(uuid4()),
                    None if queued else "2000-01-01T00:00:00.000000Z",
                    json.dumps(tampered, sort_keys=True, separators=(",", ":")),
                    job_id,
                ),
            )
        connection.execute(
            """
            UPDATE cosmos_attempts SET state=?, lease_owner=(SELECT owner FROM cosmos_jobs WHERE id=?),
                lease_expires_at=? WHERE job_id=?
            """,
            (
                "queued" if queued else "requesting",
                job_id,
                None if queued else "2000-01-01T00:00:00.000000Z",
                job_id,
            ),
        )
    before = _job_evidence_snapshot(case.database, job_id)
    process = _run_entrypoint(
        case,
        command,
        job_id,
        extra_environment={"EXFILTRATE_SENTINEL_SECRET": "must-never-be-disclosed"},
    )

    assert process.returncode == 2
    assert process.stderr == ""
    assert _status_lines(process)[-1] == {"error": "invalid_configuration", "job_id": job_id}
    assert case.server.post_count == 0
    assert _job_evidence_snapshot(case.database, job_id) == before


def test_exact_authority_tamper_never_sends_sentinel_authorization_to_redirect_hosts(
    entrypoint_case: _EntrypointCase,
) -> None:
    case = entrypoint_case
    redirect_target = _FakeCosmosServer()
    redirector = _FakeCosmosServer()
    redirect_target.start()
    redirector.redirect_url = redirect_target.base_url + "/chat/completions"
    redirector.start()
    try:
        job = _start_via_api(case)
        job_id = str(job["job_id"])
        tampered = _tamper_authority_configuration(
            "base_url",
            dict(job["configuration"]),
            attacker_source=case.workspace.parent / "attacker-source",
            attacker_base_url=redirector.base_url,
        )
        tampered["cosmos"]["api_key_env"] = "EXFILTRATE_SENTINEL_SECRET"
        with case.database._write() as connection:
            with _disabled_trigger(connection, "cosmos_jobs_configuration_immutable"):
                connection.execute(
                    "UPDATE cosmos_jobs SET configuration_json=? WHERE id=?",
                    (json.dumps(tampered, sort_keys=True, separators=(",", ":")), job_id),
                )
        before = _job_evidence_snapshot(case.database, job_id)

        process = _run_entrypoint(
            case,
            "run",
            job_id,
            extra_environment={"EXFILTRATE_SENTINEL_SECRET": "sentinel-bearer-must-never-leave"},
        )

        assert process.returncode == 2
        assert process.stderr == ""
        assert _status_lines(process)[-1] == {"error": "invalid_configuration", "job_id": job_id}
        assert redirector.get_count == 0
        assert redirector.post_count == 0
        assert redirector.authorization_headers == []
        assert redirect_target.get_count == 0
        assert redirect_target.post_count == 0
        assert redirect_target.authorization_headers == []
        assert _job_evidence_snapshot(case.database, job_id) == before
    finally:
        redirector.close()
        redirect_target.close()


def test_exact_entrypoint_contact_sheet_conflict_is_json_exit_one_and_clears_leases(
    entrypoint_case: _EntrypointCase,
) -> None:
    case = entrypoint_case
    completed_job = _start_via_api(case)
    completed = _run_entrypoint(case, "run", str(completed_job["job_id"]))
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    dataset = case.database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    identity = ContactSheetDatasetIdentity(
        dataset_id=dataset["id"],
        dataset_alias=dataset["alias"],
        source_manifest_sha256=dataset["source_manifest_sha256"],
    )
    with case.database.open_connection() as connection:
        proposal_id = connection.execute(
            "SELECT id FROM cosmos_proposals ORDER BY created_at, id LIMIT 1"
        ).fetchone()["id"]
    sheet_path = proposal_contact_sheet_path(identity, proposal_id)
    receipt = case.workspace / receipt_path(sheet_path)
    replacement = receipt.with_name("replacement.json")
    replacement.write_text("{}", encoding="utf-8")
    os.replace(replacement, receipt)

    retry = case.client.post(
        f"/api/curation/batches/{completed_job['job_id']}/retry",
        headers={"Authorization": "Bearer token"},
        json={"episode_indices": [0]},
    ).json()
    failed = _run_entrypoint(case, "run", str(retry["job_id"]))

    assert failed.returncode == 1
    assert failed.stderr == ""
    assert _status_lines(failed)[-1] == {
        "error": "contact_sheet_conflict",
        "job_id": retry["job_id"],
        "state": "failed",
    }
    assert case.server.post_count == 1
    with case.database.open_connection() as connection:
        job = connection.execute(
            "SELECT state, owner, lease_expires_at FROM cosmos_jobs WHERE id=?",
            (retry["job_id"],),
        ).fetchone()
        attempts = connection.execute(
            "SELECT lease_owner, lease_expires_at FROM cosmos_attempts WHERE job_id=?",
            (retry["job_id"],),
        ).fetchall()
    assert dict(job) == {"state": "failed", "owner": None, "lease_expires_at": None}
    assert all(row["lease_owner"] is None and row["lease_expires_at"] is None for row in attempts)


def test_exact_entrypoint_run_retry_and_all_nonsignal_exit_codes(entrypoint_case: _EntrypointCase) -> None:
    case = entrypoint_case
    parent = _start_via_api(case)
    assert parent["state"] == "queued"
    assert case.server.post_count == 0

    completed = _run_entrypoint(case, "run", str(parent["job_id"]))
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert _status_lines(completed)[-1]["state"] == "completed"
    assert case.server.post_count == 1
    assert case.service.status(str(parent["job_id"]))["state"] == "completed"

    repeated_run = _run_entrypoint(case, "run", str(parent["job_id"]))
    assert repeated_run.returncode == 2
    assert _status_lines(repeated_run)[-1]["error"] == "invalid_job_state"

    retry_response = case.client.post(
        f"/api/curation/batches/{parent['job_id']}/retry",
        headers={"Authorization": "Bearer token"},
        json={"episode_indices": [0]},
    )
    assert retry_response.status_code == 201
    child = retry_response.json()
    assert child["job_id"] != parent["job_id"]
    child_completed = _run_entrypoint(case, "run", str(child["job_id"]))
    assert child_completed.returncode == 0, (child_completed.stdout, child_completed.stderr)
    assert case.server.post_count == 2

    conflict_response = case.client.post(
        f"/api/curation/batches/{child['job_id']}/retry",
        headers={"Authorization": "Bearer token"},
        json={"episode_indices": [0]},
    )
    conflict_job = conflict_response.json()
    owner = str(uuid4())
    case.service.repository.start_job(str(conflict_job["job_id"]), owner=owner)
    conflict = _run_entrypoint(case, "resume", str(conflict_job["job_id"]))
    assert conflict.returncode == 3
    assert _status_lines(conflict)[-1]["error"] == "live_lease_conflict"
    status, _ = case.service.cancel(str(conflict_job["job_id"]))
    assert status == 202
    with case.database._write() as connection:
        connection.execute(
            "UPDATE cosmos_jobs SET lease_expires_at='2000-01-01T00:00:00.000000Z' WHERE id=?",
            (conflict_job["job_id"],),
        )
    cancelled = _run_entrypoint(case, "resume", str(conflict_job["job_id"]), include_key=False)
    assert cancelled.returncode == 0
    assert _status_lines(cancelled)[-1]["state"] == "cancelled"

    failing = case.client.post(
        f"/api/curation/batches/{child['job_id']}/retry",
        headers={"Authorization": "Bearer token"},
        json={"episode_indices": [0]},
    ).json()
    failing_attempt = case.service.repository.list_attempts(str(failing["job_id"]))[0]
    response_path = case.workspace / "artifacts" / "cosmos" / str(failing_attempt["id"]) / "response.txt"

    def inject_runtime_artifact_conflict() -> None:
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text("conflicting response artifact", encoding="utf-8")

    case.server.replies.append(_FakeReply(before_response=inject_runtime_artifact_conflict))
    failed = _run_entrypoint(case, "run", str(failing["job_id"]))
    assert failed.returncode == 1
    assert failed.stderr == ""
    assert _status_lines(failed)[-1]["event"] == "worker_failed"
    assert case.service.status(str(failing["job_id"]))["state"] == "failed"
    assert case.server.post_count == 3
    case.database.validate_worker_compatibility()


@pytest.mark.parametrize("reply", [_FakeReply(status=500, block=True), _FakeReply(content="{invalid", block=True)])
def test_exact_entrypoint_signal_persists_once_then_expired_resume_finishes(
    entrypoint_case: _EntrypointCase, reply: _FakeReply
) -> None:
    case = entrypoint_case
    job = _start_via_api(case)
    case.server.replies.append(reply)
    process = subprocess.Popen(
        _cli_command(case.workspace, "run", str(job["job_id"])),
        cwd=Path(__file__).parents[2],
        env={**os.environ, "TEST_COSMOS_KEY": "integration-secret"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert case.server.request_received.wait(timeout=15)
    process.send_signal(signal.SIGTERM)
    case.server.release_response.set()
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 130, (stdout, stderr)
    lines = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    assert lines[-1]["event"] == "worker_interrupted"
    assert case.server.post_count == 1
    persisted = case.service.repository.list_attempts(str(job["job_id"]))[0]
    assert len(json.loads(persisted["http_exchange_history_json"])) == 1
    assert persisted["state"] == "requesting"

    with case.database._write() as connection:
        connection.execute(
            "UPDATE cosmos_jobs SET lease_expires_at='2000-01-01T00:00:00.000000Z' WHERE id=?",
            (job["job_id"],),
        )
        connection.execute(
            "UPDATE cosmos_attempts SET lease_expires_at='2000-01-01T00:00:00.000000Z' WHERE job_id=?",
            (job["job_id"],),
        )
    case.server.request_received.clear()
    case.server.release_response.clear()
    resumed = _run_entrypoint(case, "resume", str(job["job_id"]))
    assert resumed.returncode == 0, (resumed.stdout, resumed.stderr)
    assert _status_lines(resumed)[-1]["state"] in {"completed", "completed_with_failures"}
    assert case.server.post_count == 2


def test_exact_cancel_only_resume_ignores_missing_key_and_changed_source(entrypoint_case: _EntrypointCase) -> None:
    case = entrypoint_case
    job = _start_via_api(case)
    owner = str(uuid4())
    case.service.repository.start_job(str(job["job_id"]), owner=owner)
    status, _ = case.service.cancel(str(job["job_id"]))
    assert status == 202
    with case.database._write() as connection:
        connection.execute(
            "UPDATE cosmos_jobs SET lease_expires_at='2000-01-01T00:00:00.000000Z' WHERE id=?",
            (job["job_id"],),
        )
    (case.source / "meta" / "info.json").write_text(json.dumps({"changed": True}))
    calls_before = case.server.post_count
    resumed = _run_entrypoint(case, "resume", str(job["job_id"]), include_key=False)
    assert resumed.returncode == 0, (resumed.stdout, resumed.stderr)
    assert _status_lines(resumed)[-1]["state"] == "cancelled"
    assert case.server.post_count == calls_before
