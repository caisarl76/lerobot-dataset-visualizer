from __future__ import annotations

import multiprocessing
from pathlib import Path
from queue import Empty
import sqlite3
import time
from typing import Any

import curation.db as db_module
from curation.db import (
    CurationDatabase,
    OptimisticConflict,
    RetryableDatabaseError,
    StateTransitionConflict,
    canonical_json,
)
from curation.models import ExportState, JobState, ReviewState
import pytest


def _db(tmp_path: Path) -> CurationDatabase:
    database = CurationDatabase(tmp_path / "curation.sqlite3")
    database.initialize()
    return database


def _dataset(database: CurationDatabase, alias: str = "local/pnp_trash") -> int:
    return database.register_dataset(
        alias=alias,
        source_path="/immutable/source",
        source_manifest_sha256="a" * 64,
        prompt_template_version="pnp-trash-prompts-v1",
        prompt_template_sha256="b" * 64,
    )["id"]


def _hold_immediate_lock(path: str, ready: Any, release: Any) -> None:
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")
        ready.set()
        release.wait(10)
        connection.execute("COMMIT")
    finally:
        connection.close()


def _concurrent_register(path: str, alias: str, attempting: Any, result: Any) -> None:
    database = CurationDatabase(Path(path))
    attempting.set()
    started = time.monotonic()
    try:
        database.register_dataset(
            alias=alias,
            source_path=f"/immutable/{alias}",
            source_manifest_sha256="a" * 64,
            prompt_template_version="pnp-trash-prompts-v1",
            prompt_template_sha256="b" * 64,
        )
    except Exception as error:  # pragma: no cover - returned to the parent process
        result.put(("error", type(error).__name__, time.monotonic() - started))
    else:  # pragma: no cover - returned to the parent process
        result.put(("ok", None, time.monotonic() - started))


def _snapshot_from_order(path: str, order: tuple[int, ...], result: Any) -> None:
    database = CurationDatabase(Path(path))
    database.initialize()
    dataset_id = _dataset(database)
    for source_episode_index in order:
        episode = database.create_episode(
            dataset_id=dataset_id,
            source_episode_index=source_episode_index,
            source_length=100,
        )
        database.update_episode(
            dataset_id=dataset_id,
            source_episode_index=source_episode_index,
            expected_revision=episode["revision"],
            changes={
                "review_state": ReviewState.APPROVED_REJECT,
                "reviewer": "reviewer",
                "approval_revision": 1,
                "approved_at": "2026-08-21T00:00:00Z",
            },
            actor="reviewer",
        )
    result.put(database.approval_snapshot(dataset_id=dataset_id)["sha256"])


def _competing_episode_update(
    path: str,
    dataset_id: int,
    source_episode_index: int,
    object_name: str,
    ready: Any,
    release: Any,
    result: Any,
) -> None:
    database = CurationDatabase(Path(path))
    ready.set()
    release.wait(10)
    try:
        episode = database.update_episode(
            dataset_id=dataset_id,
            source_episode_index=source_episode_index,
            expected_revision=0,
            changes={"review_state": ReviewState.DRAFT, "object_name": object_name},
            actor=f"curator-{object_name}",
        )
    except OptimisticConflict as error:  # pragma: no cover - returned to parent process
        result.put(("conflict", error.current_episode["object_name"], error.current_episode["revision"]))
    except Exception as error:  # pragma: no cover - returned to parent process
        result.put(("error", type(error).__name__, str(error)))
    else:  # pragma: no cover - returned to parent process
        result.put(("committed", episode["object_name"], episode["revision"]))


def test_independent_connections_apply_required_pragmas_and_idempotent_migration(tmp_path: Path) -> None:
    path = tmp_path / "curation.sqlite3"
    database = CurationDatabase(path)
    database.initialize()
    database.initialize()

    for connection in (database.open_connection(), database.open_connection()):
        try:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
            assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
            table_names = {
                row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert {
                "datasets",
                "episodes",
                "cosmos_jobs",
                "cosmos_attempts",
                "cosmos_proposals",
                "artifacts",
                "audit_events",
                "exports",
                "export_episodes",
            }.issubset(table_names)
            episode_columns = {row["name"] for row in connection.execute("PRAGMA table_info(episodes)")}
            assert {f"step_{step}_start_frame" for step in range(2, 8)}.issubset(episode_columns)
        finally:
            connection.close()


def test_failed_migration_rolls_back_schema_and_user_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = CurationDatabase(tmp_path / "curation.sqlite3")
    monkeypatch.setattr(
        db_module,
        "_migration_v1_statements",
        lambda: ("CREATE TABLE must_not_survive (id INTEGER PRIMARY KEY)", "this is not sql"),
    )

    with pytest.raises(sqlite3.OperationalError):
        database.initialize()

    with database.open_connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='must_not_survive'"
            ).fetchone()
            is None
        )


def test_read_lock_exhaustion_is_retryable_and_closes_the_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _db(tmp_path)

    class LockedConnection:
        closed = False

        def execute(self, _statement: str, _parameters: object = ()) -> None:
            raise sqlite3.OperationalError("database is locked")

        def close(self) -> None:
            self.closed = True

    connection = LockedConnection()
    monkeypatch.setattr(database, "open_connection", lambda: connection)
    with pytest.raises(RetryableDatabaseError):
        database.get_dataset(alias="local/pnp_trash")
    assert connection.closed


def test_read_connection_open_lock_exhaustion_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _db(tmp_path)

    def locked_open_connection() -> None:
        raise sqlite3.OperationalError("database is busy")

    monkeypatch.setattr(database, "open_connection", locked_open_connection)
    with pytest.raises(RetryableDatabaseError):
        database.get_dataset(alias="local/pnp_trash")


def test_v1_schema_enforces_foreign_keys_states_nonnegative_values_and_uniqueness(tmp_path: Path) -> None:
    database = _db(tmp_path)
    dataset_id = _dataset(database)
    episode = database.create_episode(dataset_id=dataset_id, source_episode_index=0, source_length=10)
    job_id = database.create_cosmos_job(dataset_id=dataset_id, configuration={})["id"]

    with database.open_connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO episodes(
                    dataset_id, source_episode_index, source_length, review_state, revision, created_at, updated_at
                ) VALUES (999, 99, 10, 'pending', 0, '2026-08-21T00:00:00Z', '2026-08-21T00:00:00Z')
                """
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO episodes(
                    dataset_id, source_episode_index, source_length, review_state, revision, created_at, updated_at
                ) VALUES (?, 0, 10, 'pending', 0, '2026-08-21T00:00:00Z', '2026-08-21T00:00:00Z')
                """,
                (dataset_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO episodes(
                    dataset_id, source_episode_index, source_length, review_state, revision, created_at, updated_at
                ) VALUES (?, 1, -1, 'pending', 0, '2026-08-21T00:00:00Z', '2026-08-21T00:00:00Z')
                """,
                (dataset_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE episodes SET review_state='not-a-review-state' WHERE id=?", (episode["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE cosmos_jobs SET state='not-a-job-state' WHERE id=?", (job_id,))
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) VALUES ('attempt-1', ?, 0, 0, 'queued', '2026-08-21T00:00:00Z', '2026-08-21T00:00:00Z')
            """,
            (job_id,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO cosmos_attempts(
                    id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
                ) VALUES ('attempt-2', ?, 0, 0, 'queued', '2026-08-21T00:00:00Z', '2026-08-21T00:00:00Z')
                """,
                (job_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE cosmos_attempts SET state='not-an-attempt-state' WHERE id='attempt-1'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO cosmos_proposals(
                    id, attempt_id, model_response_json, validation_warnings_json, state, created_at
                ) VALUES ('proposal-1', 'attempt-1', '{}', '[]', 'not-a-proposal-state', '2026-08-21T00:00:00Z')
                """
            )


def test_episode_updates_are_optimistic_and_report_the_current_http_ready_payload(tmp_path: Path) -> None:
    database = _db(tmp_path)
    dataset_id = _dataset(database)
    episode = database.create_episode(dataset_id=dataset_id, source_episode_index=4, source_length=120)

    updated = database.update_episode(
        dataset_id=dataset_id,
        source_episode_index=4,
        expected_revision=episode["revision"],
        changes={"review_state": ReviewState.DRAFT, "object_name": "can"},
        actor="curator-a",
    )
    assert updated["revision"] == 1
    with pytest.raises(OptimisticConflict) as raised:
        database.update_episode(
            dataset_id=dataset_id,
            source_episode_index=4,
            expected_revision=0,
            changes={"object_name": "cup"},
            actor="curator-b",
        )
    assert raised.value.status_code == 409
    assert raised.value.payload == {"error": "revision_conflict", "episode": updated}
    assert database.get_episode(dataset_id=dataset_id, source_episode_index=4) == updated


def test_competing_process_updates_with_the_same_revision_yield_one_commit_and_one_conflict(
    tmp_path: Path,
) -> None:
    database = _db(tmp_path)
    dataset_id = _dataset(database)
    database.create_episode(dataset_id=dataset_id, source_episode_index=4, source_length=120)
    context = multiprocessing.get_context("spawn")
    first_ready = context.Event()
    second_ready = context.Event()
    release = context.Event()
    result = context.Queue()
    first = context.Process(
        target=_competing_episode_update,
        args=(str(database.path), dataset_id, 4, "can", first_ready, release, result),
    )
    second = context.Process(
        target=_competing_episode_update,
        args=(str(database.path), dataset_id, 4, "cup", second_ready, release, result),
    )
    try:
        first.start()
        second.start()
        assert first_ready.wait(10)
        assert second_ready.wait(10)
        release.set()
        first.join(15)
        second.join(15)
        assert first.exitcode == 0
        assert second.exitcode == 0
        outcomes = [result.get(timeout=2), result.get(timeout=2)]
        assert sorted(outcome[0] for outcome in outcomes) == ["committed", "conflict"]
        winner = next(outcome for outcome in outcomes if outcome[0] == "committed")
        conflict = next(outcome for outcome in outcomes if outcome[0] == "conflict")
        assert winner[2] == conflict[2] == 1
        assert winner[1] == conflict[1]
        assert database.get_episode(dataset_id=dataset_id, source_episode_index=4)["object_name"] == winner[1]
    finally:
        release.set()
        for process in (first, second):
            if process.pid is None:
                continue
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(2)
        result.close()
        result.join_thread()


def test_partial_unique_indexes_allow_only_one_active_job_and_export_per_dataset(tmp_path: Path) -> None:
    database = _db(tmp_path)
    dataset_id = _dataset(database)
    older_job = database.create_cosmos_job(dataset_id=dataset_id, configuration={})
    with pytest.raises(sqlite3.IntegrityError):
        database.create_cosmos_job(dataset_id=dataset_id, configuration={})
    database.set_cosmos_job_state(job_id=older_job["id"], expected_state=JobState.QUEUED, state=JobState.COMPLETED)
    newer_job = database.create_cosmos_job(dataset_id=dataset_id, configuration={})
    database.set_cosmos_job_state(job_id=older_job["id"], expected_state=JobState.COMPLETED, state=JobState.FAILED)
    with pytest.raises(StateTransitionConflict):
        database.set_cosmos_job_state(
            job_id=older_job["id"], expected_state=JobState.QUEUED, state=JobState.RUNNING
        )
    with database.open_connection() as connection:
        assert (
            connection.execute("SELECT state FROM cosmos_jobs WHERE id=?", (newer_job["id"],)).fetchone()[0]
            == "queued"
        )

    assert not hasattr(database, "create_export")
    older_export = database.create_export_snapshot(
        dataset_id=dataset_id, staging_path="/tmp/staging-a", final_path="/tmp/final-a"
    )
    with pytest.raises(sqlite3.IntegrityError):
        database.create_export_snapshot(
            dataset_id=dataset_id, staging_path="/tmp/staging-b", final_path="/tmp/final-b"
        )
    database.set_export_state(
        export_id=older_export["id"], expected_state=ExportState.QUEUED, state=ExportState.FAILED
    )
    newer_export = database.create_export_snapshot(
        dataset_id=dataset_id, staging_path="/tmp/staging-b", final_path="/tmp/final-b"
    )
    database.set_export_state(
        export_id=older_export["id"], expected_state=ExportState.FAILED, state=ExportState.FAILED
    )
    with pytest.raises(StateTransitionConflict):
        database.set_export_state(
            export_id=older_export["id"], expected_state=ExportState.QUEUED, state=ExportState.BUILDING
        )
    with database.open_connection() as connection:
        assert (
            connection.execute("SELECT state FROM exports WHERE id=?", (newer_export["id"],)).fetchone()[0]
            == "queued"
        )


def test_audit_events_are_append_only_and_artifacts_are_workspace_relative(tmp_path: Path) -> None:
    database = _db(tmp_path)
    dataset_id = _dataset(database)
    database.create_episode(dataset_id=dataset_id, source_episode_index=0, source_length=10)
    job_id = database.create_cosmos_job(dataset_id=dataset_id, configuration={})["id"]
    event = database.append_audit_event(dataset_id=dataset_id, actor="curator", operation="saved_draft")

    with database.open_connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE audit_events SET actor='other' WHERE id=?", (event["id"],))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM audit_events WHERE id=?", (event["id"],))

    for unsafe in ("/absolute/request.json", "../escape.json", "cosmos/../escape.json", "cosmos\\escape.json"):
        with pytest.raises(ValueError):
            database.create_artifact(
                kind="request",
                relative_path=unsafe,
                media_type="application/json",
                byte_size=0,
                sha256="e" * 64,
            )

    with database.open_connection() as connection:
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) VALUES ('artifact-owner', ?, 0, 0, 'queued', '2026-08-21T00:00:00Z', '2026-08-21T00:00:00Z')
            """,
            (job_id,),
        )
        for unsafe in ("/absolute/request.json", "../escape.json", "cosmos/../escape.json"):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        id, attempt_id, kind, relative_path, media_type, byte_size, sha256, created_at
                    ) VALUES (?, 'artifact-owner', 'request', ?, 'application/json', 0, ?, '2026-08-21T00:00:00Z')
                    """,
                    (f"artifact-{unsafe}", unsafe, "e" * 64),
                )


def test_schema_rejects_cross_dataset_references_and_wrong_artifact_ownership(tmp_path: Path) -> None:
    database = _db(tmp_path)
    first_dataset = _dataset(database, "local/first")
    second_dataset = _dataset(database, "local/second")
    first_episode = database.create_episode(dataset_id=first_dataset, source_episode_index=0, source_length=10)
    second_episode = database.create_episode(dataset_id=second_dataset, source_episode_index=1, source_length=10)
    first_job = database.create_cosmos_job(dataset_id=first_dataset, configuration={})
    second_job = database.create_cosmos_job(dataset_id=second_dataset, configuration={})
    first_export = database.create_export_snapshot(
        dataset_id=first_dataset, staging_path="/tmp/first-staging", final_path="/tmp/first-final"
    )
    second_export = database.create_export_snapshot(
        dataset_id=second_dataset, staging_path="/tmp/second-staging", final_path="/tmp/second-final"
    )
    timestamp = "2026-08-21T00:00:00Z"

    with database.open_connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO cosmos_jobs(
                    id, dataset_id, parent_job_id, configuration_json, state, total_attempts,
                    succeeded_attempts, manual_only_attempts, cancel_requested, created_at, updated_at
                ) VALUES ('cross-parent', ?, ?, '{}', 'queued', 0, 0, 0, 0, ?, ?)
                """,
                (second_dataset, first_job["id"], timestamp, timestamp),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO cosmos_attempts(
                    id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
                ) VALUES ('cross-attempt', ?, 1, 0, 'queued', ?, ?)
                """,
                (first_job["id"], timestamp, timestamp),
            )
        connection.execute(
            """
            INSERT INTO cosmos_jobs(
                id, dataset_id, parent_job_id, configuration_json, state, total_attempts,
                succeeded_attempts, manual_only_attempts, cancel_requested, created_at, updated_at
            ) VALUES ('same-parent', ?, ?, '{}', 'failed', 0, 0, 0, 0, ?, ?)
            """,
            (first_dataset, first_job["id"], timestamp, timestamp),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE cosmos_jobs SET dataset_id=? WHERE id=?", (second_dataset, first_job["id"]))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE cosmos_jobs SET configuration_json='{\"changed\":true}' WHERE id=?", (first_job["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE cosmos_jobs SET parent_job_id=NULL WHERE id='same-parent'")
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) VALUES ('first-attempt', ?, 0, 0, 'queued', ?, ?)
            """,
            (first_job["id"], timestamp, timestamp),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE episodes SET source_episode_index=2 WHERE id=?", (first_episode["id"],))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM episodes WHERE id=?", (first_episode["id"],))
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) VALUES ('second-attempt', ?, 0, 1, 'queued', ?, ?)
            """,
            (first_job["id"], timestamp, timestamp),
        )
        for identifier, attempt_id, kind in (
            ("first-request", "first-attempt", "request"),
            ("first-response", "first-attempt", "response"),
            ("second-request", "second-attempt", "request"),
            ("first-wrong-kind", "first-attempt", "response"),
        ):
            connection.execute(
                """
                INSERT INTO artifacts(
                    id, attempt_id, kind, relative_path, media_type, byte_size, sha256, created_at
                ) VALUES (?, ?, ?, ?, 'application/json', 0, ?, ?)
                """,
                (identifier, attempt_id, kind, f"cosmos/{identifier}.json", "e" * 64, timestamp),
            )
        connection.execute(
            """
            UPDATE cosmos_attempts
            SET request_artifact_id='first-request', response_artifact_id='first-response'
            WHERE id='first-attempt'
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE cosmos_attempts SET request_artifact_id='second-request' WHERE id='first-attempt'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE cosmos_attempts SET request_artifact_id='first-wrong-kind' WHERE id='first-attempt'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE artifacts SET kind='request' WHERE id='first-response'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM artifacts WHERE id='first-response'")

        for identifier, export_id, kind in (
            ("first-structural", first_export["id"], "structural_report"),
            ("first-stats", first_export["id"], "gr00t_stats_report"),
            ("first-loader", first_export["id"], "gr00t_loader_report"),
            ("first-final", first_export["id"], "final_consistency_report"),
            ("second-structural", second_export["id"], "structural_report"),
            ("first-wrong-export-kind", first_export["id"], "gr00t_stats_report"),
        ):
            connection.execute(
                """
                INSERT INTO artifacts(
                    id, export_id, kind, relative_path, media_type, byte_size, sha256, created_at
                ) VALUES (?, ?, ?, ?, 'application/json', 0, ?, ?)
                """,
                (identifier, export_id, kind, f"exports/{identifier}.json", "f" * 64, timestamp),
            )
        connection.execute(
            """
            UPDATE exports
            SET structural_artifact_id='first-structural', gr00t_stats_artifact_id='first-stats',
                gr00t_loader_artifact_id='first-loader', final_consistency_artifact_id='first-final'
            WHERE id=?
            """,
            (first_export["id"],),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE exports SET structural_artifact_id='second-structural' WHERE id=?", (first_export["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE exports SET structural_artifact_id='first-wrong-export-kind' WHERE id=?",
                (first_export["id"],),
            )

        connection.execute(
            """
            INSERT INTO audit_events(
                id, dataset_id, actor, operation, episode_id, job_id, export_id, details_json, created_at
            ) VALUES ('valid-audit', ?, 'curator', 'valid_reference', ?, ?, ?, '{}', ?)
            """,
            (first_dataset, first_episode["id"], first_job["id"], first_export["id"], timestamp),
        )

        for identifier, column, foreign_id in (
            ("cross-audit-episode", "episode_id", second_episode["id"]),
            ("cross-audit-job", "job_id", second_job["id"]),
            ("cross-audit-export", "export_id", second_export["id"]),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"""
                    INSERT INTO audit_events(id, dataset_id, actor, operation, {column}, details_json, created_at)
                    VALUES (?, ?, 'curator', 'cross_reference', ?, '{{}}', ?)
                    """,
                    (identifier, first_dataset, foreign_id, timestamp),
                )

    assert first_episode["dataset_id"] == first_dataset


def test_approval_snapshot_uses_canonical_json_and_is_independent_of_insert_order(tmp_path: Path) -> None:
    first = _db(tmp_path / "first")
    second = _db(tmp_path / "second")
    first_dataset = _dataset(first)
    second_dataset = _dataset(second)
    for database, dataset_id, order in ((first, first_dataset, (9, 2)), (second, second_dataset, (2, 9))):
        for source_episode_index in order:
            record = database.create_episode(
                dataset_id=dataset_id,
                source_episode_index=source_episode_index,
                source_length=100,
            )
            database.update_episode(
                dataset_id=dataset_id,
                source_episode_index=source_episode_index,
                expected_revision=record["revision"],
                changes={
                    "review_state": ReviewState.APPROVED_REJECT,
                    "reviewer": "reviewer",
                    "approval_revision": 1,
                    "approved_at": "2026-08-21T00:00:00Z",
                },
                actor="reviewer",
            )

    assert canonical_json({"z": "한글", "a": [2, 1]}) == '{"a":[2,1],"z":"한글"}'
    assert first.approval_snapshot(dataset_id=first_dataset) == second.approval_snapshot(dataset_id=second_dataset)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    first_process = context.Process(
        target=_snapshot_from_order, args=(str(tmp_path / "process-a.sqlite3"), (9, 2), result)
    )
    second_process = context.Process(
        target=_snapshot_from_order, args=(str(tmp_path / "process-b.sqlite3"), (2, 9), result)
    )
    try:
        first_process.start()
        second_process.start()
        first_process.join(10)
        second_process.join(10)
        assert first_process.exitcode == 0
        assert second_process.exitcode == 0
        assert result.get(timeout=2) == result.get(timeout=2)
    finally:
        for process in (first_process, second_process):
            if process.pid is None:
                continue
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(2)
        result.close()
        result.join_thread()


def test_export_snapshot_copies_approved_rows_immutably_in_the_same_transaction(tmp_path: Path) -> None:
    database = _db(tmp_path)
    dataset_id = _dataset(database)
    episode = database.create_episode(dataset_id=dataset_id, source_episode_index=3, source_length=100)
    database.update_episode(
        dataset_id=dataset_id,
        source_episode_index=3,
        expected_revision=episode["revision"],
        changes={
            "review_state": ReviewState.APPROVED_REJECT,
            "reviewer": "reviewer",
            "approval_revision": 1,
            "approved_at": "2026-08-21T00:00:00Z",
        },
        actor="reviewer",
    )

    exported = database.create_export_snapshot(
        dataset_id=dataset_id, staging_path="/tmp/staging", final_path="/tmp/final"
    )
    assert exported["approval_snapshot_sha256"] == database.approval_snapshot(dataset_id=dataset_id)["sha256"]
    database.set_export_state(
        export_id=exported["id"], expected_state=ExportState.QUEUED, state=ExportState.FAILED
    )
    database.update_episode(
        dataset_id=dataset_id,
        source_episode_index=3,
        expected_revision=1,
        changes={"rejection_reason": "later edit"},
        actor="reviewer",
    )
    with database.open_connection() as connection:
        copied = connection.execute(
            "SELECT rejection_reason, revision FROM export_episodes WHERE export_id=?", (exported["id"],)
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE exports SET approval_snapshot_sha256=? WHERE id=?", ("c" * 64, exported["id"])
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE exports SET staging_path='/tmp/other-staging' WHERE id=?", (exported["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE exports SET final_path='/tmp/other-final' WHERE id=?", (exported["id"],))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE export_episodes SET rejection_reason='mutated' WHERE export_id=?", (exported["id"],)
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM export_episodes WHERE export_id=?", (exported["id"],))
    assert dict(copied) == {"rejection_reason": None, "revision": 1}


def test_lock_exhaustion_is_retryable_and_a_real_process_writer_waits_for_the_lock(tmp_path: Path) -> None:
    database = _db(tmp_path)
    path = str(database.path)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    first_ready = context.Event()
    first_release = context.Event()
    second_ready = context.Event()
    second_release = context.Event()
    writer_attempting = context.Event()
    first_holder = context.Process(target=_hold_immediate_lock, args=(path, first_ready, first_release))
    second_holder: multiprocessing.Process | None = None
    writer: multiprocessing.Process | None = None
    try:
        first_holder.start()
        assert first_ready.wait(10)
        writer = context.Process(
            target=_concurrent_register,
            args=(path, "local/writer", writer_attempting, result),
        )
        writer.start()
        assert writer_attempting.wait(10)
        time.sleep(0.25)
        assert writer.is_alive()
        with pytest.raises(Empty):
            result.get(timeout=0.05)
        first_release.set()
        first_holder.join(10)
        writer.join(10)
        assert first_holder.exitcode == 0
        assert writer.exitcode == 0
        writer_result = result.get(timeout=2)
        assert writer_result[0] == "ok"
        assert writer_result[2] >= 0.2
        assert database.get_dataset(alias="local/writer")["alias"] == "local/writer"

        second_holder = context.Process(target=_hold_immediate_lock, args=(path, second_ready, second_release))
        second_holder.start()
        assert second_ready.wait(10)
        started = time.monotonic()
        with pytest.raises(RetryableDatabaseError):
            database.register_dataset(
                alias="local/locked",
                source_path="/immutable/locked",
                source_manifest_sha256="a" * 64,
                prompt_template_version="pnp-trash-prompts-v1",
                prompt_template_sha256="b" * 64,
            )
        assert time.monotonic() - started >= 4.5
    finally:
        first_release.set()
        second_release.set()
        for process in (first_holder, writer, second_holder):
            if process is None:
                continue
            if process.pid is None:
                continue
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(2)
        result.close()
        result.join_thread()
