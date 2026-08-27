from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from curation.db import CurationDatabase
from curation.exporter import ExportError, ExportService
from curation.models import ExportState, ReviewState
from curation.prompts import PROMPT_TEMPLATE_SHA256, PROMPT_TEMPLATE_VERSION
from curation.router import build_curation_router
from curation.source import SourceRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


def _case(tmp_path: Path, *, count: int = 3) -> tuple[ExportService, CurationDatabase, Path, Path]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": count,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
    )
    (source / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": index, "length": 10}) + "\n" for index in range(count))
    )
    workspace = tmp_path / "workspace"
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
    database = CurationDatabase(workspace / "curation.sqlite3")
    database.initialize()
    database.open_review_workspace(
        alias="local/pnp_trash",
        source_path=str(source.resolve()),
        source_manifest_sha256=registry.records["local/pnp_trash"].fingerprint,
        episode_lengths={index: 10 for index in range(count)},
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        actor="curator",
    )
    output = tmp_path / "out" / "pnp_trash_cleaned"
    service = ExportService(
        database=database,
        source_registry=registry,
        workspace=workspace,
        final_path=output,
    )
    return service, database, source, output


def _approve(
    database: CurationDatabase,
    *,
    dataset_id: int,
    source_episode_index: int,
    keep: bool,
) -> None:
    changes: dict[str, object] = {
        "review_state": ReviewState.APPROVED_KEEP if keep else ReviewState.APPROVED_REJECT,
        "approval_revision": 1,
        "reviewer": "human",
        "approved_at": "2026-08-27T00:00:00Z",
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
    }
    if keep:
        changes.update(
            {
                "object_name": "crumpled can",
                "pickup_hand": "left",
                "turn_direction": "right",
                "step_2_start_frame": 1,
                "step_3_start_frame": 2,
                "step_4_start_frame": 3,
                "step_5_start_frame": 4,
                "step_6_start_frame": 5,
                "step_7_start_frame": 6,
            }
        )
    database.update_episode(
        dataset_id=dataset_id,
        source_episode_index=source_episode_index,
        expected_revision=0,
        changes=changes,
        actor="human",
    )


def _approve_all(database: CurationDatabase, *, keep: set[int]) -> int:
    dataset_id = database.get_dataset(alias="local/pnp_trash")["id"]
    for row in database.list_episodes(dataset_id=dataset_id):
        _approve(
            database,
            dataset_id=dataset_id,
            source_episode_index=row["source_episode_index"],
            keep=row["source_episode_index"] in keep,
        )
    return dataset_id


@pytest.mark.parametrize("state", [ReviewState.PENDING, ReviewState.DRAFT])
def test_export_creation_requires_every_source_episode_approved(tmp_path: Path, state: ReviewState) -> None:
    service, database, _, _ = _case(tmp_path)
    dataset_id = database.get_dataset(alias="local/pnp_trash")["id"]
    _approve(database, dataset_id=dataset_id, source_episode_index=0, keep=True)
    if state is ReviewState.DRAFT:
        database.update_episode(
            dataset_id=dataset_id,
            source_episode_index=1,
            expected_revision=0,
            changes={"review_state": state},
            actor="human",
        )

    with pytest.raises(ExportError) as raised:
        service.create("local/pnp_trash")

    assert raised.value.payload == {"error": "export_reviews_incomplete", "unapproved_episode_indices": [1, 2]}


def test_export_creation_rejects_zero_kept_episodes(tmp_path: Path) -> None:
    service, database, _, _ = _case(tmp_path)
    _approve_all(database, keep=set())

    with pytest.raises(ExportError) as raised:
        service.create("local/pnp_trash")

    assert raised.value.payload == {"error": "export_zero_kept_episodes"}


def test_export_creation_rejects_invalid_approval_revision(tmp_path: Path) -> None:
    service, database, _, _ = _case(tmp_path)
    dataset_id = _approve_all(database, keep={0})
    with database.open_connection() as connection:
        connection.execute(
            "UPDATE episodes SET approval_revision=0 WHERE dataset_id=? AND source_episode_index=0",
            (dataset_id,),
        )

    with pytest.raises(ExportError) as raised:
        service.create("local/pnp_trash")

    assert raised.value.payload == {"error": "export_invalid_approval", "source_episode_indices": [0]}


def test_export_creation_rejects_source_fingerprint_change_and_existing_final(tmp_path: Path) -> None:
    service, database, source, output = _case(tmp_path)
    _approve_all(database, keep={0})
    (source / "meta" / "info.json").write_text('{"changed":true}')
    with pytest.raises(ExportError) as changed:
        service.create("local/pnp_trash")
    assert changed.value.payload == {"error": "source_fingerprint_mismatch", "dataset_alias": "local/pnp_trash"}

    service, database, _, output = _case(tmp_path / "other")
    _approve_all(database, keep={0})
    output.mkdir(parents=True)
    with pytest.raises(ExportError) as exists:
        service.create("local/pnp_trash")
    assert exists.value.payload == {"error": "export_destination_exists"}


def test_export_creation_rejects_symlinked_output_parent_without_resolving_into_victim(tmp_path: Path) -> None:
    service, database, _, output = _case(tmp_path)
    _approve_all(database, keep={0})
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel"
    sentinel.write_bytes(b"victim")
    output.parent.symlink_to(victim, target_is_directory=True)

    with pytest.raises(ExportError) as raised:
        service.create("local/pnp_trash")

    assert raised.value.payload == {"error": "export_path_invalid"}
    assert list(victim.iterdir()) == [sentinel]


def test_snapshot_and_rows_commit_together_and_active_export_blocks(tmp_path: Path) -> None:
    service, database, _, output = _case(tmp_path)
    _approve_all(database, keep={2, 0})

    created = service.create("local/pnp_trash")

    assert created["state"] == "queued"
    assert created["approval_snapshot_sha256"]
    assert created["run_command"].endswith(f"run --export-id {created['export_id']}")
    staging = Path(created["staging_path"])
    assert staging.parent == output.parent.resolve()
    assert staging.name == f".{output.name}.staging-{created['export_id']}"
    with database.open_connection() as connection:
        export_count = connection.execute("SELECT count(*) FROM exports").fetchone()[0]
        frozen = connection.execute(
            "SELECT source_episode_index FROM export_episodes WHERE export_id=? ORDER BY source_episode_index",
            (created["export_id"],),
        ).fetchall()
    assert export_count == 1
    assert [row[0] for row in frozen] == [0, 1, 2]

    with pytest.raises(ExportError) as active:
        service.create("local/pnp_trash")
    assert active.value.payload == {"error": "export_active"}

    status = service.status(created["export_id"])
    assert status == created


def test_export_build_claim_is_compare_and_swap_and_exposes_frozen_rows(tmp_path: Path) -> None:
    service, database, _, _ = _case(tmp_path)
    _approve_all(database, keep={2, 0})
    created = service.create("local/pnp_trash")

    claimed = database.claim_export_build(export_id=created["export_id"])

    assert claimed["export"]["state"] == ExportState.BUILDING.value
    assert [row["source_episode_index"] for row in claimed["episodes"]] == [0, 1, 2]
    with pytest.raises(Exception):
        database.claim_export_build(export_id=created["export_id"])


def test_checked_snapshot_rolls_back_export_if_frozen_row_insert_fails(tmp_path: Path) -> None:
    service, database, _, _ = _case(tmp_path)
    _approve_all(database, keep={0})
    with database.open_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_export_episode_insert BEFORE INSERT ON export_episodes
            BEGIN SELECT RAISE(ABORT, 'injected snapshot failure'); END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected snapshot failure"):
        service.create("local/pnp_trash")

    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM exports").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM export_episodes").fetchone()[0] == 0


def test_authenticated_export_api_only_freezes_snapshot_and_returns_persisted_status() -> None:
    class FakeExportService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def create(self, dataset_alias: str) -> dict[str, Any]:
            self.calls.append(("create", dataset_alias))
            return {
                "export_id": "11111111-1111-1111-1111-111111111111",
                "dataset_alias": dataset_alias,
                "state": "queued",
                "approval_snapshot_sha256": "a" * 64,
                "staging_path": "/output/.cleaned.staging-id",
                "final_path": "/output/cleaned",
                "failure_summary": None,
                "created_at": "2026-08-27T00:00:00Z",
                "updated_at": "2026-08-27T00:00:00Z",
                "run_command": (
                    "backend/.venv/bin/python backend/curation_export.py --workspace /workspace run --export-id id"
                ),
            }

        def status(self, export_id: str) -> dict[str, Any]:
            self.calls.append(("status", export_id))
            if export_id == "00000000-0000-0000-0000-000000000000":
                raise ExportError("export not found", {"error": "export_not_found"})
            response = self.create("local/pnp_trash")
            response["export_id"] = export_id
            return response

    service = FakeExportService()
    app = FastAPI()
    app.include_router(build_curation_router(export_service=service, bearer_token="server-secret"))

    with TestClient(app) as client:
        assert client.post("/api/curation/exports", json={"dataset_alias": "local/pnp_trash"}).status_code == 401
        created = client.post(
            "/api/curation/exports",
            json={"dataset_alias": "local/pnp_trash"},
            headers={"Authorization": "Bearer server-secret"},
        )
        status = client.get(
            "/api/curation/exports/11111111-1111-1111-1111-111111111111",
            headers={"Authorization": "Bearer server-secret"},
        )
        missing = client.get(
            "/api/curation/exports/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": "Bearer server-secret"},
        )

    assert created.status_code == 201
    assert created.json()["state"] == "queued"
    assert status.status_code == 200
    assert status.json()["export_id"] == "11111111-1111-1111-1111-111111111111"
    assert missing.status_code == 404
    assert missing.json() == {"error": "export_not_found"}
    assert service.calls[0] == ("create", "local/pnp_trash")
    assert service.calls[1] == ("status", "11111111-1111-1111-1111-111111111111")
