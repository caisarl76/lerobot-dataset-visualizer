from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

from curation.db import CurationDatabase
from curation.review import ReviewService
from curation.router import build_curation_router
from curation.source import SourceRegistry
from curation.worker import BatchService
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def batch_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, BatchService, CurationDatabase]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True)
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "total_episodes": 2,
                "total_frames": 16,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
                "features": {"observation.images.ego_view": {"dtype": "video"}},
            }
        )
    )
    with (source / "meta" / "episodes.jsonl").open("w") as handle:
        for episode_index in range(2):
            handle.write(json.dumps({"episode_index": episode_index, "length": 8}) + "\n")
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [episode_index] * 8,
                        "frame_index": list(range(8)),
                        "timestamp": [frame / 10 for frame in range(8)],
                    }
                ),
                source / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            )
            (
                source
                / "videos"
                / "chunk-000"
                / "observation.images.ego_view"
                / f"episode_{episode_index:06d}.mp4"
            ).write_bytes(b"fake video")

    workspace = tmp_path / "workspace"
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
    database = CurationDatabase(workspace / "curation.sqlite3")
    database.initialize()
    review = ReviewService(database=database, source_registry=registry)
    review.open_workspace("local/pnp_trash", actor="test")
    monkeypatch.setenv("TEST_COSMOS_KEY", "not-secret-in-db")

    def capability(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://cosmos.test/v1/models")
        assert request.headers["Authorization"] == "Bearer not-secret-in-db"
        return httpx.Response(200, json={"data": [{"id": "cosmos3-nano"}]})

    batch = BatchService(
        database=database,
        source_registry=registry,
        workspace=workspace,
        cosmos_base_url="http://cosmos.test/v1",
        cosmos_model="cosmos3-nano",
        cosmos_api_key_env="TEST_COSMOS_KEY",
        cosmos_endpoint_identity="h100-test",
        capability_client=httpx.Client(transport=httpx.MockTransport(capability), trust_env=False),
        clock=lambda: datetime(2026, 8, 24, tzinfo=timezone.utc),
    )
    app = FastAPI()
    app.include_router(
        build_curation_router(review_service=review, batch_service=batch, bearer_token="secret-token")
    )
    return TestClient(app, base_url="http://127.0.0.1"), batch, database


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret-token"}


def test_start_freezes_configuration_and_only_queues_attempts(
    batch_components: tuple[TestClient, BatchService, CurationDatabase],
) -> None:
    client, _, database = batch_components
    response = client.post(
        "/api/curation/batches",
        headers=_headers(),
        json={"dataset_alias": "local/pnp_trash", "episode_indices": [1, 0]},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["state"] == "queued"
    assert body["configuration"]["episode_indices"] == [0, 1]
    assert body["configuration"]["source_manifest_sha256"]
    assert body["configuration"]["cosmos"] == {
        "api_key_env": "TEST_COSMOS_KEY",
        "base_url": "http://cosmos.test/v1",
        "endpoint_identity": "h100-test",
        "model": "cosmos3-nano",
    }
    assert "not-secret-in-db" not in json.dumps(body)
    status = client.get(f"/api/curation/batches/{body['job_id']}", headers=_headers())
    assert status.status_code == 200
    assert status.json()["counts"] == {"queued": 2}
    assert status.json()["episodes"] == [
        {
            "attempt_id": status.json()["episodes"][0]["attempt_id"],
            "attempt_number": 0,
            "source_episode_index": 0,
            "state": "queued",
        },
        {
            "attempt_id": status.json()["episodes"][1]["attempt_id"],
            "attempt_number": 0,
            "source_episode_index": 1,
            "state": "queued",
        },
    ]
    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM cosmos_proposals").fetchone()[0] == 0


def test_start_rejects_duplicate_active_batch_and_bad_model_capability(
    batch_components: tuple[TestClient, BatchService, CurationDatabase],
) -> None:
    client, batch, _ = batch_components
    first = client.post(
        "/api/curation/batches",
        headers=_headers(),
        json={"dataset_alias": "local/pnp_trash", "episode_indices": [0]},
    )
    duplicate = client.post(
        "/api/curation/batches",
        headers=_headers(),
        json={"dataset_alias": "local/pnp_trash", "episode_indices": [1]},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"] == "active_batch_exists"
    assert duplicate.json()["job_id"] == first.json()["job_id"]

    with batch.database._write() as connection:
        connection.execute(
            "UPDATE cosmos_attempts SET state='cancelled' WHERE job_id=?", (first.json()["job_id"],)
        )
        connection.execute("UPDATE cosmos_jobs SET state='cancelled' WHERE id=?", (first.json()["job_id"],))
    batch.capability_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "other"}]})),
        trust_env=False,
    )
    missing = client.post(
        "/api/curation/batches",
        headers=_headers(),
        json={"dataset_alias": "local/pnp_trash", "episode_indices": [0]},
    )
    assert missing.status_code == 422
    assert missing.json() == {"error": "cosmos_model_unavailable", "model": "cosmos3-nano"}


def test_cancel_exact_idempotent_state_machine_and_audit_counts(
    batch_components: tuple[TestClient, BatchService, CurationDatabase],
) -> None:
    client, batch, database = batch_components
    unknown = "00000000-0000-0000-0000-000000000000"
    response = client.post(f"/api/curation/batches/{unknown}/cancel", headers=_headers())
    assert response.status_code == 404
    assert response.json() == {"error": "batch_not_found", "job_id": unknown}

    queued = client.post(
        "/api/curation/batches",
        headers=_headers(),
        json={"dataset_alias": "local/pnp_trash", "episode_indices": [0, 1]},
    ).json()
    cancelled = client.post(f"/api/curation/batches/{queued['job_id']}/cancel", headers=_headers())
    assert cancelled.status_code == 200
    assert cancelled.json() == {"job_id": queued["job_id"], "state": "cancelled", "changed": True}
    repeated = client.post(f"/api/curation/batches/{queued['job_id']}/cancel", headers=_headers())
    assert repeated.status_code == 200
    assert repeated.json()["changed"] is False

    running = batch.start("local/pnp_trash", [0])
    batch.repository.start_job(running["job_id"], owner=str(uuid4()))
    requested = client.post(f"/api/curation/batches/{running['job_id']}/cancel", headers=_headers())
    assert requested.status_code == 202
    assert requested.json()["state"] == "cancel_requested"
    assert requested.json()["changed"] is True
    again = client.post(f"/api/curation/batches/{running['job_id']}/cancel", headers=_headers())
    assert again.status_code == 202
    assert again.json()["changed"] is False
    with database.open_connection() as connection:
        audit_count = connection.execute(
            "SELECT count(*) FROM audit_events WHERE job_id=? AND operation='batch_cancel_requested'",
            (running["job_id"],),
        ).fetchone()[0]
    assert audit_count == 1


@pytest.mark.parametrize("state", ["completed", "completed_with_failures", "failed"])
def test_terminal_cancel_is_conflict(
    batch_components: tuple[TestClient, BatchService, CurationDatabase], state: str
) -> None:
    client, batch, _ = batch_components
    job = batch.start("local/pnp_trash", [0])
    with batch.database._write() as connection:
        connection.execute("UPDATE cosmos_attempts SET state='manual_only' WHERE job_id=?", (job["job_id"],))
        connection.execute("UPDATE cosmos_jobs SET state=? WHERE id=?", (state, job["job_id"]))
    response = client.post(f"/api/curation/batches/{job['job_id']}/cancel", headers=_headers())
    assert response.status_code == 409
    assert response.json() == {"error": "batch_terminal", "job_id": job["job_id"], "state": state}


def test_retry_creates_immutable_child_with_explicit_episodes(
    batch_components: tuple[TestClient, BatchService, CurationDatabase],
) -> None:
    client, batch, database = batch_components
    parent = batch.start("local/pnp_trash", [0, 1])
    with database._write() as connection:
        connection.execute("UPDATE cosmos_attempts SET state='manual_only' WHERE job_id=?", (parent["job_id"],))
        connection.execute(
            "UPDATE cosmos_jobs SET state='completed_with_failures', manual_only_attempts=2 WHERE id=?",
            (parent["job_id"],),
        )
    before = batch.status(parent["job_id"])
    response = client.post(
        f"/api/curation/batches/{parent['job_id']}/retry",
        headers=_headers(),
        json={"episode_indices": [1]},
    )
    assert response.status_code == 201
    child = response.json()
    assert child["parent_job_id"] == parent["job_id"]
    assert child["configuration"]["episode_indices"] == [1]
    assert child["job_id"] != parent["job_id"]
    assert batch.status(parent["job_id"]) == before


def test_batch_routes_are_authenticated_and_strict(
    batch_components: tuple[TestClient, BatchService, CurationDatabase],
) -> None:
    client, _, _ = batch_components
    assert client.post("/api/curation/batches", json={}).status_code == 401
    invalid = client.post(
        "/api/curation/batches",
        headers=_headers(),
        json={"dataset_alias": "local/pnp_trash", "episode_indices": [0], "secret": os.getenv("TEST_COSMOS_KEY")},
    )
    assert invalid.status_code == 422
