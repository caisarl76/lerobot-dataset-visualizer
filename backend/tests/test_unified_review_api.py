# ruff: noqa: F811
import asyncio
import json
from pathlib import Path

import annotation_history as history
from annotation_runs import RunStore
import app
import httpx
import pytest
from test_annotation_publish import reviewed_run  # noqa: F401


@pytest.fixture
def workflow(reviewed_run, tmp_path, monkeypatch, clear_dataset_state):
    root = Path(reviewed_run["root"])
    snap = root / "meta/annotation_predictions/first.json"
    snap.write_text(json.dumps({**json.loads(snap.read_text()), "created_at": "2026-09-10"}))
    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    alias = app._register_local_dataset(root).split("/")[1]
    store = RunStore(app.EXPORT_ROOT)
    run = store.create(root, root, None, None, "v3.1", [0, 1, 2])
    run["episodes"]["1"].update(decision="delete", decision_reason="junk")
    run["episodes"]["2"].update(decision="keep", decision_reason="inspected")
    run = store.save(run, run["revision"])
    history.write_reviews(root, {})
    return alias, root, store, run


def test_bulk_keep_and_explicit_review_preserve_deletions(workflow):
    alias, root, store, run = workflow

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as c:
            url = "/api/workflow/" + alias
            before = (await c.get(url)).json()
            response = await c.post(url + "/keep-remaining", json={"expected_revision": run["revision"]})
            assert response.status_code == 200, response.text
            kept = response.json()
            assert kept["episodes"]["1"]["decision"] == "delete"
            assert kept["episodes"]["1"]["decision_reason"] == "junk"
            assert kept["episodes"]["2"]["decision_reason"] == "inspected"
            assert kept["episodes"]["0"]["decision"] == "keep"
            assert (
                await c.post(url + "/keep-remaining", json={"expected_revision": run["revision"]})
            ).status_code == 409
            body = {
                "expected_revision": kept["revision"],
                "expected_review_sha256": kept["review_snapshot_sha256"],
                "confirmed": False,
            }
            assert (await c.post(url + "/review-retained", json=body)).status_code == 422
            assert history.read_reviews(root) == {}
            body.update(confirmed=True, expected_review_sha256=before["review_snapshot_sha256"])
            assert (await c.post(url + "/review-retained", json=body)).status_code == 409
            body["expected_review_sha256"] = kept["review_snapshot_sha256"]
            result = await c.post(url + "/review-retained", json=body)
            assert result.status_code == 200, result.text
            rows = result.json()["episodes"]
            assert rows["0"]["review"]["status"] == rows["2"]["review"]["status"] == "reviewed"
            assert "1" not in history.read_reviews(root)

    asyncio.run(scenario())


def test_bulk_review_detects_labels_changed_without_revision(workflow):
    alias, root, store, run = workflow

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as c:
            url = "/api/workflow/" + alias
            before = (await c.get(url)).json()
            path = root / "meta/lerobot_annotations.json"
            data = json.loads(path.read_text())
            data["episodes"]["0"]["atoms"][0]["content"] = "changed"
            path.write_text(json.dumps(data))
            response = await c.post(
                url + "/review-retained",
                json={
                    "expected_revision": run["revision"],
                    "expected_review_sha256": before["review_snapshot_sha256"],
                    "confirmed": True,
                },
            )
            assert response.status_code == 409
            assert history.read_reviews(root) == {}

    asyncio.run(scenario())


def test_registered_frozen_export_refuses_atom_and_review_mutations(workflow):
    alias, root, store, run = workflow
    manifest = root.parent / "manifest.json"
    manifest.write_text(json.dumps({"dataset_name": root.name, "format": "rich"}))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as c:
            ref = {"repo_id": "local/" + alias}
            review = (await c.get("/api/episodes/0/review", params=ref)).json()
            body = {
                **ref,
                "episode_index": 0,
                "reviewed": True,
                "annotation_sha256": review["annotation_sha256"],
                "expected_exclusions_sha256": review["exclusions_sha256"],
            }
            response = await c.post("/api/episodes/0/review", json=body)
            assert response.status_code == 409 and "read-only" in response.text
            atoms = json.loads((root / "meta/lerobot_annotations.json").read_text())["episodes"]["0"]["atoms"]
            response = await c.post(
                "/api/episodes/0/atoms",
                json={
                    **ref,
                    "episode_index": 0,
                    "atoms": atoms,
                    "expected_annotation_sha256": review["annotation_sha256"],
                },
            )
            assert response.status_code == 409 and "read-only" in response.text

    asyncio.run(scenario())


def test_bulk_actions_reject_active_job_and_hosted_paths_remain_hidden(workflow, monkeypatch):
    alias, root, store, run = workflow
    job_id = "b" * 32
    run["current_job_id"] = job_id
    run["export"] = {"manifest_sha256": "x", "local_path": "/private/server/export", "format": "rich"}
    run = store.save(run, run["revision"])
    monkeypatch.setitem(app._annotation_jobs, job_id, {"status": "running"})
    monkeypatch.setenv("ANNOTATION_BACKEND_TOKEN", "test-secret")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app.app),
            base_url="http://test",
            headers={"Authorization": "Bearer test-secret"},
        ) as c:
            url = "/api/workflow/" + alias
            current = (await c.get(url)).json()
            assert "local_path" not in current["export"]
            for route, body in [
                ("keep-remaining", {}),
                (
                    "review-retained",
                    {"confirmed": True, "expected_review_sha256": current["review_snapshot_sha256"]},
                ),
            ]:
                response = await c.post(url + "/" + route, json={"expected_revision": run["revision"], **body})
                assert response.status_code == 409
            assert history.read_reviews(root) == {}

    asyncio.run(scenario())
