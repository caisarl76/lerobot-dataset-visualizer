# ruff: noqa: F811
import asyncio
import json
from pathlib import Path

import annotation_history as history
from annotation_runs import RunStore
import app
import httpx
from test_annotation_publish import reviewed_run  # noqa: F401


def test_exclusions_persist_validate_revision_and_invalidate_review(
    reviewed_run, tmp_path, monkeypatch, clear_dataset_state
):
    root = Path(reviewed_run["root"])
    snapshot = root / "meta/annotation_predictions/first.json"
    snapshot.write_text(json.dumps({**json.loads(snapshot.read_text()), "created_at": "2026-09-09T00:00:00Z"}))
    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    alias = app._register_local_dataset(root)
    store = RunStore(app.EXPORT_ROOT)
    run = store.create(root, root, None, None, "v3.1", [0, 1, 2])
    run["publication_state"] = "exported"
    run["export"] = {"manifest_sha256": "old"}
    run = store.save(run, run["revision"])

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            url = "/api/workflow/" + alias.split("/")[1] + "/exclusions"
            before = (await client.get("/api/episodes/0/review", params={"repo_id": alias})).json()
            body = {
                "episode_index": 0,
                "expected_revision": run["revision"],
                "excluded_intervals": [{"start_frame": 0, "end_frame": 1}],
            }
            response = await client.post(url, json=body)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["episodes"]["0"]["excluded_intervals"] == body["excluded_intervals"]
            assert result["publication_state"] == "draft" and not result.get("export")
            assert result["episodes"]["0"]["review"]["status"] == "unreviewed"
            legacy_export = await client.post("/api/export", json={"repo_id": alias})
            assert legacy_export.status_code == 409
            assert "Preview export" in legacy_export.json()["detail"]
            assert (await client.post(url, json=body)).status_code == 409
            status = (await client.get("/api/episodes/0/review", params={"repo_id": alias})).json()
            assert status["exclusions_sha256"] != before["exclusions_sha256"]
            review = {
                "repo_id": alias,
                "episode_index": 0,
                "reviewed": True,
                "annotation_sha256": status["annotation_sha256"],
                "expected_exclusions_sha256": before["exclusions_sha256"],
            }
            assert (await client.post("/api/episodes/0/review", json=review)).status_code == 409
            review["expected_exclusions_sha256"] = status["exclusions_sha256"]
            assert (await client.post("/api/episodes/0/review", json=review)).status_code == 200
            revision = store.for_root(root)["revision"]
            for intervals in [
                [{"start_frame": 0, "end_frame": 4}],
                [{"start_frame": -1, "end_frame": 1}],
                [{"start_frame": 0, "end_frame": 1.5}],
                [{"start_frame": True, "end_frame": 2}],
            ]:
                failed = await client.post(
                    url, json={**body, "expected_revision": revision, "excluded_intervals": intervals}
                )
                assert failed.status_code == 422, failed.text
                assert store.for_root(root)["revision"] == revision
            undo = await client.post(url, json={**body, "expected_revision": revision, "excluded_intervals": []})
            assert undo.status_code == 200
            assert undo.json()["episodes"]["0"]["excluded_intervals"] == []
            assert undo.json()["episodes"]["0"]["review"]["status"] == "unreviewed"
            assert history.read_reviews(root).get("0") is None

    asyncio.run(scenario())
