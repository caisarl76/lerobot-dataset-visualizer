import json

import pytest


def test_revision_conflict_and_decision_do_not_touch_dataset(tmp_path):
    from annotation_runs import RunStore

    store = RunStore(tmp_path / "workspace")
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    labels = root / "meta/lerobot_annotations.json"
    labels.write_text('{"episodes": {}}')
    run = store.create(root, root, "org/data", "a" * 40, "v2.1", [0, 1])
    original = labels.read_bytes()
    run["episodes"]["0"]["decision"] = "delete"
    saved = store.save(run, 1)
    assert saved["revision"] == 2
    assert labels.read_bytes() == original
    with pytest.raises(ValueError, match="conflict"):
        store.save(run, 1)
    assert "status" not in saved
    assert saved["current_job_id"] is None
    assert store.for_root(root)["run_id"] == run["run_id"]


def test_recover_job_is_persisted_and_keeps_results(tmp_path):
    from annotation_runs import recover_jobs

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    p = jobs / ("b" * 32 + ".json")
    p.write_text(json.dumps({"job_id": "b" * 32, "status": "running", "completed_episodes": [0]}))
    recover_jobs(tmp_path)
    job = json.loads(p.read_text())
    assert job["status"] == "interrupted"
    assert job["completed_episodes"] == [0]


def test_workflow_decision_version_and_save_conflict(
    clear_dataset_state, legacy_v31_dataset, tmp_path, monkeypatch
):
    import asyncio

    from annotation_runs import RunStore
    import app
    import httpx

    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    alias = app._register_local_dataset(legacy_v31_dataset)
    store = RunStore(app.EXPORT_ROOT)
    store.create(legacy_v31_dataset, legacy_v31_dataset, "org/data", "a" * 40, "v3.1", [0])

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as c:
            path = "/api/workflow/" + alias.split("/")[1]
            first = await c.get(path)
            assert first.status_code == 200
            assert "source_root" not in first.json()
            decision = {"episode_index": 0, "decision": "delete", "reason": "junk", "expected_revision": 1}
            result = await c.post(path + "/decision", json=decision)
            assert result.status_code == 200
            assert result.json()["revision"] == 2
            assert (await c.post(path + "/decision", json=decision)).status_code == 409

    asyncio.run(scenario())


def test_creation_binds_original_source_bytes_but_not_editor_metadata(tmp_path):
    import hashlib

    from annotation_runs import RunStore

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    (root / "meta/info.json").write_text('{"fps":50}')
    (root / "data/episode.parquet").write_bytes(b"original data")
    (root / "videos/episode.mp4").write_bytes(b"original video")
    (root / "meta/lerobot_annotations.json").write_text('{"episodes":{}}')
    (root / "meta/annotation_predictions").mkdir()
    (root / "meta/annotation_predictions/first.json").write_text("{}")
    store = RunStore(tmp_path / "workspace")
    run = store.create(root, root, "org/data", "a" * 40, "v2.1", [0])
    assert run["source_file_hashes"] == {
        "meta/info.json": hashlib.sha256(b'{"fps":50}').hexdigest(),
        "data/episode.parquet": hashlib.sha256(b"original data").hexdigest(),
        "videos/episode.mp4": hashlib.sha256(b"original video").hexdigest(),
    }
    assert store.read(run["run_id"])["source_file_hashes"] == run["source_file_hashes"]


def test_save_cannot_rebind_source_hashes_or_identity(tmp_path):
    from annotation_runs import RunStore

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    store = RunStore(tmp_path / "workspace")
    run = store.create(root, root, "org/data", "a" * 40, "v2.1", [0])
    changed = {**run, "source_commit": "b" * 40}
    with pytest.raises(ValueError, match="immutable|source"):
        store.save(changed, 1)
    changed = {**run, "source_file_hashes": {"data/fake.parquet": "fake"}}
    with pytest.raises(ValueError, match="immutable|source"):
        store.save(changed, 1)


def test_editor_and_history_changes_do_not_rebind_legacy_source(tmp_path):
    from annotation_runs import RunStore, source_inventory

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"fps":50}')
    store = RunStore(tmp_path / "workspace")
    run = store.create(root, root, "org/data", "a" * 40, "v2.1", [0])
    for name in ("lerobot_annotations.json", "annotation_reviews.json", "annotation_run.json"):
        (root / "meta" / name).write_text('{"changed":true}')
    history = root / "meta/annotation_predictions"
    history.mkdir()
    (history / "new.json").write_text('{"prediction":"new"}')
    assert source_inventory(root) == run["source_file_hashes"]
