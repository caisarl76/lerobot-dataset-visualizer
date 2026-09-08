"""Exercise review decisions, optimistic saves and export through the actual API."""
# ruff: noqa: F811

import asyncio
import json
from pathlib import Path

from annotation_runs import RunStore
import app
import httpx
from test_annotation_publish import reviewed_run  # noqa: F401


def test_review_queue_to_frozen_export(reviewed_run, tmp_path, monkeypatch, clear_dataset_state):  # noqa: F811
    dataset = Path(reviewed_run["root"])
    snapshot = dataset / "meta/annotation_predictions/first.json"
    snapshot.write_text(json.dumps({**json.loads(snapshot.read_text()), "created_at": "2026-09-08T00:00:00Z"}))
    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    alias = app._register_local_dataset(dataset)
    RunStore(app.EXPORT_ROOT).create(dataset, dataset, "org/data", "a" * 40, "v3.1", [0, 1, 2])

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            ident = {"repo_id": alias}
            workflow = "/api/workflow/" + alias.split("/")[1]
            for ep in range(3):
                atoms = [
                    {"role": "assistant", "style": "subtask", "content": "approach", "timestamp": 0.0},
                    {"role": "assistant", "style": "subtask", "content": "grasp", "timestamp": 0.2},
                ]
                before = (await client.get(f"/api/episodes/{ep}/atoms", params=ident)).json()
                result = await client.post(
                    f"/api/episodes/{ep}/atoms",
                    json={
                        **ident,
                        "episode_index": ep,
                        "atoms": atoms,
                        "expected_annotation_sha256": before["annotation_sha256"],
                    },
                )
                assert result.status_code == 200, result.text
                assert (
                    await client.post(
                        f"/api/episodes/{ep}/atoms",
                        json={
                            **ident,
                            "episode_index": ep,
                            "atoms": atoms,
                            "expected_annotation_sha256": before["annotation_sha256"],
                        },
                    )
                ).status_code == 409
                status = (await client.get(f"/api/episodes/{ep}/review", params=ident)).json()
                reviewed = await client.post(
                    f"/api/episodes/{ep}/review",
                    json={
                        **ident,
                        "episode_index": ep,
                        "reviewed": True,
                        "annotation_sha256": status["annotation_sha256"],
                    },
                )
                assert reviewed.status_code == 200
            run = (await client.get(workflow)).json()
            assert all(ep["review"]["status"] == "reviewed" for ep in run["episodes"].values())
            original = (dataset / "meta/lerobot_annotations.json").read_bytes()
            flagged = await client.post(
                workflow + "/decision",
                json={
                    "episode_index": 1,
                    "decision": "delete",
                    "reason": "junk",
                    "expected_revision": run["revision"],
                },
            )
            assert flagged.status_code == 200
            assert (dataset / "meta/lerobot_annotations.json").read_bytes() == original
            # Undo only changes the flag. The fixture's tiny metadata intentionally
            # omits official deletion statistics; deletion itself has dedicated tests.
            undo = await client.post(
                workflow + "/decision",
                json={
                    "episode_index": 1,
                    "decision": "keep",
                    "reason": "reviewed",
                    "expected_revision": flagged.json()["revision"],
                },
            )
            started = await client.post(workflow + "/export", json={"expected_revision": undo.json()["revision"]})
            assert started.status_code == 200, started.text
            for _ in range(100):
                job = (await client.get("/api/annotation/jobs/" + started.json()["job_id"])).json()
                if job["status"] not in ("queued", "running"):
                    break
                await asyncio.sleep(0.05)
            assert job["status"] == "completed", job
            frozen = (await client.get(workflow)).json()
            assert frozen["publication_state"] == "exported"
            assert frozen["export"]["manifest_sha256"]
            assert "root" not in frozen["export"]
            assert "meta/lerobot_annotations.json" in frozen["export"]["managed_changes"]["added_or_updated"]
            # Editing invalidates the frozen publication and review independently.
            loaded = (await client.get("/api/episodes/0/atoms", params=ident)).json()
            loaded["atoms"][1]["timestamp"] = 0.25
            changed = await client.post(
                "/api/episodes/0/atoms",
                json={
                    **ident,
                    "episode_index": 0,
                    "atoms": loaded["atoms"],
                    "expected_annotation_sha256": loaded["annotation_sha256"],
                },
            )
            assert changed.status_code == 200
            current = (await client.get(workflow)).json()
            assert current["publication_state"] == "draft" and "export" not in current
            assert current["episodes"]["0"]["review"]["status"] == "unreviewed"

    asyncio.run(scenario())


from test_annotation_prepare_recovery import raw_v21  # noqa: E402,F401


def test_corrupt_prepare_exposes_original_decisions_and_good_episode(
    raw_v21,
    tmp_path,
    monkeypatch,
    clear_dataset_state,  # noqa: F811
):
    import json

    from test_annotation_run_generation import SynchronousPool

    corrupt = raw_v21 / "videos/chunk-000/observation.images.cam/episode_000000.mp4"
    corrupt.write_bytes(b"invalid mp4")
    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(app, "_annotation_pool", SynchronousPool())
    started = app.prepare_annotation_dataset(app.DatasetRef(local_path=str(raw_v21)))
    job = app.get_annotation_job(started["job_id"])
    assert job["status"] == "completed", job
    assert job["result"]["first_episode_index"] == 1
    alias = job["result"]["repo_id"].split("/")[1]
    _, run = app._workflow(alias)
    view = app._workflow_payload(run)
    assert view["preparation_mode"] == "source_review"
    assert set(view["episodes"]) == {"0", "1"}
    assert view["episodes"]["0"]["issues"]
    state = app._ensure_state(app.DatasetRef(repo_id=job["result"]["repo_id"]))
    assert app._frame_timestamps(state, 1) == [0, 0.25, 0.5, 0.75]
    assert json.loads(app.get_episode_atoms(1, local_path=str(state.root)).body)["atoms"] == []
    assert corrupt.read_bytes() == b"invalid mp4"


def test_prepare_pins_hub_download_before_conversion(raw_v21, tmp_path, monkeypatch):  # noqa: F811
    import shutil
    from types import SimpleNamespace

    from test_annotation_run_generation import SynchronousPool

    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(app, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(app, "_annotation_pool", SynchronousPool())
    requested = []
    monkeypatch.setattr(
        app, "HfApi", lambda: SimpleNamespace(dataset_info=lambda repo, revision: SimpleNamespace(sha="a" * 40))
    )

    def download(repo, **kwargs):
        requested.append((repo, kwargs["revision"]))
        shutil.copytree(raw_v21, kwargs["local_dir"])

    monkeypatch.setattr(app, "snapshot_download", download)
    started = app.prepare_annotation_dataset(app.DatasetRef(repo_id="team/test", revision="main"))
    job = app.get_annotation_job(started["job_id"])
    assert job["status"] == "completed", job
    assert requested == [("team/test", "a" * 40)]
    _, run = app._workflow(job["result"]["repo_id"].split("/")[1])
    assert run["source_commit"] == "a" * 40 and run["source_file_hashes"]
    assert "source_file_hashes" not in app._workflow_payload(run)


def test_prepare_rejects_source_changed_during_conversion(raw_v21, tmp_path, monkeypatch):  # noqa: F811
    from test_annotation_run_generation import SynchronousPool

    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(app, "_annotation_pool", SynchronousPool())
    engine = app._official_engine()
    original = engine.prepare_dataset

    def prepare(source, output):
        result = original(source, output)
        (source / "meta/new_source_data.json").write_text("{}")
        return result

    monkeypatch.setattr(engine, "prepare_dataset", prepare)
    started = app.prepare_annotation_dataset(app.DatasetRef(local_path=str(raw_v21)))
    job = app.get_annotation_job(started["job_id"])
    assert job["status"] == "failed", job
    assert "changed during preparation" in job["error"]
