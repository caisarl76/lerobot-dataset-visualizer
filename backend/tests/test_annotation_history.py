import asyncio
import json
import shutil

import annotation_history as history
import app
import httpx


def test_snapshot_survives_edits_and_retry(tmp_path):
    atoms = [{"role": "assistant", "style": "subtask", "content": "approach", "timestamp": 0}]
    root = tmp_path / "first"
    (root / "meta").mkdir(parents=True)
    sidecar = root / "meta/lerobot_annotations.json"
    sidecar.write_text(json.dumps({"episodes": {"0": {"atoms": atoms}, "1": {"atoms": atoms}}}))
    config = {"vlm": {"model": "qwen", "api_key": "secret", "serve_command": "secret"}}
    path = history.snapshot_predictions(root, tmp_path / "source", {0}, {1: atoms}, config, "revision")
    frozen = path.read_bytes()
    status = history.review_status(root, 0, atoms)
    assert status["prediction_available"] and status["status"] == "unreviewed"
    history.save_review(root, 0, atoms, True, status["annotation_sha256"])
    assert history.review_status(root, 0, atoms)["status"] == "reviewed"
    edited = [{**atoms[0], "timestamp": 1.0}]
    sidecar.write_text(json.dumps({"episodes": {"0": {"atoms": edited}}}))
    assert history.review_status(root, 0, edited)["status"] == "unreviewed"
    retry = tmp_path / "retry"
    shutil.copytree(root, retry)
    history.snapshot_predictions(retry, root, {0}, {}, config, "revision")
    assert history.review_status(retry, 0, atoms)["status"] == "unreviewed"
    assert path.read_bytes() == frozen == (retry / path.relative_to(root)).read_bytes()
    assert len(list((retry / "meta/annotation_predictions").glob("*.json"))) == 2
    assert "secret" not in frozen.decode()


def test_explicit_review_and_stale_edit(clear_dataset_state, legacy_v31_dataset):
    async def scenario():
        identity = {"local_path": str(legacy_v31_dataset)}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            path = "/api/episodes/0/review"
            status = (await client.get(path, params=identity)).json()
            assert status["status"] == "unreviewed" and not status["prediction_available"]
            payload = {
                **identity,
                "episode_index": 0,
                "reviewed": True,
                "annotation_sha256": status["annotation_sha256"],
            }
            assert (await client.post(path, json=payload)).json()["status"] == "reviewed"
            app._states.clear()
            assert (await client.get(path, params=identity)).json()["status"] == "reviewed"
            atoms = [{"role": "assistant", "style": "subtask", "content": "edited", "timestamp": 0.0}]
            saved = await client.post(
                "/api/episodes/0/atoms", json={**identity, "episode_index": 0, "atoms": atoms}
            )
            assert saved.status_code == 200
            assert (await client.get(path, params=identity)).json()["status"] == "unreviewed"
            assert (await client.post(path, json=payload)).status_code == 409
            assert (await client.get("/api/episodes/99/review", params=identity)).status_code == 404

    asyncio.run(scenario())


def test_first_failure_without_sidecar_has_atomic_history(tmp_path, monkeypatch):
    from pathlib import Path

    real_replace = Path.replace
    observed = []

    def replace(source, target):
        if source.parent.name == "annotation_predictions":
            assert source.suffix != ".json"
            assert not list(source.parent.glob("*.json"))
            observed.append(json.loads(source.read_text()))
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    results = {"0": {"generation_status": "failed", "issues": [{"code": "decode_failed"}]}}
    path = history.snapshot_predictions(tmp_path, tmp_path, set(), {}, {}, "revision", episode_results=results)
    saved = json.loads(path.read_text())
    assert saved["episodes"] == {}
    assert saved["episode_results"] == results
    assert observed == [saved]
