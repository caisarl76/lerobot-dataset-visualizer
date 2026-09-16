"""Read-only monitor snapshots, HTTP boundaries and explicit remote checks."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import annotation_monitor as monitor
import httpx
from monitor_fixtures import review_fixture as review_fixture, write_json, write_run, write_v21
import pytest


@pytest.fixture
def service(review_fixture):
    dataset, _, workspace = review_fixture
    return monitor.MonitorService(Path(dataset["path"]).parent, workspace)


def request(module, method, path, **kwargs):
    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=module.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(call())


@pytest.fixture
def app_module(monkeypatch, review_fixture):
    import app

    dataset, _, workspace = review_fixture
    monkeypatch.delenv("ANNOTATION_BACKEND_TOKEN", raising=False)
    monkeypatch.setenv("LEROBOT_MONITOR_ROOT", str(Path(dataset["path"]).parent))
    monkeypatch.setattr(app, "EXPORT_ROOT", workspace)
    return app


def assert_public(value):
    if isinstance(value, dict):
        assert all(not key.startswith("_") for key in value)
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)


def test_service_preserves_checkpoint_metrics_and_read_only_bytes(service, review_fixture, monkeypatch):
    _, run, workspace = review_fixture
    write_json(
        Path(run["_path"]),
        dict(run, publication={"repo_id": "team/data", "revision": "main", "main_commit": "old"}),
    )
    before = {str(p): p.read_bytes() for p in workspace.parent.rglob("*") if p.is_file()}
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: pytest.fail("routine HF client"))
    snapshot = service.summary()
    row = snapshot["datasets"][0]
    assert row["collected"] == 4
    assert row["runs"][0]["metrics"]["imported"] == 3
    assert "episodes" not in row["runs"][0]
    assert row["runs"][0]["current_repo_id"] is None
    detail = service.detail(row["id"])
    assert detail["metrics"]["retained"] == 2
    assert detail["findings"] == row["runs"][0]["findings"]
    assert detail["exclusions"] == row["runs"][0]["exclusions"]
    assert detail["exports"] == []
    assert detail["publications"][0]["remote_check"] is None
    assert_public(snapshot)
    assert_public(detail)
    assert before == {str(p): p.read_bytes() for p in workspace.parent.rglob("*") if p.is_file()}


def test_detail_current_identity_and_run_ownership(service, review_fixture):
    dataset, _, workspace = review_fixture
    other = write_v21(Path(dataset["path"]).parent / "other", [10])
    write_run(workspace, other, episodes=[0], run_id="b" * 32)
    row = next(d for d in service.summary()["datasets"] if d["name"] == "source")
    with pytest.raises(KeyError):
        service.detail(row["id"], "b" * 32)
    with pytest.raises(KeyError):
        service.detail("../../etc")
    Path(dataset["path"]).rename(Path(dataset["path"]).with_name("renamed"))
    with pytest.raises(KeyError):
        service.detail(row["id"])


def test_summary_cache_and_refresh(service, monkeypatch):
    first = service.summary()
    original = monitor.summarize_run
    monkeypatch.setattr(monitor, "summarize_run", lambda *a: pytest.fail("unchanged summary recomputed"))
    assert service.summary() == first
    monkeypatch.setattr(monitor, "summarize_run", original)
    assert service.summary(refresh=True)["scanned_at"] >= first["scanned_at"]


def test_detail_invalidates_exclusions_and_parquet(service, review_fixture):
    _, run, _ = review_fixture
    row = service.summary()["datasets"][0]
    first = service.detail(row["id"])
    path = Path(run["_path"])
    content = json.loads(path.read_text())
    content["episodes"]["0"]["excluded_intervals"] = [{"start_frame": 0, "end_frame": 5}]
    write_json(path, content)
    second = service.detail(row["id"])
    assert second["signature"] != first["signature"]
    assert second["exclusions"]["frames"] == 5
    shard = next((Path(run["root"]) / "data").rglob("*.parquet"))
    import os

    os.utime(shard, ns=(shard.stat().st_atime_ns, shard.stat().st_mtime_ns + 10_000))
    assert service.detail(row["id"])["signature"] != second["signature"]


def test_snapshot_retries_then_preserves_old_timestamp(service, review_fixture, monkeypatch):
    first = service.summary()
    real = monitor.summarize_run
    _, run, _ = review_fixture
    path = Path(run["_path"])
    changes = [True, False]

    def changing(*args):
        result = real(*args)
        if changes.pop(0):
            value = json.loads(path.read_text())
            value["task_prompt"] += " changed"
            write_json(path, value)
        return result

    monkeypatch.setattr(monitor, "summarize_run", changing)
    retry = service.summary(refresh=True)
    assert changes == []
    assert retry["updating"] is False
    changes[:] = [True, True]
    stale = service.summary(refresh=True)
    assert stale["updating"] is True
    assert stale["scanned_at"] == retry["scanned_at"]
    assert stale["datasets"] == retry["datasets"]
    assert (
        first["datasets"][0]["runs"][0]["detail_signature"] != retry["datasets"][0]["runs"][0]["detail_signature"]
    )


def test_detail_retries_and_keeps_prior_success(service, review_fixture, monkeypatch):
    row = service.summary()["datasets"][0]
    old = service.detail(row["id"])
    _, run, _ = review_fixture
    path = Path(run["root"]) / "meta/annotation_reviews.json"
    actual = monitor.prompt_distribution
    count = []

    def changing(*args):
        result = actual(*args)
        content = json.loads(path.read_text())
        content["0"]["reviewed_at"] += "x"
        write_json(path, content)
        count.append(1)
        return result

    content = json.loads(path.read_text())
    content["0"]["reviewed_at"] += "initial"
    write_json(path, content)
    monkeypatch.setattr(monitor, "prompt_distribution", changing)
    stale = service.detail(row["id"])
    assert len(count) == 2
    assert stale["updating"] is True
    assert stale["scanned_at"] == old["scanned_at"]
    assert stale["signature"] == old["signature"]


def test_corrupt_folder_and_run_are_diagnostic_rows(service, review_fixture):
    dataset, _, workspace = review_fixture
    bad = Path(dataset["path"]).parent / "broken/meta/info.json"
    write_json(bad, {})
    write_json(workspace / "runs/bad/run.json", {"source_root": dataset["path"]})
    summary = service.summary()
    assert len(summary["datasets"]) == 2
    broken = next(d for d in summary["datasets"] if d["name"] == "broken")
    assert broken["state"] == "Updating" and broken["diagnostics"]
    assert next(d for d in summary["datasets"] if d["name"] == "source")["diagnostics"]


def test_remote_validation_then_unlocked_request_and_cached_display(service, review_fixture):
    _, run, _ = review_fixture
    run["publication"] = {"repo_id": "team/data", "revision": "branch", "main_commit": "old"}
    write_json(Path(run["_path"]), run)
    calls = []

    class API:
        def repo_info(self, repo, **kwargs):
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                assert pool.submit(service.summary).result(timeout=3)["configured"]
            calls.append((repo, kwargs))
            return SimpleNamespace(sha="old")

    service.api_factory = lambda: API()
    with pytest.raises(KeyError):
        service.check("unregistered")
    pub = service.summary()["datasets"][0]["publications"][0]
    result = service.check(pub["id"])
    assert result["status"] == "match"
    assert calls == [("team/data", {"repo_type": "dataset", "revision": "branch", "timeout": 10})]
    assert service.summary()["datasets"][0]["publications"][0]["remote_check"] == result
    assert service.detail(service.summary()["datasets"][0]["id"])["publications"][0]["remote_check"] == result
    run.pop("publication")
    write_json(Path(run["_path"]), run)
    with pytest.raises(KeyError):
        service.check(pub["id"])


def test_caches_bounded(service, review_fixture, monkeypatch):
    _, run, workspace = review_fixture
    for n in range(35):
        value = dict(run, run_id=f"{n:032x}")
        write_json(workspace / "runs" / value["run_id"] / "run.json", value)
    row = service.summary()["datasets"][0]
    for value in row["runs"]:
        service.detail(row["id"], value["run_id"])
    assert len(service._details) == 32
    service.api_factory = lambda: SimpleNamespace(repo_info=lambda *a, **kw: SimpleNamespace(sha="old"))
    for n in range(66):
        write_json(
            workspace / "jobs" / f"{n:032x}.json",
            {
                "run_id": run["run_id"],
                "status": "completed",
                "result": {"repo_id": f"team/data{n}", "revision": "main", "main_commit": "old"},
            },
        )
    for pub in service.summary()["datasets"][0]["publications"]:
        service.check(pub["id"])
    assert len(service._remote_checks) == 64


def test_unconfigured_api_lazy_pair_and_no_arbitrary_roots(app_module, monkeypatch, tmp_path):
    monkeypatch.delenv("LEROBOT_MONITOR_ROOT")
    response = request(app_module, "GET", "/api/monitor?root=/etc")
    assert response.status_code == 200
    assert response.json() == dict(
        configured=False, root=None, scanned_at=None, updating=False, diagnostics=[], datasets=[]
    )
    assert request(app_module, "GET", "/api/monitor/datasets/nope").status_code == 404
    monkeypatch.setenv("LEROBOT_MONITOR_ROOT", str(tmp_path))
    first = app_module._monitor_service()
    assert first is app_module._monitor_service()
    monkeypatch.setattr(app_module, "EXPORT_ROOT", tmp_path / "new-workspace")
    assert app_module._monitor_service() is not first


def test_http_unknown_ids_run_ownership_and_root503(app_module, monkeypatch, tmp_path):
    response = request(app_module, "GET", "/api/monitor")
    assert response.status_code == 200
    assert_public(response.json())
    row = response.json()["datasets"][0]
    assert request(app_module, "GET", "/api/monitor/datasets/" + row["id"] + "?run_id=foreign").status_code == 404
    assert (
        request(
            app_module, "POST", "/api/monitor/publications/unknown/check", json={"repo_id": "evil/repo"}
        ).status_code
        == 404
    )
    monkeypatch.setenv("LEROBOT_MONITOR_ROOT", str(tmp_path / "missing"))
    assert request(app_module, "GET", "/api/monitor").status_code == 503
    assert request(app_module, "GET", "/api/monitor/datasets/anything").status_code == 503
    assert request(app_module, "POST", "/api/monitor/publications/anything/check").status_code == 503


def test_post_uses_recorded_destination_and_hosted_boundary(app_module, review_fixture, monkeypatch):
    _, run, _ = review_fixture
    run["publication"] = {"repo_id": "team/data", "revision": "main", "main_commit": "old"}
    write_json(Path(run["_path"]), run)
    calls = []
    service = app_module._monitor_service()
    service.api_factory = lambda: SimpleNamespace(
        repo_info=lambda repo, **kw: calls.append(repo) or SimpleNamespace(sha="old")
    )
    pub = service.summary()["datasets"][0]["publications"][0]
    url = "/api/monitor/publications/" + pub["id"] + "/check"
    assert request(app_module, "POST", url, json={"repo_id": "evil/repo", "root": "/etc"}).status_code == 200
    assert calls == ["team/data"]
    monkeypatch.setenv("ANNOTATION_BACKEND_TOKEN", "secret")
    for method, path in [("GET", "/api/monitor"), ("GET", "/api/monitor/datasets/id"), ("POST", url)]:
        assert request(app_module, method, path).status_code == 401
        assert request(app_module, method, path, headers={"Authorization": "Bearer secret"}).status_code == 403


def test_default_client_only_created_after_validated_id(service, review_fixture, monkeypatch):
    _, run, _ = review_fixture
    clients = []
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        lambda: clients.append(1) or SimpleNamespace(repo_info=lambda *a, **kw: SimpleNamespace(sha="old")),
    )
    with pytest.raises(KeyError):
        service.check("invalid")
    assert clients == []
    run["publication"] = {"repo_id": "team/data", "revision": "main", "main_commit": "old"}
    write_json(Path(run["_path"]), run)
    before = {str(p): p.read_bytes() for p in service.workspace.parent.rglob("*") if p.is_file()}
    publication = service.summary()["datasets"][0]["publications"][0]
    assert service.check(publication["id"])["status"] == "match"
    assert clients == [1]
    assert before == {str(p): p.read_bytes() for p in service.workspace.parent.rglob("*") if p.is_file()}


def test_root_discovery_sentinel_raises503(app_module, monkeypatch):
    def unavailable(root, workspace):
        row = monitor._empty_dataset(root)
        row["diagnostics"] = [monitor._diag("root_unavailable", "Permission denied")]
        return [row]

    monkeypatch.setattr(monitor, "discover_datasets", unavailable)
    response = request(app_module, "GET", "/api/monitor?refresh=true")
    assert response.status_code == 503
    assert "LEROBOT_MONITOR_ROOT" in response.json()["detail"]


def test_no_run_detail_and_unknown_factory(service, review_fixture):
    dataset, _, _ = review_fixture
    write_v21(Path(dataset["path"]).parent / "unprepared", [])
    row = next(d for d in service.summary()["datasets"] if d["name"] == "unprepared")
    detail = service.detail(row["id"])
    assert detail["run_id"] is None and detail["metrics"] is None
    assert detail["prompts"] is None and detail["exports"] == []


def test_first_changing_snapshot_does_not_publish_mixed_counts(service, monkeypatch):
    real = service._signature
    counter = []

    def changing():
        counter.append(1)
        return real(), len(counter)

    monkeypatch.setattr(service, "_signature", changing)
    snapshot = service.summary()
    assert len(counter) == 4
    assert snapshot["updating"] is True
    assert snapshot["scanned_at"] is None
    assert snapshot["datasets"] == []


def test_nested_private_fields_removed(service, monkeypatch):
    real = monitor.summarize_run

    def summarize(*args):
        summary = real(*args)
        summary["metrics"]["_hidden"] = {"secret": "never HTTP"}
        return summary

    monkeypatch.setattr(monitor, "summarize_run", summarize)
    summary = service.summary()
    assert_public(summary)
    assert_public(service.detail(summary["datasets"][0]["id"]))


def test_detail_refreshes_export_metadata_without_run_edit(service, review_fixture):
    _, run, workspace = review_fixture
    root = write_v21(workspace / "exports/output", [10])
    run["export"] = {"local_path": str(root), "retained_frames": 10}
    write_json(Path(run["_path"]), run)
    row = service.summary()["datasets"][0]
    first = service.detail(row["id"])
    assert first["exports"][0]["seconds"] == 1
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["fps"] = 5
    write_json(path, info)
    second = service.detail(row["id"])
    assert second["exports"][0]["seconds"] == 2
    assert second["signature"] != first["signature"]


def test_detail_tracks_dataset_diagnostics_after_corrupt_run(service, review_fixture):
    dataset, _, workspace = review_fixture
    first = service.detail(dataset["id"])
    assert first["metrics"]["counts_complete"] is True
    assert first["diagnostics"] == []
    write_json(workspace / "runs/bad/run.json", {"source_root": dataset["canonical_path"]})
    row = service.summary()["datasets"][0]
    assert row["runs"][0]["metrics"]["counts_complete"] is False
    second = service.detail(dataset["id"])
    assert second["metrics"] == row["runs"][0]["metrics"]
    assert any(d["code"] == "invalid_run" for d in second["diagnostics"])
    assert second["signature"] != first["signature"]
    (workspace / "runs/bad/run.json").unlink()
    assert service.detail(dataset["id"])["metrics"]["counts_complete"] is True
    assert service.detail(dataset["id"])["diagnostics"] == []


def test_summary_tracks_external_provenance_retarget_and_removal(service, review_fixture):
    dataset, _, workspace = review_fixture
    source = Path(dataset["path"])
    alternate = write_v21(source.parent / "alternate", [10])
    child = write_v21(source.parent / "child", [10])
    intermediate = workspace / "exports/intermediate"
    evidence = write_json(intermediate / "meta/source_episode_mapping.json", {"original_root": str(source)})
    write_json(child / "meta/source_episode_mapping.json", {"original_root": str(intermediate)})

    def rows():
        return {row["name"]: row for row in service.summary()["datasets"]}

    initial = rows()
    assert initial["child"]["parent_ids"] == [initial["source"]["id"]]
    write_json(evidence, {"original_root": str(alternate)})
    changed = rows()
    assert changed["child"]["parent_ids"] == [changed["alternate"]["id"]]
    assert changed["source"]["child_ids"] == []
    assert changed["alternate"]["child_ids"] == [changed["child"]["id"]]
    evidence.unlink()
    removed = rows()
    assert removed["child"]["provenance"]["status"] == "unknown"
    assert removed["child"]["parent_ids"] == []
    assert removed["alternate"]["child_ids"] == []
    write_json(evidence, {"original_root": str(source)})
    assert rows()["child"]["parent_ids"] == [initial["source"]["id"]]


def test_no_run_detail_tracks_metadata_diagnostics(service, review_fixture):
    dataset, _, _ = review_fixture
    root = write_v21(Path(dataset["path"]).parent / "unprepared", [10])
    row = next(d for d in service.summary()["datasets"] if d["name"] == "unprepared")
    first = service.detail(row["id"])
    assert first["run_id"] is None and first["diagnostics"] == []
    info = root / "meta/info.json"
    saved = info.read_text()
    info.write_text("{")
    second = service.detail(row["id"])
    assert second["run_id"] is None
    assert any(d["code"] == "invalid_metadata" for d in second["diagnostics"])
    info.write_text(saved)
    assert service.detail(row["id"])["diagnostics"] == []


@pytest.mark.parametrize("warm", [False, True])
def test_changing_external_provenance_preserves_consistent_snapshot(service, review_fixture, monkeypatch, warm):
    dataset, _, workspace = review_fixture
    source = Path(dataset["path"])
    alternate = write_v21(source.parent / "alternate", [10])
    child = write_v21(source.parent / "child", [10])
    intermediate = workspace / "exports/intermediate"
    evidence = write_json(intermediate / "meta/source_episode_mapping.json", {"original_root": str(source)})
    write_json(child / "meta/source_episode_mapping.json", {"original_root": str(intermediate)})
    old = service.summary() if warm else None
    original = monitor.summarize_run
    calls = []

    def change_after_relationships(*args):
        result = original(*args)
        calls.append(1)
        write_json(evidence, {"original_root": str(alternate if len(calls) % 2 else source)})
        return result

    monkeypatch.setattr(monitor, "summarize_run", change_after_relationships)
    current = service.summary(refresh=True)
    assert len(calls) == 2
    assert current["updating"] is True
    assert current["scanned_at"] == (old["scanned_at"] if old else None)
    assert current["datasets"] == (old["datasets"] if old else [])
