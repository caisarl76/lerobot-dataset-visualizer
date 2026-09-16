"""Frozen export evidence, durable publication receipts and explicit Hub checks."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import annotation_monitor as monitor
import httpx
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from monitor_fixtures import review_fixture as review_fixture, write_json, write_v21
import pytest


def receipt(commit="old", **kwargs):
    return dict(
        repo_id="team/data",
        revision="260915",
        main_commit=commit,
        urls={"main": "https://huggingface.co/datasets/team/data/tree/260915"},
        **kwargs,
    )


def freeze(run, workspace, *, delivery=True):
    from annotation_publish import _review_digest

    output = workspace / "exports/frozen"
    root = write_v21(output / ("training" if delivery else "main"), [10])
    manifest = dict(run_id=run["run_id"], review_sha256=_review_digest(run), files={})
    if delivery:
        nested = write_json(output / "frozen/manifest.json", dict(manifest, run_revision=run["revision"]))
        manifest.update(
            dataset_name="training",
            format="groot_v21",
            instruction_mode="subtask",
            destination={"repo_id": "team/data", "revision": "260915"},
            frozen_manifest_sha256=sha256(nested.read_bytes()).hexdigest(),
            files={"meta/info.json": {}, "data/chunk-000/episode_000000.parquet": {}},
        )
    else:
        manifest.update(
            repo_id=run.get("repo_id"),
            source_commit=run.get("source_commit"),
            source_format=run["source_format"],
            files={"main": {"meta/info.json": {}}, "rich": {}},
        )
    path = write_json(output / "manifest.json", manifest)
    run["export"] = dict(
        root=str(root),
        local_path=str(root),
        export_root=str(output),
        manifest_sha256=sha256(path.read_bytes()).hexdigest(),
        retained_frames=10,
        output_repo_id="local/frozen",
    )
    if delivery:
        run["export"].update(format="groot_v21", instruction_mode="subtask", destination=manifest["destination"])
    run["publication"] = receipt()
    run["publication_state"] = "published"
    return path


def job(workspace, run, result, *, job_id="b" * 32, linked=True, **extra):
    value = dict(job_id=job_id, status="completed", result=result, **extra)
    if linked:
        run["current_job_id"] = job_id
    return write_json(workspace / "jobs" / f"{job_id}.json", value)


def test_delivery_publication_keeps_frozen_counts_after_review_edit(review_fixture):
    dataset, run, workspace = review_fixture
    freeze(run, workspace)
    before = monitor.publication_records(run, workspace)[0]
    assert before["exported_frames"] == 10
    assert before["format"] == "groot_v21"
    assert before["instruction_mode"] == "subtask"
    assert before["linked_run_id"] == run["run_id"]
    assert before["export_available"] is True
    assert before["remote_check"] is None
    assert len(before["id"]) == 24
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["local_changes"] is False
    run["task_prompt"] = "Edited locally after publication"
    after = monitor.publication_records(run, workspace)[0]
    assert after == before
    freshness = monitor.summarize_run(dataset, run, workspace)["freshness"]
    assert freshness["local_changes"] is True
    assert freshness["verifiable"] is True
    assert freshness["publication_state"] == "unpublished_changes"
    assert freshness["source_changed"] is True
    assert freshness["metadata_only"] is True


def test_receipt_survives_normal_edit_that_removes_export(review_fixture):
    dataset, run, workspace = review_fixture
    freeze(run, workspace)
    run.pop("export")
    run["publication_state"] = "draft"
    publication = monitor.publication_records(run, workspace)[0]
    assert publication["commit"] == "old"
    assert publication["export_path"] is None
    assert publication["export_available"] is False
    freshness = monitor.summarize_run(dataset, run, workspace)["freshness"]
    assert freshness["local_changes"] is None
    assert freshness["verifiable"] is False
    assert freshness["publication_state"] == "unpublished_changes"


def test_new_unpublished_preview_never_attaches_to_old_same_destination_receipt(review_fixture):
    dataset, run, workspace = review_fixture
    freeze(run, workspace)
    run["publication_state"] = "exported"
    publication = monitor.publication_records(run, workspace)[0]
    assert publication["export_path"] is None
    assert publication["manifest_sha256"] is None
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["verifiable"] is False
    assert len(monitor.export_records(run, workspace)) == 1


def test_explicit_frozen_hash_links_receipt_even_with_draft_state(review_fixture):
    dataset, run, workspace = review_fixture
    freeze(run, workspace)
    run["publication"]["manifest_sha256"] = run["export"]["manifest_sha256"]
    run["publication_state"] = "draft"
    run["task_prompt"] = "changed"
    assert monitor.publication_records(run, workspace)[0]["exported_frames"] == 10
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["local_changes"] is True


@pytest.mark.parametrize(
    "damage",
    [
        "manifest_missing",
        "manifest_tampered",
        "wrong_run",
        "wrong_destination",
        "nested_wrong_run",
        "review_missing",
        "export_missing",
        "file_missing",
    ],
)
def test_unreliable_or_missing_export_never_claims_usable_training_path(review_fixture, damage):
    dataset, run, workspace = review_fixture
    path = freeze(run, workspace)
    if damage == "manifest_missing":
        path.unlink()
    elif damage == "manifest_tampered":
        path.write_text(path.read_text() + " ")
    elif damage in {"wrong_run", "wrong_destination"}:
        value = json.loads(path.read_text())
        value["run_id" if damage == "wrong_run" else "destination"] = (
            "c" * 32 if damage == "wrong_run" else {"repo_id": "other/repo", "revision": "main"}
        )
        write_json(path, value)
        run["export"]["manifest_sha256"] = sha256(path.read_bytes()).hexdigest()
    elif damage == "nested_wrong_run":
        nested = path.parent / "frozen/manifest.json"
        value = json.loads(nested.read_text())
        value["run_id"] = "c" * 32
        write_json(nested, value)
        outer = json.loads(path.read_text())
        outer["frozen_manifest_sha256"] = sha256(nested.read_bytes()).hexdigest()
        write_json(path, outer)
        run["export"]["manifest_sha256"] = sha256(path.read_bytes()).hexdigest()
    elif damage == "review_missing":
        (Path(run["root"]) / "meta/lerobot_annotations.json").unlink()
    elif damage == "export_missing":
        run["export"]["root"] = run["export"]["local_path"] = str(workspace / "missing")
    else:
        (Path(run["export"]["root"]) / "data/chunk-000/episode_000000.parquet").unlink()
    publication = monitor.publication_records(run, workspace)[0]
    assert publication["commit"] == "old"
    if damage != "review_missing":
        assert publication["export_available"] is False
    if damage not in {"export_missing", "file_missing"}:
        assert monitor.summarize_run(dataset, run, workspace)["freshness"]["verifiable"] is False


def test_unpublished_export_has_independent_frozen_metadata(review_fixture):
    _, run, workspace = review_fixture
    freeze(run, workspace)
    run.pop("publication")
    run["publication_state"] = "exported"
    assert monitor.publication_records(run, workspace) == []
    exports = monitor.export_records(run, workspace)
    assert len(exports) == 1
    assert exports[0] == dict(
        id=exports[0]["id"],
        run_id=run["run_id"],
        path=run["export"]["local_path"],
        available=True,
        format="groot_v21",
        instruction_mode="subtask",
        frames=10,
        seconds=1.0,
        manifest_sha256=run["export"]["manifest_sha256"],
        output_repo_id="local/frozen",
    )
    run["episodes"].clear()
    assert monitor.export_records(run, workspace) == exports


def test_legacy_frozen_manifest_does_not_invent_destination_or_format(review_fixture):
    dataset, run, workspace = review_fixture
    freeze(run, workspace, delivery=False)
    run["publication"]["manifest_sha256"] = run["export"]["manifest_sha256"]
    pub = monitor.publication_records(run, workspace)[0]
    assert pub["export_path"] == run["export"]["root"]
    assert pub["format"] is None
    assert pub["instruction_mode"] is None
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["local_changes"] is False


def test_current_job_shape_attaches_and_deduplicates_receipts(review_fixture):
    _, run, workspace = review_fixture
    run["publication"] = receipt()
    job(workspace, run, receipt())  # No kind/run_id in actual saved jobs.
    assert len(monitor.publication_records(run, workspace)) == 1
    job(workspace, run, receipt("older"), job_id="c" * 32, linked=False, run_id=run["run_id"], kind="publish")
    job(workspace, run, receipt("unrelated"), job_id="d" * 32, linked=False, run_id="e" * 32, kind="publish")
    assert [p["commit"] for p in monitor.publication_records(run, workspace)] == ["old", "older"]
    assert monitor.publication_records(run, workspace)[1]["export_path"] is None


@pytest.mark.parametrize("extra", [{}, {"run_id": "e" * 32}, {"kind": "generate"}, {"status": "failed"}])
def test_unlinked_wrong_run_generation_or_failed_jobs_do_not_publish(review_fixture, extra):
    _, run, workspace = review_fixture
    linked = bool(extra)
    job(workspace, run, receipt(), linked=linked)
    path = workspace / "jobs" / (("b" * 32) + ".json")
    value = json.loads(path.read_text())
    value.update(extra)
    write_json(path, value)
    assert monitor.publication_records(run, workspace) == []


def test_completed_generation_destination_is_not_publication(review_fixture):
    dataset, run, workspace = review_fixture
    job(workspace, run, {"repo_id": "team/data", "revision": "260915", "episodes": [0]})
    assert monitor.publication_records(run, workspace) == []
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["publication_state"] == "draft"


def test_current_publish_job_can_recover_missing_run_receipt(review_fixture):
    dataset, run, workspace = review_fixture
    freeze(run, workspace)
    run.pop("publication")
    job(workspace, run, receipt())
    pub = monitor.publication_records(run, workspace)[0]
    assert pub["commit"] == "old"
    assert pub["exported_frames"] == 10
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["verifiable"] is True


def test_explicitly_linked_export_job_remains_separate_from_publications(review_fixture):
    _, run, workspace = review_fixture
    freeze(run, workspace)
    result = run.pop("export")
    run.pop("publication")
    job(workspace, run, result)
    assert monitor.publication_records(run, workspace) == []
    assert monitor.export_records(run, workspace)[0]["frames"] == 10


class FakeHub:
    def __init__(self, commit="old", error=None):
        self.commit, self.error, self.calls = commit, error, []

    def repo_info(self, repo_id, *, repo_type, revision, timeout):
        self.calls.append((repo_id, repo_type, revision, timeout))
        if self.error:
            raise self.error
        return SimpleNamespace(sha=self.commit)

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected HF operation: {name}")


@pytest.mark.parametrize(("head", "status"), [("old", "match"), ("new", "changed")])
def test_remote_check_uses_only_recorded_destination(head, status):
    publication = dict(repo_id="team/data", revision="260915", commit="old")
    saved = deepcopy(publication)
    api = FakeHub(head)
    result = monitor.check_publication(publication, api)
    assert result["status"] == status
    assert result["current_commit"] == head
    assert result["checked_at"]
    assert api.calls == [("team/data", "dataset", "260915", 10)]
    assert publication == saved


def hub_error(cls, code):
    return cls(
        "Private server detail must not be exposed",
        response=httpx.Response(
            code, request=httpx.Request("GET", "https://huggingface.co/api/datasets/team/data")
        ),
    )


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (hub_error(RevisionNotFoundError, 404), "missing"),
        (hub_error(HfHubHTTPError, 401), "access_denied"),
        (hub_error(HfHubHTTPError, 403), "access_denied"),
        (hub_error(GatedRepoError, 403), "access_denied"),
        (hub_error(RepositoryNotFoundError, 404), "unavailable"),
        (hub_error(RepositoryNotFoundError, 401), "unavailable"),
        (hub_error(HfHubHTTPError, 500), "unavailable"),
        (httpx.ReadTimeout("timeout"), "unavailable"),
        (ConnectionError("network"), "unavailable"),
    ],
)
def test_remote_failure_preserves_receipt_with_precise_status(error, status):
    pub = dict(repo_id="team/data", revision="260915", commit="old")
    saved = deepcopy(pub)
    result = monitor.check_publication(pub, FakeHub(error=error))
    assert result["status"] == status
    assert result["current_commit"] is None
    assert result["checked_at"]
    assert "Private server detail" not in result["message"]
    if type(error) is RepositoryNotFoundError:
        assert "inaccessible" in result["message"].lower()
    assert pub == saved


@pytest.mark.parametrize("key", ["repo_id", "revision", "commit"])
def test_incomplete_receipt_cannot_trigger_hub_lookup(key):
    pub = dict(repo_id="team/data", revision="260915", commit="old")
    pub.pop(key)
    api = FakeHub()
    assert monitor.check_publication(pub, api)["status"] == "unavailable"
    assert api.calls == []


def test_summary_and_local_records_are_read_only_without_remote_or_inventory(review_fixture, monkeypatch):
    import annotation_publish
    import huggingface_hub

    dataset, run, workspace = review_fixture
    freeze(run, workspace)

    def forbidden(*args, **kwargs):
        raise AssertionError("Monitoring must not call HF or hash source/data/video bytes")

    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", forbidden)
    monkeypatch.setattr(annotation_publish, "_inventory", forbidden)
    monkeypatch.setattr(annotation_publish, "_hash_file", forbidden)
    before = {str(p): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    saved = deepcopy(run)
    assert monitor.publication_records(run, workspace)[0]["commit"] == "old"
    assert monitor.export_records(run, workspace)[0]["frames"] == 10
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["verifiable"] is True
    assert run == saved
    assert before == {str(p): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}


def test_foreign_receipt_and_foreign_destination_cannot_borrow_export(review_fixture):
    _, run, workspace = review_fixture
    freeze(run, workspace)
    run["publication"]["run_id"] = "e" * 32
    assert monitor.publication_records(run, workspace) == []
    run["publication"].pop("run_id")
    run["publication"]["repo_id"] = "foreign/repo"
    run["publication"]["manifest_sha256"] = run["export"]["manifest_sha256"]
    assert monitor.publication_records(run, workspace)[0]["export_path"] is None


def test_wrong_recorded_training_path_never_appears_available(review_fixture):
    _, run, workspace = review_fixture
    freeze(run, workspace)
    wrong = write_v21(workspace / "other-training", [10])
    run["export"]["local_path"] = run["export"]["root"] = str(wrong)
    assert monitor.export_records(run, workspace)[0]["available"] is False
    assert monitor.publication_records(run, workspace)[0]["export_available"] is False


def test_missing_nested_manifest_and_no_frozen_receipt_report_unknown(review_fixture):
    dataset, run, workspace = review_fixture
    path = freeze(run, workspace)
    (path.parent / "frozen/manifest.json").unlink()
    assert monitor.summarize_run(dataset, run, workspace)["freshness"]["verifiable"] is False
    run["export"] = {}
    export = monitor.export_records(run, workspace)[0]
    assert export["path"] is None
    assert export["available"] is False
    assert export["frames"] is None
    assert export["seconds"] is None


def test_missing_hub_head_returns_unavailable():
    class NoHead(FakeHub):
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace()

    result = monitor.check_publication(dict(repo_id="team/data", revision="260915", commit="old"), NoHead())
    assert result["status"] == "unavailable"
    assert result["current_commit"] is None


def test_generation_result_with_source_commit_is_not_publish_receipt(review_fixture):
    _, run, workspace = review_fixture
    job(workspace, run, dict(repo_id="team/data", revision="260915", commit="source-commit", episodes=[0]))
    assert monitor.publication_records(run, workspace) == []
