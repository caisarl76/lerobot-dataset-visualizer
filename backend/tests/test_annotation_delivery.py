# ruff: noqa: F811
from pathlib import Path
from types import SimpleNamespace

import annotation_delivery as delivery
import httpx
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
import pytest
from test_annotation_publish import FakeHub, reviewed_run  # noqa: F401


class DestinationHub(FakeHub):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.repo_exists = True
        self.created = []

    def repo_info(self, repo_id, *, revision="main", **kwargs):
        if not self.repo_exists:
            raise RepositoryNotFoundError(
                "missing repo", response=httpx.Response(404, request=httpx.Request("GET", "https://hub.test"))
            )
        if revision not in self.heads and revision not in self.commits:
            raise RevisionNotFoundError(
                "missing revision", response=httpx.Response(404, request=httpx.Request("GET", "https://hub.test"))
            )
        return SimpleNamespace(sha=self.heads.get(revision, revision), private=False)

    def list_repo_refs(self, *args, **kwargs):
        return SimpleNamespace(
            tags=[], branches=[SimpleNamespace(name=k, target_commit=v) for k, v in self.heads.items()]
        )

    def create_repo(self, repo_id, **kwargs):
        assert not self.repo_exists
        self.created.append((repo_id, kwargs))
        self.repo_exists = True
        self.heads = {"main": "initial"}
        self.commits["initial"] = {}


@pytest.fixture
def options(monkeypatch):
    monkeypatch.delenv("ANNOTATION_HUB_REPOS", raising=False)
    monkeypatch.delenv("ANNOTATION_BACKEND_TOKEN", raising=False)
    return dict(
        export_format="rich",
        instruction_mode="task",
        dataset_name="retained",
        destination_repo_id="team/correct-dataset",
        destination_revision="ver.260909",
        destination_private=True,
    )


def ready(run, tmp_path, options, api):
    run["export"] = delivery.prepare_delivery(run, tmp_path / "delivery", options, api=api)
    return run["export"]["manifest_sha256"]


def test_preview_is_read_only_and_publish_targets_frozen_other_repo(reviewed_run, tmp_path, options):
    api = DestinationHub(tmp_path)
    digest = ready(reviewed_run, tmp_path, options, api)
    frozen = reviewed_run["export"]
    assert frozen["destination"]["repo_id"] == "team/correct-dataset"
    assert frozen["destination"]["revision"] == "ver.260909"
    assert frozen["local_path"].endswith("/retained")
    assert frozen["retained_frames"] == 12
    assert api.operations == [] and api.created == [] and list(api.heads) == ["main"]
    result = delivery.publish_delivery(reviewed_run, digest, api=api)
    assert result["repo_id"] == "team/correct-dataset"
    assert result["revision"] == "ver.260909"
    assert api.heads["main"] == "source-sha"
    assert api.operations[-1][:2] == ("ver.260909", "source-sha")
    files = api.commits[result["main_commit"]]
    assert b"12 frames" in files["README.md"]
    assert "data/chunk-000/obsolete.parquet" not in files
    assert files["assets/team.png"] == b"keep asset"
    assert delivery.publish_delivery(reviewed_run, digest, api=api) == result
    assert len(api.operations) == 1


def test_new_repo_creation_only_on_publish(reviewed_run, tmp_path, options):
    api = DestinationHub(tmp_path)
    api.repo_exists = False
    options["destination_revision"] = "main"
    digest = ready(reviewed_run, tmp_path, options, api)
    assert not api.created and not reviewed_run["export"]["destination"]["exists"]
    result = delivery.publish_delivery(reviewed_run, digest, api=api)
    assert api.created == [("team/correct-dataset", dict(repo_type="dataset", private=True, exist_ok=False))]
    assert result["revision"] == "main"


@pytest.mark.parametrize("change", ["target", "files", "review", "branch", "new_branch"])
def test_stale_preview_never_publishes(reviewed_run, tmp_path, options, change):
    api = DestinationHub(tmp_path)
    digest = ready(reviewed_run, tmp_path, options, api)
    if change == "target":
        reviewed_run["export"]["destination"]["repo_id"] = "team/wrong"
    elif change == "files":
        (Path(reviewed_run["export"]["local_path"]) / "README.md").write_text("changed")
    elif change == "review":
        reviewed_run["episodes"]["0"]["decision"] = "delete"
    elif change == "branch":
        api.heads["main"] = "changed"
    else:
        api.heads["ver.260909"] = "source-sha"
    with pytest.raises(ValueError):
        delivery.publish_delivery(reviewed_run, digest, api=api)
    assert api.operations == []


def test_local_preview_and_hosted_allowlist(reviewed_run, tmp_path, options, monkeypatch):
    options["destination_repo_id"] = None
    digest = ready(reviewed_run, tmp_path, options, DestinationHub(tmp_path))
    with pytest.raises(ValueError, match="local only"):
        delivery.publish_delivery(reviewed_run, digest)
    options["destination_repo_id"] = "team/unlisted"
    monkeypatch.setenv("ANNOTATION_BACKEND_TOKEN", "secret")
    with pytest.raises(ValueError, match="allowlist"):
        delivery.validate_options(options)


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", "a\\b", "", "frozen", "manifest.json"])
def test_invalid_output_name_is_rejected(options, name):
    options["dataset_name"] = name
    with pytest.raises(ValueError):
        delivery.validate_options(options)


def test_remote_commit_timeout_retry_is_idempotent(reviewed_run, tmp_path, options):
    api = DestinationHub(tmp_path)
    api.timeout_main_once = True
    options["destination_revision"] = "main"
    digest = ready(reviewed_run, tmp_path, options, api)
    with pytest.raises(TimeoutError):
        delivery.publish_delivery(reviewed_run, digest, api=api)
    result = delivery.publish_delivery(reviewed_run, digest, api=api)
    assert len(api.operations) == 1 and result["main_commit"] == api.heads["main"]


def test_groot_delivery_materializes_retained_prompts_and_metadata(reviewed_run, tmp_path, options):
    import json

    import annotation_history as history
    from annotation_runs import source_inventory
    import pyarrow.parquet as pq

    root = Path(reviewed_run["root"])
    (root / "meta/modality.json").write_text(json.dumps({"state": {"body": {"start": 0, "end": 2}}}))
    labels = json.loads((root / "meta/lerobot_annotations.json").read_text())
    for ep, row in labels["episodes"].items():
        row["atoms"] = [
            dict(
                style="task_aug",
                role="user",
                content=f"verified object {ep}",
                timestamp=0.0,
                camera=None,
                tool_calls=None,
            )
        ]
        history.save_review(root, int(ep), row["atoms"], True, history.annotation_hash(row["atoms"]))
    (root / "meta/lerobot_annotations.json").write_text(json.dumps(labels))
    reviewed_run["source_file_hashes"] = source_inventory(root)
    reviewed_run["episodes"]["1"]["decision"] = "delete"
    reviewed_run["subtask_prompts"] = []
    options.update(export_format="groot_v21", destination_repo_id=None)
    result = delivery.prepare_delivery(reviewed_run, tmp_path / "delivery", options)
    dataset = Path(result["local_path"])
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_frames"] == 8 and info["total_episodes"] == 2
    tasks = {
        r["task_index"]: r["task"]
        for r in map(json.loads, (dataset / "meta/tasks.jsonl").read_text().splitlines())
    }
    for new, old in enumerate([0, 2]):
        data = pq.read_table(dataset / f"data/chunk-000/episode_{new:06d}.parquet")
        assert {tasks[i] for i in data["task_index"].to_pylist()} == {f"verified object {old}"}
    assert result["old_to_new"] == {"0": 0, "2": 1}
    assert str(root) not in (dataset / "meta/groot_instruction_export.json").read_text()
