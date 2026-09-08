"""Publication freezes reviewed data and uses conflict-safe bounded Hub commits."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import annotation_history as history
import annotation_publish as publish
from annotation_runs import source_inventory
import datasets
from huggingface_hub import CommitOperationAdd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import official_annotations as engine
import pyarrow.parquet as pq
import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def reviewed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    root = tmp_path / "draft"
    dataset = LeRobotDataset.create(
        repo_id="local/test",
        fps=10,
        root=root,
        use_videos=False,
        features={"observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}},
    )
    for ep in range(3):
        for frame in range(4):
            dataset.add_frame({"observation.state": np.array([ep, frame], dtype=np.float32), "task": "pick trash"})
        dataset.save_episode()
    dataset.finalize()
    atoms = [
        {
            "style": "subtask",
            "role": "assistant",
            "content": "pick trash",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        }
    ]
    write_json(
        root / "meta/lerobot_annotations.json",
        {"version": 2, "episodes": {str(ep): {"atoms": atoms} for ep in range(3)}},
    )
    for ep in range(3):
        history.save_review(root, ep, atoms, True, history.annotation_hash(atoms))
    write_json(root / "meta/annotation_run.json", {"run_id": "abc", "root": str(root)})
    write_json(
        root / "meta/annotation_predictions/first.json",
        {
            "source": str(root),
            "config": {"root": str(root), "vlm": {"api_key": "secret", "model": "qwen"}},
            "episodes": {"0": {"atoms": atoms}},
            "episode_results": {"1": {"generation_status": "failed"}},
        },
    )
    return {
        "root": str(root),
        "source_root": str(root),
        "source_format": "v3.1",
        "source_file_hashes": source_inventory(root),
        "source_commit": "source-sha",
        "repo_id": "team/data",
        "run_id": "a" * 32,
        "revision": 4,
        "subtask_prompts": ["pick trash"],
        "episodes": {str(ep): {"decision": "keep", "issues": []} for ep in range(3)},
    }


@pytest.mark.parametrize("problem", ["unreviewed", "stale_review", "bad_atom", "prompt_mismatch", "pending_issue"])
def test_export_blocks_unreviewed_or_unresolved_retained_episodes(reviewed_run, tmp_path, problem):
    run = reviewed_run
    root = Path(run["root"])
    if problem == "unreviewed":
        history.write_reviews(root, {})
    elif problem == "pending_issue":
        run["episodes"]["0"].update(decision="pending", issues=[{"code": "wrong_task"}])
    else:
        path = root / "meta/lerobot_annotations.json"
        labels = json.loads(path.read_text())
        atoms = labels["episodes"]["0"]["atoms"]
        atoms[0]["content"] = "wrong label"
        if problem == "bad_atom":
            atoms[0]["timestamp"] = float("nan")
        write_json(path, labels)
        if problem != "stale_review":
            history.save_review(root, 0, atoms, True, history.annotation_hash(atoms))
    with pytest.raises(ValueError):
        publish.prepare_export(run, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_frozen_export_applies_delete_once_and_preserves_source_history(reviewed_run, tmp_path, monkeypatch):
    run = reviewed_run
    root = Path(run["root"])
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    run["episodes"]["0"]["decision"] = run["episodes"]["1"]["decision"] = "delete"
    original = engine.delete_dataset_episodes
    calls = []

    def deleting(*args, **kwargs):
        calls.append(list(args[3]))
        return original(*args, **kwargs)

    monkeypatch.setattr(publish, "delete_dataset_episodes", deleting)
    result = publish.prepare_export(run, tmp_path / "export")
    assert calls == [[0, 1]]
    assert result["old_to_new"] == {"2": 0}
    assert result["retained_episodes"] == 1
    rich = Path(result["rich_root"])
    record = list(engine.iter_episodes(rich))[0]
    assert pq.read_table(record.data_path)["observation.state"].to_pylist() == [[2.0, float(i)] for i in range(4)]
    assert history.review_status(rich, 0, engine.read_atoms(record))["status"] == "reviewed"
    snapshot = (rich / "meta/annotation_predictions/first.json").read_text()
    assert str(root) not in snapshot and "secret" not in snapshot and "qwen" in snapshot
    assert not (rich / "meta/annotation_run.json").exists()
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_deleting_all_episodes_is_rejected(reviewed_run, tmp_path):
    for episode in reviewed_run["episodes"].values():
        episode["decision"] = "delete"
    with pytest.raises(ValueError, match="all"):
        publish.prepare_export(reviewed_run, tmp_path / "export")


class FakeHub:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.heads = {"main": "source-sha"}
        self.commits = {
            "source-sha": {
                "README.md": b"keep card",
                "LICENSE": b"keep license",
                "assets/team.png": b"keep asset",
                "data/chunk-000/obsolete.parquet": b"old",
                "meta/episodes.jsonl": b"old source",
            }
        }
        self.operations = []
        self.timeout_main_once = False

    def repo_info(self, repo_id, *, revision="main", **kwargs):
        return SimpleNamespace(sha=self.heads.get(revision, revision))

    def list_repo_files(self, repo_id, *, revision="main", **kwargs):
        return list(self.commits[self.heads.get(revision, revision)])

    def get_paths_info(self, repo_id, paths, *, revision="main", **kwargs):
        files = self.commits[self.heads.get(revision, revision)]
        return [
            SimpleNamespace(
                path=p,
                size=len(files[p]),
                lfs=None,
                blob_id=hashlib.sha1(f"blob {len(files[p])}\0".encode() + files[p]).hexdigest(),
            )
            for p in paths
            if p in files
        ]

    def create_branch(self, repo_id, *, branch, revision, **kwargs):
        self.heads.setdefault(branch, self.heads.get(revision, revision))

    def create_commit(self, repo_id, *, operations, revision, parent_commit, **kwargs):
        assert self.heads[revision] == parent_commit
        files = dict(self.commits[parent_commit])
        operations = list(operations)
        for operation in operations:
            if isinstance(operation, CommitOperationAdd):
                value = operation.path_or_fileobj
                files[operation.path_in_repo] = value if isinstance(value, bytes) else Path(value).read_bytes()
            else:
                files.pop(operation.path_in_repo)
        sha = f"commit-{len(self.operations)}"
        self.operations.append((revision, parent_commit, operations))
        self.commits[sha] = files
        self.heads[revision] = sha
        if revision == "main" and self.timeout_main_once:
            self.timeout_main_once = False
            raise TimeoutError("Server committed but client timed out")
        return SimpleNamespace(oid=sha)

    def hf_hub_download(self, repo_id, filename, *, revision, **kwargs):
        value = self.commits[self.heads.get(revision, revision)][filename]
        path = self.tmp_path / "download.json"
        path.write_bytes(value)
        return str(path)


def ready(run, tmp_path, monkeypatch):
    run["export"] = publish.prepare_export(run, tmp_path / "export")
    monkeypatch.setenv("ANNOTATION_HUB_REPOS", "team/data")
    return FakeHub(tmp_path)


def test_publish_preserves_assets_and_deletes_obsolete_managed_files_with_parent(
    reviewed_run, tmp_path, monkeypatch
):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    result = publish.publish_export(reviewed_run, reviewed_run["export"]["manifest_sha256"], api=api)
    main = api.commits[result["main_commit"]]
    assert main["README.md"] == b"keep card" and main["assets/team.png"] == b"keep asset"
    assert "data/chunk-000/obsolete.parquet" not in main and "meta/episodes.jsonl" not in main
    assert api.operations[-1][0:2] == ("main", "source-sha")
    assert api.heads["raw/" + reviewed_run["run_id"]] == "source-sha"
    assert api.operations[0][0] == result["rich_revision"]
    assert not any(path.startswith(".annotate_staging/") for path in api.commits[result["rich_commit"]])
    assert result["urls"]["raw"].endswith("raw%2F" + reviewed_run["run_id"])
    assert reviewed_run["export"]["validation"]["ok"]


def test_remote_change_blocks_all_publication_commits(reviewed_run, tmp_path, monkeypatch):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    api.heads["main"] = "someone-else"
    api.commits["someone-else"] = {}
    with pytest.raises(ValueError, match="conflict"):
        publish.publish_export(reviewed_run, reviewed_run["export"]["manifest_sha256"], api=api)
    assert api.operations == []


def test_retry_after_successful_commit_timeout_reuses_receipt(reviewed_run, tmp_path, monkeypatch):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    api.timeout_main_once = True
    digest = reviewed_run["export"]["manifest_sha256"]
    with pytest.raises(TimeoutError):
        publish.publish_export(reviewed_run, digest, api=api)
    result = publish.publish_export(reviewed_run, digest, api=api)
    assert result["main_commit"] == api.heads["main"]
    assert len(api.operations) == 2


def test_manifest_tampering_and_unallowlisted_repo_block_upload(reviewed_run, tmp_path, monkeypatch):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    digest = reviewed_run["export"]["manifest_sha256"]
    monkeypatch.delenv("ANNOTATION_HUB_REPOS")
    with pytest.raises(ValueError, match="allowlist"):
        publish.publish_export(reviewed_run, digest, api=api)
    monkeypatch.setenv("ANNOTATION_HUB_REPOS", "team/data")
    (Path(reviewed_run["export"]["root"]) / "meta/info.json").write_text("tampered")
    with pytest.raises(ValueError, match="changed|manifest"):
        publish.publish_export(reviewed_run, digest, api=api)
    assert api.operations == []


def make_v21_source(run, tmp_path):
    source = tmp_path / "original-v21"
    root = Path(run["root"])
    info = json.loads((root / "meta/info.json").read_text())
    info.update(
        codebase_version="v2.1",
        chunks_size=1000,
        total_chunks=1,
        total_videos=3,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    info["features"]["observation.images.front"] = {"dtype": "video", "shape": [24, 32, 3]}
    write_json(source / "meta/info.json", info)
    write_json(source / "meta/modality.json", {"state": {"joint": {"start": 0, "end": 2}}})
    episodes, stats = [], []
    for record in engine.iter_episodes(root):
        ep = record.episode_index
        table = pq.read_table(record.data_path).slice(record.row_offset, record.row_count)
        path = source / f"data/chunk-000/episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        video = source / f"videos/chunk-000/observation.images.front/episode_{ep:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        import av

        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.height, stream.width, stream.pix_fmt = 24, 32, "yuv420p"
            for _ in range(record.row_count):
                frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), ep * 30, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        episodes.append({"episode_index": ep, "length": record.row_count, "tasks": ["pick trash"]})
        values = np.asarray(table["observation.state"].to_pylist())
        stats.append(
            {
                "episode_index": ep,
                "stats": {
                    "observation.state": {
                        "min": values.min(0).tolist(),
                        "max": values.max(0).tolist(),
                        "mean": values.mean(0).tolist(),
                        "std": values.std(0).tolist(),
                        "count": [len(table)],
                    }
                },
            }
        )
    for name, rows in (
        ("episodes", episodes),
        ("episodes_stats", stats),
        ("tasks", [{"task_index": 0, "task": "pick trash"}]),
    ):
        (source / f"meta/{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(source / "meta/stats.json", {"observation.state": stats[0]["stats"]["observation.state"]})
    run.update(source_root=str(source), source_format="v2.1", source_file_hashes=source_inventory(source))
    return source


def test_v21_deletion_uses_one_mapping_and_preserves_noninstruction_frames(reviewed_run, tmp_path):
    run = reviewed_run
    source = make_v21_source(run, tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    run["episodes"]["0"]["decision"] = run["episodes"]["1"]["decision"] = "delete"
    result = publish.prepare_export(run, tmp_path / "export-v21")
    main = Path(result["root"])
    assert json.loads((main / "meta/info.json").read_text())["codebase_version"] == "v2.1"
    actual = pq.read_table(main / "data/chunk-000/episode_000000.parquet")
    original = pq.read_table(source / "data/chunk-000/episode_000002.parquet")
    for column in original.column_names:
        if column not in {"episode_index", "index", "task_index"}:
            assert actual[column].equals(original[column]), column
    assert actual["episode_index"].to_pylist() == [0] * 4
    assert actual["index"].to_pylist() == list(range(4))
    assert (main / "videos/chunk-000/observation.images.front/episode_000000.mp4").read_bytes() == (
        source / "videos/chunk-000/observation.images.front/episode_000002.mp4"
    ).read_bytes()
    assert json.loads((main / "meta/stats.json").read_text())["observation.state"]["mean"] == [2.0, 1.5]
    assert result["old_to_new"] == {"2": 0}
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


def test_operation_cap_rejects_before_creating_any_commit(reviewed_run, tmp_path, monkeypatch):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    monkeypatch.setattr(publish, "MAX_OPERATIONS", 2)
    with pytest.raises(ValueError, match="cap"):
        publish.publish_export(reviewed_run, reviewed_run["export"]["manifest_sha256"], api=api)
    assert api.operations == [] and set(api.heads) == {"main"}


def test_old_publication_receipt_is_replaced_not_frozen_as_dataset_content(reviewed_run, tmp_path, monkeypatch):
    write_json(Path(reviewed_run["root"]) / "meta/annotation_publication.json", {"manifest_sha256": "old"})
    api = ready(reviewed_run, tmp_path, monkeypatch)
    result = publish.publish_export(reviewed_run, reviewed_run["export"]["manifest_sha256"], api=api)
    receipt = json.loads(api.commits[result["main_commit"]]["meta/annotation_publication.json"])
    assert receipt["manifest_sha256"] == reviewed_run["export"]["manifest_sha256"]


def test_changed_review_after_export_invalidates_frozen_publication(reviewed_run, tmp_path, monkeypatch):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    history.write_reviews(Path(reviewed_run["root"]), {})
    with pytest.raises(ValueError, match="review|stale"):
        publish.publish_export(reviewed_run, reviewed_run["export"]["manifest_sha256"], api=api)
    assert api.operations == []


def test_retry_after_rich_commit_timeout_verifies_and_reuses_rich_branch(reviewed_run, tmp_path, monkeypatch):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    original = api.create_commit
    timed_out = False

    def commit(*args, **kwargs):
        nonlocal timed_out
        result = original(*args, **kwargs)
        if kwargs["revision"] != "main" and not timed_out:
            timed_out = True
            raise TimeoutError("Rich commit completed")
        return result

    api.create_commit = commit
    digest = reviewed_run["export"]["manifest_sha256"]
    with pytest.raises(TimeoutError):
        publish.publish_export(reviewed_run, digest, api=api)
    result = publish.publish_export(reviewed_run, digest, api=api)
    assert len(api.operations) == 2
    assert result["rich_commit"] == "commit-0"


def test_keep_cannot_publish_corrupt_video(reviewed_run, tmp_path):
    run = reviewed_run
    source = make_v21_source(run, tmp_path)
    run["episodes"]["0"].update(
        decision="keep", issues=[{"code": "invalid_video", "source": "deterministic", "severity": "error"}]
    )
    next(source.rglob("*.mp4")).write_bytes(b"corrupt video")
    run["source_file_hashes"] = source_inventory(source)  # The source was already corrupt when pinned.
    with pytest.raises(ValueError, match="video|structur"):
        publish.prepare_export(run, tmp_path / "corrupt-export")
    assert not (tmp_path / "corrupt-export").exists()


def test_keep_cannot_publish_nonfinite_robot_data(reviewed_run, tmp_path):
    root = Path(reviewed_run["root"])
    record = next(engine.iter_episodes(root))
    table = pq.read_table(record.data_path)
    values = table["observation.state"].to_pylist()
    values[0][0] = float("nan")
    import pyarrow as pa

    column = table.schema.get_field_index("observation.state")
    table = table.set_column(
        column, table.schema.field(column), pa.array(values, type=table.schema.field(column).type)
    )
    pq.write_table(table, record.data_path)
    reviewed_run["source_file_hashes"] = source_inventory(root)  # Invalid source supplied at preparation.
    with pytest.raises(ValueError, match="finite|structur"):
        publish.prepare_export(reviewed_run, tmp_path / "nan-export")


def test_frozen_media_are_independent_of_source(reviewed_run, tmp_path):
    source = make_v21_source(reviewed_run, tmp_path)
    result = publish.prepare_export(reviewed_run, tmp_path / "independent-export")
    original = next(source.rglob("*.mp4"))
    frozen = Path(result["root"]) / original.relative_to(source)
    before = frozen.read_bytes()
    original.write_bytes(b"source was changed after preparation")
    assert frozen.read_bytes() == before


@pytest.mark.parametrize("branch", ["raw", "annotations"])
def test_completed_retry_checks_named_revision_integrity(reviewed_run, tmp_path, monkeypatch, branch):
    api = ready(reviewed_run, tmp_path, monkeypatch)
    digest = reviewed_run["export"]["manifest_sha256"]
    publish.publish_export(reviewed_run, digest, api=api)
    api.heads[f"{branch}/{reviewed_run['run_id']}"] = "moved-by-another-writer"
    with pytest.raises(ValueError, match="revision|conflict"):
        publish.publish_export(reviewed_run, digest, api=api)
    assert len(api.operations) == 2


def test_published_provenance_keeps_issue_evidence_and_decision_reason(reviewed_run, tmp_path):
    finding = {
        "code": "wrong_task",
        "source": "vlm",
        "severity": "warning",
        "message": "uncertain grasp",
        "start": 0.0,
        "end": 0.1,
    }
    reviewed_run["episodes"]["0"].update(
        issues=[finding], decision="keep", decision_reason="Human checked the grasp"
    )
    result = publish.prepare_export(reviewed_run, tmp_path / "export-with-reasons")
    for key in ("root", "rich_root"):
        metadata = json.loads((Path(result[key]) / "meta/source_episode_mapping.json").read_text())
        assert metadata["episode_decisions"]["0"]["decision_reason"] == "Human checked the grasp"
        assert metadata["episode_decisions"]["0"]["issues"] == [finding]


def add_v3_shared_video(run, *, frames=12):
    import av
    import pyarrow as pa

    root = Path(run["root"])
    camera = "observation.images.front"
    info = json.loads((root / "meta/info.json").read_text())
    info["features"][camera] = {"dtype": "video", "shape": [24, 32, 3]}
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    write_json(root / "meta/info.json", info)
    for path in (root / "meta/episodes").rglob("*.parquet"):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            ep = row["episode_index"]
            row.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": ep * 0.4,
                    f"videos/{camera}/to_timestamp": (ep + 1) * 0.4,
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), path)
    video = root / f"videos/{camera}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.height, stream.width, stream.pix_fmt = 24, 32, "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 10, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    run["source_file_hashes"] = source_inventory(root)
    return video


def test_v3_shared_video_offsets_validate_and_freeze(reviewed_run, tmp_path):
    video = add_v3_shared_video(reviewed_run)
    result = publish.prepare_export(reviewed_run, tmp_path / "export-shared-video")
    for key in ("root", "rich_root"):
        frozen = Path(result[key]) / video.relative_to(reviewed_run["root"])
        assert frozen.read_bytes() == video.read_bytes()
        assert frozen.stat().st_ino != video.stat().st_ino


def test_v3_video_missing_last_frame_blocks_publication(reviewed_run, tmp_path):
    add_v3_shared_video(reviewed_run, frames=11)
    with pytest.raises(ValueError, match="video"):
        publish.prepare_export(reviewed_run, tmp_path / "export-short-video")


def test_source_mutation_during_freeze_is_rejected(reviewed_run, tmp_path, monkeypatch):
    original = engine.export_dataset

    def export_and_mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        info_path = Path(reviewed_run["root"]) / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["concurrent_edit"] = True
        write_json(info_path, info)
        return result

    monkeypatch.setattr(engine, "export_dataset", export_and_mutate)
    with pytest.raises(ValueError, match="changed"):
        publish.prepare_export(reviewed_run, tmp_path / "export-during-mutation")
    assert not (tmp_path / "export-during-mutation").exists()


def test_source_changed_after_run_creation_blocks_export(reviewed_run, tmp_path):
    root = Path(reviewed_run["source_root"])
    reviewed_run["source_file_hashes"] = {
        name: record["sha256"]
        for name, record in publish._inventory(root).items()
        if not name.startswith("meta/annotation_") and name != "meta/lerobot_annotations.json"
    }
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = "changed after pinning"
    write_json(info_path, info)
    with pytest.raises(ValueError, match="source.*changed|source.*hash"):
        publish.prepare_export(reviewed_run, tmp_path / "source-drift-export")
    assert not (tmp_path / "source-drift-export").exists()


@pytest.mark.parametrize("corruption", ["video", "parquet"])
def test_raw_v21_recovery_filters_only_frozen_export(reviewed_run, tmp_path, monkeypatch, corruption):
    import shutil

    run = reviewed_run
    source = make_v21_source(run, tmp_path)
    if corruption == "video":
        (source / "videos/chunk-000/observation.images.front/episode_000001.mp4").write_bytes(b"corrupt")
    else:
        (source / "data/chunk-000/episode_000001.parquet").write_bytes(b"corrupt")
    raw = tmp_path / "raw-review"
    shutil.copytree(source, raw)
    for name in ("lerobot_annotations.json", "annotation_reviews.json"):
        shutil.copy2(Path(run["root"]) / "meta" / name, raw / "meta" / name)
    run.update(root=str(raw), preparation_mode="source_review", source_file_hashes=source_inventory(source))
    run["episodes"]["1"].update(decision="delete", decision_reason="Corrupt recording")
    before = {
        root: {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        for root in (source, raw)
    }
    calls = []
    original_prepare = engine.prepare_dataset

    def converting(*args, **kwargs):
        calls.append("convert")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(engine, "prepare_dataset", converting)

    def cannot_delete_raw(*args, **kwargs):
        raise AssertionError("Official v3 delete cannot be called on raw v2")

    monkeypatch.setattr(publish, "delete_dataset_episodes", cannot_delete_raw)
    result = publish.prepare_export(run, tmp_path / "recovered-export")
    assert calls == ["convert"]
    assert result["old_to_new"] == {"0": 0, "2": 1}
    assert result["retained_episodes"] == 2
    for label in ("root", "rich_root"):
        records = list(engine.iter_episodes(Path(result[label])))
        assert [record.episode_index for record in records] == [0, 1]
        last = pq.read_table(records[1].data_path).slice(records[1].row_offset, records[1].row_count)
        assert last["observation.state"].to_pylist() == [[2.0, float(i)] for i in range(4)]
    for root in (source, raw):
        assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before[root]
