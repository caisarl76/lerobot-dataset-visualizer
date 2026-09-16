import json
import os
from pathlib import Path

from annotation_monitor import attach_relationships, discover_datasets, read_runs
from monitor_fixtures import write_json, write_run, write_v3, write_v21
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def codes(row):
    return {d["code"] for d in row["diagnostics"]}


def discover(root, workspace):
    return {
        row["name"]: row for row in attach_relationships(discover_datasets(root, workspace), read_runs(workspace))
    }


def test_discovery_deduplicates_alias_and_keeps_empty(tmp_path):
    root, workspace = tmp_path / "collection", tmp_path / "workspace"
    source = write_v21(root / "source", [4, 6])
    write_v21(root / "empty", [])
    (root / "alias").symlink_to(source, target_is_directory=True)
    rows = discover_datasets(root, workspace)
    assert sorted(r["collected"] for r in rows) == [0, 2]
    assert all(r["state"] == "Ready" for r in rows)
    first_ids = {r["canonical_path"]: r["id"] for r in rows}
    (root / "alias").unlink()
    assert {r["canonical_path"]: r["id"] for r in discover_datasets(root, workspace)} == first_ids


@pytest.mark.parametrize("writer", [write_v21, write_v3])
def test_sparse_unique_valid_completed_ids_counted(writer, tmp_path):
    root = writer(tmp_path / "source", [4, 6])
    records = [
        {"episode_index": 3, "length": 4},
        {"episode_index": 8, "length": 6},
        {"episode_index": 3, "length": 4},
        {"episode_index": -1, "length": 3},
        {"episode_index": 10, "length": -1},
    ]
    if writer == write_v21:
        (root / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    else:
        pq.write_table(pa.Table.from_pylist(records), root / "meta/episodes/chunk-000/file-000.parquet")
    row = discover_datasets(tmp_path, tmp_path / "workspace")[0]
    assert row["collected"] == 2
    assert row["episode_lengths"] == {"3": 4, "8": 6}
    assert {"duplicate_episode_id", "invalid_episode_metadata"} <= codes(row)
    assert row["state"] == "Updating"


@pytest.mark.parametrize("writer", [write_v21, write_v3])
def test_v3_and_v21_ready_metadata_counts(writer, tmp_path):
    writer(tmp_path / "source", [4, 6])
    row = discover_datasets(tmp_path, tmp_path / "workspace")[0]
    assert row["collected"] == row["reported_collected"] == 2
    assert row["episode_lengths"] == {"0": 4, "1": 6}
    assert row["state"] == "Ready"


@pytest.mark.parametrize("info", [[], None, {}, {"total_episodes": True}, "partial"])
def test_bad_info_keeps_diagnostic_row_and_healthy_sibling(info, tmp_path):
    source = write_v21(tmp_path / "broken", [2])
    write_v21(tmp_path / "healthy", [4])
    if info == "partial":
        (source / "meta/info.json").write_text('{"total_episodes":')
    else:
        write_json(source / "meta/info.json", info)
    rows = discover(tmp_path, tmp_path / "workspace")
    assert rows["broken"]["state"] == "Updating"
    assert rows["broken"]["diagnostics"]
    assert rows["healthy"]["state"] == "Ready"


def test_missing_metadata_and_count_mismatch_are_visible(tmp_path):
    missing_info = write_v21(tmp_path / "no-info", [2])
    (missing_info / "meta/info.json").unlink()
    missing_episodes = write_v21(tmp_path / "no-episodes", [2])
    (missing_episodes / "meta/episodes.jsonl").unlink()
    partial = write_v21(tmp_path / "partial", [2, 3])
    (partial / "meta/episodes.jsonl").write_text('{"episode_index":7,"length":2}\n{"episode_index":')
    rows = discover(tmp_path, tmp_path / "workspace")
    assert all(r["state"] == "Updating" for r in rows.values())
    assert rows["partial"]["collected"] == 1
    assert rows["partial"]["reported_collected"] == 2
    assert "episode_count_mismatch" in codes(rows["partial"])
    assert rows["no-episodes"]["collected"] == 0


def test_hidden_infrastructure_and_escaping_links_excluded(tmp_path, monkeypatch):
    root = tmp_path / "collection"
    workspace = root / "work"
    for name in ["source", ".hidden", "cache", "exports", "staging", "drafts", "work", "configured-cache"]:
        write_v21(root / name, [2])
    external = write_v21(tmp_path / "external", [2])
    (root / "escape").symlink_to(external, target_is_directory=True)
    (root / "cache-alias").symlink_to(root / "configured-cache", target_is_directory=True)
    monkeypatch.setenv("LEROBOT_ANNOTATE_CACHE", str(root / "configured-cache"))
    assert [r["name"] for r in discover_datasets(root, workspace)] == ["source"]


def test_run_fixture_matches_production_and_does_not_imply_self_parent(tmp_path):
    source = write_v21(tmp_path / "source", [2])
    workspace = tmp_path / "workspace"
    checkpoint = workspace / "drafts/checkpoint"
    write_run(workspace, source, checkpoint, [0])
    runs = read_runs(workspace)
    assert runs[0]["root"] == str(checkpoint)
    assert runs[0]["episodes"]["0"]["original_episode_index"] == 0
    row = attach_relationships(discover_datasets(tmp_path, workspace), runs)[0]
    assert row["default_run_id"] == "a" * 32
    assert row["parent_ids"] == []
    assert row["provenance"] == {"status": "unknown", "sources": []}


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {"run_id": "broken", "source_root": None},
        {"run_id": "broken", "source_root": "relative", "root": "/tmp/checkpoint", "episodes": {}},
        {"run_id": "broken", "source_root": "/tmp/source", "root": "/tmp/checkpoint", "episodes": []},
    ],
)
def test_corrupt_run_isolated_with_evidence(bad, tmp_path):
    source = write_v21(tmp_path / "source", [2])
    workspace = tmp_path / "workspace"
    write_run(workspace, source, episodes=[0])
    write_json(workspace / "runs/broken/run.json", bad)
    truncated = workspace / "runs/truncated/run.json"
    truncated.parent.mkdir(parents=True)
    truncated.write_text('{"run_id":')
    runs = read_runs(workspace)
    assert len([r for r in runs if r.get("run_id") == "a" * 32]) == 1
    failures = [r for r in runs if r.get("_diagnostic")]
    assert len(failures) == 2
    assert all(r["_path"] in r["_diagnostic"]["message"] for r in failures)
    assert attach_relationships(discover_datasets(tmp_path, workspace), runs)[0]["default_run_id"] == "a" * 32


def test_run_order_mtime_then_identity_and_read_only(tmp_path):
    source = write_v21(tmp_path / "source", [2])
    workspace = tmp_path / "workspace"
    paths = [write_run(workspace, source, episodes=[0], run_id=x * 32) for x in "cba"]
    for path, stamp in zip(paths, [1, 2, 2]):
        os.utime(path, ns=(stamp, stamp))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    row = discover(tmp_path, workspace)["source"]
    assert [r["run_id"] for r in row["runs"]] == ["a" * 32, "b" * 32, "c" * 32]
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["original", "run", "merge", "export", "checkpoint"])
def test_confirmed_recorded_parent(kind, tmp_path):
    source = write_v21(tmp_path / "source", [2])
    child = write_v21(tmp_path / "child", [2])
    workspace = tmp_path / "workspace"
    write_run(workspace, source, child if kind == "checkpoint" else None, [0])
    if kind == "original":
        write_json(
            child / "meta/source_episode_mapping.json", {"original_root": str(source), "old_to_new": {"0": 0}}
        )
    elif kind == "run":
        write_json(child / "meta/source_episode_mapping.json", {"run_id": "a" * 32, "old_to_new": {"0": 0}})
    elif kind == "merge":
        write_json(child / "meta/merge_manifest.json", {"sources": [{"root": str(source)}]})
    elif kind == "export":
        write_json(child / "meta/groot_instruction_export.json", {"source_root": str(source), "mode": "subtask"})
    rows = discover(tmp_path, workspace)
    assert rows["child"]["provenance"] == {"status": "confirmed", "sources": [str(source)]}
    assert rows["child"]["parent_ids"] == [rows["source"]["id"]]
    assert rows["source"]["child_ids"] == [rows["child"]["id"]]


def test_merge_multiple_sources_stays_top_level(tmp_path):
    a, b, merged = [write_v21(tmp_path / name, [2]) for name in ["a", "b", "merged"]]
    write_json(merged / "meta/merge_manifest.json", {"sources": [{"root": str(a)}, {"root": str(b)}]})
    rows = discover(tmp_path, tmp_path / "workspace")
    assert rows["merged"]["provenance"] == {"status": "confirmed", "sources": [str(a), str(b)]}
    assert set(rows["merged"]["parent_ids"]) == {rows["a"]["id"], rows["b"]["id"]}
    assert rows["a"]["child_ids"] == rows["b"]["child_ids"] == []


def test_export_chain_resolves_only_recorded_source_and_not_staging(tmp_path):
    root = tmp_path / "collection"
    source, child, unknown = [write_v21(root / name, [2]) for name in ["source", "child", "unknown"]]
    staging = tmp_path / "staging"
    write_json(staging / "meta/source_episode_mapping.json", {"original_root": str(source)})
    write_json(child / "meta/groot_instruction_export.json", {"source_root": str(staging)})
    write_json(unknown / "meta/groot_instruction_export.json", {"source_root": str(tmp_path / "unrecorded")})
    rows = discover(root, tmp_path / "workspace")
    assert len(rows) == 3
    assert rows["child"]["provenance"] == {"status": "confirmed", "sources": [str(source)]}
    assert rows["unknown"]["parent_ids"] == []
    assert rows["unknown"]["provenance"]["status"] == "unknown"
    assert "unresolved_provenance" in codes(rows["unknown"])


@pytest.mark.parametrize("cycle", [False, True])
def test_conflicts_and_cycles_remain_ungrouped(cycle, tmp_path):
    a, b, child = [write_v21(tmp_path / name, [2]) for name in ["a", "b", "child"]]
    write_json(child / "meta/source_episode_mapping.json", {"original_root": str(a)})
    if cycle:
        write_json(a / "meta/source_episode_mapping.json", {"original_root": str(child)})
    else:
        write_json(child / "meta/groot_instruction_export.json", {"source_root": str(b)})
    rows = discover(tmp_path, tmp_path / "workspace")
    assert rows["child"]["provenance"]["status"] == "conflict"
    assert rows["child"]["parent_ids"] == []
    assert rows["child"]["diagnostics"]
    if cycle:
        assert rows["a"]["provenance"]["status"] == "conflict"
        assert rows["a"]["parent_ids"] == []


def test_episode_mapping_alone_does_not_infer_parent(tmp_path):
    child = write_v21(tmp_path / "source_full_task", [2])
    write_v21(tmp_path / "source", [2])
    write_json(child / "meta/source_episode_mapping.json", {"old_to_new": {"0": 0}})
    row = discover(tmp_path, tmp_path / "workspace")["source_full_task"]
    assert row["provenance"] == {"status": "unknown", "sources": []}
    assert row["parent_ids"] == []


@pytest.mark.parametrize("continues", [False, True])
def test_metadata_changes_retry_once_then_updating(tmp_path, monkeypatch, continues):
    source = write_v21(tmp_path / "source", [2])
    target = source / "meta/info.json"
    original = Path.read_text
    reads = []

    def changing_read(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == target:
            reads.append(1)
            if continues or len(reads) == 1:
                path.write_text(text + " ")
        return text

    monkeypatch.setattr(Path, "read_text", changing_read)
    row = discover_datasets(tmp_path, tmp_path / "workspace")[0]
    assert len(reads) == 2
    assert row["state"] == ("Updating" if continues else "Ready")
    assert ("metadata_changing" in codes(row)) == continues


def test_metadata_file_added_during_scan_is_retried(tmp_path, monkeypatch):
    source = write_v3(tmp_path / "source", [2])
    target = source / "meta/info.json"
    original = Path.read_text
    added = []

    def changing_read(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == target and not added:
            added.append(1)
            pq.write_table(
                pa.table({"episode_index": [4], "length": [3]}),
                source / "meta/episodes/chunk-000/file-001.parquet",
            )
            value = json.loads(text)
            value["total_episodes"] = 2
            write_json(target, value)
        return text

    monkeypatch.setattr(Path, "read_text", changing_read)
    row = discover_datasets(tmp_path, tmp_path / "workspace")[0]
    assert row["collected"] == row["reported_collected"] == 2
    assert row["state"] == "Ready"


def test_missing_root_not_successful_empty(tmp_path):
    rows = discover_datasets(tmp_path / "missing", tmp_path / "workspace")
    assert len(rows) == 1
    assert rows[0]["state"] == "Updating"
    assert rows[0]["diagnostics"]


@pytest.mark.parametrize("continues", [False, True])
def test_run_changes_retry_and_never_attach_unstable_identity(tmp_path, monkeypatch, continues):
    source = write_v21(tmp_path / "source", [2])
    workspace = tmp_path / "workspace"
    target = write_run(workspace, source, episodes=[0])
    original = Path.read_text
    reads = []

    def changing_read(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == target:
            reads.append(1)
            if continues or len(reads) == 1:
                path.write_text(text + " ")
        return text

    monkeypatch.setattr(Path, "read_text", changing_read)
    runs = read_runs(workspace)
    assert len(reads) == 2
    if continues:
        assert runs[0]["_diagnostic"]["code"] == "runs_changing"
        assert not runs[0].get("run_id")
    else:
        assert runs[0]["run_id"] == "a" * 32


def test_run_added_during_read_and_invalid_related_record_diagnostic(tmp_path, monkeypatch):
    source = write_v21(tmp_path / "source", [2])
    workspace = tmp_path / "workspace"
    target = write_run(workspace, source, episodes=[0])
    original = Path.read_text
    added = []

    def changing_read(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == target and not added:
            added.append(1)
            write_run(workspace, source, episodes=[0], run_id="b" * 32)
        return text

    write_json(workspace / "runs/broken/run.json", {"run_id": "broken", "source_root": str(source)})
    monkeypatch.setattr(Path, "read_text", changing_read)
    rows = discover(tmp_path, workspace)
    assert {r["run_id"] for r in rows["source"]["runs"]} == {"a" * 32, "b" * 32}
    assert "invalid_run" in codes(rows["source"])


def test_corrupt_v3_shard_keeps_healthy_completed_records(tmp_path):
    source = write_v3(tmp_path / "source", [2])
    (source / "meta/episodes/chunk-000/file-001.parquet").write_bytes(b"partial parquet")
    row = discover_datasets(tmp_path, tmp_path / "workspace")[0]
    assert row["collected"] == 1
    assert "invalid_episode_metadata" in codes(row)
    assert row["state"] == "Updating"


@pytest.mark.parametrize(
    "filename,value",
    [
        ("source_episode_mapping.json", []),
        ("source_episode_mapping.json", {"run_id": []}),
        ("merge_manifest.json", {"sources": [None]}),
        ("groot_instruction_export.json", {"source_root": {}}),
    ],
)
def test_invalid_provenance_is_visible_without_crashing(filename, value, tmp_path):
    source = write_v21(tmp_path / "source", [2])
    write_json(source / "meta" / filename, value)
    row = discover(tmp_path, tmp_path / "workspace")["source"]
    assert row["provenance"]["status"] == "unknown"
    assert row["diagnostics"]
    assert row["parent_ids"] == []


def test_external_provenance_cycle_retains_discovered_dataset(tmp_path):
    root = tmp_path / "collection"
    child = write_v21(root / "child", [2])
    a, b = tmp_path / "staging-a", tmp_path / "staging-b"
    write_json(child / "meta/groot_instruction_export.json", {"source_root": str(a)})
    write_json(a / "meta/source_episode_mapping.json", {"original_root": str(b)})
    write_json(b / "meta/source_episode_mapping.json", {"original_root": str(a)})
    row = discover(root, tmp_path / "workspace")["child"]
    assert row["provenance"]["status"] == "conflict"
    assert "provenance_cycle" in codes(row)
    assert row["parent_ids"] == []


def test_projected_metadata_reads_do_not_read_frame_data(tmp_path, monkeypatch):
    write_v3(tmp_path / "source", [2])
    read_table = pq.read_table

    def metadata_only(path, *args, **kwargs):
        assert "/meta/episodes/" in str(path)
        assert kwargs["columns"] == ["episode_index", "length"]
        return read_table(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", metadata_only)
    assert discover_datasets(tmp_path, tmp_path / "workspace")[0]["collected"] == 1


@pytest.mark.parametrize("indirect", [False, True])
def test_mixed_confirmed_and_unresolved_provenance_never_nests(tmp_path, indirect):
    root = tmp_path / "collection"
    source = write_v21(root / "source", [2])
    child = write_v21(root / "child", [2])
    evidence_root = tmp_path / "staging" if indirect else child
    write_json(
        evidence_root / "meta/source_episode_mapping.json",
        {"original_root": str(source), "run_id": "missing-run"},
    )
    if indirect:
        write_json(
            child / "meta/groot_instruction_export.json",
            {"source_root": str(evidence_root)},
        )
    rows = discover(root, tmp_path / "workspace")
    assert rows["child"]["provenance"]["status"] == "unknown"
    assert "unresolved_provenance" in codes(rows["child"])
    assert rows["child"]["parent_ids"] == []
    assert rows["source"]["child_ids"] == []


@pytest.mark.parametrize("suffix", [".tmp", ".staging"])
def test_alias_cannot_discover_canonical_temporary_directory(tmp_path, suffix):
    root = tmp_path / "collection"
    target = write_v21(root / f"import{suffix}", [2])
    write_v21(root / "healthy", [2])
    (root / "alias").symlink_to(target, target_is_directory=True)
    assert [row["name"] for row in discover_datasets(root, tmp_path / "workspace")] == ["healthy"]


def test_looping_child_symlink_does_not_hide_healthy_collections(tmp_path):
    root = tmp_path / "collection"
    write_v21(root / "healthy", [2])
    (root / "bad-link").symlink_to("bad-link")
    rows = discover_datasets(root, tmp_path / "workspace")
    assert [row["name"] for row in rows] == ["healthy"]
    assert rows[0]["state"] == "Ready"


def test_confirmed_source_with_unreadable_competing_provenance_stays_ungrouped(
    tmp_path,
):
    source = write_v21(tmp_path / "source", [2])
    child = write_v21(tmp_path / "child", [2])
    write_json(child / "meta/source_episode_mapping.json", {"original_root": str(source)})
    (child / "meta/merge_manifest.json").write_text('{"sources":')
    rows = discover(tmp_path, tmp_path / "workspace")
    assert rows["child"]["provenance"]["status"] == "unknown"
    assert "invalid_provenance" in codes(rows["child"])
    assert rows["child"]["parent_ids"] == []
    assert rows["source"]["child_ids"] == []
