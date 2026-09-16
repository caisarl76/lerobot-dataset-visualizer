from monitor_fixtures import write_v21
from annotation_monitor import attach_relationships, discover_datasets, read_runs


def test_discovery_deduplicates_alias_and_keeps_empty(tmp_path):
    root, workspace = tmp_path / "collection", tmp_path / "workspace"
    source = write_v21(root / "source", [4, 6])
    write_v21(root / "empty", [])
    root.mkdir(exist_ok=True)
    (root / "alias").symlink_to(source, target_is_directory=True)
    rows = discover_datasets(root, workspace)
    assert sorted(r["collected"] for r in rows) == [0, 2]
    assert len({r["canonical_path"] for r in rows}) == 2


def test_runs_and_relationships_use_recorded_source(tmp_path):
    source = write_v21(tmp_path / "source", [2])
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text('{"run_id":"r1","source_root":"' + str(source) + '"}')
    runs = read_runs(workspace)
    rows = attach_relationships(discover_datasets(tmp_path, workspace), runs)
    row = next(r for r in rows if r["canonical_path"] == str(source.resolve()))
    assert row["default_run_id"] == "r1"
    assert row["provenance"]["status"] == "confirmed"
