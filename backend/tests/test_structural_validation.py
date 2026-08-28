from __future__ import annotations

from pathlib import Path

from curation.exporter import StagingExporter
from curation.validation import StructuralValidationError, validate_structural_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_exporter import _rich_case


def _built_case(tmp_path: Path):
    source, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    StagingExporter(database=database, source_registry=registry).run(created["export_id"])
    export = database.get_export(export_id=created["export_id"])
    rows = database.list_export_episodes(export_id=created["export_id"])
    return source, workspace, registry.records["local/pnp_trash"], created, export, rows


def test_structural_gate_covers_snapshot_runs_metadata_preservation_and_source(tmp_path: Path) -> None:
    source, _, record, created, export, rows = _built_case(tmp_path)

    report = validate_structural_dataset(
        staging_path=Path(created["staging_path"]),
        source=record,
        export=export,
        export_episodes=rows,
        video_frame_counter=lambda path: 8,
    )

    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert report["approval"] == {"approved": 3, "kept": 2, "rejected": 1}
    assert report["output"] == {"episodes": 2, "frames": 16, "videos": 2, "tasks": 11}
    assert report["episodes"][0]["prompt_runs"] == 7
    assert report["episodes"][0]["full_coverage"] is True
    assert report["checks"]["untouched_arrow_columns_equal"] is True
    assert report["checks"]["independent_regular_files"] is True
    assert report["source_manifest_sha256"] == record.fingerprint
    assert record.verify_current_inventory()
    assert (source / "meta" / "tasks.jsonl").read_text() == '{"task_index":0,"task":"original"}\n'


def test_structural_report_is_persisted_workspace_first_and_copied_only_after_success(tmp_path: Path) -> None:
    _, workspace, record, created, export, rows = _built_case(tmp_path)
    from curation.validation import persist_structural_report

    artifact = persist_structural_report(
        workspace=workspace,
        export_id=created["export_id"],
        staging_path=Path(created["staging_path"]),
        report=validate_structural_dataset(
            staging_path=Path(created["staging_path"]),
            source=record,
            export=export,
            export_episodes=rows,
            video_frame_counter=lambda path: 8,
        ),
    )

    workspace_report = workspace / "exports" / created["export_id"] / "structural-report.json"
    staged_report = Path(created["staging_path"]) / artifact["path"]
    assert workspace_report.read_bytes() == staged_report.read_bytes()
    assert artifact["kind"] == "structural_report"
    assert artifact["sha256"]

    (Path(created["staging_path"]) / "meta" / "tasks.jsonl").write_text("{}\n")
    with pytest.raises(StructuralValidationError):
        validate_structural_dataset(
            staging_path=Path(created["staging_path"]),
            source=record,
            export=export,
            export_episodes=rows,
            video_frame_counter=lambda path: 8,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "omitted_snapshot_row",
        "snapshot_hash",
        "extra_column",
        "reordered_column",
        "fabricated_stats",
        "extra_parquet",
        "extra_video",
    ],
)
def test_structural_gate_rejects_unauthenticated_or_extra_dataset_bytes(tmp_path: Path, mutation: str) -> None:
    _, _, record, created, export, rows = _built_case(tmp_path)
    staging = Path(created["staging_path"])
    if mutation == "omitted_snapshot_row":
        rows = rows[:-1]
    elif mutation == "snapshot_hash":
        export = dict(export)
        export["approval_snapshot_sha256"] = "0" * 64
    elif mutation in {"extra_column", "reordered_column"}:
        path = staging / "data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        if mutation == "extra_column":
            table = table.append_column("fabricated", pa.array([1] * len(table), type=pa.int8()))
        else:
            names = [*table.column_names[1:], table.column_names[0]]
            table = table.select(names)
        pq.write_table(table, path)
    elif mutation == "fabricated_stats":
        (staging / "meta/episodes_stats.jsonl").write_text(
            '{"episode_index":0,"stats":{}}\n{"episode_index":1,"stats":{}}\n'
        )
    elif mutation == "extra_parquet":
        pq.write_table(pa.table({"x": [1]}), staging / "data/chunk-000/extra.parquet")
    else:
        (staging / "videos/chunk-000/observation.images.ego_view/extra.mp4").write_bytes(b"extra")

    with pytest.raises(StructuralValidationError):
        validate_structural_dataset(
            staging_path=staging,
            source=record,
            export=export,
            export_episodes=rows,
            video_frame_counter=lambda path: 8,
        )
