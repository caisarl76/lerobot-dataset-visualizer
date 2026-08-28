from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

from curation.db import CurationDatabase, RetryableDatabaseError, StateTransitionConflict
from curation.exporter import ExportError, ExportService, StagingExporter
from curation.models import ReviewState
from curation.prompts import PROMPT_TEMPLATE_SHA256, PROMPT_TEMPLATE_VERSION, expand_prompts
from curation.publication import PublicationError
from curation.source import SourceRegistry
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest


def _hold_export_lock(workspace: str, export_id: str, ready: Any, release: Any) -> None:
    from curation.exporter import _exclusive_export_execution

    with _exclusive_export_execution(Path(workspace), export_id):
        ready.set()
        release.wait(10)


def _crash_with_export_lock(workspace: str, export_id: str, ready: Any) -> None:
    from curation.exporter import _exclusive_export_execution

    with _exclusive_export_execution(Path(workspace), export_id):
        ready.set()
        os._exit(0)


def _write_parquet(path: Path, source_episode_index: int) -> pa.Table:
    size = 8
    base = source_episode_index * 100
    schema = pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float64(), 2), nullable=True, metadata={b"unit": b"rad"}),
            pa.field("action", pa.list_(pa.float32(), 2), nullable=False),
            pa.field("nullable_unknown", pa.int16(), nullable=True),
            pa.field("label_unknown", pa.string(), nullable=True),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("frame_index", pa.uint32(), nullable=True),
            pa.field("episode_index", pa.int32(), nullable=True),
            pa.field("index", pa.int64(), nullable=True),
            pa.field("task_index", pa.int16(), nullable=True),
        ],
        metadata={b"fixture": b"rich-v2.1", b"unicode": "한글".encode()},
    )
    table = pa.Table.from_arrays(
        [
            pa.array(
                [[float(base + index), float(base + index) + 0.5] for index in range(size)],
                type=schema.field("observation.state").type,
            ),
            pa.array([[index / 10, -index / 10] for index in range(size)], type=schema.field("action").type),
            pa.array(range(size), type=pa.int16()),
            pa.array(["a", None, "한", "d", "e", "f", None, "h"], type=pa.string()),
            pa.array([index / 50 for index in range(size)], type=pa.float32()),
            pa.array(range(size), type=pa.uint32()),
            pa.array([source_episode_index] * size, type=pa.int32()),
            pa.array(range(base, base + size), type=pa.int64()),
            pa.array([0] * size, type=pa.int16()),
        ],
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(table.slice(0, 3))
        writer.write_table(table.slice(3, 5))
    return pq.read_table(path)


def _rich_case(
    tmp_path: Path,
) -> tuple[Path, Path, CurationDatabase, SourceRegistry, ExportService, dict[int, pa.Table]]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fixture_bot",
        "total_episodes": 3,
        "total_frames": 24,
        "total_tasks": 1,
        "total_videos": 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 50,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [2, 2, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {"dtype": "float64", "shape": [2], "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
            "nullable_unknown": {"dtype": "int16", "shape": [1], "names": None},
            "label_unknown": {"dtype": "string", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "uint32", "shape": [1], "names": None},
            "episode_index": {"dtype": "int32", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int16", "shape": [1], "names": None},
        },
        "script_config": {"preserve": [1, None, 3.5]},
        "discarded_episode_indices": [1, 7],
    }
    (source / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    (source / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": ["original"], "length": 8}) + "\n" for index in range(3)
        )
    )
    (source / "meta" / "tasks.jsonl").write_text('{"task_index":0,"task":"original"}\n')
    (source / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps({"episode_index": index, "stats": {}}) + "\n" for index in range(3))
    )
    (source / "meta/stats.json").write_text('{"stale":{"mean":[999]}}\n')
    (source / "meta/relative_stats.json").write_text('{"stale_action":{}}\n')
    modality = {
        "state": {"joint": {"start": 0, "end": 2, "original_key": "observation.state"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (source / "meta" / "modality.json").write_text(json.dumps(modality, indent=2))
    tables: dict[int, pa.Table] = {}
    for index in range(3):
        tables[index] = _write_parquet(source / "data" / "chunk-000" / f"episode_{index:06d}.parquet", index)
        video = source / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{index:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"mp4-source-{index}".encode())
        extra = source / "extras" / "chunk-000" / f"episode_{index:06d}.bin"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(bytes([index, 0, 255]))
    (source / "LICENSE.txt").write_text("global ancillary bytes\n")
    (source / "data" / "chunk-000" / "calibration.bin").write_bytes(b"supplemental-data-asset")

    workspace = tmp_path / "workspace"
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
    database = CurationDatabase(workspace / "curation.sqlite3")
    database.initialize()
    opened = database.open_review_workspace(
        alias="local/pnp_trash",
        source_path=str(source.resolve()),
        source_manifest_sha256=registry.records["local/pnp_trash"].fingerprint,
        episode_lengths={0: 8, 1: 8, 2: 8},
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        actor="curator",
    )
    dataset_id = opened["dataset"]["id"]
    for index in (2, 0, 1):
        keep = index in {0, 2}
        changes: dict[str, object] = {
            "review_state": ReviewState.APPROVED_KEEP if keep else ReviewState.APPROVED_REJECT,
            "approval_revision": 1,
            "reviewer": "human",
            "approved_at": "2026-08-27T00:00:00Z",
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        }
        if keep:
            changes.update(
                {
                    "object_name": "can" if index == 0 else "paper cup",
                    "pickup_hand": "left" if index == 0 else "right",
                    "turn_direction": "right" if index == 0 else "left",
                    "step_2_start_frame": 1,
                    "step_3_start_frame": 2,
                    "step_4_start_frame": 3,
                    "step_5_start_frame": 4,
                    "step_6_start_frame": 5,
                    "step_7_start_frame": 6,
                }
            )
        database.update_episode(
            dataset_id=dataset_id,
            source_episode_index=index,
            expected_revision=0,
            changes=changes,
            actor="human",
        )
    final = tmp_path / "publication" / "pnp_trash_cleaned"
    service = ExportService(
        database=database,
        source_registry=registry,
        workspace=workspace,
        final_path=final,
    )
    return source, workspace, database, registry, service, tables


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ipc_hash(column: pa.ChunkedArray, field: pa.Field) -> str:
    table = pa.Table.from_arrays([column.combine_chunks()], schema=pa.schema([field]))
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def test_staging_export_preserves_arrow_semantics_and_rewrites_only_four_columns(tmp_path: Path) -> None:
    source, _, database, registry, service, source_tables = _rich_case(tmp_path)
    manifest_before = registry.records["local/pnp_trash"].manifest_path.read_bytes()
    source_hashes_before = {
        path: _sha256(source / path) for path in registry.records["local/pnp_trash"].file_hashes
    }
    created = service.create("local/pnp_trash")

    result = StagingExporter(database=database, source_registry=registry).run(created["export_id"])

    staging = Path(created["staging_path"])
    assert result == {
        "export_id": created["export_id"],
        "state": "building",
        "staging_path": str(staging),
        "kept_episodes": 2,
        "total_frames": 16,
    }
    assert database.get_export(export_id=created["export_id"])["state"] == "building"
    assert not (staging / "meta/stats.json").exists()
    assert not (staging / "meta/relative_stats.json").exists()
    assert registry.records["local/pnp_trash"].manifest_path.read_bytes() == manifest_before
    assert {path: _sha256(source / path) for path in source_hashes_before} == source_hashes_before

    global_offset = 0
    for output_index, source_index in enumerate((0, 2)):
        output_path = staging / "data" / "chunk-000" / f"episode_{output_index:06d}.parquet"
        parquet = pq.ParquetFile(output_path)
        output = parquet.read()
        source_table = source_tables[source_index]
        assert parquet.num_row_groups == 2
        assert output.schema.metadata == source_table.schema.metadata
        for name in source_table.column_names:
            assert output.schema.field(name) == source_table.schema.field(name)
            if name in {"episode_index", "frame_index", "index", "task_index"}:
                continue
            assert output.column(name).combine_chunks().equals(source_table.column(name).combine_chunks())
            assert _ipc_hash(output.column(name), output.schema.field(name)) == _ipc_hash(
                source_table.column(name), source_table.schema.field(name)
            )
        assert output.column("episode_index").to_pylist() == [output_index] * 8
        assert output.column("frame_index").to_pylist() == list(range(8))
        assert output.column("index").to_pylist() == list(range(global_offset, global_offset + 8))
        global_offset += 8
        assert output.column("nullable_unknown").to_pylist() == list(range(8))
        for name in {"episode_index", "frame_index", "index", "task_index"}:
            assert output.schema.field(name).nullable is True


def test_staging_export_writes_exact_metadata_and_independent_assets(tmp_path: Path) -> None:
    source, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    StagingExporter(database=database, source_registry=registry).run(created["export_id"])
    staging = Path(created["staging_path"])

    info = json.loads((staging / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 16
    assert info["total_tasks"] == 11
    assert info["total_videos"] == 2
    assert info["total_chunks"] == 1
    assert info["splits"] == {"train": "0:2"}
    assert info["features"] == json.loads((source / "meta" / "info.json").read_text())["features"]
    assert info["script_config"] == {"preserve": [1, None, 3.5]}
    assert info["discarded_episode_indices"] == []

    episodes = [json.loads(line) for line in (staging / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert episodes == [
        {"episode_index": 0, "length": 8, "tasks": expand_prompts(object_name="can", hand="left", turn="right")},
        {
            "episode_index": 1,
            "length": 8,
            "tasks": expand_prompts(object_name="paper cup", hand="right", turn="left"),
        },
    ]
    tasks = [json.loads(line) for line in (staging / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert [row["task_index"] for row in tasks] == list(range(11))
    referenced = {
        value
        for index in (0, 1)
        for value in pq.read_table(
            staging / "data" / "chunk-000" / f"episode_{index:06d}.parquet", columns=["task_index"]
        )
        .column("task_index")
        .to_pylist()
    }
    assert referenced == {row["task_index"] for row in tasks}
    task_lookup = {row["task_index"]: row["task"] for row in tasks}
    for output_index, expected_prompts in enumerate((episodes[0]["tasks"], episodes[1]["tasks"])):
        values = (
            pq.read_table(
                staging / "data" / "chunk-000" / f"episode_{output_index:06d}.parquet",
                columns=["task_index"],
            )
            .column("task_index")
            .to_pylist()
        )
        runs = [
            task_lookup[value]
            for position, value in enumerate(values)
            if position == 0 or value != values[position - 1]
        ]
        assert runs == expected_prompts
    assert json.loads((staging / "meta" / "modality.json").read_text()) == json.loads(
        (source / "meta" / "modality.json").read_text()
    )
    stats = [json.loads(line) for line in (staging / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in stats] == [0, 1]
    assert all(row["stats"]["index"]["count"] == [8] for row in stats)
    assert stats[1]["stats"]["episode_index"]["min"] == [1]

    pairs = [
        (
            source / "videos/chunk-000/observation.images.ego_view/episode_000002.mp4",
            staging / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4",
        ),
        (source / "extras/chunk-000/episode_000002.bin", staging / "extras/chunk-000/episode_000001.bin"),
        (source / "LICENSE.txt", staging / "LICENSE.txt"),
        (
            source / "data/chunk-000/calibration.bin",
            staging / "data/chunk-000/calibration.bin",
        ),
    ]
    for source_asset, output_asset in pairs:
        assert output_asset.is_file() and not output_asset.is_symlink()
        assert _sha256(output_asset) == _sha256(source_asset)
        assert (output_asset.stat().st_dev, output_asset.stat().st_ino) != (
            source_asset.stat().st_dev,
            source_asset.stat().st_ino,
        )
    assert (
        staging / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4"
    ).read_bytes() == b"mp4-source-2"
    assert not (staging / "videos/chunk-000/observation.images.ego_view/episode_000002.mp4").exists()


def test_asset_verification_hashes_reopened_destination_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import curation.exporter as exporter_module

    _, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    original = exporter_module._hash_regular_destination
    corrupted = False

    def corrupt_then_hash(
        staging: Any,
        relative_path: str,
        *,
        expected_identity: tuple[int, int, int],
    ) -> str:
        nonlocal corrupted
        if not corrupted:
            corrupted = True
            (staging.path / relative_path).write_bytes(b"corrupted-after-copy")
        return original(staging, relative_path, expected_identity=expected_identity)

    monkeypatch.setattr(exporter_module, "_hash_regular_destination", corrupt_then_hash)

    with pytest.raises(ExportError) as raised:
        StagingExporter(database=database, source_registry=registry).run(created["export_id"])

    assert raised.value.payload["error"] == "export_asset_copy"
    assert corrupted


def test_episode_statistics_match_population_moments_exactly() -> None:
    from curation.exporter import _episode_statistics

    table = pa.table(
        {
            "scalar": pa.array([1, 2, 5], type=pa.int16()),
            "vector": pa.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]], type=pa.list_(pa.float32(), 2)),
            "ignored": pa.array(["a", None, "c"]),
        }
    )

    assert _episode_statistics(table) == {
        "scalar": {
            "min": [1],
            "max": [5],
            "mean": [8 / 3],
            "std": [float(np.std(np.asarray([1, 2, 5], dtype=np.float64)))],
            "count": [3],
        },
        "vector": {
            "min": [1, 10],
            "max": [5, 18],
            "mean": [3, 14],
            "std": [float(np.std(np.asarray([1, 3, 5]))), float(np.std(np.asarray([10, 14, 18])))],
            "count": [3],
        },
    }


@pytest.mark.parametrize(
    "column",
    [
        pa.array([1, None, 3], type=pa.int16()),
        pa.array([[1.0, 2.0], [3.0, None], [5.0, 6.0]], type=pa.list_(pa.float32(), 2)),
    ],
)
def test_episode_statistics_reject_nullable_numeric_values(column: pa.Array) -> None:
    from curation.exporter import _episode_statistics

    with pytest.raises(ExportError) as raised:
        _episode_statistics(pa.table({"nullable": column}))

    assert raised.value.payload == {"error": "export_source_stats_null", "column": "nullable"}


def test_staging_export_rejects_any_source_symlink(tmp_path: Path) -> None:
    source, _, database, registry, service, _ = _rich_case(tmp_path)
    (source / "unsafe-link").symlink_to(source / "LICENSE.txt")

    with pytest.raises(ExportError) as raised:
        service.create("local/pnp_trash")

    assert raised.value.payload == {"error": "export_source_symlink", "path": "unsafe-link"}
    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM exports").fetchone()[0] == 0


def test_staging_creation_never_reuses_an_existing_path(tmp_path: Path) -> None:
    _, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    staging = Path(created["staging_path"])
    staging.mkdir(parents=True)
    (staging / "sentinel").write_bytes(b"do not overwrite")

    with pytest.raises(ExportError) as raised:
        StagingExporter(database=database, source_registry=registry).run(created["export_id"])

    assert raised.value.payload == {"error": "export_staging_exists"}
    assert (staging / "sentinel").read_bytes() == b"do not overwrite"


@pytest.mark.parametrize("race", ["root_symlink", "parent_symlink", "parquet_hardlink"])
def test_staging_parquet_writes_are_descriptor_anchored_against_path_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    import curation.exporter as exporter_module

    source, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    staging = Path(created["staging_path"])
    displaced = staging.with_name(f"{staging.name}.displaced")
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel"
    sentinel.write_bytes(b"victim")
    source_license_before = (source / "LICENSE.txt").read_bytes()
    original = exporter_module._rewrite_episode_parquet
    injected = False

    def inject(*args: Any, **kwargs: Any) -> pa.Table:
        nonlocal injected
        if not injected:
            injected = True
            if race == "root_symlink":
                staging.rename(displaced)
                staging.symlink_to(victim, target_is_directory=True)
            elif race == "parent_symlink":
                (staging / "data").symlink_to(victim, target_is_directory=True)
            else:
                target = staging / "data/chunk-000/episode_000000.parquet"
                target.parent.mkdir(parents=True)
                os.link(source / "LICENSE.txt", target)
        return original(*args, **kwargs)

    monkeypatch.setattr(exporter_module, "_rewrite_episode_parquet", inject)

    with pytest.raises(ExportError) as raised:
        StagingExporter(database=database, source_registry=registry).run(created["export_id"])

    assert raised.value.payload["error"] == "export_path_invalid"
    assert sentinel.read_bytes() == b"victim"
    assert (source / "LICENSE.txt").read_bytes() == source_license_before
    assert list(victim.iterdir()) == [sentinel]


def test_resume_rebuilds_only_the_owned_building_staging_directory(tmp_path: Path) -> None:
    _, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = StagingExporter(database=database, source_registry=registry)
    first = exporter.run(created["export_id"])
    staging = Path(first["staging_path"])
    (staging / "crash-junk.tmp").write_bytes(b"partial")

    resumed = exporter.resume(created["export_id"])

    assert resumed == first
    assert not (staging / "crash-junk.tmp").exists()
    assert pq.read_table(staging / "data/chunk-000/episode_000001.parquet").num_rows == 8


def test_resume_reconciles_crash_after_receipt_install_before_temp_unlink(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = StagingExporter(database=database, source_registry=registry)
    exporter.run(created["export_id"])
    receipt = workspace / "exports" / created["export_id"] / "staging-ownership.json"
    receipt_temp = receipt.with_name(f".{receipt.name}.tmp")
    os.link(receipt, receipt_temp)

    resumed = exporter.resume(created["export_id"])

    assert resumed["state"] == "building"
    assert receipt.is_file()
    assert not receipt_temp.exists()


def test_resume_cleanup_is_descriptor_anchored_if_root_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import curation.exporter as exporter_module

    _, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = StagingExporter(database=database, source_registry=registry)
    exporter.run(created["export_id"])
    staging = Path(created["staging_path"])
    displaced = staging.with_name(f"{staging.name}.cleanup-displaced")
    victim = tmp_path / "cleanup-victim"
    victim.mkdir()
    sentinel = victim / "sentinel"
    sentinel.write_bytes(b"victim")
    original_walk = exporter_module.os.walk
    original_fwalk = exporter_module.os.fwalk
    injected = False

    def replace_root() -> None:
        nonlocal injected
        if not injected:
            injected = True
            staging.rename(displaced)
            staging.symlink_to(victim, target_is_directory=True)

    def racing_walk(top: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(top) == staging:
            replace_root()
        return original_walk(top, *args, **kwargs)

    def racing_fwalk(top: Any, *args: Any, **kwargs: Any) -> Any:
        replace_root()
        return original_fwalk(top, *args, **kwargs)

    monkeypatch.setattr(exporter_module.os, "walk", racing_walk)
    monkeypatch.setattr(exporter_module.os, "fwalk", racing_fwalk)

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])

    assert raised.value.payload["error"] == "export_path_invalid"
    assert sentinel.read_bytes() == b"victim"
    assert list(victim.iterdir()) == [sentinel]


def test_executor_lock_is_process_exclusive_and_crash_releasing(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_export_lock,
        args=(str(workspace), created["export_id"], ready, release),
    )
    holder.start()
    assert ready.wait(10)
    try:
        with pytest.raises(ExportError) as busy:
            StagingExporter(database=database, source_registry=registry).run(created["export_id"])
        assert busy.value.payload == {"error": "export_executor_busy", "export_id": created["export_id"]}
        with pytest.raises(ExportError) as resume_busy:
            StagingExporter(database=database, source_registry=registry).resume(created["export_id"])
        assert resume_busy.value.payload == {
            "error": "export_executor_busy",
            "export_id": created["export_id"],
        }
    finally:
        release.set()
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join(2)
    assert holder.exitcode == 0

    crashed_ready = context.Event()
    crashed = context.Process(
        target=_crash_with_export_lock,
        args=(str(workspace), created["export_id"], crashed_ready),
    )
    crashed.start()
    assert crashed_ready.wait(10)
    crashed.join(10)
    assert crashed.exitcode == 0
    assert (
        StagingExporter(database=database, source_registry=registry).run(created["export_id"])["state"]
        == "building"
    )


def test_executor_and_receipt_parent_reject_symlink_components(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    victim = tmp_path / "control-victim"
    victim.mkdir()
    sentinel = victim / "sentinel"
    sentinel.write_bytes(b"victim")
    (workspace / "exports").symlink_to(victim, target_is_directory=True)

    with pytest.raises(ExportError) as raised:
        StagingExporter(database=database, source_registry=registry).run(created["export_id"])

    assert raised.value.payload == {"error": "export_path_invalid"}
    assert list(victim.iterdir()) == [sentinel]


def test_resume_preserves_foreign_staging_after_crash_between_claim_and_receipt(tmp_path: Path) -> None:
    _, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    database.claim_export_build(export_id=created["export_id"])
    staging = Path(created["staging_path"])
    staging.mkdir(parents=True)
    sentinel = staging / "foreign-sentinel"
    sentinel.write_bytes(b"foreign")

    with pytest.raises(ExportError) as raised:
        StagingExporter(database=database, source_registry=registry).resume(created["export_id"])

    assert raised.value.payload == {
        "error": "export_staging_ownership_missing",
        "export_id": created["export_id"],
    }
    assert sentinel.read_bytes() == b"foreign"
    failed = database.get_export(export_id=created["export_id"])
    assert failed["state"] == "failed"
    assert failed["failure_summary"] == "export_staging_ownership_missing"


def test_resume_recovers_crash_after_claim_before_staging_creation(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    database.claim_export_build(export_id=created["export_id"])
    staging = Path(created["staging_path"])
    receipt = workspace / "exports" / created["export_id"] / "staging-ownership.json"
    assert not staging.exists()
    assert not receipt.exists()

    resumed = StagingExporter(database=database, source_registry=registry).resume(created["export_id"])

    assert resumed["state"] == "building"
    assert pq.read_table(staging / "data/chunk-000/episode_000001.parquet").num_rows == 8
    receipt_document = json.loads(receipt.read_text())
    staging_stat = staging.stat()
    assert receipt_document == {
        "approval_snapshot_sha256": created["approval_snapshot_sha256"],
        "device": staging_stat.st_dev,
        "export_id": created["export_id"],
        "inode": staging_stat.st_ino,
        "schema_version": 1,
        "staging_path": str(staging),
    }


def test_resume_preserves_partial_receipt_temp_and_crash_after_mkdir_evidence(
    tmp_path: Path,
) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    database.claim_export_build(export_id=created["export_id"])
    staging = Path(created["staging_path"])
    staging.mkdir(parents=True)
    sentinel = staging / "crash-after-mkdir"
    sentinel.write_bytes(b"preserve")
    receipt = workspace / "exports" / created["export_id"] / "staging-ownership.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_temp = receipt.with_name(f".{receipt.name}.tmp")
    receipt_temp.write_bytes(b'{"partial":')

    with pytest.raises(ExportError) as raised:
        StagingExporter(database=database, source_registry=registry).resume(created["export_id"])

    assert raised.value.payload == {
        "error": "export_staging_ownership_missing",
        "export_id": created["export_id"],
    }
    assert sentinel.read_bytes() == b"preserve"
    assert not receipt.exists()
    assert receipt_temp.read_bytes() == b'{"partial":'
    failed = database.get_export(export_id=created["export_id"])
    assert failed["state"] == "failed"
    assert failed["failure_summary"] == "export_staging_ownership_missing"


def test_resume_preserves_temp_only_ambiguous_state_and_transitions_terminal(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    database.claim_export_build(export_id=created["export_id"])
    receipt = workspace / "exports" / created["export_id"] / "staging-ownership.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_temp = receipt.with_name(f".{receipt.name}.tmp")
    receipt_temp.write_bytes(b'{"partial":')

    with pytest.raises(ExportError) as raised:
        StagingExporter(database=database, source_registry=registry).resume(created["export_id"])

    assert raised.value.payload == {
        "error": "export_staging_ownership_mismatch",
        "export_id": created["export_id"],
    }
    assert not receipt.exists()
    assert receipt_temp.read_bytes() == b'{"partial":'
    failed = database.get_export(export_id=created["export_id"])
    assert failed["state"] == "failed"
    assert failed["failure_summary"] == "export_staging_ownership_mismatch"


def test_resume_preserves_staging_when_durable_ownership_receipt_mismatches(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = StagingExporter(database=database, source_registry=registry)
    exporter.run(created["export_id"])
    staging = Path(created["staging_path"])
    sentinel = staging / "preserve-on-mismatch"
    sentinel.write_bytes(b"preserve")
    receipt = workspace / "exports" / created["export_id"] / "staging-ownership.json"
    receipt.write_text('{"schema_version":1,"tampered":true}\n')

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])

    assert raised.value.payload == {
        "error": "export_staging_ownership_mismatch",
        "export_id": created["export_id"],
    }
    assert sentinel.read_bytes() == b"preserve"
    failed = database.get_export(export_id=created["export_id"])
    assert failed["state"] == "failed"
    assert failed["failure_summary"] == "export_staging_ownership_mismatch"


def test_export_cli_freezes_run_and_resume_commands_and_runs_out_of_process_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import curation_export as cli_module

    source, workspace, _, _, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    settings = type(
        "Settings",
        (),
        {
            "workspace": workspace.resolve(),
            "dataset_aliases": {"local/pnp_trash": source},
            "isaac_groot_root": tmp_path,
            "cosmos_model": "cosmos",
            "cosmos_endpoint_identity": "h100",
        },
    )()
    monkeypatch.setattr(cli_module.CurationSettings, "from_env", classmethod(lambda cls: settings))

    class FakeValidatedExporter:
        def __init__(self, **kwargs):
            pass

        def run(self, export_id: str) -> dict[str, object]:
            return {
                "export_id": export_id,
                "state": "published",
                "final_path": created["final_path"],
                "approval_snapshot_sha256": created["approval_snapshot_sha256"],
            }

        resume = run

    monkeypatch.setattr(cli_module, "ValidatedDatasetExporter", FakeValidatedExporter)
    zero = "00000000-0000-0000-0000-000000000000"
    parser = cli_module.build_cli_parser()
    assert parser.parse_args(["--workspace", str(workspace), "run", "--export-id", zero]).command == "run"
    assert parser.parse_args(["--workspace", str(workspace), "resume", "--export-id", zero]).command == "resume"

    assert cli_module.cli_main(["--workspace", str(workspace), "run", "--export-id", created["export_id"]]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["event"] == "export_published"
    assert output["export_id"] == created["export_id"]
    assert output["state"] == "published"


def test_export_cli_sanitizes_retryable_database_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import curation_export as cli_module

    _, workspace, _, _, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")

    def busy(self: CurationDatabase) -> None:
        raise RetryableDatabaseError("secret database path")

    monkeypatch.setattr(cli_module.CurationDatabase, "validate_worker_compatibility", busy)

    assert cli_module.cli_main(["--workspace", str(workspace), "run", "--export-id", created["export_id"]]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "database_busy",
        "export_id": created["export_id"],
        "retryable": True,
    }


def test_export_cli_sanitizes_lifecycle_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import curation_export as cli_module

    source, workspace, _, _, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")

    settings = type(
        "Settings",
        (),
        {
            "workspace": workspace.resolve(),
            "dataset_aliases": {"local/pnp_trash": source},
            "isaac_groot_root": tmp_path,
            "cosmos_model": "cosmos",
            "cosmos_endpoint_identity": "h100",
        },
    )()
    monkeypatch.setattr(cli_module.CurationSettings, "from_env", classmethod(lambda cls: settings))

    def conflict(self: object, export_id: str) -> dict[str, object]:
        raise StateTransitionConflict(
            entity="export",
            identifier=export_id,
            expected_state="queued",
            current_state="building",
        )

    monkeypatch.setattr(cli_module.ValidatedDatasetExporter, "run", conflict)

    assert cli_module.cli_main(["--workspace", str(workspace), "run", "--export-id", created["export_id"]]) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "export_state_conflict",
        "export_id": created["export_id"],
    }


def test_export_cli_emits_exact_sanitized_json_for_unsupported_publication_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import curation_export as cli_module

    source, workspace, database, _, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    settings = type(
        "Settings",
        (),
        {
            "workspace": workspace.resolve(),
            "dataset_aliases": {"local/pnp_trash": source},
            "isaac_groot_root": tmp_path,
            "cosmos_model": "cosmos",
            "cosmos_endpoint_identity": "h100",
        },
    )()
    monkeypatch.setattr(cli_module.CurationSettings, "from_env", classmethod(lambda cls: settings))
    exporter_type = cli_module.ValidatedDatasetExporter

    def exporter_factory(**kwargs):
        exporter = exporter_type(**kwargs)
        exporter.publication_preflight = lambda parent: (_ for _ in ()).throw(
            PublicationError(
                "publish_noreplace_unsupported",
                "secret /srv/private/path authorization=token",
            )
        )
        return exporter

    monkeypatch.setattr(cli_module, "ValidatedDatasetExporter", exporter_factory)
    exit_code = cli_module.cli_main(["--workspace", str(workspace), "run", "--export-id", created["export_id"]])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    assert captured.out == (
        '{"error":"publish_noreplace_unsupported","export_id":"' + created["export_id"] + '"}\n'
    )
    assert "secret" not in captured.out
    assert "/srv/private" not in captured.out
    assert database.get_export(export_id=created["export_id"])["state"] == "queued"


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"error": "export_not_found"}, 404),
        ({"error": "dataset_alias_not_found"}, 404),
        ({"error": "export_executor_busy", "export_id": "id"}, 423),
        ({"error": "export_destination_exists"}, 409),
    ],
)
def test_export_errors_have_typed_http_statuses(payload: dict[str, object], expected_status: int) -> None:
    assert ExportError("sanitized", payload).status_code == expected_status
