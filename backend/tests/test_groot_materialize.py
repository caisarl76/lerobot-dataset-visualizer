"""A standalone GR00T export must work from real retained v3 rows and media."""

import json

from annotation_publish import _validate_structure
from annotation_runs import source_inventory
from annotation_source import v21_records
import av
from groot_materialize import materialize_groot_v21
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
import numpy as np
from official_annotations import export_dataset
import pyarrow.parquet as pq
import pytest
from test_annotation_clipping import atom, video_source  # noqa: F401


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def rich(video_source, tmp_path):  # noqa: F811 - imported pytest fixture
    root = tmp_path / "rich"
    export_dataset(
        video_source,
        root,
        {ep: [atom("task_aug", 0, f"Full task {ep}"), atom("subtask", 0.3, "Reach object")] for ep in range(2)},
        copy_videos=True,
    )
    return root


@pytest.mark.parametrize("mode", ["task", "subtask"])
def test_real_shared_videos_saved_instructions_and_numeric_integrity(rich, tmp_path, mode):
    sidecar = rich / "meta/lerobot_annotations.json"
    saved = json.loads(sidecar.read_text())
    saved["episodes"]["1"]["atoms"][0]["content"] = "Edited full task"
    sidecar.write_text(json.dumps(saved))
    before = source_inventory(rich)
    source_meta = LeRobotDatasetMetadata(repo_id="local/source", root=rich)
    key = "observation.images.left"
    assert source_meta.get_video_file_path(0, key) == source_meta.get_video_file_path(1, key)
    assert source_meta.episodes[1][f"videos/{key}/from_timestamp"] > 0
    output = tmp_path / "groot"
    report = materialize_groot_v21(rich, output, instruction_mode=mode)
    assert (report["episodes"], report["frames"]) == (2, 16)
    info = json.loads((output / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["features"]["observation.state"]["shape"] == [2]
    assert "language_persistent" not in info["features"]
    assert json.loads((output / "meta/modality.json").read_text())["annotation"]["human.task_description"] == {
        "original_key": "task_index"
    }
    tasks = {r["task_index"]: r["task"] for r in read_jsonl(output / "meta/tasks.jsonl")}
    for ep, record in enumerate(iter_episodes(rich)):
        original = pq.read_table(record.data_path).slice(record.row_offset, record.row_count)
        table = pq.read_table(output / f"data/chunk-000/episode_{ep:06d}.parquet")
        for column in ("observation.state", "action", "timestamp"):
            assert table[column].equals(original[column])
        assert table["index"].to_pylist() == list(range(ep * 8, (ep + 1) * 8))
        full_task = "Full task 0" if ep == 0 else "Edited full task"
        expected = [full_task] * 8 if mode == "task" else [full_task] * 3 + ["Reach object"] * 5
        assert [tasks[i] for i in table["task_index"].to_pylist()] == expected
        for camera, extra in (("left", 0), ("right", 10)):
            video = output / f"videos/chunk-000/observation.images.{camera}/episode_{ep:06d}.mp4"
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                assert float(stream.average_rate) == 10
                frames = list(container.decode(stream))
                assert len(frames) == 8
                assert all((frame.height, frame.width) == (16, 16) for frame in frames)
                np.testing.assert_allclose(
                    [frame.to_ndarray(format="rgb24").mean() for frame in frames],
                    [ep * 60 + i * 15 + extra for i in range(8)],
                    atol=3,
                )
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["observation.state"]["mean"] == [0.5, 3.5]
    assert stats["observation.state"]["count"] == [16]
    assert "task_index" not in stats
    assert len(read_jsonl(output / "meta/episodes_stats.jsonl")) == 2
    _validate_structure(output, v21_records(output, [0, 1]))
    assert source_inventory(rich) == before
    source_inodes = {p.stat().st_ino for p in rich.rglob("*") if p.is_file()}
    assert all(
        not p.is_symlink() and p.stat().st_ino not in source_inodes for p in output.rglob("*") if p.is_file()
    )


def test_ambiguous_task_variants_and_empty_saved_override(rich, tmp_path):
    path = rich / "meta/lerobot_annotations.json"
    saved = json.loads(path.read_text())
    saved["episodes"]["0"]["atoms"].append(atom("task_aug", 0, "Another full task"))
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="[Aa]mbiguous.*task_aug"):
        materialize_groot_v21(rich, tmp_path / "ambiguous")
    assert not (tmp_path / "ambiguous").exists()
    saved["episodes"]["0"]["atoms"] = []
    path.write_text(json.dumps(saved))
    output = tmp_path / "empty"
    materialize_groot_v21(rich, output)
    tasks = {r["task_index"]: r["task"] for r in read_jsonl(output / "meta/tasks.jsonl")}
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert [tasks[i] for i in table["task_index"].to_pylist()] == (
        ["the entire first task label"] * 4 + ["a distinct complete second task label"] * 4
    )


@pytest.mark.parametrize("unsupported", ["depth", "audio", "image"])
def test_reject_existing_nested_and_unsupported_media(rich, tmp_path, unsupported):
    existing = tmp_path / "existing"
    existing.mkdir()
    for target in (existing, rich / "nested", rich):
        with pytest.raises((ValueError, FileExistsError)):
            materialize_groot_v21(rich, target)
    info_path = rich / "meta/info.json"
    info = json.loads(info_path.read_text())
    if unsupported == "depth":
        info["features"]["observation.images.left"]["info"]["is_depth_map"] = True
    else:
        info["features"]["observation.images.left"]["dtype"] = unsupported
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="unsupported|RGB"):
        materialize_groot_v21(rich, tmp_path / "depth")
    assert not (tmp_path / "depth").exists()


def test_parquet_annotations_work_without_an_editor_sidecar(rich, tmp_path):
    (rich / "meta/lerobot_annotations.json").unlink()
    output = tmp_path / "parquet-only"
    materialize_groot_v21(rich, output)
    assert [row["task"] for row in read_jsonl(output / "meta/tasks.jsonl")] == ["Full task 0", "Full task 1"]


def test_corrupt_shared_video_cleans_partial_output(rich, tmp_path):
    next((rich / "videos").rglob("*.mp4")).write_bytes(b"broken video")
    before = source_inventory(rich)
    with pytest.raises(Exception):
        materialize_groot_v21(rich, tmp_path / "broken")
    assert not (tmp_path / "broken").exists()
    assert source_inventory(rich) == before
