"""Clipping preserves source/frame/media/language alignment in real v3 datasets."""

import json

import annotation_clipping as clipping
from annotation_runs import source_inventory
import datasets
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames
import numpy as np
import pyarrow.parquet as pq
import pytest


def exclusion(start, end):
    return {"start_frame": start, "end_frame": end}


def atom(style, timestamp, content):
    return {
        "style": style,
        "role": "user" if style == "task_aug" else "assistant",
        "timestamp": timestamp,
        "content": content,
        "camera": None,
        "tool_calls": None,
    }


@pytest.mark.parametrize(
    "intervals",
    [
        [exclusion(True, 2)],
        [exclusion(0, 1.0)],
        [exclusion(-1, 1)],
        [exclusion(0, 9)],
        [exclusion(2, 2)],
        [exclusion(3, 2)],
        [exclusion(0, 8)],
        [exclusion(0, 3), exclusion(3, 8)],
        None,
    ],
)
def test_invalid_exclusions(intervals):
    with pytest.raises(ValueError):
        clipping.normalize_exclusions(intervals, 8)


def test_exclusions_are_merged_without_mutating_input():
    spans = [exclusion(5, 7), exclusion(1, 3), exclusion(2, 5)]
    assert clipping.normalize_exclusions(spans, 8) == [exclusion(1, 7)]
    assert clipping.retained_indices(8, spans) == [0, 7]
    assert spans[1] == exclusion(1, 3)


def test_language_state_and_variants_survive_join_but_excluded_events_do_not():
    times = [i / 10 for i in range(8)]
    atoms = [
        atom("subtask", 0, "first"),
        atom("subtask", 0.3, "second"),
        atom("plan", 0.1, "plan"),
        atom("memory", 0.4, "remember"),
        atom("task_aug", 0, "variant a"),
        atom("task_aug", 0.7, "variant b"),
        atom("interjection", 0.3, "deleted event"),
        atom("interjection", 0.5, "retained event"),
    ]
    result = clipping.remap_atoms(atoms, times, [1, 2, 5, 6], 10)
    assert [(a["content"], a["timestamp"]) for a in result if a["style"] == "subtask"] == [
        ("first", 0),
        ("second", 0.2),
    ]
    assert [(a["content"], a["timestamp"]) for a in result if a["style"] == "task_aug"] == [
        ("variant a", 0),
        ("variant b", 0),
    ]
    assert [(a["content"], a["timestamp"]) for a in result if a["style"] == "interjection"] == [
        ("retained event", 0.2)
    ]
    assert next(a for a in result if a["style"] == "memory")["timestamp"] == 0.2
    assert atoms[1]["timestamp"] == 0.3


def test_event_cannot_be_silently_snapped_to_another_source_frame():
    with pytest.raises(ValueError, match="exact source frame"):
        clipping.remap_atoms([atom("interjection", 0.05, "off frame")], [0, 0.1, 0.2], [0, 1, 2], 10)


@pytest.fixture
def video_source(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    root = tmp_path / "source"
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["episode", "frame"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["command"]},
    }
    for camera in ("left", "right"):
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (16, 16, 3),
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id="local/source",
        root=root,
        fps=10,
        features=features,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=0),
        video_backend="pyav",
    )
    for ep in range(2):
        for index in range(8):
            frame = {
                "observation.state": np.array([ep, index], dtype=np.float32),
                "action": np.array([index * 2], dtype=np.float32),
                "task": "the entire first task label" if index < 4 else "a distinct complete second task label",
            }
            for camera, extra in (("left", 0), ("right", 10)):
                frame[f"observation.images.{camera}"] = np.full(
                    (16, 16, 3), ep * 60 + index * 15 + extra, np.uint8
                )
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    (root / "meta/modality.json").write_text(json.dumps({"state": {"body": {"start": 0, "end": 2}}}))
    return root


@pytest.mark.parametrize(
    "intervals,kept",
    [
        ([exclusion(0, 2)], list(range(2, 8))),
        ([exclusion(2, 5)], [0, 1, 5, 6, 7]),
        ([exclusion(6, 8)], list(range(6))),
    ],
)
def test_real_writer_clips_rows_both_cameras_tasks_and_stats(video_source, tmp_path, intervals, kept):
    before = source_inventory(video_source)
    source_meta = LeRobotDatasetMetadata(repo_id="local/source", root=video_source)
    # Episode 1 lives after episode 0 in the same video file, exercising offsets.
    assert source_meta.episodes[1]["videos/observation.images.left/from_timestamp"] > 0
    source_record = list(iter_episodes(video_source))[1]
    annotations = {
        1: [
            atom("subtask", 0, "first"),
            atom("subtask", source_record.frame_timestamps[3], "second"),
            {
                **atom("vqa", source_record.frame_timestamps[4], '{"label":"event","count":1}'),
                "camera": "observation.images.left",
            },
        ]
    }
    output = tmp_path / "clipped"
    result = clipping.clip_v3_dataset(video_source, output, {1: intervals}, annotations, [1])
    assert result["old_to_new"] == {1: 0}
    record = list(iter_episodes(output))[0]
    rows = pq.read_table(record.data_path).slice(record.row_offset, record.row_count).to_pylist()
    assert [row["observation.state"] for row in rows] == [[1.0, float(index)] for index in kept]
    assert [row["frame_index"] for row in rows] == list(range(len(kept)))
    assert [row["index"] for row in rows] == list(range(len(kept)))
    assert [row["episode_index"] for row in rows] == [0] * len(kept)
    np.testing.assert_allclose([row["timestamp"] for row in rows], np.arange(len(kept)) / 10, atol=1e-7)
    target_meta = LeRobotDatasetMetadata(repo_id="local/clipped", root=output)
    tasks = {int(row.task_index): task for task, row in target_meta.tasks.iterrows()}
    assert [tasks[row["task_index"]] for row in rows] == [
        "the entire first task label" if index < 4 else "a distinct complete second task label" for index in kept
    ]
    for camera, extra in (("left", 0), ("right", 10)):
        key = f"observation.images.{camera}"
        frames = decode_video_frames(
            output / target_meta.get_video_file_path(0, key),
            list(record.frame_timestamps),
            tolerance_s=1e-4,
            backend="pyav",
            return_uint8=True,
        )
        np.testing.assert_allclose(
            frames.numpy().mean(axis=(1, 2, 3)), [60 + index * 15 + extra for index in kept], atol=3
        )
    stats = json.loads((output / "meta/stats.json").read_text())
    np.testing.assert_allclose(stats["observation.state"]["mean"], [1, np.mean(kept)])
    assert stats["observation.state"]["count"] == [len(kept)]
    assert target_meta.total_frames == len(kept)
    assert target_meta.episodes[0]["length"] == len(kept)
    assert (output / "meta/modality.json").read_bytes() == (video_source / "meta/modality.json").read_bytes()
    assert source_inventory(video_source) == before
    from official_annotations import export_dataset

    rich = tmp_path / "rich"
    export_dataset(output, rich, result["annotations"])
    assert list(iter_episodes(rich))[0].row_count == len(kept)


def test_existing_output_and_nested_output_are_rejected(video_source, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("untouched")
    for target in (existing, video_source / "nested", video_source):
        with pytest.raises(ValueError, match="new directory"):
            clipping.clip_v3_dataset(video_source, target, {}, {}, [0])
    assert (existing / "keep").read_text() == "untouched"


def test_decode_failure_cleans_only_new_output(video_source, tmp_path, monkeypatch):
    before = source_inventory(video_source)

    def fail(*args, **kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr("lerobot.datasets.video_utils.decode_video_frames", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="decoder failed"):
        clipping.clip_v3_dataset(video_source, output, {}, {}, [0])
    assert not output.exists()
    assert source_inventory(video_source) == before


@pytest.mark.parametrize("kept", [[3, 4], [0, 3, 4]])
def test_clipping_rejects_new_mixed_role_active_state_ambiguity(kept):
    from lerobot.datasets.language_render import active_at

    atoms = [atom("subtask", 1.0, "old"), {**atom("subtask", 2.0, "new"), "role": "user"}]
    assert active_at(3.0, persistent=atoms, style="subtask")["content"] == "new"
    with pytest.raises(ValueError, match="ambiguous persistent subtask"):
        clipping.remap_atoms(atoms, [0.0, 1.0, 2.0, 3.0, 4.0], kept, 1)


@pytest.mark.parametrize("mixed_roles", [False, True])
def test_active_state_transitions_remain_valid_when_no_new_tie_is_created(mixed_roles):
    from lerobot.datasets.language_render import active_at

    atoms = [atom("subtask", 0.0, "first"), atom("subtask", 1.0, "second"), atom("subtask", 3.0, "third")]
    kept = [2, 3, 4]
    if mixed_roles:
        atoms[1]["role"] = "user"
        kept = [0, 1, 2, 3, 4]
    result = clipping.remap_atoms(atoms, [0.0, 1.0, 2.0, 3.0, 4.0], kept, 1)
    for new, old in enumerate(kept):
        assert (
            active_at(new, persistent=result, style="subtask")["content"]
            == active_at(old, persistent=atoms, style="subtask")["content"]
        )


def test_custom_tool_schemas_survive_base_rebuild_and_annotation_export(video_source, tmp_path):
    from official_annotations import export_dataset

    custom_tool = {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move to a position",
            "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
        },
    }
    source_meta = LeRobotDatasetMetadata(repo_id="local/source", root=video_source)
    source_meta.tools = [custom_tool]
    before = source_inventory(video_source)
    motion = {
        **atom("motion", 0.0, "move"),
        "tool_calls": [{"type": "function", "function": {"name": "move", "arguments": {"x": 1.0}}}],
    }
    output = tmp_path / "base"
    result = clipping.clip_v3_dataset(video_source, output, {1: [exclusion(0, 2)]}, {1: [motion]}, [1])
    assert json.loads((output / "meta/info.json").read_text())["tools"] == [custom_tool]
    rich = tmp_path / "rich"
    export_dataset(output, rich, result["annotations"])
    rich_meta = LeRobotDatasetMetadata(repo_id="local/rich", root=rich)
    assert custom_tool in rich_meta.tools
    assert any(tool["function"]["name"] == "say" for tool in rich_meta.tools)
    assert result["annotations"][0][0]["tool_calls"] == motion["tool_calls"]
    assert source_inventory(video_source) == before
