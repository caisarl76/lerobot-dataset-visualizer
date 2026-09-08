from __future__ import annotations

import json

from groot_export import export_groot_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def atom(style, content, timestamp, role="assistant"):
    return dict(style=style, content=content, timestamp=timestamp, role=role)


@pytest.fixture
def datasets(tmp_path):
    source, rich, output = (tmp_path / name for name in ("source", "rich", "output"))
    write_json(
        source / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "chunks_size": 1000,
            "fps": 4,
            "total_episodes": 2,
            "total_frames": 8,
            "total_tasks": 1,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        },
    )
    write_json(
        source / "meta/modality.json",
        {
            "state": {"joint": {"start": 0, "end": 1}},
            "action": {"joint": {"start": 0, "end": 1}},
            "annotation": {"human.task_description": {"original_key": "old_task"}},
        },
    )
    write_json(
        source / "meta/stats.json",
        {
            "observation.state": {"mean": [0.5]},
            "action": {"mean": [0.75]},
            "task_index": {"mean": [9]},
        },
    )
    write_json(source / "meta/relative_stats.json", {"joint": {"mean": [0.0]}})
    write_jsonl(source / "meta/tasks.jsonl", [{"task_index": 9, "task": "Tidy table"}])
    write_jsonl(
        source / "meta/episodes.jsonl",
        [{"episode_index": ep, "length": 4, "tasks": ["Tidy table"]} for ep in range(2)],
    )
    write_jsonl(
        source / "meta/episodes_stats.jsonl",
        [
            {"episode_index": ep, "stats": {"action": {"mean": [ep]}, "task_index": {"mean": [9]}}}
            for ep in range(2)
        ],
    )
    write_json(rich / "meta/info.json", {"codebase_version": "v3.1"})
    (rich / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 1]}), rich / "meta/episodes/chunk-000/file-000.parquet")
    persistent = [
        atom("task_aug", "Arrange table", 0.0, "user"),
        atom("task_aug", "Later phrasing", 0.5, "user"),
        atom("subtask", "Reach", 0.25),
        atom("subtask", "Release", 0.75),
        atom("plan", "Move to bin", 0.25),
        atom("memory", "Grasped object", 0.5),
    ]
    rich_rows = []
    for ep in range(2):
        source_rows = [
            dict(
                episode_index=ep,
                frame_index=i,
                timestamp=i / 4,
                task_index=9,
                index=ep * 4 + i,
                action=[float(i)],
                **{"observation.state": [i * 2.0]},
            )
            for i in range(4)
        ]
        data_path = source / f"data/chunk-000/episode_{ep:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(source_rows), data_path)
        video = source / f"videos/chunk-000/front/episode_{ep:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"opaque video bytes " + bytes([ep]))
        for row in source_rows:
            events = []
            if row["frame_index"] == 1:
                events = [{"style": "interjection", "content": "Careful", "role": "user"}]
            if row["frame_index"] == 0:
                events = [
                    {"style": "vqa", "content": "SECRET VQA answer", "role": "assistant"},
                    {"style": None, "content": "SECRET speech", "role": "assistant"},
                ]
            rich_rows.append(
                {
                    **{k: row[k] for k in ("episode_index", "frame_index", "timestamp")},
                    "language_persistent": persistent,
                    "language_events": events,
                }
            )
    rich_data = rich / "data/chunk-000/file-000.parquet"
    rich_data.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rich_rows), rich_data)
    return rich, source, output


def exported_texts(output, episode=0):
    # Exact GR00T loader mapping: annotation subkey -> original_key -> tasks_map.
    modality = json.loads((output / "meta/modality.json").read_text())
    key = modality["annotation"]["human.task_description"]["original_key"]
    tasks_map = {row["task_index"]: row["task"] for row in read_jsonl(output / "meta/tasks.jsonl")}
    table = pq.read_table(output / f"data/chunk-000/episode_{episode:06d}.parquet")
    return [tasks_map[index] for index in table[key].to_pylist()]


def test_export_preserves_inputs_and_nonlanguage_data(datasets):
    rich, source, output = datasets
    before = {p: p.read_bytes() for root in (rich, source) for p in root.rglob("*") if p.is_file()}
    report = export_groot_dataset(rich, source, output)
    assert report["frames"] == 8 and report["episodes"] == 2 and report["tasks"] == 3
    assert exported_texts(output) == ["Tidy table", "Reach", "Reach", "Release"]
    assert exported_texts(output, 1) == exported_texts(output)
    for path, original in before.items():
        assert path.read_bytes() == original
    for path in (source / "data").rglob("*.parquet"):
        original = pq.read_table(path).drop(["task_index"])
        exported = pq.read_table(output / path.relative_to(source)).drop(["task_index"])
        assert original.equals(exported, check_metadata=True)
    for path in (source / "videos").rglob("*.mp4"):
        target = output / path.relative_to(source)
        assert target.read_bytes() == path.read_bytes()
        assert target.stat().st_ino == path.stat().st_ino
    assert (source / "meta/relative_stats.json").read_bytes() == (output / "meta/relative_stats.json").read_bytes()
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats == {"observation.state": {"mean": [0.5]}, "action": {"mean": [0.75]}}
    assert read_jsonl(output / "meta/episodes_stats.jsonl")[1]["stats"] == {"action": {"mean": [1]}}
    assert read_jsonl(output / "meta/episodes.jsonl")[0]["tasks"] == ["Tidy table", "Reach", "Release"]
    assert json.loads((output / "meta/info.json").read_text())["total_tasks"] == 3


def test_context_is_causal_at_exact_boundaries_and_excludes_vqa_and_say(datasets):
    rich, source, output = datasets
    export_groot_dataset(rich, source, output, mode="context")
    assert exported_texts(output) == [
        "Task: Tidy table",
        "Task: Tidy table\nSubtask: Reach\nPlan: Move to bin\nInterjection: Careful",
        "Task: Tidy table\nSubtask: Reach\nPlan: Move to bin\nMemory: Grasped object\nInterjection: Careful",
        "Task: Tidy table\nSubtask: Release\nPlan: Move to bin\nMemory: Grasped object\nInterjection: Careful",
    ]


def test_saved_edits_and_explicit_empty_episode_override_parquet(datasets):
    rich, source, output = datasets
    write_json(
        rich / "meta/lerobot_annotations.json",
        {
            "version": 2,
            "episodes": {
                "0": {"atoms": [atom("subtask", "Edited", 0.5)]},
                "1": {"atoms": []},
            },
        },
    )
    export_groot_dataset(rich, source, output)
    assert exported_texts(output) == ["Tidy table", "Tidy table", "Edited", "Edited"]
    assert exported_texts(output, 1) == ["Tidy table"] * 4


def test_task_variant_never_uses_future_phrasing(datasets):
    rich, source, output = datasets
    export_groot_dataset(rich, source, output, mode="task", task_variant=1)
    assert exported_texts(output) == ["Tidy table", "Tidy table", "Later phrasing", "Later phrasing"]


def test_video_copy_fallback(datasets, monkeypatch):
    import groot_export

    rich, source, output = datasets

    def cannot_link(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr(groot_export.os, "link", cannot_link)
    export_groot_dataset(rich, source, output)
    original = source / "videos/chunk-000/front/episode_000000.mp4"
    exported = output / original.relative_to(source)
    assert original.read_bytes() == exported.read_bytes()
    assert original.stat().st_ino != exported.stat().st_ino


@pytest.mark.parametrize(
    "failure", ["timestamp", "missing_frame", "duplicate_frame", "missing_episode", "extra_episode"]
)
def test_alignment_rejects_missing_or_mismatched_frames_before_writing(datasets, failure):
    rich, source, output = datasets
    path = rich / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    if failure == "timestamp":
        rows[1]["timestamp"] += 1e-10  # A rounded timestamp join would wrongly accept this.
    elif failure == "missing_frame":
        rows.pop(1)
    elif failure == "duplicate_frame":
        rows[1]["frame_index"] = 0
    elif failure == "missing_episode":
        rows = rows[:4]
    else:
        rows.append({**rows[0], "episode_index": 2})
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="[Ff]rame|episode|timestamp"):
        export_groot_dataset(rich, source, output)
    assert not output.exists()


def test_rejects_overwrite_and_source_descendant(datasets):
    rich, source, output = datasets
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        export_groot_dataset(rich, source, output)
    assert (output / "keep").read_text() == "existing"
    with pytest.raises(ValueError, match="separate"):
        export_groot_dataset(rich, source, source / "export")


def test_invalid_variant_and_ambiguous_atoms_fail_before_writing(datasets):
    rich, source, output = datasets
    with pytest.raises(ValueError, match="unavailable"):
        export_groot_dataset(rich, source, output, task_variant=10)
    write_json(
        rich / "meta/lerobot_annotations.json",
        {
            "episodes": {
                "0": {"atoms": [atom("subtask", "A", 0.0), atom("subtask", "B", 0.0)]},
            }
        },
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        export_groot_dataset(rich, source, output)
    assert not output.exists()
