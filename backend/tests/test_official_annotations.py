"""Integration checks against the pinned LeRobot modules, writer and validator."""

from __future__ import annotations

import json

import av
import datasets.config
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
from lerobot.annotations.steerable_pipeline.writer import speech_atom
import numpy as np
import official_annotations as engine
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

CAMERA = "observation.images.top"
DATA = "data/chunk-000/file-000.parquet"


def atom(style, content, *, timestamp=0.0, role="assistant", camera=None):
    return dict(style=style, content=content, timestamp=timestamp, role=role, camera=camera, tool_calls=None)


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_deployment_config_defaults_are_shared_and_nested(monkeypatch, tmp_path):
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps({"vlm": {"api_base": "https://deploy.example"}}))
    monkeypatch.setenv("LEROBOT_ANNOTATE_CONFIG", str(path))
    monkeypatch.setenv("LEROBOT_VLM_API_KEY", "secret")

    defaults = engine.default_config()
    config = engine.parse_config({"vlm": {"client_concurrency": 2}})
    assert defaults["vlm"]["api_base"] == config.vlm.api_base == "https://deploy.example"
    assert config.vlm.client_concurrency == 2
    assert "api_key" not in defaults["vlm"] and "serve_command" not in defaults["vlm"]
    assert config.vlm.api_key == "secret"


@pytest.mark.parametrize("contents", ["not json", "[]"])
def test_invalid_deployment_config_is_rejected(monkeypatch, tmp_path, contents):
    path = tmp_path / "invalid.json"
    path.write_text(contents)
    monkeypatch.setenv("LEROBOT_ANNOTATE_CONFIG", str(path))
    with pytest.raises(ValueError, match="LEROBOT_ANNOTATE_CONFIG"):
        engine.default_config()


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.1",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 24,
        "total_tasks": 1,
        "data_path": DATA,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAMERA: {
                "dtype": "video",
                "shape": [24, 32, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": 10, "video.height": 24, "video.width": 32, "video.is_depth_map": False},
            }
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["Place the cup on the table"], name="task")).to_parquet(
        root / "meta/tasks.parquet"
    )
    rows = []
    for ep in range(2):
        rows.append(
            {
                "episode_index": ep,
                "length": 12,
                "tasks": ["Place the cup on the table"],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": ep * 12,
                "dataset_to_index": (ep + 1) * 12,
                f"videos/{CAMERA}/chunk_index": 0,
                f"videos/{CAMERA}/file_index": 0,
                f"videos/{CAMERA}/from_timestamp": ep * 1.2,
                f"videos/{CAMERA}/to_timestamp": (ep + 1) * 1.2,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
    frames = pa.table(
        {
            "episode_index": [ep for ep in range(2) for _ in range(12)],
            "frame_index": list(range(12)) * 2,
            "timestamp": [i / 10 for i in range(12)] * 2,
            "task_index": [0] * 24,
            "subtask_index": [0] * 24,
            "observation.state": [[float(i), -float(i)] for i in range(24)],
        }
    )
    persistent = [[atom("task_aug", f"original episode {ep}")] for ep in range(2) for _ in range(12)]
    event = atom("vqa", '{"label":"cup","count":1}', camera=CAMERA)
    event.pop("timestamp")
    frames = frames.append_column("language_persistent", pa.array(persistent))
    frames = frames.append_column("language_events", pa.array([[event] if i % 12 == 0 else [] for i in range(24)]))
    pq.write_table(frames, root / DATA)
    (root / "meta/lerobot_annotations.json").write_text('{"version":2,"episodes":{"0":{"atoms":[]}}}')
    video = root / f"videos/{CAMERA}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for i in range(24):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), i * 10, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return root


@pytest.mark.parametrize("all_modules", [True, False], ids=["all-official-modules", "preserve-disabled-modules"])
@pytest.mark.parametrize("few_shot", [False, True])
def test_selected_generation_preserves_shared_shard_and_manual_edits(dataset, tmp_path, all_modules, few_shot):
    source_bytes = snapshot(dataset)
    original_records = list(iter_episodes(dataset))
    untouched = engine.read_atoms(original_records[1])
    if few_shot:
        untouched = [*untouched, atom("subtask", "human reviewed cup handling")]
    manual = atom("vqa", '{"label":"manual cup","count":7}', camera=CAMERA)
    interjection = atom("interjection", "now place the cup", timestamp=0.4, role="user")
    speech = {"camera": None, **speech_atom(0.4, "Placing it.")}
    saved = [manual]
    if not all_modules:
        saved += [interjection, speech, atom("plan", "1. old manual plan", timestamp=0.4)]
        assert engine.validate_atoms(dataset, original_records, {0: saved})["ok"]
    calls = []

    def respond(messages):
        blocks = [b for m in messages for b in m["content"] if isinstance(b, dict)]
        prompt = "\n".join(b.get("text", "") for b in blocks)
        calls.append((prompt, any(b.get("type") == "image" for b in blocks)))
        if "COMPLETED manipulation events" in prompt:
            return {
                "subtasks": [
                    {"text": "grasp the cup", "start": 0.0, "end": 0.5},
                    {"text": "place the cup", "start": 0.5, "end": 1.1},
                ]
            }
        if "compressed semantic memory" in prompt:
            return {"memory": "grasped the cup"}
        if "acknowledgement the robot" in prompt:
            return {"text": "On it."}
        if "Write ONE compact interjection" in prompt:
            return {"interjection": "now place the cup please", "speech": "Placing it."}
        if "frame-grounded visual question" in prompt:
            return {"question": "How many cups?", "answer": {"label": "cup", "count": 2}}
        raise AssertionError(f"Unexpected official prompt: {prompt[:200]}")

    config = engine.parse_config(
        {
            "plan": {"n_task_rephrasings": 0, "subtask_describe_first": False, "min_subtask_seconds": 0.1},
            "interjections": {
                "enabled": all_modules,
                "max_interjections_per_episode": 1,
                "interjection_min_t": 0.2,
            },
            "vqa": {"enabled": all_modules, "question_types": ["count"]},
            "executor": {"episode_parallelism": 1},
        }
    )
    output = tmp_path / "generated"
    result = engine.generate_dataset(
        dataset,
        output,
        {0: saved, **({1: untouched} if few_shot else {})},
        config,
        None if few_shot else [0],
        example_episode_indices=[1] if few_shot else [],
        vlm=StubVlmClient(responder=respond),
    )
    assert result["validation"]["ok"] and result["validation"]["episodes_checked"] == 2
    assert result["first_generated_episode_index"] == 0
    from pathlib import Path

    baseline = json.loads(Path(result["prediction_snapshot"]).read_text())
    assert set(baseline["episodes"]) == {"0"}
    assert baseline["example_episode_indices"] == ([1] if few_shot else [])
    assert "api_key" not in baseline["config"]["vlm"]
    assert (
        baseline["episodes"]["0"]
        == json.loads((output / "meta/lerobot_annotations.json").read_text())["episodes"]["0"]
    )
    generated, preserved = list(iter_episodes(output))
    assert sorted(json.dumps(a, sort_keys=True) for a in engine.read_atoms(preserved)) == sorted(
        json.dumps(a, sort_keys=True) for a in untouched
    )
    if few_shot:
        assert all('"episode_index": 1' in prompt and "END OF EXAMPLES" in prompt for prompt, _ in calls)
        assert result["example_episode_indices"] == [1]
        assert all("human reviewed cup handling" in prompt for prompt, _ in calls)
        provenance = json.loads((output / "meta/annotation_pipeline.json").read_text())
        assert provenance["few_shot"]["annotation_sha256"]["1"]
    atoms = engine.read_atoms(generated)
    assert {"subtask", "plan", "memory"} <= {a["style"] for a in atoms}
    assert any("COMPLETED manipulation events" in prompt and images for prompt, images in calls)
    if all_modules:
        assert {"interjection", None, "vqa"} <= {a["style"] for a in atoms}
        assert any("frame-grounded visual question" in prompt and images for prompt, images in calls)
        assert any(a["style"] == "vqa" and '"count": 2' in a["content"] for a in atoms)
    else:
        assert manual in atoms
        assert [a for a in atoms if a["style"] == "interjection"] == [interjection]
        assert [a for a in atoms if a["style"] is None] == [speech]
        assert any(
            a["style"] == "plan" and a["timestamp"] == 0.4 and a["content"] == "1. place the cup" for a in atoms
        )
    sidecar = json.loads((output / "meta/lerobot_annotations.json").read_text())
    assert sidecar["episodes"]["0"]["atoms"]
    assert {a["content"] for a in sidecar["episodes"]["0"]["atoms"]} == {a["content"] for a in atoms}
    assert engine.validate_atoms(output, list(iter_episodes(output)), {})["ok"]
    table = pq.read_table(output / DATA)
    assert "subtask_index" not in table.column_names
    assert table["observation.state"] == pq.read_table(dataset / DATA)["observation.state"]
    assert snapshot(dataset) == source_bytes


@pytest.mark.parametrize(
    "examples,targets,overrides,error",
    [
        ([1, 1], None, {}, "distinct"),
        ([99], None, {}, "existing"),
        ([1], [1], {}, "No target"),
        ([1], None, {1: []}, "saved annotations"),
        ([1], None, {1: [atom("interjection", "stop", role="user")]}, "official validation"),
    ],
)
def test_invalid_examples_fail_before_vlm_calls(dataset, tmp_path, examples, targets, overrides, error):
    def unexpected(_):
        pytest.fail("Invalid examples must not reach the VLM")

    before = snapshot(dataset)
    with pytest.raises(ValueError, match=error):
        engine.generate_dataset(
            dataset,
            tmp_path / "bad-example",
            overrides,
            engine.parse_config({}),
            targets,
            example_episode_indices=examples,
            vlm=StubVlmClient(responder=unexpected),
        )
    assert snapshot(dataset) == before


@pytest.mark.parametrize(
    ("bad_atom", "error"),
    [
        (atom("interjection", "stop", timestamp=0.5, role="user"), "paired speech"),
        (atom("vqa", '{"label":"cup","count":1}'), "camera"),
        (atom("vqa", '{"label":"cup","count":1}', camera="observation.images.missing"), "not one of"),
        (atom("vqa", "not JSON", camera=CAMERA), "not valid JSON"),
        (atom("vqa", '{"label":"cup","count":1}', timestamp=0.123, camera=CAMERA), "source frame timestamp"),
    ],
)
def test_export_uses_official_validator_before_writing(dataset, tmp_path, bad_atom, error):
    before = snapshot(dataset)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=error):
        engine.export_dataset(dataset, output, {0: [bad_atom]})
    assert (output / DATA).read_bytes() == (dataset / DATA).read_bytes()
    assert snapshot(dataset) == before


def test_empty_override_removes_language_and_overlapping_output_is_rejected(dataset, tmp_path):
    before = snapshot(dataset)
    for output in (dataset, dataset / "nested", dataset.parent):
        with pytest.raises(ValueError, match="outside the source"):
            engine.export_dataset(dataset, output, {})
    output = tmp_path / "exported"
    engine.export_dataset(dataset, output, {0: []})
    first, second = list(iter_episodes(output))
    assert engine.read_atoms(first) == []
    assert engine.read_atoms(second) == engine.read_atoms(list(iter_episodes(dataset))[1])
    assert json.loads((output / "meta/lerobot_annotations.json").read_text())["episodes"]["0"]["atoms"] == []
    assert snapshot(dataset) == before


def test_prepare_converts_v21_with_official_converter_without_changing_source(dataset, tmp_path):
    source = tmp_path / "v21"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    table = pq.read_table(dataset / DATA).select(
        ["episode_index", "frame_index", "timestamp", "task_index", "observation.state"]
    )
    for ep in range(2):
        pq.write_table(table.slice(ep * 12, 12), source / f"data/chunk-000/episode_{ep:06d}.parquet")
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 24,
        "total_tasks": 1,
        "total_chunks": 1,
        "total_videos": 0,
        "video_path": None,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {"observation.state": {"dtype": "float32", "shape": [2], "names": None}},
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Place the cup on the table"}) + "\n"
    )
    (source / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": ep, "tasks": ["Place the cup on the table"], "length": 12}) + "\n"
            for ep in range(2)
        )
    )
    stats = []
    for ep in range(2):
        values = np.array(table.slice(ep * 12, 12)["observation.state"].to_pylist())
        stats.append(
            {
                "episode_index": ep,
                "stats": {
                    "observation.state": {
                        "min": values.min(0).tolist(),
                        "max": values.max(0).tolist(),
                        "mean": values.mean(0).tolist(),
                        "std": values.std(0).tolist(),
                        "count": [12],
                    }
                },
            }
        )
    (source / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    (source / "meta/modality.json").write_text('{"state":{"joints":{"start":0,"end":2}}}')
    before = snapshot(source)
    output = tmp_path / "converted"
    assert engine.prepare_dataset(source, output)["output_dir"] == str(output)
    assert json.loads((output / "meta/info.json").read_text())["codebase_version"] == "v3.0"
    records = list(iter_episodes(output))
    assert [r.row_count for r in records] == [12, 12]
    assert all(r.episode_task == "Place the cup on the table" for r in records)
    assert pq.read_table(records[0].data_path)["observation.state"] == table["observation.state"]
    assert json.loads((output / "meta/stats.json").read_text())["observation.state"]["count"] == [24]
    assert (output / "meta/modality.json").read_bytes() == (source / "meta/modality.json").read_bytes()
    assert snapshot(source) == before


def guided_config():
    return engine.parse_config(
        {
            "plan": {
                "n_task_rephrasings": 0,
                "subtask_describe_first": False,
                "min_subtask_seconds": 0.1,
                "emit_memory": False,
                "emit_plan": False,
            },
            "interjections": {"enabled": False},
            "vqa": {"enabled": False},
            "executor": {"episode_parallelism": 1},
        }
    )


def test_guided_generation_snapshots_prompts_and_missing_labels_without_relabeling(dataset, tmp_path):
    prompts = []

    def respond(messages):
        prompts.append("\n".join(b.get("text", "") for m in messages for b in m["content"]))
        return {"subtasks": [{"text": "approach", "start": 0.0, "end": 1.1}]}

    result = engine.generate_dataset(
        dataset,
        tmp_path / "guided",
        {},
        guided_config(),
        [0],
        task_prompt="approach and grasp",
        subtask_prompts=["approach", "grasp"],
        vlm=StubVlmClient(responder=respond),
    )
    assert '"ordered_subtask_prompts": ["approach", "grasp"]' in prompts[0]
    assert '"task": "approach and grasp"' in prompts[0]
    assert result["episode_results"]["0"]["generation_status"] == "generated"
    assert result["episode_results"]["0"]["issues"][0]["code"] == "missing_subtask"
    baseline = json.loads(__import__("pathlib").Path(result["prediction_snapshot"]).read_text())
    assert baseline["task_prompt"] == "approach and grasp"
    assert baseline["subtask_prompts"] == ["approach", "grasp"]
    assert [a["content"] for a in baseline["episodes"]["0"]["atoms"] if a["style"] == "subtask"] == ["approach"]


@pytest.mark.parametrize("prompts", [[""], ["a", "a"], ["a", " a "], [None], "a"])
def test_invalid_subtask_prompts_rejected_before_copy(dataset, tmp_path, prompts):
    output = tmp_path / "invalid-prompts"
    with pytest.raises(ValueError, match="prompt"):
        engine.generate_dataset(dataset, output, {}, guided_config(), subtask_prompts=prompts)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["vlm", "validation", "timestamp", "missing_subtasks", "writer"])
def test_failed_episode_preserved_beside_success_without_false_prediction(dataset, tmp_path, monkeypatch, failure):
    original_run = engine.PlanSubtasksMemoryModule.run_episode

    def run(module, record, staging):
        if record.episode_index == 0:
            staging.write("plan", [atom("subtask", "partial write")])
            if failure == "vlm":
                raise RuntimeError("video could not decode")
            if failure == "validation":
                staging.write("vqa", [atom("vqa", "malformed", timestamp=0.123, camera=CAMERA)])
            elif failure == "timestamp":
                staging.write("plan", [atom("subtask", "bad time", timestamp=float("nan"))])
            elif failure == "missing_subtasks":
                staging.write("plan", [])
            elif failure == "writer":
                staging.write("plan", [{"style": "subtask", "content": "missing role", "timestamp": 0.0}])
        else:
            original_run(module, record, staging)

    monkeypatch.setattr(engine.PlanSubtasksMemoryModule, "run_episode", run)
    result = engine.generate_dataset(
        dataset,
        tmp_path / "partial",
        {},
        guided_config(),
        vlm=StubVlmClient(responder=lambda _: {"subtasks": [{"text": "grasp", "start": 0, "end": 1.1}]}),
    )
    assert result["validation"]["ok"]
    assert result["episode_results"]["0"]["generation_status"] == "failed"
    assert result["episode_results"]["0"]["issues"]
    assert result["episode_results"]["1"] == {"generation_status": "generated", "issues": []}
    assert result["first_generated_episode_index"] == 1
    records = list(iter_episodes(tmp_path / "partial"))
    assert engine.read_atoms(records[0]) == engine.read_atoms(list(iter_episodes(dataset))[0])
    baseline = json.loads(__import__("pathlib").Path(result["prediction_snapshot"]).read_text())
    assert set(baseline["episodes"]) == {"1"}
    assert baseline["episode_results"]["0"]["generation_status"] == "failed"


def test_qa_failure_is_a_finding_and_retry_retains_original_snapshot(dataset, tmp_path):
    from pathlib import Path

    def respond(messages):
        text = "\n".join(b.get("text", "") for m in messages for b in m["content"])
        if "Findings are advisory" in text:
            return {"issues": [{"code": "wrong_task", "message": "Wrong object", "start": 0, "end": 5}]}
        return {"subtasks": [{"text": "grasp", "start": 0, "end": 1.1}]}

    first = engine.generate_dataset(
        dataset,
        tmp_path / "first",
        {},
        guided_config(),
        [0],
        assess_quality=True,
        vlm=StubVlmClient(responder=respond),
    )
    assert first["episode_results"]["0"]["generation_status"] == "generated"
    assert [i["code"] for i in first["episode_results"]["0"]["issues"]] == ["assessment_failed"]
    prediction = Path(first["prediction_snapshot"])
    original = prediction.read_bytes()
    second = engine.generate_dataset(
        tmp_path / "first", tmp_path / "retry", {}, guided_config(), [0], vlm=StubVlmClient(responder=respond)
    )
    assert second["episode_results"]["0"]["issues"] == []
    assert (tmp_path / "retry/meta/annotation_predictions" / prediction.name).read_bytes() == original
    assert len(list((tmp_path / "retry/meta/annotation_predictions").glob("*.json"))) == 2


def test_task_context_is_scoped_to_plan_client(dataset, tmp_path):
    def respond(messages):
        text = "\n".join(b.get("text", "") for m in messages for b in m["content"])
        if "frame-grounded visual question" in text:
            assert '"ordered_subtask_prompts"' not in text
            return {"question": "How many cups?", "answer": {"label": "cup", "count": 1}}
        assert '"ordered_subtask_prompts": ["grasp"]' in text
        return {"subtasks": [{"text": "grasp", "start": 0, "end": 1.1}]}

    config = guided_config()
    config.vqa.enabled = True
    config.vqa.question_types = ["count"]
    result = engine.generate_dataset(
        dataset,
        tmp_path / "scope",
        {},
        config,
        [0],
        subtask_prompts=["grasp"],
        vlm=StubVlmClient(responder=respond),
    )
    assert result["episode_results"]["0"] == {"generation_status": "generated", "issues": []}
