"""Corrupt episodes remain visible by source identity until a human export decision."""

import json

import av
import datasets
import numpy as np
import official_annotations as engine
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def raw_v21(tmp_path, monkeypatch, request):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    count = getattr(request, "param", 2)
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 4,
        "chunks_size": 1000,
        "total_episodes": count,
        "total_frames": count * 4,
        "total_tasks": 1,
        "total_chunks": 1,
        "total_videos": count,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [1], "names": None},
            "observation.images.cam": {
                "dtype": "video",
                "shape": [24, 32, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": 4, "video.height": 24, "video.width": 32, "video.is_depth_map": False},
            },
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Pick the cup"}) + "\n")
    rows, stats = [], []
    for ep in range(count):
        rows.append({"episode_index": ep, "length": 4, "tasks": ["Pick the cup"]})
        stats.append(
            {
                "episode_index": ep,
                "stats": {
                    "observation.state": {
                        "min": [0.0],
                        "max": [3.0],
                        "mean": [1.5],
                        "std": [1.118],
                        "count": [4],
                    }
                },
            }
        )
        data = root / f"data/chunk-000/episode_{ep:06d}.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": [ep] * 4,
                    "frame_index": list(range(4)),
                    "index": list(range(ep * 4, (ep + 1) * 4)),
                    "timestamp": [0.0, 0.25, 0.5, 0.75],
                    "task_index": [0] * 4,
                    "observation.state": [[float(i)] for i in range(4)],
                }
            ),
            data,
        )
        video = root / f"videos/chunk-000/observation.images.cam/episode_{ep:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=4)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
            for i in range(4):
                frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), i * 40, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    for name, values in (("episodes", rows), ("episodes_stats", stats)):
        (root / f"meta/{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    return root


@pytest.mark.parametrize("kind", ["video", "parquet", "missing_video"])
def test_prepare_keeps_corrupt_source_episode_in_raw_review_workspace(raw_v21, tmp_path, kind):
    damaged = raw_v21 / (
        "data/chunk-000/episode_000001.parquet"
        if kind == "parquet"
        else "videos/chunk-000/observation.images.cam/episode_000001.mp4"
    )
    if kind == "missing_video":
        damaged.unlink()
    else:
        damaged.write_bytes(b"corrupt source bytes")
    before = {p.relative_to(raw_v21): p.read_bytes() for p in raw_v21.rglob("*") if p.is_file()}
    output = tmp_path / "review"
    result = engine.prepare_dataset(raw_v21, output)
    assert result["preparation_mode"] == "source_review"
    assert result["episode_indices"] == [0, 1]
    assert result["episode_results"]["0"] == {"generation_status": "pending", "issues": []}
    failed = result["episode_results"]["1"]
    assert failed["generation_status"] == "failed"
    assert failed["issues"][0]["code"] == ("unreadable_data" if kind == "parquet" else "unreadable_video")
    assert not result["source_validation"]["ok"]
    assert {p.relative_to(raw_v21): p.read_bytes() for p in raw_v21.rglob("*") if p.is_file()} == before
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == before


def test_valid_source_still_uses_official_conversion(raw_v21, tmp_path):
    result = engine.prepare_dataset(raw_v21, tmp_path / "converted")
    assert result["preparation_mode"] == "ready"
    assert result["episode_indices"] == [0, 1]
    assert result["source_validation"]["ok"]
    assert json.loads((tmp_path / "converted/meta/info.json").read_text())["codebase_version"] == "v3.0"
    assert [r.episode_index for r in engine.iter_episodes(tmp_path / "converted")] == [0, 1]


def test_v3_corrupt_data_is_reported_for_all_affected_original_episodes(raw_v21, tmp_path):
    converted = tmp_path / "converted"
    engine.prepare_dataset(raw_v21, converted)
    next((converted / "data").rglob("*.parquet")).write_bytes(b"unreadable shared shard")
    result = engine.prepare_dataset(converted, tmp_path / "review")
    assert result["preparation_mode"] == "source_review"
    assert result["episode_indices"] == [0, 1]
    assert all(ep["generation_status"] == "failed" for ep in result["episode_results"].values())


def test_raw_generation_keeps_failed_episode_and_uses_official_modules_for_good_episode(raw_v21, tmp_path):
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
    from test_official_annotations import guided_config

    damaged = raw_v21 / "data/chunk-000/episode_000001.parquet"
    damaged.write_bytes(b"corrupt source episode")
    result = engine.generate_dataset(
        raw_v21,
        tmp_path / "generated",
        {},
        guided_config(),
        subtask_prompts=["grasp"],
        vlm=StubVlmClient(responder=lambda _: {"subtasks": [{"text": "grasp", "start": 0.0, "end": 0.75}]}),
    )
    assert result["episode_results"]["0"]["generation_status"] == "generated"
    assert result["episode_results"]["1"]["generation_status"] == "failed"
    output = tmp_path / "generated"
    assert (output / damaged.relative_to(raw_v21)).read_bytes() == damaged.read_bytes()
    assert json.loads((output / "meta/info.json").read_text())["total_episodes"] == 2
    snapshot = json.loads(__import__("pathlib").Path(result["prediction_snapshot"]).read_text())
    assert set(snapshot["episodes"]) == {"0"}
    assert snapshot["episode_results"]["1"]["generation_status"] == "failed"


@pytest.mark.parametrize("raw_v21", [3], indirect=True)
def test_raw_generation_maps_examples_predictions_and_retry_history_to_source_ids(raw_v21, tmp_path):
    from annotation_history import annotation_hash, save_review, snapshot_predictions
    from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
    from test_official_annotations import guided_config

    example = [{"role": "assistant", "style": "subtask", "content": "example grasp", "timestamp": 0.0}]
    (raw_v21 / "meta/lerobot_annotations.json").write_text(
        json.dumps({"version": 2, "episodes": {"0": {"atoms": example}}})
    )
    save_review(raw_v21, 0, example, True, annotation_hash(example))
    original = snapshot_predictions(
        raw_v21,
        raw_v21,
        set(),
        {},
        {},
        engine.REVISION,
        episode_results={"1": {"generation_status": "failed", "issues": []}},
    )
    original_bytes = original.read_bytes()
    damaged = raw_v21 / "videos/chunk-000/observation.images.cam/episode_000001.mp4"
    damaged.write_bytes(b"bad original video")
    source = raw_v21
    for name in ("generated", "retry"):
        output = tmp_path / name
        result = engine.generate_dataset(
            source,
            output,
            {0: example},
            guided_config(),
            [2],
            example_episode_indices=[0],
            subtask_prompts=["grasp"],
            vlm=StubVlmClient(responder=lambda _: {"subtasks": [{"text": "grasp", "start": 0.0, "end": 0.75}]}),
        )
        snapshot = json.loads(__import__("pathlib").Path(result["prediction_snapshot"]).read_text())
        assert set(result["episode_results"]) == {"2"}
        assert result["first_generated_episode_index"] == 2
        assert snapshot["original_episode_indices"] == {"2": 2}
        assert snapshot["example_episode_indices"] == [0]
        assert set(snapshot["episodes"]) == {"2"}
        assert (output / "meta/annotation_predictions" / original.name).read_bytes() == original_bytes
        assert (output / damaged.relative_to(raw_v21)).read_bytes() == b"bad original video"
        assert json.loads((output / "meta/annotation_reviews.json").read_text())["0"][
            "annotation_sha256"
        ] == annotation_hash(example)
        assert (output / "data/chunk-000/episode_000002.parquet").read_bytes() == (
            raw_v21 / "data/chunk-000/episode_000002.parquet"
        ).read_bytes()
        assert result["output_dir"] == str(output)
        source = output
    assert len(list((source / "meta/annotation_predictions").glob("*.json"))) == 3
