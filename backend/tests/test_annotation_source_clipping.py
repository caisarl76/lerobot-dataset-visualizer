import hashlib
import json

from annotation_source import align_source_v21
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _video(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "yuv420p"
        for value in values:
            frame = av.VideoFrame.from_ndarray(np.full((12, 16, 3), value, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [1]},
        "observation.images.left": {"dtype": "video", "shape": [12, 16, 3]},
        "observation.images.right": {"dtype": "video", "shape": [12, 16, 3]},
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 4,
        "chunks_size": 1000,
        "total_episodes": 1,
        "total_frames": 5,
        "total_chunks": 1,
        "total_videos": 2,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = {
        "episode_index": [0] * 5,
        "frame_index": list(range(5)),
        "index": list(range(5)),
        "timestamp": [i / 4 for i in range(5)],
        "task_index": [7] * 5,
        "observation.state": [[float(i)] for i in range(5)],
    }
    data = root / "data/chunk-000/episode_000000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(pa.table(rows), data)
    for camera, base in [("observation.images.left", 10), ("observation.images.right", 100)]:
        _video(root / f"videos/chunk-000/{camera}/episode_000000.mp4", [base + i * 10 for i in range(5)])
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 5, "tasks": ["task"]}) + "\n"
    )
    (root / "meta/episodes_stats.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "stats": {
                    "observation.state": {"min": [0], "max": [4], "mean": [2], "std": [1.4], "count": [5]},
                    "observation.images.left": {
                        "min": [0, 0, 0],
                        "max": [1, 1, 1],
                        "mean": [0, 0, 0],
                        "std": [1, 1, 1],
                        "count": [5],
                    },
                    "observation.images.right": {
                        "min": [0, 0, 0],
                        "max": [1, 1, 1],
                        "mean": [0, 0, 0],
                        "std": [1, 1, 1],
                        "count": [5],
                    },
                },
            }
        )
        + "\n"
    )
    return root


def test_clip_rewrites_rows_stats_and_both_videos(source, tmp_path):
    before = {
        p.relative_to(source): hashlib.sha256(p.read_bytes()).digest() for p in source.rglob("*") if p.is_file()
    }
    output = tmp_path / "output"
    align_source_v21(source, output, {0: 0}, kept_frames={0: [1, 3]})
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert table["frame_index"].to_pylist() == [0, 1]
    assert table["timestamp"].to_pylist() == pytest.approx([0.0, 0.25])
    assert table["task_index"].to_pylist() == [7, 7]
    assert table["observation.state"].to_pylist() == [[1.0], [3.0]]
    stats = json.loads((output / "meta/episodes_stats.jsonl").read_text())["stats"]
    assert stats["observation.state"]["min"] == [1.0]
    assert stats["observation.state"]["max"] == [3.0]
    for camera, expected in [("observation.images.left", [20, 40]), ("observation.images.right", [110, 130])]:
        with av.open(str(output / f"videos/chunk-000/{camera}/episode_000000.mp4")) as container:
            decoded = list(container.decode(video=0))
            assert all((frame.width, frame.height) == (16, 12) for frame in decoded)
            assert [float(frame.pts * frame.time_base) for frame in decoded] == pytest.approx([0, 0.25])
            frames = [frame.to_ndarray(format="rgb24")[0, 0, 0] for frame in decoded]
        assert len(frames) == 2
        assert frames == pytest.approx(expected, abs=8)
    after = {
        p.relative_to(source): hashlib.sha256(p.read_bytes()).digest() for p in source.rglob("*") if p.is_file()
    }
    assert after == before


@pytest.mark.parametrize("kept", [{0: []}, {0: [2, 1]}, {0: [9]}, {1: [0]}, {0: [True]}, {0: [1.5]}])
def test_invalid_kept_frames_rejected_before_output(source, tmp_path, kept):
    with pytest.raises(ValueError):
        align_source_v21(source, tmp_path / "output", {0: 0}, kept_frames=kept)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kept", [[1, 2, 3, 4], [0, 1, 2], [4]])
def test_prefix_tail_and_single_retained_frame(source, tmp_path, kept):
    out = tmp_path / "out"
    align_source_v21(source, out, {0: 0}, kept_frames={0: kept})
    data = pq.read_table(out / "data/chunk-000/episode_000000.parquet")
    assert data["observation.state"].to_pylist() == [[float(i)] for i in kept]
    assert data["timestamp"].to_pylist() == pytest.approx([i / 4 for i in range(len(kept))])
    for camera in ["left", "right"]:
        with av.open(str(out / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4")) as container:
            assert len(list(container.decode(video=0))) == len(kept)
