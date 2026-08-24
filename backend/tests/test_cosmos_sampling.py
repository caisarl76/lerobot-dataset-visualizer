from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import av
from curation.cosmos_transport import (
    SamplingLimits,
    prepare_episode_samples,
    prove_alignment_and_select,
)
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_video(path: Path, *, fps: int, frame_count: int, width: int = 64, height: int = 48) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            pixels = np.empty((height, width, 3), dtype=np.uint8)
            pixels[:, :, 0] = index % 256
            pixels[:, :, 1] = (index * 3) % 256
            pixels[:, :, 2] = (index * 7) % 256
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_parquet(path: Path, *, fps: float, frame_count: int) -> None:
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array(range(frame_count), type=pa.int64()),
                "timestamp": pa.array(
                    np.arange(frame_count, dtype=np.float64) / np.float64(fps),
                    type=pa.float64(),
                ),
            }
        ),
        path,
    )


@pytest.fixture
def aligned_50hz_episode(tmp_path: Path) -> tuple[Path, Path]:
    parquet_path = tmp_path / "episode.parquet"
    video_path = tmp_path / "episode.mkv"
    _write_parquet(parquet_path, fps=50.0, frame_count=101)
    _write_video(video_path, fps=50, frame_count=101)
    return parquet_path, video_path


def test_synthetic_50hz_parquet_and_video_prove_alignment_and_sample_exact_frames(
    aligned_50hz_episode: tuple[Path, Path],
) -> None:
    parquet_path, video_path = aligned_50hz_episode

    result = prepare_episode_samples(parquet_path=parquet_path, video_path=video_path, source_fps=50.0)

    assert result.status == "ready"
    assert result.reason is None
    assert result.sample is not None
    assert result.sample.source_fps == 50.0
    assert result.sample.total_num_frames == 101
    assert result.sample.duration_s == pytest.approx(101 / 50)
    assert result.sample.frame_indices == (0, 25, 50, 75, 100)
    assert result.sample.parquet_timestamps == pytest.approx((0.0, 0.5, 1.0, 1.5, 2.0))
    assert len(result.sample.jpeg_frames) == 5


@pytest.mark.parametrize(
    ("frame_indices", "timestamps", "video_rate", "video_count", "message"),
    [
        ([0, 2], [0.0, 0.02], 50.0, 2, "frame_index"),
        ([0, 1], [0.0, 0.04], 50.0, 2, "timestamp"),
        ([0, 1], [0.0, 0.02], 49.999, 2, "rate"),
        ([0, 1], [0.0, 0.02], 50.0, 3, "count"),
    ],
)
def test_alignment_proof_rejects_each_unproven_invariant(
    frame_indices: list[int],
    timestamps: list[float],
    video_rate: float,
    video_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prove_alignment_and_select(
            frame_indices=frame_indices,
            timestamps=timestamps,
            source_fps=50.0,
            video_rate=video_rate,
            video_frame_count=video_count,
        )


def test_float64_targets_break_nearest_ties_lower_and_deduplicate() -> None:
    proof = prove_alignment_and_select(
        frame_indices=[0, 1],
        timestamps=np.asarray([0.0, 1.0], dtype=np.float64),
        source_fps=1.0,
        video_rate=1.0,
        video_frame_count=2,
    )

    # Targets are 0.0, 0.5, 1.0.  The 0.5 tie selects index 0, then dedupes.
    assert proof.frame_indices == (0, 1)
    assert proof.parquet_timestamps == (0.0, 1.0)


def test_exact_rgb_jpegs_are_deterministic_never_upscale_and_resize_long_edge(tmp_path: Path) -> None:
    small_parquet = tmp_path / "small.parquet"
    small_video = tmp_path / "small.mkv"
    large_parquet = tmp_path / "large.parquet"
    large_video = tmp_path / "large.mkv"
    _write_parquet(small_parquet, fps=2.0, frame_count=2)
    _write_video(small_video, fps=2, frame_count=2, width=32, height=16)
    _write_parquet(large_parquet, fps=2.0, frame_count=2)
    _write_video(large_video, fps=2, frame_count=2, width=800, height=400)

    first = prepare_episode_samples(parquet_path=small_parquet, video_path=small_video, source_fps=2.0)
    repeated = prepare_episode_samples(parquet_path=small_parquet, video_path=small_video, source_fps=2.0)
    resized = prepare_episode_samples(parquet_path=large_parquet, video_path=large_video, source_fps=2.0)

    assert first.sample is not None and repeated.sample is not None and resized.sample is not None
    assert [hashlib.sha256(value).hexdigest() for value in first.sample.jpeg_frames] == [
        hashlib.sha256(value).hexdigest() for value in repeated.sample.jpeg_frames
    ]
    with Image.open(BytesIO(first.sample.jpeg_frames[0])) as image:
        assert image.mode == "RGB"
        assert image.size == (32, 16)
    with Image.open(BytesIO(resized.sample.jpeg_frames[0])) as image:
        assert image.mode == "RGB"
        assert image.size == (640, 320)


def test_preflight_limits_and_unproven_alignment_are_manual_only(
    aligned_50hz_episode: tuple[Path, Path], tmp_path: Path
) -> None:
    parquet_path, video_path = aligned_50hz_episode

    too_many = prepare_episode_samples(
        parquet_path=parquet_path,
        video_path=video_path,
        source_fps=50.0,
        limits=SamplingLimits(max_sampled_frames=4),
    )
    too_large = prepare_episode_samples(
        parquet_path=parquet_path,
        video_path=video_path,
        source_fps=50.0,
        limits=SamplingLimits(max_payload_bytes=1),
    )
    long_parquet = tmp_path / "long.parquet"
    _write_parquet(long_parquet, fps=50.0, frame_count=6_001)
    too_long = prepare_episode_samples(parquet_path=long_parquet, video_path=video_path, source_fps=50.0)
    bad_parquet = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table({"frame_index": [0, 2], "timestamp": [0.0, 0.02]}),
        bad_parquet,
    )
    unproven = prepare_episode_samples(parquet_path=bad_parquet, video_path=video_path, source_fps=50.0)

    assert (too_long.status, too_long.reason) == ("manual_only", "duration_limit")
    assert (too_many.status, too_many.reason) == ("manual_only", "sample_count_limit")
    assert (too_large.status, too_large.reason) == ("manual_only", "payload_size_limit")
    assert (unproven.status, unproven.reason) == ("manual_only", "alignment_unproven")
    assert all(not outcome.reject_episode for outcome in (too_long, too_many, too_large, unproven))
