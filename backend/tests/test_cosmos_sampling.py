from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
import shutil

import av
from curation import cosmos_transport
from curation.cosmos_transport import SamplingLimits, prepare_episode_samples, prove_alignment_and_select
from curation.source import SourceRecord, SourceRegistry
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_video(path: Path, *, fps: int, frame_count: int, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_parquet(path: Path, *, fps: float, frame_count: int, frame_indices: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array(
                    range(frame_count) if frame_indices is None else frame_indices,
                    type=pa.int64(),
                ),
                "timestamp": pa.array(
                    np.arange(frame_count, dtype=np.float64) / np.float64(fps),
                    type=pa.float64(),
                ),
            }
        ),
        path,
    )


def _registered_episode(
    tmp_path: Path,
    *,
    name: str,
    fps: int,
    frame_count: int,
    width: int = 64,
    height: int = 48,
    frame_indices: list[int] | None = None,
) -> tuple[SourceRecord, str, str, Path, Path]:
    source = tmp_path / name
    parquet_relative = "data/episode.parquet"
    video_relative = "videos/episode.mkv"
    parquet_path = source / parquet_relative
    video_path = source / video_relative
    _write_parquet(parquet_path, fps=float(fps), frame_count=frame_count, frame_indices=frame_indices)
    _write_video(video_path, fps=fps, frame_count=frame_count, width=width, height=height)
    registry = SourceRegistry.from_paths({"org/data": source}, workspace=tmp_path / f"{name}-workspace")
    return registry.records["org/data"], parquet_relative, video_relative, parquet_path, video_path


@pytest.fixture
def aligned_50hz_episode(tmp_path: Path) -> tuple[SourceRecord, str, str, Path, Path]:
    return _registered_episode(tmp_path, name="aligned", fps=50, frame_count=101)


def _prepare(
    episode: tuple[SourceRecord, str, str, Path, Path],
    *,
    source_fps: float,
    limits: SamplingLimits | None = None,
):
    source_record, parquet_relative, video_relative, _, _ = episode
    return prepare_episode_samples(
        source_record=source_record,
        parquet_asset_path=parquet_relative,
        video_asset_path=video_relative,
        source_fps=source_fps,
        limits=limits,
    )


def test_synthetic_50hz_registered_parquet_and_video_sample_exact_pinned_frames(
    aligned_50hz_episode: tuple[SourceRecord, str, str, Path, Path],
) -> None:
    result = _prepare(aligned_50hz_episode, source_fps=50.0)

    assert result.status == "ready"
    assert result.reason is None
    assert result.sample is not None
    assert result.sample.source_fps == 50.0
    assert result.sample.total_num_frames == 101
    assert result.sample.duration_s == pytest.approx(101 / 50)
    assert result.sample.frame_indices == (0, 25, 50, 75, 100)
    assert result.sample.parquet_timestamps == pytest.approx((0.0, 0.5, 1.0, 1.5, 2.0))
    assert len(result.sample.jpeg_frames) == 5
    assert result.sample.source_video_sha256 == aligned_50hz_episode[0].file_hashes["videos/episode.mkv"]


@pytest.mark.parametrize(
    ("frame_indices", "timestamps", "source_fps", "video_rate", "video_count", "message"),
    [
        ([0, 2], [0.0, 0.02], 50.0, 50.0, 2, "frame_index"),
        ([0, 1], [0.0, 0.04], 50.0, 50.0, 2, "timestamp"),
        ([0, 1], [0.0, 0.02], 50.0, 49.999, 2, "rate"),
        ([0, 1], [0.0, 0.02], 50.0, 50.0, 3, "count"),
        ([0, 1], [0.0, 0.02], 50.0, 50.0, 2.0, "count"),
        ([0, 1], [0.0, 0.02], 50.0, 50.0, 2.5, "count"),
        ([0, 1], [0.0, 1.0], 1.0, True, 2, "rate"),
        ([0, 1], [0.0, 1.0], True, 1.0, 2, "FPS"),
        ([0, 1], [0.0, 0.02], "50", 50.0, 2, "FPS"),
        ([0, 1], [0.0, 0.02], 50.0, "50", 2, "rate"),
    ],
)
def test_alignment_proof_rejects_each_unproven_invariant(
    frame_indices: list[int],
    timestamps: list[float],
    source_fps: float,
    video_rate: float,
    video_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prove_alignment_and_select(
            frame_indices=frame_indices,
            timestamps=timestamps,
            source_fps=source_fps,
            video_rate=video_rate,
            video_frame_count=video_count,
        )


def test_float64_targets_break_nearest_ties_lower_deduplicate_and_obey_exact_boundary() -> None:
    proof = prove_alignment_and_select(
        frame_indices=[0, 1],
        timestamps=np.asarray([0.0, 1.0], dtype=np.float64),
        source_fps=1.0,
        video_rate=1.0,
        video_frame_count=2,
    )
    below_boundary = prove_alignment_and_select(
        frame_indices=[0, 1, 2],
        timestamps=np.asarray([0.0, 0.5, np.nextafter(1.0, 0.0)], dtype=np.float64),
        source_fps=2.0,
        video_rate=2.0,
        video_frame_count=3,
    )

    assert proof.frame_indices == (0, 1)
    assert proof.parquet_timestamps == (0.0, 1.0)
    assert below_boundary.frame_indices == (0, 1)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_duration_seconds": True}, "finite and positive"),
        ({"max_duration_seconds": 0}, "finite and positive"),
        ({"max_duration_seconds": -1.0}, "finite and positive"),
        ({"max_duration_seconds": float("nan")}, "finite and positive"),
        ({"max_duration_seconds": float("inf")}, "finite and positive"),
        ({"max_duration_seconds": np.float64(120)}, "built-in"),
        ({"max_duration_seconds": 10**10_000}, "finite and positive"),
        ({"max_sampled_frames": True}, "built-in positive integer"),
        ({"max_sampled_frames": 240.0}, "built-in positive integer"),
        ({"max_sampled_frames": np.int64(240)}, "built-in positive integer"),
        ({"max_payload_bytes": True}, "built-in positive integer"),
        ({"max_payload_bytes": 1.5}, "built-in positive integer"),
        ({"max_payload_bytes": np.int64(100)}, "built-in positive integer"),
    ],
)
def test_sampling_limits_require_canonical_built_in_finite_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingLimits(**changes)


def test_exact_rgb_jpegs_are_deterministic_never_upscale_and_resize_long_edge(tmp_path: Path) -> None:
    small = _registered_episode(tmp_path, name="small", fps=2, frame_count=2, width=32, height=16)
    large = _registered_episode(tmp_path, name="large", fps=2, frame_count=2, width=800, height=400)

    first = _prepare(small, source_fps=2.0)
    repeated = _prepare(small, source_fps=2.0)
    resized = _prepare(large, source_fps=2.0)

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


def test_limits_and_unproven_registered_alignment_are_manual_only(
    aligned_50hz_episode: tuple[SourceRecord, str, str, Path, Path], tmp_path: Path
) -> None:
    too_many = _prepare(
        aligned_50hz_episode,
        source_fps=50.0,
        limits=SamplingLimits(max_sampled_frames=4),
    )
    too_large = _prepare(
        aligned_50hz_episode,
        source_fps=50.0,
        limits=SamplingLimits(max_payload_bytes=1),
    )
    long_root = tmp_path / "long"
    _write_parquet(long_root / "data/episode.parquet", fps=50.0, frame_count=6_001)
    _write_video(long_root / "videos/episode.mkv", fps=50, frame_count=2)
    long_record = SourceRegistry.from_paths(
        {"org/data": long_root}, workspace=tmp_path / "long-workspace"
    ).records["org/data"]
    long_episode = (
        long_record,
        "data/episode.parquet",
        "videos/episode.mkv",
        long_root / "data/episode.parquet",
        long_root / "videos/episode.mkv",
    )
    too_long = _prepare(long_episode, source_fps=50.0)
    bad_episode = _registered_episode(
        tmp_path,
        name="bad",
        fps=50,
        frame_count=2,
        frame_indices=[0, 2],
    )
    unproven = _prepare(bad_episode, source_fps=50.0)

    assert (too_long.status, too_long.reason) == ("manual_only", "duration_limit")
    assert (too_many.status, too_many.reason) == ("manual_only", "sample_count_limit")
    assert (too_large.status, too_large.reason) == ("manual_only", "payload_size_limit")
    assert (unproven.status, unproven.reason) == ("manual_only", "alignment_unproven")
    assert all(not outcome.reject_episode for outcome in (too_long, too_many, too_large, unproven))


def test_pathname_swap_after_pinned_decode_is_manual_only_without_mixed_sample(
    aligned_50hz_episode: tuple[SourceRecord, str, str, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, video_path = aligned_50hz_episode
    replacement = tmp_path / "replacement.mkv"
    displaced = tmp_path / "displaced.mkv"
    _write_video(replacement, fps=50, frame_count=101)
    original_decode = cosmos_transport._decode_selected_frames

    def decode_then_swap(*args, **kwargs):
        decoded = original_decode(*args, **kwargs)
        os.replace(video_path, displaced)
        shutil.copyfile(replacement, video_path)
        return decoded

    monkeypatch.setattr(cosmos_transport, "_decode_selected_frames", decode_then_swap)
    result = _prepare(aligned_50hz_episode, source_fps=50.0)

    assert (result.status, result.reason, result.sample) == ("manual_only", "source_changed", None)


def test_in_place_mutation_after_pinned_decode_is_manual_only_without_mixed_jpeg_hash(
    aligned_50hz_episode: tuple[SourceRecord, str, str, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, video_path = aligned_50hz_episode
    original_decode = cosmos_transport._decode_selected_frames

    def decode_then_mutate(*args, **kwargs):
        decoded = original_decode(*args, **kwargs)
        with video_path.open("r+b", buffering=0) as handle:
            handle.seek(-1, os.SEEK_END)
            value = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([value[0] ^ 0x01]))
            os.fsync(handle.fileno())
        return decoded

    monkeypatch.setattr(cosmos_transport, "_decode_selected_frames", decode_then_mutate)
    result = _prepare(aligned_50hz_episode, source_fps=50.0)

    assert (result.status, result.reason, result.sample) == ("manual_only", "source_changed", None)
