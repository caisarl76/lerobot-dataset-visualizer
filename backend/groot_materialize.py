"""Materialize a frozen rich v3 dataset as independent GR00T LeRobot v2.1."""

from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .groot_export import _atoms, _instruction, _json, _prepare_atoms, _within
except ImportError:
    from groot_export import _atoms, _instruction, _json, _prepare_atoms, _within


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def _lists(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


def _episode_video(source, target, timestamps, offset, end, fps, shape):
    """Match source timestamps strictly and encode online, retaining one RGB frame."""
    tolerance = min(1e-4, 0.1 / fps)
    expected = np.asarray(timestamps, dtype=float) + offset
    if not math.isfinite(offset) or not math.isfinite(end) or offset < 0 or end <= expected[-1]:
        raise ValueError("Invalid source video episode offsets")
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    target.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as inp:
        if len(inp.streams.video) != 1 or inp.streams.audio:
            raise ValueError("Only single-stream RGB video is supported; audio is unsupported")
        stream = inp.streams.video[0]
        if stream.average_rate is None or not math.isclose(float(stream.average_rate), fps, rel_tol=1e-4):
            raise ValueError("Source video FPS mismatch")
        # FFmpeg must report corrupted packets instead of concealing damage.
        stream.codec_context.options = {"err_detect": "explode"}
        if offset:
            # Shared v3 videos can contain many episodes. Start at the preceding
            # keyframe, then require an exact match for every requested frame.
            inp.seek(int(expected[0] / stream.time_base), stream=stream, backward=True)
        with av.open(str(target), "w") as out:
            encoder = out.add_stream("libx264", rate=rate)
            encoder.width, encoder.height = shape[1], shape[0]
            encoder.pix_fmt = "yuv420p" if shape[0] % 2 == shape[1] % 2 == 0 else "yuv444p"
            encoder.time_base = 1 / rate
            encoder.codec_context.max_b_frames = 0
            encoder.options = {"crf": "18"}
            index, previous = 0, None
            for frame in inp.decode(stream):
                timestamp = frame.time
                if (
                    timestamp is None
                    or not math.isfinite(timestamp)
                    or (previous is not None and timestamp <= previous)
                ):
                    raise ValueError("Invalid decoded video timestamps")
                previous = timestamp
                if [frame.height, frame.width, 3] != list(shape) or frame.is_corrupt:
                    raise ValueError("Corrupt video frame or RGB dimensions mismatch")
                if timestamp < expected[index] - tolerance:
                    continue
                if abs(timestamp - expected[index]) > tolerance:
                    raise ValueError(f"Missing video frame at {expected[index]} in {source}")
                fresh = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                fresh.pts, fresh.time_base = index, encoder.time_base
                for packet in encoder.encode(fresh):
                    out.mux(packet)
                index += 1
                if index == len(expected):
                    break
            if index != len(expected):
                raise ValueError(f"Video has fewer frames than requested: {source}")
            for packet in encoder.encode():
                out.mux(packet)


def materialize_groot_v21(annotated_root: Path, output_root: Path, instruction_mode="task") -> dict:
    """Write v2.1 robot rows, episode videos, instructions, and training metadata.

    Saved sidecar atoms override parquet atoms, including explicit empty edits.
    A single task_aug supplies the task prompt; multiple variants are rejected in
    task mode. Subtask mode uses active subtask text and falls back to the single
    task variant (or the canonical frame task when variants are ambiguous).
    """
    from lerobot.annotations.steerable_pipeline.reader import iter_episodes
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import get_video_info

    if instruction_mode not in {"task", "subtask"}:
        raise ValueError("instruction_mode must be task or subtask")
    annotated_root, output_root = (Path(p).expanduser().resolve() for p in (annotated_root, output_root))
    if output_root.is_relative_to(annotated_root) or annotated_root.is_relative_to(output_root):
        raise ValueError("Output must be separate from the annotated source")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    info = _json(annotated_root / "meta/info.json")
    if info.get("codebase_version") not in {"v3.0", "v3.1"}:
        raise ValueError("annotated_root must be a v3 dataset")
    fps = info.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid dataset FPS")
    features = {
        key: deepcopy(value)
        for key, value in info["features"].items()
        if key not in {"language_persistent", "language_events"}
    }
    cameras = []
    for key, feature in features.items():
        dtype = feature.get("dtype")
        if (
            dtype in {"audio", "image", "language"}
            or feature.get("info", {}).get("is_depth_map")
            or feature.get("is_depth_map")
        ):
            raise ValueError(f"Feature {key}: audio/depth/embedded image/language modalities are unsupported")
        if dtype == "video":
            if len(feature["shape"]) != 3 or feature["shape"][-1] != 3:
                raise ValueError(f"Feature {key} requires three-channel RGB video")
            cameras.append(key)
    modality = _json(annotated_root / "meta/modality.json")
    modality.setdefault("annotation", {}).setdefault("human.task_description", {})["original_key"] = "task_index"
    metadata = LeRobotDatasetMetadata(repo_id="local/annotated", root=annotated_root)
    records = sorted(iter_episodes(annotated_root), key=lambda r: r.episode_index)
    ids = [record.episode_index for record in records]
    # The metadata reader deliberately drops stats columns from ``episodes``.
    # Read its underlying parquet metadata to retain the flattened statistics.
    rows = [
        row
        for path in sorted((annotated_root / "meta/episodes").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]
    episode_rows = {row["episode_index"]: row for row in rows}
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(episode_rows) or len(episode_rows) != len(rows):
        raise ValueError("Missing, duplicate or fragmented source episode identities")
    original_tasks = {int(row.task_index): str(task) for task, row in metadata.tasks.iterrows()}
    sidecar_path = annotated_root / "meta/lerobot_annotations.json"
    sidecar = _json(sidecar_path).get("episodes", {}) if sidecar_path.exists() else {}
    if not set(sidecar) <= {str(ep) for ep in ids}:
        raise ValueError("Sidecar contains unknown episodes")
    if sum(r.row_count for r in records) != info["total_frames"] or len(records) != info["total_episodes"]:
        raise ValueError("Source metadata frame/episode counts do not match")
    data_path = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_path = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    chunks = 1000
    tasks, episodes, episode_stats = {}, [], []
    offset = 0
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        for new, record in enumerate(records):
            ep = record.episode_index
            source_path = _within(annotated_root, str(record.data_path.relative_to(annotated_root)))
            table = pq.read_table(source_path).slice(record.row_offset, record.row_count)
            count = table.num_rows
            times = table["timestamp"].to_pylist()
            episode_meta = episode_rows[ep]
            if (
                count <= 0
                or count != episode_meta["length"]
                or table["episode_index"].to_pylist() != [ep] * count
                or table["frame_index"].to_pylist() != list(range(count))
                or not np.isfinite(times).all()
                or not np.allclose(times, np.arange(count) / fps, atol=1e-4, rtol=1e-5)
            ):
                raise ValueError(f"Invalid episode/frame/timestamp identities in episode {ep}")
            atoms = sidecar[str(ep)]["atoms"] if str(ep) in sidecar else _atoms(table.to_pylist())
            atoms = _prepare_atoms(atoms)
            variants = sum(atom["style"] == "task_aug" for atom in atoms)
            if instruction_mode == "task" and variants > 1:
                raise ValueError(f"Ambiguous task_aug variants in episode {ep}")
            indices, episode_tasks = [], []
            for task_index, timestamp in zip(table["task_index"].to_pylist(), times, strict=True):
                task = original_tasks.get(task_index)
                if not task or not task.strip():
                    raise ValueError(f"Missing canonical frame task in episode {ep}")
                text = _instruction(task, timestamp, atoms, instruction_mode, 0 if variants == 1 else None)
                indices.append(tasks.setdefault(text, len(tasks)))
                if text not in episode_tasks:
                    episode_tasks.append(text)
            table = table.drop(
                [key for key in ("language_persistent", "language_events") if key in table.column_names]
            )
            for key, values in (
                ("episode_index", [new] * count),
                ("frame_index", range(count)),
                ("index", range(offset, offset + count)),
                ("task_index", indices),
            ):
                index = table.schema.get_field_index(key)
                field = table.schema.field(index)
                table = table.set_column(index, field, pa.array(values, type=field.type))
            stats = {}
            for key, feature in features.items():
                if key in cameras or key == "task_index" or feature["dtype"] == "string":
                    continue
                values = np.asarray(table[key].to_pylist())
                shape = tuple(feature["shape"])
                if (
                    values.dtype.kind not in "biuf"
                    or not np.isfinite(values).all()
                    or (values.shape != (count, *shape) and not (shape == (1,) and values.shape == (count,)))
                ):
                    raise ValueError(f"Invalid numeric robot feature {key} in episode {ep}")
                stats[key] = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)
            target = output_root / data_path.format(episode_chunk=new // chunks, episode_index=new)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            for key in cameras:
                source_video = _within(annotated_root, str(metadata.get_video_file_path(ep, key)))
                target = _within(
                    output_root, video_path.format(episode_chunk=new // chunks, episode_index=new, video_key=key)
                )
                _episode_video(
                    source_video,
                    target,
                    times,
                    float(episode_meta[f"videos/{key}/from_timestamp"]),
                    float(episode_meta[f"videos/{key}/to_timestamp"]),
                    fps,
                    features[key]["shape"],
                )
                features[key]["info"] = get_video_info(target)
                # Frozen v3 metadata stores already-retained RGB statistics in
                # flattened columns; never reuse the whole source dataset stats.
                prefix = f"stats/{key}/"
                stats[key] = {
                    name[len(prefix) :]: np.asarray(value)
                    for name, value in episode_meta.items()
                    if name.startswith(prefix) and value is not None
                }
                if not {"min", "max", "mean", "std", "count"} <= stats[key].keys():
                    raise ValueError(f"Missing per-episode video statistics for {key}")
            episodes.append({"episode_index": new, "tasks": episode_tasks, "length": count})
            episode_stats.append(stats)
            offset += count
        result_info = {
            "codebase_version": "v2.1",
            "robot_type": info.get("robot_type"),
            "fps": fps,
            "features": features,
            "total_episodes": len(records),
            "total_frames": offset,
            "total_tasks": len(tasks),
            "total_videos": len(records) * len(cameras),
            "total_chunks": (len(records) + chunks - 1) // chunks,
            "chunks_size": chunks,
            "splits": {"train": f"0:{len(records)}"},
            "data_path": data_path,
            "video_path": video_path,
        }
        _write(output_root / "meta/info.json", result_info)
        _write(output_root / "meta/modality.json", modality)
        _write(output_root / "meta/stats.json", _lists(aggregate_stats(episode_stats)))
        _jsonl(
            output_root / "meta/tasks.jsonl",
            [{"task_index": index, "task": text} for text, index in tasks.items()],
        )
        _jsonl(output_root / "meta/episodes.jsonl", episodes)
        _jsonl(
            output_root / "meta/episodes_stats.jsonl",
            [{"episode_index": ep, "stats": _lists(stats)} for ep, stats in enumerate(episode_stats)],
        )
        report = {
            "output_dir": str(output_root),
            "annotated_root": str(annotated_root),
            "instruction_mode": instruction_mode,
            "episodes": len(records),
            "frames": offset,
            "tasks": len(tasks),
            "old_to_new": {old: new for new, old in enumerate(ids)},
        }
        _write(output_root / "meta/groot_instruction_export.json", report)
        return report
    except BaseException:
        shutil.rmtree(output_root)
        raise
