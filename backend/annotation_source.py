"""Materialize retained v2.1 episodes without opening excluded damaged media/shards.

This helper is shared by raw-source generation and frozen publication. It never
modifies the source and uses one explicit original-to-output episode mapping.
"""

from fractions import Fraction
import json
from pathlib import Path
import shutil

import av
from lerobot.annotations.steerable_pipeline.reader import EpisodeRecord
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _source_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe source dataset file: {relative}")
    return path


def _trim_video(source, target, frame_indices, fps):
    """Decode the complete source stream and encode exactly the requested frames."""
    wanted = set(frame_indices)
    with av.open(str(source)) as inp:
        stream = inp.streams.video[0]
        width, height = stream.width, stream.height
        target.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(target), mode="w") as out:
            out_stream = out.add_stream("libx264", rate=fps)
            out_stream.width, out_stream.height, out_stream.pix_fmt = width, height, "yuv420p"
            frame_time_base = Fraction(1, int(round(fps)))
            out_stream.time_base = frame_time_base
            out_stream.codec_context.max_b_frames = 0
            output_index = 0
            for input_index, frame in enumerate(inp.decode(stream)):
                if input_index not in wanted:
                    continue
                packet_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                packet_frame.pts = output_index
                # MP4 muxing may change the stream time base after the first packet.
                # Input frame indices must keep using the original frame clock.
                packet_frame.time_base = frame_time_base
                for packet in out_stream.encode(packet_frame):
                    out.mux(packet)
                output_index += 1
            if output_index != len(frame_indices):
                raise ValueError(f"Video has fewer frames than requested: {source}")
            for packet in out_stream.encode():
                out.mux(packet)


def align_source_v21(source, output, mapping, *, kept_frames=None):
    """Apply the official export's one identity map to original v2.1 frames and media."""
    info = _json(source / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Pinned source format does not match v2.1")
    episodes = {row["episode_index"]: row for row in _jsonl(source / "meta/episodes.jsonl")}
    episode_stats = {row["episode_index"]: row["stats"] for row in _jsonl(source / "meta/episodes_stats.jsonl")}
    if not {int(old) for old in mapping} <= episodes.keys() & episode_stats.keys():
        raise ValueError("Retained source episodes or their per-episode stats are missing")
    chunks = info["chunks_size"]
    cameras = [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]
    if kept_frames is not None:
        for old, indices in kept_frames.items():
            old = int(old)
            values = list(indices)
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
                raise ValueError(f"kept_frames[{old}] must contain integer frame indices")
            if not values or values != sorted(set(values)) or values[0] < 0:
                raise ValueError(f"kept_frames[{old}] must be nonempty, unique, and ascending")
            if old not in episodes or values[-1] >= episodes[old]["length"]:
                raise ValueError(f"kept_frames[{old}] contains an out-of-range frame")
        unknown = {int(old) for old in kept_frames} - set(episodes)
        unknown |= set(int(old) for old in kept_frames) - {int(old) for old in mapping}
        if unknown:
            raise ValueError(f"kept_frames contains unknown episodes: {sorted(unknown)}")
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    retained_episodes, retained_stats = [], []
    offset = 0
    for old, new in sorted(mapping.items(), key=lambda item: item[1]):
        old = int(old)
        original_relative = info["data_path"].format(episode_chunk=old // chunks, episode_index=old)
        new_relative = info["data_path"].format(episode_chunk=new // chunks, episode_index=new)
        table = pq.read_table(_source_path(source, original_relative))
        count = table.num_rows
        if count != episodes[old]["length"] or table["frame_index"].to_pylist() != list(range(count)):
            raise ValueError(f"Invalid source frame counts or order for episode {old}")
        if table["episode_index"].to_pylist() != [old] * count:
            raise ValueError(f"Source episode identity mismatch for {old}")
        selected = list(range(count)) if kept_frames is None else list(kept_frames.get(old, range(count)))
        if kept_frames is not None:
            table = table.take(pa.array(selected, type=pa.int64()))
            fps = float(info.get("fps", info.get("video_fps", 30)))
            for column, values in (
                ("frame_index", list(range(len(selected)))),
                ("timestamp", [i / fps for i in range(len(selected))]),
                ("episode_index", [new] * len(selected)),
                ("index", list(range(offset, offset + len(selected)))),
            ):
                index = table.schema.get_field_index(column)
                if index >= 0:
                    field = table.schema.field(index)
                    cast = pa.array(values, type=field.type)
                    table = table.set_column(index, field, cast)
            count = table.num_rows
        stats = json.loads(json.dumps(episode_stats[old]))
        if kept_frames is not None:
            # Recompute statistics from the retained rows; stale source statistics
            # would describe frames that are absent from the materialized episode.
            for column, feature in info.get("features", {}).items():
                if column not in table.column_names or feature.get("dtype") in {
                    "string",
                    "language",
                    "video",
                    "image",
                }:
                    continue
                values = np.asarray(table[column].to_pylist())
                if values.dtype.kind not in "biufc":
                    continue
                stats[column] = {
                    k: v.tolist() for k, v in get_feature_stats(values, axis=0, keepdims=values.ndim == 1).items()
                }
        for column, values in (
            (("episode_index", [new] * count), ("index", list(range(offset, offset + count))))
            if kept_frames is None
            else ()
        ):
            index = table.schema.get_field_index(column)
            if index < 0:
                continue
            table = table.set_column(
                index, table.schema.field(index), pa.array(values, type=table.schema.field(index).type)
            )
            array = np.asarray(values, dtype=float)
            stats[column] = {
                "min": [float(array.min())],
                "max": [float(array.max())],
                "mean": [float(array.mean())],
                "std": [float(array.std())],
                "count": [count],
            }
        target = output / new_relative
        if not target.resolve().is_relative_to(output):
            raise ValueError("Source data template escapes the export")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        for camera in cameras:
            original_video = info["video_path"].format(
                episode_chunk=old // chunks, episode_index=old, video_key=camera
            )
            new_video = info["video_path"].format(episode_chunk=new // chunks, episode_index=new, video_key=camera)
            target = output / new_video
            if not target.resolve().is_relative_to(output):
                raise ValueError("Source video template escapes the export")
            target.parent.mkdir(parents=True, exist_ok=True)
            if kept_frames is None or old not in kept_frames:
                shutil.copy2(_source_path(source, original_video), target)
            else:
                _trim_video(_source_path(source, original_video), target, selected, info.get("fps", 30))
                feature = info["features"][camera]
                feature["info"] = {**feature.get("info", {}), "video.codec": "h264"}
                with av.open(str(target)) as clipped:
                    sample_set = set(sample_indices(count))
                    pixels = np.stack(
                        [
                            frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                            for index, frame in enumerate(clipped.decode(video=0))
                            if index in sample_set
                        ]
                    )
                media_stats = get_feature_stats(pixels, axis=(0, 2, 3), keepdims=True)
                stats[camera] = {
                    key: (
                        np.asarray(value / 255.0 if key != "count" else value)
                        .reshape((1,) if key == "count" else (3, 1, 1))
                        .tolist()
                    )
                    for key, value in media_stats.items()
                }
        retained_episodes.append({**episodes[old], "episode_index": new, "length": count})
        retained_stats.append({"episode_index": new, "stats": stats})
        offset += count
    count = len(mapping)
    info.update(
        total_episodes=count,
        total_frames=offset,
        total_chunks=(count + chunks - 1) // chunks,
        total_videos=count * len(cameras),
        splits={"train": f"0:{count}"},
    )
    _write(output / "meta/info.json", info)
    _write_jsonl(output / "meta/episodes.jsonl", retained_episodes)
    _write_jsonl(output / "meta/episodes_stats.jsonl", retained_stats)
    aggregated = aggregate_stats(
        [
            {
                key: {name: np.asarray(value) for name, value in stats.items()}
                for key, stats in row["stats"].items()
            }
            for row in retained_stats
        ]
    )
    _write(
        output / "meta/stats.json",
        {key: {name: value.tolist() for name, value in stats.items()} for key, stats in aggregated.items()},
    )


def v21_records(root: Path, episode_ids) -> list[EpisodeRecord]:
    """Read only selected per-episode parquet files; excluded files may be corrupt."""
    root = Path(root).resolve()
    info = _json(root / "meta/info.json")
    rows = _jsonl(root / "meta/episodes.jsonl")
    metadata = {row["episode_index"]: row for row in rows}
    if len(metadata) != len(rows) or not set(episode_ids) <= metadata.keys():
        raise ValueError("Missing or duplicate v2 source episode identities")
    records = []
    for ep in sorted(episode_ids):
        row = metadata[ep]
        relative = info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        path = _source_path(root, relative)
        try:
            table = pq.read_table(path, columns=["timestamp", "frame_index", "episode_index"])
            records.append(
                EpisodeRecord(
                    episode_index=ep,
                    episode_task="; ".join(row.get("tasks", [])),
                    frame_timestamps=tuple(table["timestamp"].to_pylist()),
                    frame_indices=tuple(table["frame_index"].to_pylist()),
                    data_path=path,
                    row_offset=0,
                    row_count=table.num_rows,
                )
            )
        except Exception as error:
            raise ValueError(f"Cannot read retained source episode {ep}: {error}") from error
    return records
