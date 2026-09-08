"""Materialize retained v2.1 episodes without opening excluded damaged media/shards.

This helper is shared by raw-source generation and frozen publication. It never
modifies the source and uses one explicit original-to-output episode mapping.
"""

import json
from pathlib import Path
import shutil

from lerobot.annotations.steerable_pipeline.reader import EpisodeRecord
from lerobot.datasets.compute_stats import aggregate_stats
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


def align_source_v21(source, output, mapping):
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
        stats = json.loads(json.dumps(episode_stats[old]))
        for column, values in (("episode_index", [new] * count), ("index", list(range(offset, offset + count)))):
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
            shutil.copy2(_source_path(source, original_video), target)
        retained_episodes.append({**episodes[old], "episode_index": new})
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
