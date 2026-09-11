"""Reversible source-frame exclusions, materialized only into a new frozen dataset."""

from bisect import bisect_left
from contextlib import suppress
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil


def normalize_exclusions(intervals, frame_count):
    """Validate and merge half-open source-frame intervals; retain at least one frame."""
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("Frame count must be a positive integer")
    if not isinstance(intervals, list):
        raise ValueError("Exclusions must be a list")
    spans = []
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != {"start_frame", "end_frame"}:
            raise ValueError("Each exclusion requires start_frame and end_frame")
        start, end = interval["start_frame"], interval["end_frame"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= frame_count:
            raise ValueError("Exclusion must be a nonempty in-bounds integer frame interval")
        spans.append((start, end))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1]["end_frame"]:
            merged[-1]["end_frame"] = max(end, merged[-1]["end_frame"])
        else:
            merged.append({"start_frame": start, "end_frame": end})
    if sum(span["end_frame"] - span["start_frame"] for span in merged) == frame_count:
        raise ValueError("Cannot exclude all frames; use the episode deletion decision")
    return merged


def retained_indices(count, intervals):
    spans = normalize_exclusions(intervals, count)
    kept, start = [], 0
    for span in spans:
        kept.extend(range(start, span["start_frame"]))
        start = span["end_frame"]
    kept.extend(range(start, count))
    return kept


def _persistent_key(atom):
    tools = tuple(sorted(call.get("function", {}).get("name", "") for call in atom.get("tool_calls") or []))
    return atom.get("style"), atom.get("role"), atom.get("camera"), tools


def remap_atoms(atoms, source_timestamps, kept_indices, fps):
    """Preserve active language state on retained frames and discard excluded events.

    Atoms emitted between frame samples become effective on the next source frame.
    Removed changes still establish the active state at the next retained segment.
    """
    from lerobot.datasets.language import LANGUAGE_PERSISTENT, column_for_style

    times = list(source_timestamps)
    if (
        type(fps) not in (int, float)
        or not math.isfinite(fps)
        or fps <= 0
        or not times
        or any(not math.isfinite(t) for t in times)
        or any(a >= b for a, b in zip(times, times[1:]))
    ):
        raise ValueError("Source timestamps and FPS must be finite and ordered")
    if not kept_indices or any(type(i) is not int or not 0 <= i < len(times) for i in kept_indices):
        raise ValueError("Retained indices must be nonempty source frame indices")
    if any(a >= b for a, b in zip(kept_indices, kept_indices[1:])):
        raise ValueError("Retained indices must be strictly increasing")
    output_indices = {old: new for new, old in enumerate(kept_indices)}
    result, persistent = [], []
    for atom in atoms:
        timestamp = atom.get("timestamp")
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or not times[0] <= timestamp <= times[-1]
        ):
            raise ValueError("Annotation timestamp is outside source frame bounds")
        if column_for_style(atom.get("style")) == LANGUAGE_PERSISTENT:
            if atom.get("style") == "task_aug":
                result.append({**deepcopy(atom), "timestamp": 0.0})
            else:
                persistent.append(atom)
        else:
            index = bisect_left(times, timestamp)
            if index == len(times) or times[index] != timestamp:
                raise ValueError("Event timestamp must match an exact source frame")
            if index in output_indices:
                result.append({**deepcopy(atom), "timestamp": output_indices[index] / fps})
    ordered = sorted(enumerate(persistent), key=lambda item: item[1]["timestamp"])
    cursor, active, emitted = 0, {}, {}
    for new, old in enumerate(kept_indices):
        while cursor < len(ordered) and ordered[cursor][1]["timestamp"] <= times[old]:
            identity, atom = ordered[cursor]
            key = _persistent_key(atom)
            # Equal-time rows are retained so upstream validation/resolution can
            # report ambiguity instead of silently choosing one annotation.
            previous = active.get(key, [])
            active[key] = (
                previous + [(identity, atom)]
                if previous and previous[0][1]["timestamp"] == atom["timestamp"]
                else [(identity, atom)]
            )
            cursor += 1
        boundary_times = {}
        for key, rows in active.items():
            identities = tuple(identity for identity, _ in rows)
            if emitted.get(key) != identities:
                style, source_time = key[0], rows[0][1]["timestamp"]
                if style in boundary_times and boundary_times[style] != source_time:
                    # Without a role/tool/camera selector, active_at chooses the
                    # latest emission of this style. Collapsing distinct source
                    # times into a tie would make previously valid rows fail.
                    raise ValueError(
                        f"Clipping would create ambiguous persistent {style} state at output frame {new}; "
                        "different source emission times collapse to the same boundary"
                    )
                boundary_times[style] = source_time
                result.extend({**deepcopy(atom), "timestamp": new / fps} for _, atom in rows)
                emitted[key] = identities
    return sorted(result, key=lambda atom: atom["timestamp"])


def _segments(kept):
    result = []
    for new, old in enumerate(kept):
        if result and old == result[-1]["source_end_frame"]:
            result[-1]["source_end_frame"] = old + 1
        else:
            result.append({"source_start_frame": old, "source_end_frame": old + 1, "output_start_frame": new})
    return result


def clip_v3_dataset(source, output, exclusions, annotations, retained_episode_ids):
    """Rebuild base v3 rows/media with the official writer; return remapped labels.

    Video decoding is strict and bounded to 32 frames per camera. The upstream
    writer buffers numerical rows for one episode, while encoding video online.
    """
    from lerobot.annotations.steerable_pipeline.reader import iter_episodes
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames
    from lerobot.utils.constants import DEFAULT_FEATURES
    import numpy as np
    import pyarrow.parquet as pq

    try:
        from .annotation_runs import source_inventory
    except ImportError:
        from annotation_runs import source_inventory

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Clip output must be a new directory outside the source")
    before = source_inventory(source)
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") not in {"v3.0", "v3.1"}:
        raise ValueError("Clip writer requires a v3 source")
    metadata = LeRobotDatasetMetadata(repo_id="local/source", root=source)
    records = list(iter_episodes(source))
    by_episode = {record.episode_index: record for record in records}
    if len(by_episode) != len(records):
        raise ValueError("Each source episode must occupy one contiguous parquet segment")
    ids = retained_episode_ids
    if not ids or any(type(ep) is not int or ep not in by_episode for ep in ids) or len(set(ids)) != len(ids):
        raise ValueError("Select distinct existing retained episode indices")
    if set(exclusions) - set(by_episode):
        raise ValueError("Exclusions reference an unknown episode")
    ids = sorted(ids)
    mapping = {old: new for new, old in enumerate(ids)}
    features = {
        key: deepcopy(value)
        for key, value in info["features"].items()
        if key not in DEFAULT_FEATURES and key not in {"language_persistent", "language_events"}
    }
    if metadata.depth_keys or any(value["dtype"] in {"audio", "image", "language"} for value in features.values()):
        raise ValueError(
            "Clipping supports numeric/string features and RGB video; audio/depth/image features are unsupported"
        )
    video_keys = list(metadata.video_keys)
    for key in video_keys:
        if list(features[key]["shape"])[-1] != 3:
            raise ValueError("Clipping requires three-channel RGB video")
        # Source encoder details no longer describe newly encoded media.
        features[key].pop("info", None)
    task_names = {int(row.task_index): str(task) for task, row in metadata.tasks.iterrows()}
    frame_maps, labels, selections = {}, {}, {}
    for ep in ids:
        record = by_episode[ep]
        kept = retained_indices(record.row_count, exclusions.get(ep, []))
        selections[ep] = kept
        labels[mapping[ep]] = remap_atoms(annotations.get(ep, []), record.frame_timestamps, kept, info["fps"])
        frame_maps[ep] = {
            "output_episode_index": mapping[ep],
            "source_frame_count": record.row_count,
            "output_frame_count": len(kept),
            "retained_segments": _segments(kept),
        }
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id="local/clipped",
            root=output,
            fps=info["fps"],
            features=features,
            robot_type=info.get("robot_type"),
            streaming_encoding=True,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
            video_backend="pyav",
        )
        for ep in ids:
            record, kept = by_episode[ep], selections[ep]
            table = pq.read_table(record.data_path).slice(record.row_offset, record.row_count)
            if (
                table["frame_index"].to_pylist() != list(range(record.row_count))
                or table["episode_index"].to_pylist() != [ep] * record.row_count
            ):
                raise ValueError("Source frame/episode identities are invalid")
            numeric_keys = [key for key in features if key not in video_keys]
            episode_meta = metadata.episodes[ep]
            for start in range(0, len(kept), 32):
                indices = kept[start : start + 32]
                rows = table.take(indices).to_pylist()
                cameras = {}
                for key in video_keys:
                    path = (source / metadata.get_video_file_path(ep, key)).resolve()
                    if not path.is_relative_to(source) or not path.is_file():
                        raise ValueError("Missing or unsafe source video path")
                    offset = float(episode_meta[f"videos/{key}/from_timestamp"])
                    requested = [float(record.frame_timestamps[index]) + offset for index in indices]
                    frames = decode_video_frames(
                        path,
                        requested,
                        tolerance_s=min(1e-4, 0.1 / info["fps"]),
                        backend="pyav",
                        return_uint8=True,
                    )
                    if len(frames) != len(indices):
                        raise ValueError("Decoder returned an incorrect retained frame count")
                    cameras[key] = frames.permute(0, 2, 3, 1).numpy()
                for position, row in enumerate(rows):
                    task = task_names.get(row["task_index"])
                    if not task:
                        raise ValueError("Source frame has no canonical task label")
                    frame = {"task": task}
                    for key in numeric_keys:
                        feature = features[key]
                        frame[key] = (
                            row[key]
                            if feature["dtype"] == "string"
                            else np.asarray(row[key], dtype=feature["dtype"]).reshape(feature["shape"])
                        )
                    frame.update({key: values[position] for key, values in cameras.items()})
                    dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()
        if "tools" in info:
            dataset.meta.tools = deepcopy(info["tools"])
        # The recording schema stores float32 timestamps. Bind atoms to those
        # exact stored values, as the official event validator uses equality.
        for record in iter_episodes(output):
            for atom in labels[record.episode_index]:
                atom["timestamp"] = record.frame_timestamps[round(atom["timestamp"] * info["fps"])]
        if (source / "meta/modality.json").exists():
            shutil.copy2(source / "meta/modality.json", output / "meta/modality.json")
        if source_inventory(source) != before:
            raise ValueError("Source changed during clipping")
        return {
            "output_dir": str(output),
            "old_to_new": mapping,
            "annotations": labels,
            "frame_mapping": frame_maps,
        }
    except BaseException:
        if dataset is not None:
            with suppress(Exception):
                dataset.writer.cancel_pending_videos()
                dataset.finalize()
        if output.exists():
            shutil.rmtree(output)
        raise
