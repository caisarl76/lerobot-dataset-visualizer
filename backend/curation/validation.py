"""Read-only dataset validation and immutable publication artifacts.

The functions in this module deliberately operate on explicit paths and
snapshots.  They do not infer approvals from mutable episode rows and they do
not publish anything.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import traceback as traceback_module
from typing import Any
from uuid import UUID

import av
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .db import canonical_json, canonical_json_sha256
from .models import ReviewState
from .prompts import expand_prompts
from .source import SourceRecord


class StructuralValidationError(RuntimeError):
    """The staged dataset does not match its immutable export snapshot."""


class FinalConsistencyError(RuntimeError):
    """The sealed staging tree is not internally consistent."""


class ArtifactInstallConflict(RuntimeError):
    """An installed or temporary artifact disagrees with canonical bytes."""


_REPLACED_COLUMNS = frozenset({"episode_index", "frame_index", "index", "task_index"})
_MAX_REPORT_STREAM_BYTES = 4 * 1024 * 1024
_SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "STAGING_PATH",
        "TORCH_HOME",
    }
)
_SNAPSHOT_FIELDS = (
    "source_episode_index",
    "source_length",
    "review_state",
    "object_name",
    "pickup_hand",
    "turn_direction",
    "step_2_start_frame",
    "step_3_start_frame",
    "step_4_start_frame",
    "step_5_start_frame",
    "step_6_start_frame",
    "step_7_start_frame",
    "revision",
    "approval_revision",
    "reviewer",
    "approved_at",
    "rejection_reason",
    "prompt_template_sha256",
)
_EPISODE_FILENAME = re.compile(r"^episode_(\d{6})(?P<suffix>\..+)$")
_REWRITTEN_METADATA = frozenset(
    {
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/episodes_stats.jsonl",
    }
)
_SOURCE_CACHE_METADATA = frozenset({"meta/stats.json", "meta/relative_stats.json"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain JSON objects")
        rows.append(value)
    return rows


def _format_episode_path(pattern: str, *, episode_index: int, chunk_size: int) -> str:
    return pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        chunk_index=episode_index // chunk_size,
        file_index=episode_index,
    )


def _ipc_hash(column: pa.ChunkedArray, field: pa.Field) -> str:
    table = pa.Table.from_arrays([column.combine_chunks()], schema=pa.schema([field]))
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _hash_file(path: Path) -> str:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, *, directory_fd: int | None = None) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactInstallConflict(f"artifact is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_safe_directory(path: Path, *, create: bool) -> int:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ArtifactInstallConflict("artifact parent path is unsafe")
    descriptor = os.open(
        "/",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for component in path.parts[1:]:
            try:
                following = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                following = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = following
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def install_canonical_file(
    path: Path,
    contents: bytes,
    *,
    mode: int = 0o644,
    after_install: Callable[[], None] = lambda: None,
) -> str:
    """Install immutable bytes and reconcile every crash point exactly."""

    path = Path(path)
    if not isinstance(contents, bytes):
        raise TypeError("canonical artifact contents must be bytes")
    parent_fd = _open_safe_directory(path.parent, create=True)
    temporary_name = f".{path.name}.installing"

    def exact(name: str) -> bool:
        try:
            existing = _read_regular_bytes(Path(name), directory_fd=parent_fd)
        except FileNotFoundError:
            return False
        except ArtifactInstallConflict:
            raise
        except OSError as error:
            raise ArtifactInstallConflict(f"artifact entry is unsafe: {path.parent / name}") from error
        if existing != contents:
            raise ArtifactInstallConflict(f"artifact bytes disagree: {path.parent / name}")
        return True

    try:
        final_exists = exact(path.name)
        temporary_exists = exact(temporary_name)
        installed = False
        if not final_exists and not temporary_exists:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent_fd,
            )
            try:
                view = memoryview(contents)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(parent_fd)
            temporary_exists = True
        if not final_exists:
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if not exact(path.name):
                    raise ArtifactInstallConflict(f"racing artifact bytes disagree: {path}") from None
            os.fsync(parent_fd)
            final_exists = True
            installed = True
            after_install()
        if temporary_exists:
            os.unlink(temporary_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        if not exact(path.name):
            raise ArtifactInstallConflict(f"installed artifact bytes disagree: {path}")
        return "installed" if installed else "existing"
    finally:
        os.close(parent_fd)


def canonical_file_presence(path: Path) -> tuple[bool, bool]:
    """Return canonical final/temp presence without following any parent or leaf symlink."""

    path = Path(path)
    parent_fd = _open_safe_directory(path.parent, create=False)
    try:
        values: list[bool] = []
        for name in (path.name, f".{path.name}.installing"):
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                values.append(False)
            else:
                values.append(True)
        return values[0], values[1]
    finally:
        os.close(parent_fd)


def read_canonical_file(path: Path) -> bytes:
    """Read one canonical regular file through a pinned O_NOFOLLOW parent descriptor."""

    path = Path(path)
    parent_fd = _open_safe_directory(path.parent, create=False)
    try:
        try:
            return _read_regular_bytes(Path(path.name), directory_fd=parent_fd)
        except (OSError, ArtifactInstallConflict) as error:
            raise ArtifactInstallConflict(f"canonical artifact is unsafe: {path}") from error
    finally:
        os.close(parent_fd)


def _regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, directories, names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if stat.S_ISLNK(directory_path.lstat().st_mode):
            raise ValueError(f"symlink directory is prohibited: {directory_path}")
        for name in directories:
            child = directory_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError(f"unsafe directory entry: {child}")
        for name in names:
            child = directory_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError(f"non-regular file is prohibited: {child}")
            relative = child.relative_to(root).as_posix()
            if "\n" in relative or "\r" in relative:
                raise ValueError("newlines are prohibited in artifact paths")
            files.append(child)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))


def _video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise ValueError(f"expected exactly one video stream: {path}")
        return sum(1 for _ in container.decode(streams[0]))


def _json_number(value: Any) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _episode_statistics(table: pa.Table) -> dict[str, dict[str, list[Any]]]:
    import numpy as np

    result: dict[str, dict[str, list[Any]]] = {}
    for field, column in zip(table.schema, table.columns, strict=True):
        value_type = field.type.value_type if pa.types.is_fixed_size_list(field.type) else field.type
        if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
            continue
        rows = column.to_pylist()
        width = field.type.list_size if pa.types.is_fixed_size_list(field.type) else 1
        dimensions: list[list[float]] = [[] for _ in range(width)]
        for row in rows:
            values = row if isinstance(row, list) else [row]
            if row is None or any(value is None for value in values):
                raise ValueError(f"numeric null in statistics column {field.name}")
            for dimension, value in enumerate(values):
                dimensions[dimension].append(float(value))
        arrays = [np.asarray(values, dtype=np.float64) for values in dimensions]
        result[field.name] = {
            "min": [_json_number(values.min()) for values in arrays],
            "max": [_json_number(values.max()) for values in arrays],
            "mean": [_json_number(values.mean()) for values in arrays],
            "std": [_json_number(values.std()) for values in arrays],
            "count": [len(rows)],
        }
    return result


def _expected_base_files(
    source: SourceRecord,
    *,
    source_to_output: Mapping[int, int],
    chunk_size: int,
    source_parquet_paths: set[str],
) -> set[str]:
    expected = set(_REWRITTEN_METADATA)
    for source_index, output_index in source_to_output.items():
        expected.add(f"data/chunk-{output_index // chunk_size:03d}/episode_{output_index:06d}.parquet")
    for relative in source.file_hashes:
        if (
            relative in source_parquet_paths
            or relative in _REWRITTEN_METADATA
            or relative in _SOURCE_CACHE_METADATA
        ):
            continue
        parts = list(Path(relative).parts)
        match = _EPISODE_FILENAME.match(parts[-1])
        if match is not None:
            output_index = source_to_output.get(int(match.group(1)))
            if output_index is None:
                continue
            parts[-1] = f"episode_{output_index:06d}{match.group('suffix')}"
            for position, part in enumerate(parts[:-1]):
                if re.fullmatch(r"chunk-\d{3}", part):
                    parts[position] = f"chunk-{output_index // chunk_size:03d}"
        expected.add(Path(*parts).as_posix())
    return expected


def _source_inodes(source: SourceRecord) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    for relative in source.file_hashes:
        result = (source.root / relative).stat(follow_symlinks=False)
        if not stat.S_ISREG(result.st_mode):
            raise StructuralValidationError(f"source file is not regular: {relative}")
        identities.add((result.st_dev, result.st_ino))
    return identities


def validate_structural_dataset(
    *,
    staging_path: Path,
    source: SourceRecord,
    export: Mapping[str, Any],
    export_episodes: Sequence[Mapping[str, Any]],
    video_frame_counter: Callable[[Path], int] = _video_frame_count,
) -> dict[str, Any]:
    """Run the complete core structural and exact-preservation gate."""

    staging = Path(staging_path)
    try:
        if not staging.is_absolute() or not staging.is_dir() or staging.is_symlink():
            raise ValueError("staging path is not a directory")
        if export.get("approval_snapshot_sha256") is None:
            raise ValueError("approval snapshot is missing")
        source_info = _read_json(source.root / "meta/info.json")
        output_info = _read_json(staging / "meta/info.json")
        tasks_rows = _read_jsonl(staging / "meta/tasks.jsonl")
        episode_rows = _read_jsonl(staging / "meta/episodes.jsonl")
        stats_rows = _read_jsonl(staging / "meta/episodes_stats.jsonl")
        if not source.verify_current_inventory():
            raise ValueError("source manifest changed")

        approved_states = {ReviewState.APPROVED_KEEP.value, ReviewState.APPROVED_REJECT.value}
        source_episode_count = source_info.get("total_episodes")
        if type(source_episode_count) is not int or source_episode_count < 1:
            raise ValueError("source episode count is invalid")
        if (
            len(export_episodes) != source_episode_count
            or [row.get("source_episode_index") for row in export_episodes] != list(range(source_episode_count))
            or any(any(field not in row for field in _SNAPSHOT_FIELDS) for row in export_episodes)
            or any(row.get("review_state") not in approved_states for row in export_episodes)
        ):
            raise ValueError("approval snapshot is incomplete")
        snapshot_rows = [{field: row.get(field) for field in _SNAPSHOT_FIELDS} for row in export_episodes]
        snapshot_document = {
            "source_manifest_sha256": source.fingerprint,
            "prompt_template_sha256": export.get("prompt_template_sha256"),
            "episodes": snapshot_rows,
        }
        if canonical_json_sha256(snapshot_document) != export.get("approval_snapshot_sha256"):
            raise ValueError("approval snapshot hash is invalid")
        kept = [row for row in export_episodes if row["review_state"] == ReviewState.APPROVED_KEEP.value]
        rejected = [row for row in export_episodes if row["review_state"] == ReviewState.APPROVED_REJECT.value]
        if not kept:
            raise ValueError("approval snapshot contains zero kept episodes")

        chunk_size = output_info.get("chunks_size")
        data_pattern = output_info.get("data_path")
        source_pattern = source_info.get("data_path")
        if (
            type(chunk_size) is not int
            or chunk_size < 1
            or not isinstance(data_pattern, str)
            or not isinstance(source_pattern, str)
        ):
            raise ValueError("data path metadata is invalid")
        task_lookup = {row.get("task_index"): row.get("task") for row in tasks_rows}
        if len(task_lookup) != len(tasks_rows) or sorted(task_lookup) != list(range(len(tasks_rows))):
            raise ValueError("task indices are not contiguous")

        episode_reports: list[dict[str, Any]] = []
        global_cursor = 0
        video_files = [path for path in _regular_files(staging) if path.suffix == ".mp4"]
        parquet_files = [path for path in _regular_files(staging) if path.suffix == ".parquet"]
        if len(parquet_files) != len(kept):
            raise ValueError("staged parquet file count is invalid")
        video_feature_count = sum(
            1
            for feature in output_info.get("features", {}).values()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        )
        if len(video_files) != len(kept) * video_feature_count:
            raise ValueError("staged video file count is invalid")
        source_inode_set = _source_inodes(source)
        for path in _regular_files(staging):
            result = path.stat(follow_symlinks=False)
            if (result.st_dev, result.st_ino) in source_inode_set:
                raise ValueError(f"staging file is a hardlink to source: {path.relative_to(staging)}")

        if [row.get("episode_index") for row in episode_rows] != list(range(len(kept))):
            raise ValueError("episode metadata indices are not contiguous")
        if [row.get("episode_index") for row in stats_rows] != list(range(len(kept))):
            raise ValueError("episode statistics indices are not contiguous")

        referenced_tasks: set[int] = set()
        for output_index, snapshot in enumerate(kept):
            source_index = snapshot["source_episode_index"]
            length = snapshot["source_length"]
            output_parquet = staging / _format_episode_path(
                data_pattern, episode_index=output_index, chunk_size=chunk_size
            )
            source_parquet = source.root / _format_episode_path(
                source_pattern,
                episode_index=source_index,
                chunk_size=int(source_info.get("chunks_size", chunk_size)),
            )
            output_table = pq.read_table(output_parquet)
            source_table = pq.read_table(source_parquet)
            if len(output_table) != length or len(source_table) != length:
                raise ValueError(f"parquet row count mismatch for episode {output_index}")
            if episode_rows[output_index].get("length") != length:
                raise ValueError(f"episode metadata length mismatch for episode {output_index}")
            if output_table.schema.metadata != source_table.schema.metadata:
                raise ValueError(f"Arrow schema metadata changed for episode {output_index}")
            if (
                output_table.column_names != source_table.column_names
                or output_table.schema != source_table.schema
            ):
                raise ValueError(f"Arrow column order/schema changed for episode {output_index}")
            for name in source_table.column_names:
                if name not in output_table.column_names or output_table.schema.field(
                    name
                ) != source_table.schema.field(name):
                    raise ValueError(f"Arrow field changed: {name}")
                if name in _REPLACED_COLUMNS:
                    continue
                source_column = source_table.column(name)
                output_column = output_table.column(name)
                if not output_column.combine_chunks().equals(source_column.combine_chunks()):
                    raise ValueError(f"untouched Arrow values changed: {name}")
                if _ipc_hash(output_column, output_table.schema.field(name)) != _ipc_hash(
                    source_column, source_table.schema.field(name)
                ):
                    raise ValueError(f"untouched Arrow serialization changed: {name}")

            episode_values = output_table.column("episode_index").to_pylist()
            frame_values = output_table.column("frame_index").to_pylist()
            index_values = output_table.column("index").to_pylist()
            task_values = output_table.column("task_index").to_pylist()
            if episode_values != [output_index] * length:
                raise ValueError("episode_index is not contiguous")
            if frame_values != list(range(length)):
                raise ValueError("frame_index is not contiguous")
            if index_values != list(range(global_cursor, global_cursor + length)):
                raise ValueError("global index is not contiguous")
            global_cursor += length
            if any(type(value) is not int or value not in task_lookup for value in task_values):
                raise ValueError("task_index does not resolve")
            referenced_tasks.update(task_values)
            prompts = expand_prompts(
                object_name=snapshot["object_name"],
                hand=snapshot["pickup_hand"],
                turn=snapshot["turn_direction"],
            )
            resolved = [task_lookup[value] for value in task_values]
            run_starts = [
                index for index in range(length) if index == 0 or task_values[index] != task_values[index - 1]
            ]
            runs = [resolved[index] for index in run_starts]
            if runs != prompts or len(run_starts) != 7:
                raise ValueError(f"episode {output_index} does not contain seven ordered prompt runs")
            expected_starts = [0, *[snapshot[f"step_{step}_start_frame"] for step in range(2, 8)]]
            if run_starts != expected_starts or run_starts[0] != 0 or run_starts[-1] >= length:
                raise ValueError(f"episode {output_index} task coverage is invalid")
            if episode_rows[output_index].get("tasks") != prompts:
                raise ValueError(f"episode {output_index} prompt metadata is invalid")
            matching_videos = [path for path in video_files if path.name == f"episode_{output_index:06d}.mp4"]
            if not matching_videos:
                raise ValueError(f"episode {output_index} has no video")
            for video in matching_videos:
                if video_frame_counter(video) != length:
                    raise ValueError(f"video frame count mismatch: {video.relative_to(staging)}")
                relative = video.relative_to(staging)
                source_parts = list(relative.parts)
                source_parts[-1] = f"episode_{source_index:06d}.mp4"
                for position, part in enumerate(source_parts[:-1]):
                    if part.startswith("chunk-"):
                        source_parts[position] = (
                            f"chunk-{source_index // int(source_info.get('chunks_size', chunk_size)):03d}"
                        )
                source_video = source.root / Path(*source_parts)
                if not source_video.is_file() or _hash_file(source_video) != _hash_file(video):
                    raise ValueError(f"video bytes changed: {relative}")
            episode_reports.append(
                {
                    "full_coverage": True,
                    "output_episode_index": output_index,
                    "prompt_runs": 7,
                    "row_count": length,
                    "source_episode_index": source_index,
                    "video_count": len(matching_videos),
                }
            )
            if stats_rows[output_index] != {
                "episode_index": output_index,
                "stats": _episode_statistics(output_table),
            }:
                raise ValueError(f"episode statistics changed for episode {output_index}")

        if referenced_tasks != set(task_lookup):
            raise ValueError("tasks.jsonl contains unreferenced tasks")
        expected_counts = {
            "episodes": len(kept),
            "frames": global_cursor,
            "tasks": len(tasks_rows),
            "videos": len(video_files),
        }
        for field, value in {
            "total_episodes": expected_counts["episodes"],
            "total_frames": expected_counts["frames"],
            "total_tasks": expected_counts["tasks"],
            "total_videos": expected_counts["videos"],
            "total_chunks": (len(kept) + chunk_size - 1) // chunk_size,
        }.items():
            if output_info.get(field) != value:
                raise ValueError(f"info.json {field} is invalid")
        if output_info.get("splits") != {"train": f"0:{len(kept)}"}:
            raise ValueError("info.json split range is invalid")
        if len(stats_rows) != len(kept):
            raise ValueError("episode statistics row count is invalid")
        source_parquet_paths = {
            _format_episode_path(
                source_pattern,
                episode_index=row["source_episode_index"],
                chunk_size=int(source_info.get("chunks_size", chunk_size)),
            )
            for row in export_episodes
        }
        source_to_output = {row["source_episode_index"]: output_index for output_index, row in enumerate(kept)}
        expected_files = _expected_base_files(
            source,
            source_to_output=source_to_output,
            chunk_size=chunk_size,
            source_parquet_paths=source_parquet_paths,
        )
        provenance_path = staging / "meta/curation_provenance.json"
        if provenance_path.is_file():
            provenance = _read_json(provenance_path)
            expected_files.update(artifact["path"] for artifact in provenance.get("artifacts", []))
            expected_files.update(
                {
                    "meta/curation_provenance.json",
                    "meta/curation_checksums.sha256",
                }
            )
        for generated in ("meta/stats.json", "meta/relative_stats.json"):
            if (staging / generated).is_file():
                expected_files.add(generated)
        actual_files = {path.relative_to(staging).as_posix() for path in _regular_files(staging)}
        if actual_files != expected_files:
            raise ValueError("staged regular-file inventory is invalid")
        if not source.verify_current_inventory():
            raise ValueError("source manifest changed during validation")
    except StructuralValidationError:
        raise
    except Exception as error:
        raise StructuralValidationError(str(error)) from error

    return {
        "approval": {"approved": len(export_episodes), "kept": len(kept), "rejected": len(rejected)},
        "approval_snapshot_sha256": export["approval_snapshot_sha256"],
        "checks": {
            "contiguous_indices": True,
            "full_task_coverage": True,
            "independent_regular_files": True,
            "metadata_counts": True,
            "prompt_resolution": True,
            "source_manifest_unchanged": True,
            "untouched_arrow_columns_equal": True,
            "video_bytes_equal": True,
            "video_frame_counts": True,
        },
        "episodes": episode_reports,
        "output": expected_counts,
        "passed": True,
        "schema_version": 1,
        "source_manifest_sha256": source.fingerprint,
    }


def _write_new_file(path: Path, contents: bytes, *, mode: int = 0o644) -> None:
    install_canonical_file(path, contents, mode=mode)


def _artifact_descriptor(
    path: Path, *, root: Path, kind: str, media_type: str = "application/json"
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "bytes": path.stat().st_size,
        "kind": kind,
        "media_type": media_type,
        "path": relative,
        "sha256": _hash_file(path),
    }


def persist_structural_report(
    *, workspace: Path, export_id: str, staging_path: Path, report: Mapping[str, Any]
) -> dict[str, Any]:
    if report.get("passed") is not True:
        raise StructuralValidationError("structural report did not pass")
    UUID(export_id)
    contents = (canonical_json(dict(report)) + "\n").encode("utf-8")
    workspace_path = Path(workspace) / "exports" / export_id / "structural-report.json"
    _write_new_file(workspace_path, contents, mode=0o600)
    staged_path = Path(staging_path) / "meta/curation_artifacts/structural-report.json"
    _write_new_file(staged_path, contents)
    return _artifact_descriptor(staged_path, root=Path(staging_path), kind="structural_report")


def capture_repository_state(path: Path, name: str) -> dict[str, Any]:
    root = Path(path).resolve(strict=True)

    def git(*arguments: str) -> bytes:
        return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True).stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = git("diff", "--binary", "HEAD")
    untracked_raw = git("ls-files", "--others", "--exclude-standard", "-z")
    untracked: list[dict[str, Any]] = []
    for encoded in sorted(filter(None, untracked_raw.split(b"\0"))):
        relative = encoded.decode("utf-8")
        candidate = root / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError("repository contains an unsafe untracked entry")
        untracked.append({"bytes": candidate.stat().st_size, "path": relative, "sha256": _hash_file(candidate)})
    dirty = bool(tracked_diff or untracked)
    return {
        "commit": commit,
        "dirty": dirty,
        "name": name,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest() if tracked_diff else None,
        "untracked_files": untracked,
    }


def _bounded_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    data = value.encode("utf-8", "replace") if isinstance(value, str) else value
    return data[:_MAX_REPORT_STREAM_BYTES].decode("utf-8", "replace")


def _configured_secrets(environment: Mapping[str, str], explicit: Sequence[str]) -> tuple[str, ...]:
    values = set(explicit)
    for key, value in environment.items():
        lowered = key.lower()
        if any(fragment in lowered for fragment in ("token", "secret", "password", "api_key", "authorization")):
            values.add(value)
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def _redact_text(value: str | None, secrets: Sequence[str]) -> str | None:
    if value is None:
        return None
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return value


def _redact_value(value: Any, secrets: Sequence[str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, list):
        return [_redact_value(item, secrets) for item in value]
    if isinstance(value, Mapping):
        return {key: _redact_value(item, secrets) for key, item in value.items()}
    return value


def _run_validation_command(
    *,
    argv: list[str],
    cwd: Path,
    environment: Mapping[str, str] | None,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    repository_probe: Callable[[Path, str], dict[str, Any]],
    repository_name: str,
    timeout_s: int,
    secret_values: Sequence[str] = (),
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str] | None]:
    source_environment = dict(os.environ if environment is None else environment)
    secrets = _configured_secrets(source_environment, secret_values)
    process_environment = {
        key: value
        for key, value in sorted(source_environment.items())
        if key in _SAFE_ENVIRONMENT_KEYS and value not in secrets
    }
    started = _utc_now()
    exception: str | None = None
    trace: str | None = None
    result: subprocess.CompletedProcess[str] | None = None
    try:
        result = runner(
            argv,
            cwd=cwd,
            env=process_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except Exception as error:  # The report is the durable failure evidence.
        exception = _redact_text(f"{type(error).__name__}: {error}", secrets)
        trace = _redact_text(traceback_module.format_exc(), secrets)
    try:
        repository = repository_probe(cwd, repository_name)
    except Exception as error:
        repository = {
            "commit": None,
            "dirty": None,
            "name": repository_name,
            "tracked_diff_sha256": None,
            "untracked_files": [],
        }
        exception = _redact_text(f"{type(error).__name__}: {error}", secrets)
        trace = _redact_text(traceback_module.format_exc(), secrets)
    ended = _utc_now()
    report = {
        "command": {
            "argv": argv,
            "cwd": str(cwd),
            "environment": process_environment,
            "executable": argv[0],
        },
        "end_time_utc": ended,
        "exception": exception,
        "exit_code": None if result is None else result.returncode,
        "passed": result is not None and result.returncode == 0 and exception is None,
        "repository": _redact_value(repository, secrets),
        "schema_version": 1,
        "start_time_utc": started,
        "stderr": "" if result is None else _redact_text(_bounded_stream(result.stderr), secrets),
        "stdout": "" if result is None else _redact_text(_bounded_stream(result.stdout), secrets),
        "traceback": trace,
    }
    return _redact_value(report, secrets), result


def _canonical_staging(staging_path: Path) -> Path:
    if not str(staging_path):
        raise ValueError("staging path is required")
    staging = Path(staging_path).resolve(strict=True)
    if not staging.is_dir():
        raise ValueError("staging path must be a directory")
    return staging


_STATS_METRICS = frozenset({"mean", "std", "min", "max", "q01", "q99"})
_STATS_OUTPUT_PATHS = ("meta/relative_stats.json", "meta/stats.json")


def _numeric_shape(value: Any) -> tuple[int, ...]:
    if type(value) in {int, float}:
        if not math.isfinite(float(value)):
            raise ValueError("stats output contains a non-finite number")
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError("stats output values must be nonempty numeric scalars or lists")
    child_shapes = [_numeric_shape(item) for item in value]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError("stats output contains a ragged value")
    return (len(value), *child_shapes[0])


def _read_stats_output(staging: Path, relative: str) -> tuple[bytes, os.stat_result]:
    path = staging / relative
    final_present, temporary_present = canonical_file_presence(path)
    if not final_present or temporary_present:
        raise ValueError(f"stats output is missing or partial: {relative}")
    parent_fd = _open_safe_directory(path.parent, create=False)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
            raise ValueError(f"stats output is not an independent nonempty regular file: {relative}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"stats output changed while reading: {relative}")
        return b"".join(chunks), before
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def validate_gr00t_stats_outputs(staging_path: Path) -> list[dict[str, Any]]:
    """Validate and authenticate the two UNITREE_G1_SONIC stats outputs."""

    staging = _canonical_staging(staging_path)
    loaded: dict[str, tuple[bytes, os.stat_result, dict[str, Any]]] = {}
    for relative in _STATS_OUTPUT_PATHS:
        contents, identity = _read_stats_output(staging, relative)
        try:
            document = json.loads(contents)
        except Exception as error:
            raise ValueError(f"stats output is malformed JSON: {relative}") from error
        if not isinstance(document, dict):
            raise ValueError(f"stats output must contain a JSON object: {relative}")
        loaded[relative] = (contents, identity, document)
    identities = {(result.st_dev, result.st_ino) for _, result, _ in loaded.values()}
    if len(identities) != len(_STATS_OUTPUT_PATHS):
        raise ValueError("stats outputs are not independent files")

    info = _read_json(staging / "meta/info.json")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("staged info.json features are invalid")
    expected: dict[str, tuple[int, ...]] = {}
    for key, metadata in features.items():
        if not isinstance(key, str) or not isinstance(metadata, Mapping):
            raise ValueError("staged feature metadata is invalid")
        dtype = metadata.get("dtype")
        if not isinstance(dtype, str) or not (dtype.startswith("float") or dtype.startswith("bfloat")):
            continue
        shape = metadata.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension < 1 for dimension in shape)
        ):
            raise ValueError(f"staged float feature shape is invalid: {key}")
        expected[key] = tuple(shape)
    stats = loaded["meta/stats.json"][2]
    if set(stats) != set(expected):
        raise ValueError("stats.json float-feature key set is invalid")
    for feature, expected_shape in expected.items():
        evidence = stats[feature]
        if not isinstance(evidence, Mapping) or set(evidence) != _STATS_METRICS:
            raise ValueError(f"stats.json metrics are incomplete: {feature}")
        shapes = [_numeric_shape(evidence[metric]) for metric in sorted(_STATS_METRICS)]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(f"stats.json metric shapes disagree: {feature}")
        flattened_width = math.prod(expected_shape)
        if shapes[0] not in {expected_shape, (flattened_width,)} and not (
            flattened_width == 1 and shapes[0] == ()
        ):
            raise ValueError(f"stats.json metric shape is invalid: {feature}")
    if loaded["meta/relative_stats.json"][2] != {}:
        raise ValueError("relative_stats.json must be empty for UNITREE_G1_SONIC")

    return [
        {
            "bytes": len(loaded[relative][0]),
            "media_type": "application/json",
            "path": relative,
            "sha256": hashlib.sha256(loaded[relative][0]).hexdigest(),
        }
        for relative in _STATS_OUTPUT_PATHS
    ]


def run_gr00t_stats_validation(
    *,
    staging_path: Path,
    isaac_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    repository_probe: Callable[[Path, str], dict[str, Any]] = capture_repository_state,
    environment: Mapping[str, str] | None = None,
    secret_values: Sequence[str] = (),
    timeout_s: int = 3600,
) -> dict[str, Any]:
    staging = _canonical_staging(staging_path)
    root = Path(isaac_root).resolve(strict=True)
    process_environment = dict(os.environ if environment is None else environment)
    process_environment["STAGING_PATH"] = str(staging)
    argv = _validation_argv(kind="gr00t_stats_report", staging=staging, isaac_root=root)
    report, _ = _run_validation_command(
        argv=argv,
        cwd=root,
        environment=process_environment,
        runner=runner,
        repository_probe=repository_probe,
        repository_name="Isaac-GR00T",
        timeout_s=timeout_s,
        secret_values=secret_values,
    )
    report["outputs"] = []
    if report.get("passed") is True:
        try:
            report["outputs"] = validate_gr00t_stats_outputs(staging)
        except Exception as error:
            secrets = _configured_secrets(process_environment, secret_values)
            report["passed"] = False
            report["exception"] = _redact_text(f"{type(error).__name__}: {error}", secrets)
            report["traceback"] = _redact_text(traceback_module.format_exc(), secrets)
    return report


_LOADER_ACCEPTANCE_SCRIPT = r"""
import json
from pathlib import Path
import sys
import traceback

import pandas as pd
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

root = Path(sys.argv[1]).resolve(strict=True)
results = []
passed = True
try:
    task_rows = map(
        json.loads,
        (root / "meta/tasks.jsonl").read_text().splitlines(),
    )
    tasks = {row["task_index"]: row["task"] for row in task_rows}
    episodes = list(map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines()))
    info = json.loads((root / "meta/info.json").read_text())
    loader = LeRobotEpisodeLoader(root, MODALITY_CONFIGS["unitree_g1_sonic"])
    for position, episode in enumerate(episodes):
        result = {
            "episode_index": episode["episode_index"],
            "row_count": 0,
            "runs": [],
            "passed": False,
            "exception": None,
            "traceback": None,
        }
        try:
            loaded = loader[position]
            language_key = "language.annotation.human.task_description"
            actual = loaded[language_key].tolist()
            chunk = episode["episode_index"] // info["chunks_size"]
            parquet_path = root / info["data_path"].format(
                episode_chunk=chunk,
                episode_index=episode["episode_index"],
                chunk_index=chunk,
                file_index=episode["episode_index"],
            )
            raw = pd.read_parquet(parquet_path, columns=["task_index"])["task_index"].tolist()
            expected = [tasks[index] for index in raw]
            runs = [value for index, value in enumerate(expected) if index == 0 or value != expected[index - 1]]
            if len(expected) != len(actual) or actual != expected or len(runs) != 7 or runs != episode["tasks"]:
                raise AssertionError("loader language does not match seven ordered parquet task runs")
            result.update(row_count=len(actual), runs=runs, passed=True)
        except Exception as error:
            passed = False
            result["exception"] = f"{type(error).__name__}: {error}"
            result["traceback"] = traceback.format_exc()
        results.append(result)
except Exception as error:
    passed = False
    results.append({
        "episode_index": None,
        "row_count": 0,
        "runs": [],
        "passed": False,
        "exception": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    })
print(json.dumps(
    {"schema_version": 1, "episodes": results, "passed": passed},
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
))
raise SystemExit(0 if passed else 1)
""".strip()


def validate_gr00t_loader_report(report: Mapping[str, Any], *, staging_path: Path) -> None:
    """Authenticate exact per-episode loader evidence against staged metadata."""

    _require_exact_keys(report, {"schema_version", "episodes", "passed"}, "loader result")
    if report.get("schema_version") != 1 or report.get("passed") is not True:
        raise ValueError("loader result did not pass")
    rows = _read_jsonl(Path(staging_path) / "meta/episodes.jsonl")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("loader result has no episode evidence")
    expected_ids = list(range(len(rows)))
    if len(episodes) != len(rows) or [item.get("episode_index") for item in episodes] != expected_ids:
        raise ValueError("loader episode identities are incomplete, duplicated, or unordered")
    for position, (evidence, metadata) in enumerate(zip(episodes, rows, strict=True)):
        if not isinstance(evidence, Mapping):
            raise ValueError("loader episode evidence is not an object")
        _require_exact_keys(
            evidence,
            {"episode_index", "row_count", "runs", "passed", "exception", "traceback"},
            "loader episode evidence",
        )
        expected_runs = metadata.get("tasks")
        if (
            evidence["episode_index"] != position
            or evidence["row_count"] != metadata.get("length")
            or not isinstance(expected_runs, list)
            or len(expected_runs) != 7
            or evidence["runs"] != expected_runs
            or evidence["passed"] is not True
            or evidence["exception"] is not None
            or evidence["traceback"] is not None
        ):
            raise ValueError(f"loader episode evidence is invalid for episode {position}")


def run_gr00t_loader_validation(
    *,
    staging_path: Path,
    isaac_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    repository_probe: Callable[[Path, str], dict[str, Any]] = capture_repository_state,
    environment: Mapping[str, str] | None = None,
    secret_values: Sequence[str] = (),
    timeout_s: int = 3600,
) -> dict[str, Any]:
    staging = _canonical_staging(staging_path)
    root = Path(isaac_root).resolve(strict=True)
    process_environment = dict(os.environ if environment is None else environment)
    process_environment["STAGING_PATH"] = str(staging)
    argv = _validation_argv(kind="gr00t_loader_report", staging=staging, isaac_root=root)
    report, result = _run_validation_command(
        argv=argv,
        cwd=root,
        environment=process_environment,
        runner=runner,
        repository_probe=repository_probe,
        repository_name="Isaac-GR00T",
        timeout_s=timeout_s,
        secret_values=secret_values,
    )
    episodes: list[dict[str, Any]] = []
    if result is not None and result.stdout:
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            if not isinstance(payload, dict):
                raise ValueError("loader result schema is invalid")
            secrets = _configured_secrets(process_environment, secret_values)
            payload = _redact_value(payload, secrets)
            validate_gr00t_loader_report(payload, staging_path=staging)
            episodes = payload["episodes"]
            report["passed"] = bool(report["passed"])
        except Exception as error:
            report["passed"] = False
            report["exception"] = f"{type(error).__name__}: {error}"
            report["traceback"] = traceback_module.format_exc()
    report["episodes"] = episodes
    return report


def persist_validation_report(
    *,
    workspace: Path,
    export_id: str,
    staging_path: Path,
    filename: str,
    kind: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if report.get("passed") is not True:
        raise ValueError(f"{kind} did not pass")
    if filename not in {"gr00t-stats-report.json", "gr00t-loader-report.json"}:
        raise ValueError("validation report filename is invalid")
    contents = (canonical_json(dict(report)) + "\n").encode("utf-8")
    workspace_path = Path(workspace) / "exports" / export_id / filename
    _write_new_file(workspace_path, contents, mode=0o600)
    staged_path = Path(staging_path) / "meta/curation_artifacts" / filename
    _write_new_file(staged_path, contents)
    return _artifact_descriptor(staged_path, root=Path(staging_path), kind=kind)


_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "source", "approval", "software", "cosmos", "export", "episodes", "tasks", "artifacts"}
)


def _require_exact_keys(value: Mapping[str, Any], keys: set[str] | frozenset[str], label: str) -> None:
    if set(value) != set(keys):
        raise ValueError(f"{label} violates the closed schema")


_VALIDATION_REPORT_KEYS = frozenset(
    {
        "command",
        "end_time_utc",
        "exception",
        "exit_code",
        "passed",
        "repository",
        "schema_version",
        "start_time_utc",
        "stderr",
        "stdout",
        "traceback",
    }
)


def _validate_repository_evidence(repository: Any) -> None:
    if not isinstance(repository, Mapping):
        raise ValueError("validation repository evidence is invalid")
    _require_exact_keys(
        repository,
        {"name", "commit", "dirty", "tracked_diff_sha256", "untracked_files"},
        "validation repository evidence",
    )
    if repository["name"] != "Isaac-GR00T":
        raise ValueError("validation repository name is invalid")
    if not isinstance(repository["commit"], str) or re.fullmatch(r"[0-9a-f]{40}", repository["commit"]) is None:
        raise ValueError("validation repository commit is invalid")
    if type(repository["dirty"]) is not bool:
        raise ValueError("validation repository dirty flag is invalid")
    digest = repository["tracked_diff_sha256"]
    if digest is not None and (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise ValueError("validation repository diff digest is invalid")
    untracked = repository["untracked_files"]
    if not isinstance(untracked, list) or untracked != sorted(
        untracked, key=lambda item: item.get("path", "").encode("utf-8")
    ):
        raise ValueError("validation repository untracked evidence is invalid")
    for item in untracked:
        _require_exact_keys(item, {"path", "bytes", "sha256"}, "validation untracked file")
        if (
            not isinstance(item["path"], str)
            or not item["path"]
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise ValueError("validation untracked file evidence is invalid")
    if repository["dirty"]:
        if digest is None and not untracked:
            raise ValueError("dirty validation repository lacks evidence")
    elif digest is not None or untracked:
        raise ValueError("clean validation repository contains dirty evidence")


def _validation_argv(*, kind: str, staging: Path, isaac_root: Path) -> list[str]:
    executable = str(isaac_root / ".venv/bin/python")
    if kind == "gr00t_stats_report":
        return [
            executable,
            "gr00t/data/stats.py",
            "--dataset-path",
            str(staging),
            "--embodiment-tag",
            "UNITREE_G1_SONIC",
            "--modality-config-path",
            "gr00t/configs/data/embodiment_configs.py",
        ]
    if kind == "gr00t_loader_report":
        return [executable, "-c", _LOADER_ACCEPTANCE_SCRIPT, str(staging)]
    raise ValueError("validation report kind is invalid")


def _validate_gr00t_outer_report(
    report: Mapping[str, Any],
    *,
    kind: str,
    staging_path: Path,
    isaac_root: Path,
    loader: bool,
    command_staging_path: Path | None = None,
) -> None:
    expected_keys = set(_VALIDATION_REPORT_KEYS)
    if loader:
        expected_keys.add("episodes")
    else:
        expected_keys.add("outputs")
    _require_exact_keys(report, expected_keys, "GR00T validation report")
    staging = _canonical_staging(staging_path)
    command_staging = staging if command_staging_path is None else Path(command_staging_path).resolve(strict=False)
    if not command_staging.is_absolute() or ".." in command_staging.parts:
        raise ValueError("GR00T command staging path is invalid")
    root = Path(isaac_root).resolve(strict=True)
    expected_argv = _validation_argv(kind=kind, staging=command_staging, isaac_root=root)
    command = report.get("command")
    if not isinstance(command, Mapping):
        raise ValueError("GR00T validation command is invalid")
    _require_exact_keys(command, {"argv", "cwd", "environment", "executable"}, "GR00T command")
    environment = command["environment"]
    if (
        command["argv"] != expected_argv
        or command["cwd"] != str(root)
        or command["executable"] != expected_argv[0]
        or not isinstance(environment, Mapping)
        or any(key not in _SAFE_ENVIRONMENT_KEYS for key in environment)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items())
        or environment.get("STAGING_PATH") != str(command_staging)
    ):
        raise ValueError("GR00T validation command does not match the expected contract")
    if (
        report["schema_version"] != 1
        or report["passed"] is not True
        or report["exit_code"] != 0
        or report["exception"] is not None
        or report["traceback"] is not None
    ):
        raise ValueError("GR00T validation report did not pass exactly")
    parsed_times: list[datetime] = []
    for key in ("start_time_utc", "end_time_utc"):
        timestamp = report[key]
        if not isinstance(timestamp, str):
            raise ValueError("GR00T validation timestamp is invalid")
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("GR00T validation timestamp is not UTC")
        parsed_times.append(parsed)
    if parsed_times[1] < parsed_times[0]:
        raise ValueError("GR00T validation timestamps are reversed")
    for key in ("stdout", "stderr"):
        stream = report[key]
        if not isinstance(stream, str) or len(stream.encode("utf-8")) > _MAX_REPORT_STREAM_BYTES:
            raise ValueError("GR00T validation stream is invalid")
    _validate_repository_evidence(report["repository"])
    if loader:
        validate_gr00t_loader_report(
            {"schema_version": report["schema_version"], "episodes": report["episodes"], "passed": True},
            staging_path=staging,
        )
    else:
        outputs = report["outputs"]
        if not isinstance(outputs, list):
            raise ValueError("GR00T stats output descriptors are invalid")
        for descriptor in outputs:
            if not isinstance(descriptor, Mapping):
                raise ValueError("GR00T stats output descriptor is invalid")
            _require_exact_keys(
                descriptor,
                {"path", "bytes", "sha256", "media_type"},
                "GR00T stats output descriptor",
            )
        if outputs != validate_gr00t_stats_outputs(staging):
            raise ValueError("GR00T stats output descriptors do not match current files")


def validate_gr00t_stats_report(
    report: Mapping[str, Any],
    *,
    staging_path: Path,
    isaac_root: Path,
    command_staging_path: Path | None = None,
) -> None:
    _validate_gr00t_outer_report(
        report,
        kind="gr00t_stats_report",
        staging_path=staging_path,
        isaac_root=isaac_root,
        loader=False,
        command_staging_path=command_staging_path,
    )


def validate_gr00t_loader_outer_report(report: Mapping[str, Any], *, staging_path: Path, isaac_root: Path) -> None:
    _validate_gr00t_outer_report(
        report,
        kind="gr00t_loader_report",
        staging_path=staging_path,
        isaac_root=isaac_root,
        loader=True,
    )


def _reject_sensitive_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(
                fragment in lowered
                for fragment in ("api_key", "authorization", "bearer", "raw_reasoning", "chain_of_thought")
            ):
                raise ValueError("provenance contains a sensitive or reasoning field")
            _reject_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_keys(child)


def build_curation_provenance(
    *,
    source: Mapping[str, Any],
    approval: Mapping[str, Any],
    software: Mapping[str, Any],
    cosmos: Mapping[str, Any],
    export: Mapping[str, Any],
    episodes: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    software_document = dict(software)
    if isinstance(software_document.get("repositories"), list):
        software_document["repositories"] = sorted(
            (dict(item) for item in software_document["repositories"]),
            key=lambda item: str(item.get("name", "")).encode("utf-8"),
        )
    cosmos_document = dict(cosmos)
    for key in ("job_ids", "attempt_ids"):
        if isinstance(cosmos_document.get(key), list):
            cosmos_document[key] = sorted(cosmos_document[key])
    export_document = dict(export)
    if isinstance(export_document.get("source_to_output"), list):
        export_document["source_to_output"] = sorted(
            (dict(item) for item in export_document["source_to_output"]),
            key=lambda item: item.get("source_episode_index", -1),
        )
    episode_document = dict(episodes)
    for key in ("kept", "rejected"):
        if isinstance(episode_document.get(key), list):
            episode_document[key] = sorted(
                (dict(item) for item in episode_document[key]),
                key=lambda item: item.get("source_episode_index", -1),
            )
    document = {
        "approval": dict(approval),
        "artifacts": sorted(
            (dict(item) for item in artifacts),
            key=lambda item: str(item.get("path", "")).encode("utf-8"),
        ),
        "cosmos": cosmos_document,
        "episodes": episode_document,
        "export": export_document,
        "schema_version": 2,
        "software": software_document,
        "source": dict(source),
        "tasks": sorted((dict(item) for item in tasks), key=lambda item: item.get("task_index", -1)),
    }
    _validate_provenance_schema(document)
    return document


def _validate_provenance_schema(document: Mapping[str, Any]) -> None:
    _require_exact_keys(document, _TOP_LEVEL_KEYS, "provenance")
    if document.get("schema_version") != 2:
        raise ValueError("unsupported provenance schema version")
    nested_keys = {
        "source": {"dataset_alias", "manifest_sha256", "file_count", "original_tasks"},
        "approval": {"snapshot_sha256", "prompt_template_version", "prompt_template_sha256", "templates"},
        "software": {"exporter_version", "repositories"},
        "cosmos": {
            "model",
            "endpoint_identity",
            "contract_version",
            "sampling",
            "limits",
            "job_ids",
            "attempt_ids",
            "workspace_artifacts",
        },
        "export": {"export_id", "created_at_utc", "source_to_output"},
        "episodes": {"kept", "rejected"},
    }
    for key, expected in nested_keys.items():
        value = document.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"provenance {key} must be an object")
        _require_exact_keys(value, expected, f"provenance {key}")
    for key in ("tasks", "artifacts"):
        if not isinstance(document.get(key), list):
            raise ValueError(f"provenance {key} must be an array")
    _reject_sensitive_keys(document)
    source = document["source"]
    approval = document["approval"]
    software = document["software"]
    cosmos = document["cosmos"]
    export = document["export"]
    episodes = document["episodes"]
    for digest in (
        source["manifest_sha256"],
        approval["snapshot_sha256"],
        approval["prompt_template_sha256"],
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("provenance contains an invalid digest")
    if type(source["file_count"]) is not int or source["file_count"] < 1:
        raise ValueError("provenance source file count is invalid")
    if not isinstance(source["dataset_alias"], str) or not source["dataset_alias"]:
        raise ValueError("provenance dataset alias is invalid")
    if not isinstance(source["original_tasks"], list):
        raise ValueError("provenance original tasks are invalid")
    for item in source["original_tasks"]:
        _require_exact_keys(item, {"task_index", "task"}, "original task")
    templates = approval["templates"]
    if not isinstance(templates, list) or [item.get("step") for item in templates] != list(range(1, 8)):
        raise ValueError("provenance templates are not the ordered seven-step contract")
    for item in templates:
        _require_exact_keys(item, {"step", "template"}, "prompt template")
        if not isinstance(item["template"], str) or not item["template"]:
            raise ValueError("provenance prompt template is invalid")

    repositories = software["repositories"]
    if not isinstance(repositories, list) or repositories != sorted(
        repositories, key=lambda item: item.get("name", "").encode("utf-8")
    ):
        raise ValueError("provenance repositories are not sorted")
    # Canonical identity arrays must already be ordered before bytes are frozen.
    if document["tasks"] != sorted(document["tasks"], key=lambda row: row["task_index"]):
        raise ValueError("provenance tasks are not sorted")
    if [row.get("task_index") for row in document["tasks"]] != list(range(len(document["tasks"]))):
        raise ValueError("provenance task indices are not contiguous")
    for task in document["tasks"]:
        _require_exact_keys(task, {"task_index", "ordering_step", "prompt"}, "task")
        if task["ordering_step"] not in range(1, 8) or not isinstance(task["prompt"], str) or not task["prompt"]:
            raise ValueError("provenance task is invalid")
    artifacts = document["artifacts"]
    paths = [row.get("path") for row in artifacts]
    if len(paths) != len(set(paths)) or paths != sorted(paths, key=lambda value: str(value).encode("utf-8")):
        raise ValueError("provenance artifacts are not unique and sorted")
    for artifact in artifacts:
        _require_exact_keys(artifact, {"kind", "path", "bytes", "sha256", "media_type"}, "artifact")
        path = artifact["path"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or "\n" in path
            or "\r" in path
        ):
            raise ValueError("provenance artifact path is invalid")
        if type(artifact["bytes"]) is not int or artifact["bytes"] < 0:
            raise ValueError("provenance artifact byte size is invalid")
        if not isinstance(artifact["media_type"], str) or not artifact["media_type"]:
            raise ValueError("provenance artifact media type is invalid")
        if not isinstance(artifact["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is None:
            raise ValueError("provenance artifact digest is invalid")
    required_report_kinds = {"structural_report", "gr00t_stats_report", "gr00t_loader_report"}
    if not required_report_kinds.issubset({artifact["kind"] for artifact in artifacts}):
        raise ValueError("provenance is missing a validation report")
    for repository in repositories:
        _require_exact_keys(
            repository,
            {"name", "commit", "dirty", "tracked_diff_sha256", "untracked_files"},
            "repository",
        )
        if repository["dirty"]:
            if repository["tracked_diff_sha256"] is None and not repository["untracked_files"]:
                raise ValueError("dirty repository lacks dirty evidence")
        elif repository["tracked_diff_sha256"] is not None or repository["untracked_files"]:
            raise ValueError("clean repository contains dirty evidence")
        if (
            not isinstance(repository["commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", repository["commit"]) is None
        ):
            raise ValueError("repository commit is invalid")
        untracked = repository["untracked_files"]
        if not isinstance(untracked, list) or untracked != sorted(
            untracked, key=lambda item: item.get("path", "").encode("utf-8")
        ):
            raise ValueError("repository untracked files are not sorted")
        for item in untracked:
            _require_exact_keys(item, {"path", "bytes", "sha256"}, "untracked file")

    _require_exact_keys(
        cosmos["sampling"], {"target_fps", "resize_max_long_edge", "jpeg_quality"}, "Cosmos sampling"
    )
    _require_exact_keys(cosmos["limits"], {"max_duration_s", "max_frames", "max_payload_bytes"}, "Cosmos limits")
    if cosmos["contract_version"] != "pnp-trash-cosmos-v2":
        raise ValueError("Cosmos contract version is invalid")
    for key in ("job_ids", "attempt_ids"):
        values = cosmos[key]
        if not isinstance(values, list) or values != sorted(values):
            raise ValueError(f"Cosmos {key} are not sorted")
        for identifier in values:
            UUID(identifier)
    if not isinstance(cosmos["workspace_artifacts"], list):
        raise ValueError("Cosmos workspace artifacts are invalid")
    for item in cosmos["workspace_artifacts"]:
        _require_exact_keys(item, {"artifact_id", "sha256"}, "Cosmos workspace artifact")
        UUID(item["artifact_id"])

    UUID(export["export_id"])
    if not isinstance(export["source_to_output"], list) or export["source_to_output"] != sorted(
        export["source_to_output"], key=lambda item: item.get("source_episode_index", -1)
    ):
        raise ValueError("source-to-output map is not sorted")
    for item in export["source_to_output"]:
        _require_exact_keys(item, {"source_episode_index", "output_episode_index"}, "source-to-output row")
    for key, expected_keys in (
        (
            "kept",
            {
                "source_episode_index",
                "output_episode_index",
                "object",
                "hand",
                "turn",
                "transition_frames",
                "reviewer",
                "revision",
                "approved_at",
            },
        ),
        ("rejected", {"source_episode_index", "reason", "reviewer", "revision", "approved_at"}),
    ):
        rows = episodes[key]
        if not isinstance(rows, list) or rows != sorted(
            rows, key=lambda item: item.get("source_episode_index", -1)
        ):
            raise ValueError(f"provenance {key} episodes are not sorted")
        for row in rows:
            _require_exact_keys(row, expected_keys, f"{key} episode")
            if key == "kept":
                transitions = row["transition_frames"]
                if (
                    not isinstance(transitions, list)
                    or len(transitions) != 6
                    or any(type(value) is not int for value in transitions)
                    or any(left >= right for left, right in zip(transitions, transitions[1:]))
                ):
                    raise ValueError("kept episode transition frames are invalid")


def write_provenance(staging_path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    _validate_provenance_schema(document)
    path = Path(staging_path) / "meta/curation_provenance.json"
    _write_new_file(path, (canonical_json(dict(document)) + "\n").encode("utf-8"))
    return _artifact_descriptor(path, root=Path(staging_path), kind="curation_provenance")


def write_checksum_manifest(staging_path: Path) -> Path:
    root = Path(staging_path)
    checksum = root / "meta/curation_checksums.sha256"
    rows: list[str] = []
    for path in _regular_files(root):
        if path == checksum:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(f"{_hash_file(path)}  {relative}\n")
    install_canonical_file(checksum, "".join(rows).encode("utf-8"))
    return checksum


def seal_staging_tree(staging_path: Path) -> None:
    root = Path(staging_path)
    files = _regular_files(root)
    for path in files:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories: list[Path] = []
    for directory, child_directories, _ in os.walk(root, topdown=False, followlinks=False):
        directory_path = Path(directory)
        for child in child_directories:
            if (directory_path / child).is_symlink():
                raise ValueError("symlink directory is prohibited")
        directories.append(directory_path)
    for directory in directories:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    parent_fd = os.open(root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _media_type(path: str) -> str:
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".png"):
        return "image/png"
    return "application/octet-stream"


def validate_final_consistency(
    *,
    staging_path: Path,
    provenance: Mapping[str, Any],
    source_roots: Sequence[Path],
    stats_report_validator: Callable[[Path], None],
    structural_validator: Callable[[], Mapping[str, Any]],
    expectations: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a sealed tree without writing any staged bytes."""

    root = Path(staging_path)
    try:
        _require_exact_keys(
            expectations,
            {
                "approval_snapshot_sha256",
                "source_manifest_sha256",
                "prompt_template_version",
                "prompt_template_sha256",
                "templates",
                "source_to_output",
                "episodes",
                "tasks",
                "structural_report_sha256",
                "artifacts",
            },
            "final consistency expectations",
        )
        stats_report_validator(root)
        structural = structural_validator()
        if (
            structural.get("passed") is not True
            or structural.get("approval_snapshot_sha256") != expectations["approval_snapshot_sha256"]
            or structural.get("source_manifest_sha256") != expectations["source_manifest_sha256"]
        ):
            raise ValueError("repeated structural gate did not pass")
        _validate_provenance_schema(provenance)
        authenticated_links = {
            "approval_snapshot_sha256": provenance["approval"]["snapshot_sha256"],
            "source_manifest_sha256": provenance["source"]["manifest_sha256"],
            "prompt_template_version": provenance["approval"]["prompt_template_version"],
            "prompt_template_sha256": provenance["approval"]["prompt_template_sha256"],
            "templates": provenance["approval"]["templates"],
            "source_to_output": provenance["export"]["source_to_output"],
            "episodes": provenance["episodes"],
            "tasks": provenance["tasks"],
        }
        for key, value in authenticated_links.items():
            if value != expectations[key]:
                raise ValueError(f"provenance {key} does not match authoritative expectations")
        if provenance["artifacts"] != expectations["artifacts"]:
            raise ValueError("provenance artifacts do not match the authoritative exact set")
        provenance_path = root / "meta/curation_provenance.json"
        if _read_json(provenance_path) != dict(provenance):
            raise ValueError("provenance bytes do not match the validated document")

        files = _regular_files(root)
        checksum_path = root / "meta/curation_checksums.sha256"
        other_files = [path for path in files if path != checksum_path]
        manifest_lines = checksum_path.read_text(encoding="utf-8").splitlines(keepends=True)
        expected_paths = [path.relative_to(root).as_posix() for path in other_files]
        actual_paths: list[str] = []
        for line in manifest_lines:
            if not line.endswith("\n") or len(line) < 67 or line[64:66] != "  ":
                raise ValueError("checksum manifest format is invalid")
            digest, relative = line[:-1].split("  ", 1)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("checksum digest is invalid")
            actual_paths.append(relative)
            if _hash_file(root / relative) != digest:
                raise ValueError(f"checksum mismatch for {relative}")
        if actual_paths != expected_paths or len(actual_paths) != len(set(actual_paths)):
            raise ValueError("checksum manifest is incomplete, unordered, or contains unlisted paths")

        source_inodes: set[tuple[int, int]] = set()
        for source_root in source_roots:
            for path in _regular_files(Path(source_root)):
                result = path.stat(follow_symlinks=False)
                source_inodes.add((result.st_dev, result.st_ino))
        for path in other_files:
            result = path.stat(follow_symlinks=False)
            if (result.st_dev, result.st_ino) in source_inodes:
                raise ValueError(f"output hardlink to source: {path.relative_to(root)}")

        structural_artifacts = [
            artifact for artifact in provenance["artifacts"] if artifact["kind"] == "structural_report"
        ]
        if (
            len(structural_artifacts) != 1
            or structural_artifacts[0]["sha256"] != expectations["structural_report_sha256"]
        ):
            raise ValueError("provenance structural report link is invalid")
        structural_document = _read_json(root / structural_artifacts[0]["path"])
        if (
            structural_document.get("passed") is not True
            or structural_document.get("approval_snapshot_sha256") != expectations["approval_snapshot_sha256"]
            or structural_document.get("source_manifest_sha256") != expectations["source_manifest_sha256"]
        ):
            raise ValueError("structural report does not authenticate the expected inputs")

        for artifact in provenance["artifacts"]:
            relative = artifact["path"]
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("provenance artifact path is unsafe")
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"provenance artifact is missing: {relative}")
            if path.stat().st_size != artifact["bytes"] or _hash_file(path) != artifact["sha256"]:
                raise ValueError(f"provenance artifact changed: {relative}")
            expected_media_type = _media_type(relative)
            if artifact["media_type"] != expected_media_type:
                raise ValueError(f"provenance artifact media type is invalid: {relative}")
        for path in files:
            expected_mode = 0o444
            if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != expected_mode:
                raise ValueError(f"staged file is not sealed: {path.relative_to(root)}")
        for directory, directories, _ in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            if stat.S_IMODE(directory_path.stat(follow_symlinks=False).st_mode) != 0o555:
                raise ValueError(f"staged directory is not sealed: {directory_path.relative_to(root)}")
            if any((directory_path / child).is_symlink() for child in directories):
                raise ValueError("staged directory symlink is prohibited")
    except FinalConsistencyError:
        raise
    except Exception as error:
        raise FinalConsistencyError(str(error)) from error
    return {
        "approval_snapshot_sha256": provenance["approval"]["snapshot_sha256"],
        "checksum_sha256": _hash_file(root / "meta/curation_checksums.sha256"),
        "file_count": len(other_files),
        "passed": True,
        "schema_version": 1,
        "source_manifest_sha256": provenance["source"]["manifest_sha256"],
    }
