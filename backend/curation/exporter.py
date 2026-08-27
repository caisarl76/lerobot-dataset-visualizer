"""Deterministic, source-preserving v2.1 curation exports."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
from typing import Any, Sequence
from uuid import UUID, uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .db import CurationDatabase, ExportSnapshotValidationError, canonical_json
from .models import ExportState, ReviewState
from .prompts import PROMPT_TEMPLATE_SHA256, PROMPT_TEMPLATE_VERSION, expand_prompts
from .security import OpenedAsset
from .source import SourceRecord, SourceRegistry


class ExportError(RuntimeError):
    _STATUS_CODES = {
        "dataset_alias_not_found": 404,
        "export_not_found": 404,
        "export_executor_busy": 423,
    }

    def __init__(self, message: str, payload: dict[str, object]) -> None:
        self.payload = payload
        self.status_code = self._STATUS_CODES.get(str(payload.get("error")), 409)
        super().__init__(message)


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_absolute_directory(path: Path, *, create: bool) -> int:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ExportError("output directory path is invalid", {"error": "export_path_invalid"})
    current = os.open("/", _directory_open_flags())
    try:
        for component in path.parts[1:]:
            if not component:
                continue
            try:
                following = os.open(component, _directory_open_flags(), dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=current)
                os.fsync(current)
                following = os.open(component, _directory_open_flags(), dir_fd=current)
            os.close(current)
            current = following
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise OSError(errno.ENOTDIR, "output component is not a directory")
        descriptor = current
        current = -1
        return descriptor
    except ExportError:
        raise
    except OSError as error:
        raise ExportError("output directory path is unsafe", {"error": "export_path_invalid"}) from error
    finally:
        if current >= 0:
            os.close(current)


def _safe_relative_parts(relative_path: str | Path) -> tuple[str, ...]:
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ExportError("output path is invalid", {"error": "export_path_invalid"})
    return path.parts


def _secure_leaf_exists(path: Path, *, create_parent: bool) -> bool:
    parent_fd = _open_absolute_directory(Path(path).parent, create=create_parent)
    try:
        try:
            os.stat(Path(path).name, dir_fd=parent_fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False
    finally:
        os.close(parent_fd)


@dataclass
class _StagingTree:
    path: Path
    parent_fd: int
    root_fd: int
    parent_identity: tuple[int, int]
    root_identity: tuple[int, int]

    @classmethod
    def create(cls, path: Path) -> _StagingTree:
        path = Path(path)
        parent_fd = _open_absolute_directory(path.parent, create=True)
        root_fd = -1
        try:
            os.mkdir(path.name, mode=0o755, dir_fd=parent_fd)
            os.fsync(parent_fd)
            root_fd = os.open(path.name, _directory_open_flags(), dir_fd=parent_fd)
            parent_stat = os.fstat(parent_fd)
            root_stat = os.fstat(root_fd)
            tree = cls(
                path=path,
                parent_fd=parent_fd,
                root_fd=root_fd,
                parent_identity=(parent_stat.st_dev, parent_stat.st_ino),
                root_identity=(root_stat.st_dev, root_stat.st_ino),
            )
            parent_fd = -1
            root_fd = -1
            return tree
        except FileExistsError:
            raise ExportError("staging path already exists", {"error": "export_staging_exists"}) from None
        except ExportError:
            raise
        except OSError as error:
            raise ExportError("staging path is unsafe", {"error": "export_path_invalid"}) from error
        finally:
            if root_fd >= 0:
                os.close(root_fd)
            if parent_fd >= 0:
                os.close(parent_fd)

    @classmethod
    def open(cls, path: Path, *, expected_identity: tuple[int, int]) -> _StagingTree:
        path = Path(path)
        parent_fd = _open_absolute_directory(path.parent, create=False)
        root_fd = -1
        try:
            root_fd = os.open(path.name, _directory_open_flags(), dir_fd=parent_fd)
            parent_stat = os.fstat(parent_fd)
            root_stat = os.fstat(root_fd)
            if (root_stat.st_dev, root_stat.st_ino) != expected_identity:
                raise ExportError("staging identity changed", {"error": "export_path_invalid"})
            tree = cls(
                path=path,
                parent_fd=parent_fd,
                root_fd=root_fd,
                parent_identity=(parent_stat.st_dev, parent_stat.st_ino),
                root_identity=expected_identity,
            )
            parent_fd = -1
            root_fd = -1
            return tree
        except ExportError:
            raise
        except OSError as error:
            raise ExportError("staging path is unsafe", {"error": "export_path_invalid"}) from error
        finally:
            if root_fd >= 0:
                os.close(root_fd)
            if parent_fd >= 0:
                os.close(parent_fd)

    def close(self) -> None:
        if self.root_fd >= 0:
            os.close(self.root_fd)
            self.root_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1

    def __enter__(self) -> _StagingTree:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _open_directory(self, parts: Sequence[str], *, create: bool) -> int:
        current = os.dup(self.root_fd)
        try:
            for component in parts:
                try:
                    following = os.open(component, _directory_open_flags(), dir_fd=current)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o755, dir_fd=current)
                    os.fsync(current)
                    following = os.open(component, _directory_open_flags(), dir_fd=current)
                os.close(current)
                current = following
            descriptor = current
            current = -1
            return descriptor
        except OSError as error:
            raise ExportError("staging component is unsafe", {"error": "export_path_invalid"}) from error
        finally:
            if current >= 0:
                os.close(current)

    def create_file(self, relative_path: str | Path, *, mode: int = 0o644) -> int:
        parts = _safe_relative_parts(relative_path)
        parent_fd = self._open_directory(parts[:-1], create=True)
        descriptor = -1
        try:
            descriptor = os.open(
                parts[-1],
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                mode,
                dir_fd=parent_fd,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError(errno.EINVAL, "output is not a regular file")
            os.fsync(parent_fd)
            result = descriptor
            descriptor = -1
            return result
        except OSError as error:
            raise ExportError("output target is unsafe", {"error": "export_path_invalid"}) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)

    def open_regular(self, relative_path: str | Path, *, expected_identity: tuple[int, int, int]) -> int:
        parts = _safe_relative_parts(relative_path)
        parent_fd = self._open_directory(parts[:-1], create=False)
        descriptor = -1
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            result = os.fstat(descriptor)
            if (
                not stat.S_ISREG(result.st_mode)
                or (
                    result.st_dev,
                    result.st_ino,
                    result.st_size,
                )
                != expected_identity
            ):
                raise OSError(errno.EINVAL, "output identity changed")
            opened = descriptor
            descriptor = -1
            return opened
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)

    def write_bytes(self, relative_path: str | Path, contents: bytes, *, mode: int = 0o644) -> None:
        descriptor = self.create_file(relative_path, mode=mode)
        try:
            view = memoryview(contents)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def is_regular(self, relative_path: str | Path) -> bool:
        parts = _safe_relative_parts(relative_path)
        try:
            parent_fd = self._open_directory(parts[:-1], create=False)
        except ExportError:
            return False
        descriptor = -1
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            return stat.S_ISREG(os.fstat(descriptor).st_mode)
        except OSError:
            return False
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)

    def verify_path(self) -> None:
        try:
            current_parent = _open_absolute_directory(self.path.parent, create=False)
            try:
                parent_stat = os.fstat(current_parent)
                leaf_stat = os.stat(self.path.name, dir_fd=current_parent, follow_symlinks=False)
            finally:
                os.close(current_parent)
        except (OSError, ExportError) as error:
            raise ExportError("staging path identity changed", {"error": "export_path_invalid"}) from error
        if (
            (parent_stat.st_dev, parent_stat.st_ino) != self.parent_identity
            or not stat.S_ISDIR(leaf_stat.st_mode)
            or (leaf_stat.st_dev, leaf_stat.st_ino) != self.root_identity
        ):
            raise ExportError("staging path identity changed", {"error": "export_path_invalid"})

    def validate_tree(self) -> None:
        self.verify_path()
        try:
            for _, directories, filenames, directory_fd in os.fwalk(
                ".", topdown=False, follow_symlinks=False, dir_fd=self.root_fd
            ):
                for name in filenames:
                    if not stat.S_ISREG(os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode):
                        raise OSError(errno.EINVAL, "non-regular staging asset")
                for name in directories:
                    if not stat.S_ISDIR(os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode):
                        raise OSError(errno.EINVAL, "non-directory staging component")
        except OSError as error:
            raise ExportError("staging tree is unsafe", {"error": "export_path_invalid"}) from error
        self.verify_path()

    def clear(self) -> None:
        self.validate_tree()
        try:
            for _, directories, filenames, directory_fd in os.fwalk(
                ".", topdown=False, follow_symlinks=False, dir_fd=self.root_fd
            ):
                for name in filenames:
                    mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
                    if not stat.S_ISREG(mode):
                        raise OSError(errno.EINVAL, "non-regular staging asset")
                    os.unlink(name, dir_fd=directory_fd)
                for name in directories:
                    mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
                    if not stat.S_ISDIR(mode):
                        raise OSError(errno.EINVAL, "non-directory staging component")
                    os.rmdir(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
        except OSError as error:
            raise ExportError("staging cleanup is unsafe", {"error": "export_path_invalid"}) from error
        self.verify_path()


class ExportService:
    def __init__(
        self,
        *,
        database: CurationDatabase,
        source_registry: SourceRegistry,
        workspace: Path,
        final_path: Path,
    ) -> None:
        self.database = database
        self.source_registry = source_registry
        self.workspace = Path(workspace)
        self.final_path = Path(final_path)

    def create(self, dataset_alias: str) -> dict[str, object]:
        record = self.source_registry.records.get(dataset_alias)
        if record is None:
            raise ExportError(
                "dataset alias is not registered",
                {"error": "dataset_alias_not_found", "dataset_alias": dataset_alias},
            )
        dataset = self.database.get_dataset(alias=dataset_alias)
        if dataset is None:
            raise ExportError(
                "workspace has not been opened",
                {"error": "workspace_not_open", "dataset_alias": dataset_alias},
            )
        if (
            dataset["source_path"] != str(record.root)
            or dataset["source_manifest_sha256"] != record.fingerprint
            or not record.verify_current_inventory()
        ):
            raise ExportError(
                "source fingerprint does not match the registered workspace",
                {"error": "source_fingerprint_mismatch", "dataset_alias": dataset_alias},
            )
        if (
            dataset["prompt_template_version"] != PROMPT_TEMPLATE_VERSION
            or dataset["prompt_template_sha256"] != PROMPT_TEMPLATE_SHA256
        ):
            raise ExportError(
                "workspace prompt contract does not match this exporter",
                {"error": "export_prompt_contract_mismatch"},
            )
        _reject_source_symlinks(record.root)
        final_path = Path(os.path.abspath(self.final_path))
        if _secure_leaf_exists(final_path, create_parent=True):
            raise ExportError("final export path already exists", {"error": "export_destination_exists"})
        info = _read_registered_json(record, "meta/info.json")
        if info.get("codebase_version") != "v2.1":
            raise ExportError("source dataset is not v2.1", {"error": "export_source_version"})
        expected_episode_count = info.get("total_episodes")
        if type(expected_episode_count) is not int or expected_episode_count < 1:
            raise ExportError("source episode count is invalid", {"error": "source_fingerprint_mismatch"})
        identifier = str(uuid4())
        staging_path = final_path.parent / f".{final_path.name}.staging-{identifier}"
        try:
            row = self.database.create_validated_export_snapshot(
                dataset_id=dataset["id"],
                staging_path=str(staging_path),
                final_path=str(final_path),
                export_id=identifier,
                expected_source_manifest_sha256=record.fingerprint,
                expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
                expected_episode_count=expected_episode_count,
            )
        except ExportSnapshotValidationError as error:
            payload = dict(error.payload)
            if payload["error"] == "source_fingerprint_mismatch":
                payload["dataset_alias"] = dataset_alias
            raise ExportError(str(error), payload) from error
        return self._response(row | {"dataset_alias": dataset_alias})

    def status(self, export_id: str) -> dict[str, object]:
        try:
            UUID(export_id)
        except (TypeError, ValueError):
            raise ExportError("export not found", {"error": "export_not_found"}) from None
        row = self.database.get_export(export_id=export_id)
        if row is None:
            raise ExportError("export not found", {"error": "export_not_found"})
        return self._response(row)

    def _response(self, row: dict[str, object]) -> dict[str, object]:
        export_id = str(row["id"])
        workspace = shlex.quote(str(self.workspace.resolve()))
        return {
            "export_id": export_id,
            "dataset_alias": row["dataset_alias"],
            "state": row["state"],
            "approval_snapshot_sha256": row["approval_snapshot_sha256"],
            "staging_path": row["staging_path"],
            "final_path": row["final_path"],
            "failure_summary": row.get("failure_summary"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "run_command": (
                "backend/.venv/bin/python backend/curation_export.py "
                f"--workspace {workspace} run --export-id {export_id}"
            ),
        }


class StagingExporter:
    def __init__(self, *, database: CurationDatabase, source_registry: SourceRegistry) -> None:
        self.database = database
        self.source_registry = source_registry

    def run(self, export_id: str) -> dict[str, object]:
        with _exclusive_export_execution(self.database.path.parent, export_id) as control_fd:
            return self._run_locked(export_id, control_fd=control_fd)

    def _run_locked(self, export_id: str, *, control_fd: int) -> dict[str, object]:
        export = self.database.get_export(export_id=export_id)
        if export is None:
            raise ExportError("export not found", {"error": "export_not_found"})
        if export["state"] != ExportState.QUEUED.value:
            raise ExportError(
                "export is not queued",
                {"error": "export_not_queued", "state": export["state"]},
            )
        record = self._source_authority(export)
        _reject_source_symlinks(record.root)
        staging = Path(export["staging_path"])
        final = Path(export["final_path"])
        if _secure_leaf_exists(final, create_parent=True):
            raise ExportError("final export path already exists", {"error": "export_destination_exists"})
        if _secure_leaf_exists(staging, create_parent=True):
            raise ExportError("staging path already exists", {"error": "export_staging_exists"})
        receipt_path = _staging_receipt_path(self.database.path.parent, export_id)
        if _control_path_exists(control_fd, receipt_path.name) or _control_path_exists(
            control_fd, _staging_receipt_temp_path(receipt_path).name
        ):
            raise ExportError(
                "staging ownership state already exists",
                {"error": "export_staging_ownership_mismatch", "export_id": export_id},
            )
        claimed = self.database.claim_export_build(export_id=export_id)
        episodes = claimed["episodes"]
        kept = [row for row in episodes if row["review_state"] == ReviewState.APPROVED_KEEP.value]
        episode_parquet_paths = {
            _one_source_episode_parquet(record, row["source_episode_index"]) for row in episodes
        }
        with _StagingTree.create(staging) as staging_tree:
            _write_staging_ownership_receipt(
                receipt_path=receipt_path,
                control_fd=control_fd,
                export=claimed["export"],
                staging=staging,
                staging_identity=staging_tree.root_identity,
            )
            return self._build(
                record=record,
                export=claimed["export"],
                kept=kept,
                staging=staging_tree,
                episode_parquet_paths=episode_parquet_paths,
            )

    def resume(self, export_id: str) -> dict[str, object]:
        with _exclusive_export_execution(self.database.path.parent, export_id) as control_fd:
            return self._resume_locked(export_id, control_fd=control_fd)

    def _resume_locked(self, export_id: str, *, control_fd: int) -> dict[str, object]:
        export = self.database.get_export(export_id=export_id)
        if export is None:
            raise ExportError("export not found", {"error": "export_not_found"})
        if export["state"] == ExportState.QUEUED.value:
            return self._run_locked(export_id, control_fd=control_fd)
        if export["state"] != ExportState.BUILDING.value:
            raise ExportError(
                "export state is not resumable by the staging builder",
                {"error": "export_not_resumable", "state": export["state"]},
            )
        record = self._source_authority(export)
        _reject_source_symlinks(record.root)
        staging = Path(export["staging_path"])
        final = Path(export["final_path"])
        expected_staging = final.parent / f".{final.name}.staging-{export_id}"
        if staging != expected_staging:
            raise ExportError(
                "persisted staging path is not owned by this export", {"error": "export_path_invalid"}
            )
        if _secure_leaf_exists(final, create_parent=True):
            raise ExportError("final export path already exists", {"error": "export_destination_exists"})
        receipt_path = _staging_receipt_path(self.database.path.parent, export_id)
        staging_exists = _secure_leaf_exists(staging, create_parent=True)
        receipt_exists = _control_path_exists(control_fd, receipt_path.name)
        receipt_temp_exists = _control_path_exists(control_fd, _staging_receipt_temp_path(receipt_path).name)
        staging_tree: _StagingTree
        clear_identity: tuple[int, int] | None = None
        if not staging_exists and not receipt_exists:
            if receipt_temp_exists:
                self._fail_ambiguous_export(export_id, "export_staging_ownership_mismatch")
            staging_tree = _StagingTree.create(staging)
            try:
                _write_staging_ownership_receipt(
                    receipt_path=receipt_path,
                    control_fd=control_fd,
                    export=export,
                    staging=staging,
                    staging_identity=staging_tree.root_identity,
                )
            except BaseException:
                staging_tree.close()
                raise
        elif staging_exists and not receipt_exists:
            self._fail_ambiguous_export(export_id, "export_staging_ownership_missing")
        elif not staging_exists and receipt_exists:
            self._fail_ambiguous_export(export_id, "export_staging_ownership_mismatch")
        else:
            if receipt_temp_exists:
                try:
                    _remove_installed_staging_receipt_temp(receipt_path, control_fd=control_fd)
                except (OSError, ValueError) as error:
                    self._fail_ambiguous_export(
                        export_id,
                        "export_staging_ownership_mismatch",
                        cause=error,
                    )
            try:
                identity = _verified_staging_ownership(
                    receipt_path=receipt_path,
                    control_fd=control_fd,
                    export=export,
                    staging=staging,
                )
            except ExportError as error:
                self._fail_ambiguous_export(
                    export_id,
                    str(error.payload["error"]),
                    cause=error,
                )
            try:
                staging_tree = _StagingTree.open(staging, expected_identity=identity)
            except ExportError as error:
                self._fail_ambiguous_export(
                    export_id,
                    "export_staging_ownership_mismatch",
                    cause=error,
                )
            clear_identity = identity
        with staging_tree:
            if clear_identity is not None:
                _clear_owned_staging(staging_tree, expected_identity=clear_identity)
            episodes = self.database.list_export_episodes(export_id=export_id)
            kept = [row for row in episodes if row["review_state"] == ReviewState.APPROVED_KEEP.value]
            episode_parquet_paths = {
                _one_source_episode_parquet(record, row["source_episode_index"]) for row in episodes
            }
            return self._build(
                record=record,
                export=export,
                kept=kept,
                staging=staging_tree,
                episode_parquet_paths=episode_parquet_paths,
            )

    def _fail_ambiguous_export(
        self,
        export_id: str,
        error_code: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self.database.fail_export_build(export_id=export_id, failure_summary=error_code)
        message = (
            "staging ownership receipt is missing"
            if error_code == "export_staging_ownership_missing"
            else "staging ownership receipt does not match"
        )
        error = ExportError(message, {"error": error_code, "export_id": export_id})
        if cause is None:
            raise error
        raise error from cause

    def _source_authority(self, export: dict[str, Any]) -> SourceRecord:
        record = self.source_registry.records.get(export["dataset_alias"])
        if record is None or (
            str(record.root) != export["source_path"]
            or record.fingerprint != export["source_manifest_sha256"]
            or not record.verify_current_inventory()
        ):
            raise ExportError(
                "source fingerprint does not match the export snapshot",
                {"error": "source_fingerprint_mismatch", "dataset_alias": export["dataset_alias"]},
            )
        return record

    def _build(
        self,
        *,
        record: SourceRecord,
        export: dict[str, Any],
        kept: list[dict[str, Any]],
        staging: _StagingTree,
        episode_parquet_paths: set[str],
    ) -> dict[str, object]:
        source_info = _read_registered_json(record, "meta/info.json")
        chunk_size = source_info.get("chunks_size")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ExportError("source chunks_size is invalid", {"error": "export_source_metadata"})
        prompt_sets = [
            expand_prompts(
                object_name=row["object_name"],
                hand=row["pickup_hand"],
                turn=row["turn_direction"],
            )
            for row in kept
        ]
        stable_tasks = build_stable_tasks(prompt_sets)
        task_indices = {task.prompt: task.task_index for task in stable_tasks}
        source_to_output = {row["source_episode_index"]: output_index for output_index, row in enumerate(kept)}
        episode_documents: list[dict[str, Any]] = []
        stats_documents: list[dict[str, Any]] = []
        global_index = 0
        for output_index, (row, prompts) in enumerate(zip(kept, prompt_sets, strict=True)):
            source_index = row["source_episode_index"]
            source_path = _one_source_episode_parquet(record, source_index)
            output_path = (
                Path("data") / f"chunk-{output_index // chunk_size:03d}" / f"episode_{output_index:06d}.parquet"
            )
            transition_frames = [row[f"step_{step}_start_frame"] for step in range(2, 8)]
            per_frame_tasks = _task_indices_for_frames(
                source_length=row["source_length"],
                transition_frames=transition_frames,
                prompts=prompts,
                task_indices=task_indices,
            )
            rewritten = _rewrite_episode_parquet(
                record,
                staging=staging,
                source_path=source_path,
                output_path=output_path.as_posix(),
                source_episode_index=source_index,
                output_episode_index=output_index,
                global_index_start=global_index,
                per_frame_tasks=per_frame_tasks,
            )
            global_index += row["source_length"]
            episode_documents.append(
                {"episode_index": output_index, "length": row["source_length"], "tasks": prompts}
            )
            stats_documents.append({"episode_index": output_index, "stats": _episode_statistics(rewritten)})

        copied_video_count = _copy_carried_assets(
            record,
            staging=staging,
            source_to_output=source_to_output,
            chunk_size=chunk_size,
            episode_parquet_paths=episode_parquet_paths,
        )
        if not staging.is_regular("meta/modality.json"):
            raise ExportError("source modality.json is required", {"error": "export_source_metadata"})
        output_info = json.loads(json.dumps(source_info))
        output_info.update(
            {
                "total_episodes": len(kept),
                "total_frames": global_index,
                "total_tasks": len(stable_tasks),
                "total_videos": copied_video_count,
                "total_chunks": (len(kept) + chunk_size - 1) // chunk_size,
                "splits": {"train": f"0:{len(kept)}"},
                "discarded_episode_indices": [],
            }
        )
        staging.write_bytes("meta/info.json", _pretty_json_bytes(output_info))
        staging.write_bytes(
            "meta/episodes.jsonl",
            _jsonl_bytes(episode_documents),
        )
        staging.write_bytes("meta/tasks.jsonl", serialize_tasks_jsonl(stable_tasks))
        staging.write_bytes(
            "meta/episodes_stats.jsonl",
            _jsonl_bytes(stats_documents),
        )
        if not record.verify_current_inventory():
            raise ExportError(
                "source inventory changed while building staging",
                {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
            )
        staging.validate_tree()
        return {
            "export_id": export["id"],
            "state": ExportState.BUILDING.value,
            "staging_path": str(staging.path),
            "kept_episodes": len(kept),
            "total_frames": global_index,
        }


@dataclass(frozen=True)
class StableTask:
    task_index: int
    ordering_step: int
    prompt: str


def build_stable_tasks(prompt_sets: Sequence[Sequence[str]]) -> list[StableTask]:
    """Build the frozen prompt map independent of database or worker order."""
    minimum_steps: dict[str, int] = {}
    for prompts in prompt_sets:
        if len(prompts) != 7:
            raise ValueError("each kept episode must expand to exactly seven prompts")
        for step, prompt in enumerate(prompts, start=1):
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("expanded prompts must be nonempty strings")
            minimum_steps[prompt] = min(step, minimum_steps.get(prompt, step))
    ordered = sorted(minimum_steps.items(), key=lambda item: (item[1], item[0].encode("utf-8")))
    return [
        StableTask(task_index=task_index, ordering_step=step, prompt=prompt)
        for task_index, (prompt, step) in enumerate(ordered)
    ]


def serialize_tasks_jsonl(tasks: Sequence[StableTask]) -> bytes:
    """Serialize the canonical compact tasks.jsonl representation."""
    if not tasks:
        return b""
    lines = [canonical_json({"task": task.prompt, "task_index": task.task_index}) for task in tasks]
    return ("\n".join(lines) + "\n").encode("utf-8")


_EPISODE_FILE = re.compile(r"^episode_(\d{6})(?P<suffix>\..+)$")
_KNOWN_REWRITTEN_METADATA = frozenset(
    {
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/episodes_stats.jsonl",
    }
)
_REPLACED_COLUMNS = ("episode_index", "frame_index", "index", "task_index")


def _read_registered_json(record: SourceRecord, relative_path: str) -> dict[str, object]:
    try:
        value = json.loads(_read_asset_bytes(record, relative_path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError("source metadata is invalid", {"error": "export_source_metadata"}) from error
    if not isinstance(value, dict):
        raise ExportError("source metadata is invalid", {"error": "export_source_metadata"})
    return value


def _read_asset_bytes(record: SourceRecord, relative_path: str) -> bytes:
    asset = record.open_asset(relative_path)
    if asset is None:
        raise ExportError(
            "source fingerprint does not match the registered workspace",
            {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
        )
    try:
        contents = _pread_all(asset)
        digest = hashlib.sha256(contents).hexdigest()
        if not record.verify_pinned_asset(asset, sha256=digest):
            raise ExportError(
                "source fingerprint changed while reading",
                {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
            )
        return contents
    finally:
        asset.close()


def _pread_all(asset: OpenedAsset) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while chunk := os.pread(asset.fd, 1024 * 1024, offset):
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _reject_source_symlinks(root: Path) -> None:
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted([*directories, *filenames], key=lambda value: value.encode("utf-8")):
            path = directory_path / name
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise ExportError(
                    "source inventory changed while checking symlinks",
                    {"error": "source_fingerprint_mismatch"},
                ) from error
            if stat.S_ISLNK(mode):
                raise ExportError(
                    "source symlinks are prohibited",
                    {"error": "export_source_symlink", "path": path.relative_to(root).as_posix()},
                )


@contextmanager
def _exclusive_export_execution(workspace: Path, export_id: str):
    """Hold the crash-releasing, cross-process executor lock for one export."""
    try:
        parsed = UUID(export_id)
    except (TypeError, ValueError):
        raise ExportError("export not found", {"error": "export_not_found"}) from None
    if str(parsed) != export_id:
        raise ExportError("export not found", {"error": "export_not_found"})
    lock_directory = Path(workspace) / "exports" / export_id
    directory_fd = -1
    descriptor = -1
    try:
        directory_fd = _open_absolute_directory(lock_directory, create=True)
        descriptor = os.open(
            "executor.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ExportError("export executor lock is not a regular file", {"error": "export_path_invalid"})
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise ExportError(
                    "another executor is already running this export",
                    {"error": "export_executor_busy", "export_id": export_id},
                ) from None
            raise
        os.fsync(directory_fd)
        directory_stat = os.fstat(directory_fd)
        yield directory_fd
        current_directory_fd = _open_absolute_directory(lock_directory, create=False)
        try:
            current_stat = os.fstat(current_directory_fd)
        finally:
            os.close(current_directory_fd)
        if (current_stat.st_dev, current_stat.st_ino) != (
            directory_stat.st_dev,
            directory_stat.st_ino,
        ):
            raise ExportError("export control directory changed", {"error": "export_path_invalid"})
    except ExportError:
        raise
    except OSError as error:
        raise ExportError("export executor lock is unsafe", {"error": "export_path_invalid"}) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)


def _staging_receipt_path(workspace: Path, export_id: str) -> Path:
    return Path(workspace) / "exports" / export_id / "staging-ownership.json"


def _staging_receipt_temp_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f".{receipt_path.name}.tmp")


def _control_path_exists(control_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=control_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _staging_identity(staging: Path) -> tuple[int, int]:
    try:
        result = staging.lstat()
    except OSError as error:
        raise ValueError("staging directory cannot be inspected") from error
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
        raise ValueError("staging path is not a regular directory")
    return result.st_dev, result.st_ino


def _write_staging_ownership_receipt(
    *,
    receipt_path: Path,
    control_fd: int,
    export: dict[str, Any],
    staging: Path,
    staging_identity: tuple[int, int] | None = None,
) -> None:
    if staging_identity is None:
        try:
            staging_identity = _staging_identity(staging)
        except ValueError as error:
            raise ExportError("staging path is unsafe", {"error": "export_path_invalid"}) from error
    device, inode = staging_identity
    document = {
        "approval_snapshot_sha256": export["approval_snapshot_sha256"],
        "device": device,
        "export_id": export["id"],
        "inode": inode,
        "schema_version": 1,
        "staging_path": str(staging),
    }
    receipt_temp_path = _staging_receipt_temp_path(receipt_path)
    try:
        _write_new_bytes_at(
            control_fd,
            receipt_temp_path.name,
            (canonical_json(document) + "\n").encode("utf-8"),
            mode=0o600,
        )
        os.fsync(control_fd)
        os.link(
            receipt_temp_path.name,
            receipt_path.name,
            src_dir_fd=control_fd,
            dst_dir_fd=control_fd,
            follow_symlinks=False,
        )
        os.fsync(control_fd)
        os.unlink(receipt_temp_path.name, dir_fd=control_fd)
        os.fsync(control_fd)
    except FileExistsError as error:
        raise ExportError(
            "staging ownership receipt already exists",
            {"error": "export_staging_ownership_mismatch", "export_id": export["id"]},
        ) from error


def _remove_installed_staging_receipt_temp(receipt_path: Path, *, control_fd: int) -> None:
    receipt_temp_path = _staging_receipt_temp_path(receipt_path)
    temporary = os.stat(receipt_temp_path.name, dir_fd=control_fd, follow_symlinks=False)
    installed = os.stat(receipt_path.name, dir_fd=control_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(temporary.st_mode)
        or not stat.S_ISREG(installed.st_mode)
        or (temporary.st_dev, temporary.st_ino) != (installed.st_dev, installed.st_ino)
    ):
        raise ValueError("temporary receipt is not the installed receipt")
    os.unlink(receipt_temp_path.name, dir_fd=control_fd)
    os.fsync(control_fd)


def _read_regular_file(path: Path) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("path is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ValueError("file identity changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_regular_file_at(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("path is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ValueError("file identity changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verified_staging_ownership(
    *,
    receipt_path: Path,
    control_fd: int,
    export: dict[str, Any],
    staging: Path,
) -> tuple[int, int]:
    export_id = str(export["id"])
    if not _control_path_exists(control_fd, receipt_path.name):
        raise ExportError(
            "staging ownership receipt is missing",
            {"error": "export_staging_ownership_missing", "export_id": export_id},
        )
    try:
        document = json.loads(_read_regular_file_at(control_fd, receipt_path.name))
        device, inode = _staging_identity(staging)
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(
            "staging ownership receipt does not match",
            {"error": "export_staging_ownership_mismatch", "export_id": export_id},
        ) from error
    expected = {
        "approval_snapshot_sha256": export["approval_snapshot_sha256"],
        "device": device,
        "export_id": export_id,
        "inode": inode,
        "schema_version": 1,
        "staging_path": str(staging),
    }
    if type(document) is not dict or document != expected:
        raise ExportError(
            "staging ownership receipt does not match",
            {"error": "export_staging_ownership_mismatch", "export_id": export_id},
        )
    return device, inode


def _clear_owned_staging(staging: _StagingTree, *, expected_identity: tuple[int, int]) -> None:
    if staging.root_identity != expected_identity:
        raise ExportError("staging path is unsafe", {"error": "export_path_invalid"})
    staging.clear()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _delete_staging_tree(staging: Path) -> None:
    """Legacy private helper retained for callers that require complete removal."""
    try:
        identity = _staging_identity(staging)
    except ValueError as error:
        raise ExportError("staging path is not an owned directory", {"error": "export_path_invalid"}) from error
    with _StagingTree.open(staging, expected_identity=identity) as tree:
        _clear_owned_staging(tree, expected_identity=identity)
        os.rmdir(staging.name, dir_fd=tree.parent_fd)
        os.fsync(tree.parent_fd)


def _one_source_episode_parquet(record: SourceRecord, source_episode_index: int) -> str:
    expected = f"episode_{source_episode_index:06d}.parquet"
    paths = sorted(path for path in record.file_hashes if path.startswith("data/") and Path(path).name == expected)
    if len(paths) != 1:
        raise ExportError(
            "source episode must resolve to exactly one parquet file",
            {"error": "export_source_parquet", "source_episode_index": source_episode_index},
        )
    return paths[0]


def _task_indices_for_frames(
    *,
    source_length: int,
    transition_frames: Sequence[int],
    prompts: Sequence[str],
    task_indices: dict[str, int],
) -> list[int]:
    if (
        len(transition_frames) != 6
        or len(prompts) != 7
        or not all(type(value) is int for value in transition_frames)
        or not 0 < transition_frames[0]
        or not all(left < right for left, right in zip(transition_frames, transition_frames[1:]))
        or transition_frames[-1] >= source_length
    ):
        raise ExportError("frozen transitions are invalid", {"error": "export_invalid_approval"})
    boundaries = [0, *transition_frames, source_length]
    values: list[int] = []
    for step_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        values.extend([task_indices[prompts[step_index]]] * (end - start))
    return values


def _rewrite_episode_parquet(
    record: SourceRecord,
    *,
    staging: _StagingTree,
    source_path: str,
    output_path: str,
    source_episode_index: int,
    output_episode_index: int,
    global_index_start: int,
    per_frame_tasks: Sequence[int],
) -> pa.Table:
    asset = record.open_asset(source_path)
    if asset is None:
        raise ExportError(
            "source parquet identity changed",
            {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
        )
    duplicate = -1
    output_fd = -1
    output_stream: pa.PythonFile | None = None
    writer: pq.ParquetWriter | None = None
    rewritten_groups: list[pa.Table] = []
    try:
        expected_digest = record.file_hashes[source_path]
        if hashlib.sha256(_pread_all(asset)).hexdigest() != expected_digest:
            raise ExportError(
                "source parquet bytes changed",
                {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
            )
        duplicate = os.dup(asset.fd)
        with os.fdopen(duplicate, "rb") as handle:
            duplicate = -1
            parquet = pq.ParquetFile(handle)
            schema = parquet.schema_arrow
            missing = set(_REPLACED_COLUMNS).difference(schema.names)
            if missing:
                raise ExportError(
                    "source parquet is missing required index columns",
                    {"error": "export_source_parquet", "missing_columns": sorted(missing)},
                )
            output_fd = staging.create_file(output_path)
            output_handle = os.fdopen(os.dup(output_fd), "wb")
            output_stream = pa.PythonFile(output_handle, mode="w")
            writer = pq.ParquetWriter(output_stream, schema)
            row_offset = 0
            for row_group_index in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group_index)
                count = len(table)
                replacement_values: dict[str, Sequence[int]] = {
                    "episode_index": [output_episode_index] * count,
                    "frame_index": list(range(row_offset, row_offset + count)),
                    "index": list(range(global_index_start + row_offset, global_index_start + row_offset + count)),
                    "task_index": per_frame_tasks[row_offset : row_offset + count],
                }
                source_episodes = table.column("episode_index").to_pylist()
                source_frames = table.column("frame_index").to_pylist()
                if source_episodes != [source_episode_index] * count or source_frames != list(
                    range(row_offset, row_offset + count)
                ):
                    raise ExportError(
                        "source parquet row indices are inconsistent",
                        {"error": "export_source_parquet", "source_episode_index": source_episode_index},
                    )
                for name in _REPLACED_COLUMNS:
                    column_index = schema.get_field_index(name)
                    field = schema.field(column_index)
                    replacement = pa.array(replacement_values[name], type=field.type)
                    table = table.set_column(column_index, field, replacement)
                writer.write_table(table)
                rewritten_groups.append(table)
                row_offset += count
            if row_offset != len(per_frame_tasks):
                raise ExportError(
                    "source parquet length differs from frozen approval",
                    {"error": "export_source_parquet", "source_episode_index": source_episode_index},
                )
        if not record.verify_pinned_asset(asset, sha256=expected_digest):
            raise ExportError(
                "source parquet identity changed after reading",
                {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
            )
    finally:
        if writer is not None:
            writer.close()
        if output_stream is not None:
            output_stream.close()
        if output_fd >= 0:
            os.fsync(output_fd)
            os.close(output_fd)
        if duplicate >= 0:
            os.close(duplicate)
        asset.close()
    return pa.concat_tables(rewritten_groups)


def _copy_carried_assets(
    record: SourceRecord,
    *,
    staging: _StagingTree,
    source_to_output: dict[int, int],
    chunk_size: int,
    episode_parquet_paths: set[str],
) -> int:
    video_count = 0
    for relative_path in sorted(record.file_hashes, key=lambda value: value.encode("utf-8")):
        if relative_path in episode_parquet_paths or relative_path in _KNOWN_REWRITTEN_METADATA:
            continue
        destination_relative = _carried_destination(
            relative_path,
            source_to_output=source_to_output,
            chunk_size=chunk_size,
        )
        if destination_relative is None:
            continue
        _copy_registered_asset(record, relative_path, staging, destination_relative)
        if destination_relative.startswith("videos/") and destination_relative.endswith(".mp4"):
            video_count += 1
    return video_count


def _carried_destination(
    relative_path: str,
    *,
    source_to_output: dict[int, int],
    chunk_size: int,
) -> str | None:
    parts = list(Path(relative_path).parts)
    match = _EPISODE_FILE.match(parts[-1])
    if match is None:
        return relative_path
    source_index = int(match.group(1))
    output_index = source_to_output.get(source_index)
    if output_index is None:
        return None
    parts[-1] = f"episode_{output_index:06d}{match.group('suffix')}"
    for position, part in enumerate(parts[:-1]):
        if re.fullmatch(r"chunk-\d{3}", part):
            parts[position] = f"chunk-{output_index // chunk_size:03d}"
    return Path(*parts).as_posix()


def _copy_registered_asset(
    record: SourceRecord,
    relative_path: str,
    staging: _StagingTree,
    destination: str,
) -> None:
    source_asset = record.open_asset(relative_path)
    if source_asset is None:
        raise ExportError(
            "source asset identity changed",
            {"error": "source_fingerprint_mismatch", "dataset_alias": record.alias},
        )
    destination_fd = -1
    source_digest = hashlib.sha256()
    try:
        destination_fd = staging.create_file(destination)
        offset = 0
        while chunk := os.pread(source_asset.fd, 1024 * 1024, offset):
            source_digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
            offset += len(chunk)
        os.fsync(destination_fd)
        output_stat = os.fstat(destination_fd)
        output_identity = (output_stat.st_dev, output_stat.st_ino, output_stat.st_size)
        os.close(destination_fd)
        destination_fd = -1
        expected_digest = record.file_hashes[relative_path]
        try:
            copied_digest = _hash_regular_destination(
                staging,
                destination,
                expected_identity=output_identity,
            )
        except (OSError, ValueError) as error:
            raise ExportError(
                "independent asset copy verification failed",
                {"error": "export_asset_copy", "path": relative_path},
            ) from error
        if (
            source_digest.hexdigest() != expected_digest
            or copied_digest != expected_digest
            or (output_stat.st_dev, output_stat.st_ino)
            == (source_asset.stat_result.st_dev, source_asset.stat_result.st_ino)
            or not record.verify_pinned_asset(source_asset, sha256=expected_digest)
        ):
            raise ExportError(
                "independent asset copy verification failed",
                {"error": "export_asset_copy", "path": relative_path},
            )
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        source_asset.close()


def _hash_regular_destination(
    staging: _StagingTree,
    relative_path: str,
    *,
    expected_identity: tuple[int, int, int],
) -> str:
    descriptor = staging.open_regular(relative_path, expected_identity=expected_identity)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
            )
            != expected_identity
        ):
            raise ValueError("destination identity differs from the completed copy")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != expected_identity:
            raise ValueError("destination identity changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _write_new_bytes(path: Path, contents: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes_at(directory_fd: int, name: str, contents: bytes, *, mode: int) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=directory_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "receipt temporary is not a regular file")
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")


def _pretty_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _episode_statistics(table: pa.Table) -> dict[str, dict[str, list[Any]]]:
    result: dict[str, dict[str, list[Any]]] = {}
    for field, column in zip(table.schema, table.columns, strict=True):
        value_type = field.type.value_type if pa.types.is_fixed_size_list(field.type) else field.type
        if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
            continue
        rows = column.to_pylist()
        width = field.type.list_size if pa.types.is_fixed_size_list(field.type) else 1
        per_dimension: list[list[float]] = [[] for _ in range(width)]
        for row in rows:
            values = row if isinstance(row, list) else [row]
            if row is None or any(value is None for value in values):
                raise ExportError(
                    "canonical v2.1 statistics do not support numeric nulls",
                    {"error": "export_source_stats_null", "column": field.name},
                )
            for dimension, value in enumerate(values):
                per_dimension[dimension].append(float(value))
        arrays = [np.asarray(values, dtype=np.float64) for values in per_dimension]
        result[field.name] = {
            "min": [_json_number(values.min()) for values in arrays],
            "max": [_json_number(values.max()) for values in arrays],
            "mean": [_json_number(values.mean()) for values in arrays],
            "std": [_json_number(values.std()) for values in arrays],
            "count": [len(rows)],
        }
    return result


def _json_number(value: np.generic) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number
