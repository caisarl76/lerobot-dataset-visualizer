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
import subprocess
from typing import Any, Callable, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .cosmos_transport import (
    CONTRACT_VERSION,
    JPEG_QUALITY,
    MAX_DURATION_SECONDS,
    MAX_PAYLOAD_BYTES,
    MAX_SAMPLED_FRAMES,
    RESIZE_MAX_LONG_EDGE,
    TARGET_SAMPLING_FPS,
)
from .db import CurationDatabase, ExportSnapshotValidationError, canonical_json
from .models import ExportState, ReviewState
from .prompts import (
    PROMPT_TEMPLATE_BYTES,
    PROMPT_TEMPLATE_SHA256,
    PROMPT_TEMPLATE_VERSION,
    expand_prompts,
)
from .publication import (
    PublicationError,
    PublicationIdentity,
    fsync_directory,
    pin_publication_source,
    preflight_rename_noreplace,
    publish_no_clobber,
    reconcile_publication_paths,
    verify_publication_identity,
)
from .security import OpenedAsset
from .source import SourceRecord, SourceRegistry
from .validation import (
    ArtifactInstallConflict,
    FinalConsistencyError,
    _artifact_descriptor,
    _hash_file,
    _read_json,
    _read_jsonl,
    _write_new_file,
    build_curation_provenance,
    canonical_file_presence,
    capture_repository_state,
    install_canonical_file,
    persist_structural_report,
    persist_validation_report,
    read_canonical_file,
    run_gr00t_loader_validation,
    run_gr00t_stats_validation,
    seal_staging_tree,
    validate_final_consistency,
    validate_gr00t_loader_outer_report,
    validate_gr00t_stats_report,
    validate_structural_dataset,
    write_checksum_manifest,
    write_provenance,
)


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


class ValidatedDatasetExporter:
    """Resume the fixed validation sequence and publish only its sealed result."""

    def __init__(
        self,
        *,
        database: CurationDatabase,
        source_registry: SourceRegistry,
        workspace: Path,
        isaac_root: Path,
        visualizer_root: Path,
        cosmos_model: str,
        cosmos_endpoint_identity: str,
        video_frame_counter: Callable[[Path], int] | None = None,
        stats_validator: Callable[..., dict[str, Any]] = run_gr00t_stats_validation,
        loader_validator: Callable[..., dict[str, Any]] = run_gr00t_loader_validation,
        repository_probe: Callable[[Path, str], dict[str, Any]] = capture_repository_state,
        stats_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        loader_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        parent_fsync: Callable[[Path], None] = fsync_directory,
        publication_preflight: Callable[[Path], None] = preflight_rename_noreplace,
        require_contact_sheets: bool = True,
    ) -> None:
        if not cosmos_model or not cosmos_endpoint_identity:
            raise ValueError("Cosmos model and endpoint identity are required for provenance")
        self.database = database
        self.source_registry = source_registry
        self.workspace = Path(workspace).resolve()
        self.isaac_root = Path(isaac_root)
        self.visualizer_root = Path(visualizer_root)
        self.cosmos_model = cosmos_model
        self.cosmos_endpoint_identity = cosmos_endpoint_identity
        self.video_frame_counter = video_frame_counter
        self.stats_validator = stats_validator
        self.loader_validator = loader_validator
        self.repository_probe = repository_probe
        self.stats_runner = stats_runner
        self.loader_runner = loader_runner
        self.parent_fsync = parent_fsync
        self.publication_preflight = publication_preflight
        self.require_contact_sheets = require_contact_sheets
        self.builder = StagingExporter(database=database, source_registry=source_registry)

    def run(self, export_id: str) -> dict[str, object]:
        with _exclusive_export_execution(self.database.path.parent, export_id) as control_fd:
            export = self._require_export(export_id)
            if export["state"] != ExportState.QUEUED.value:
                raise ExportError("export is not queued", {"error": "export_not_queued", "state": export["state"]})
            self._run_publication_preflight(Path(export["final_path"]).parent)
            self.builder._run_locked(export_id, control_fd=control_fd)
            return self._continue(export_id)

    def resume(self, export_id: str) -> dict[str, object]:
        with _exclusive_export_execution(self.database.path.parent, export_id) as control_fd:
            export = self._require_export(export_id)
            state = ExportState(export["state"])
            if state not in {ExportState.PUBLISHING, ExportState.PUBLISHED, ExportState.FAILED}:
                self._run_publication_preflight(Path(export["final_path"]).parent)
            if state is ExportState.QUEUED:
                self.builder._run_locked(export_id, control_fd=control_fd)
            elif state is ExportState.BUILDING:
                self.builder._resume_locked(export_id, control_fd=control_fd)
            elif state is ExportState.PUBLISHING:
                self._reconcile_publishing(export)
            elif state in {ExportState.PUBLISHED, ExportState.FAILED}:
                raise ExportError(
                    "export state is not resumable",
                    {"error": "export_not_resumable", "state": state.value},
                )
            return self._continue(export_id)

    def _run_publication_preflight(self, final_parent: Path) -> None:
        try:
            self.publication_preflight(final_parent)
        except PublicationError as error:
            raise ExportError(
                "publication no-clobber capability is unavailable",
                {"error": error.code},
            ) from error

    def _require_export(self, export_id: str) -> dict[str, Any]:
        export = self.database.get_export(export_id=export_id)
        if export is None:
            raise ExportError("export not found", {"error": "export_not_found"})
        return export

    def _source_authority(self, export: Mapping[str, Any]) -> SourceRecord:
        return self.builder._source_authority(dict(export))

    def _structural_report(
        self,
        *,
        root: Path,
        source: SourceRecord,
        export: Mapping[str, Any],
        episodes: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "staging_path": root,
            "source": source,
            "export": export,
            "export_episodes": episodes,
        }
        if self.video_frame_counter is not None:
            arguments["video_frame_counter"] = self.video_frame_counter
        return validate_structural_dataset(**arguments)

    def _continue(self, export_id: str) -> dict[str, object]:
        export = self._require_export(export_id)
        state = ExportState(export["state"])
        if state is ExportState.PUBLISHED:
            return self._result(export)
        if state is ExportState.PUBLISHING:
            self._reconcile_publishing(export)
            export = self._require_export(export_id)
            state = ExportState(export["state"])
            if state is ExportState.PUBLISHED:
                return self._result(export)
        source = self._source_authority(export)
        episodes = self.database.list_export_episodes(export_id=export_id)
        staging = Path(export["staging_path"])
        if state is not ExportState.BUILDING:
            try:
                self._verify_recorded_gate_artifacts(export, staging, state)
            except Exception as error:
                self._terminal_gate_failure(export, "export_artifact_reconciliation_conflict", error)

        if state is ExportState.BUILDING:
            try:
                report = self._structural_report(root=staging, source=source, export=export, episodes=episodes)
                persist_structural_report(
                    workspace=self.workspace,
                    export_id=export_id,
                    staging_path=staging,
                    report=report,
                )
                self._advance_with_report(
                    export=export,
                    expected=ExportState.BUILDING,
                    target=ExportState.CORE_STRUCTURAL_VALIDATED,
                    kind="structural_report",
                    filename="structural-report.json",
                )
            except ArtifactInstallConflict as error:
                self._terminal_gate_failure(export, "export_artifact_reconciliation_conflict", error)
            except Exception as error:
                self._terminal_gate_failure(export, "export_structural_validation_failed", error)
            state = ExportState.CORE_STRUCTURAL_VALIDATED
            export = self._require_export(export_id)

        if state is ExportState.CORE_STRUCTURAL_VALIDATED:
            adopted = self._artifact_operation(
                export,
                lambda: self._adopt_orphan_validation_report(
                    export=export,
                    staging=staging,
                    filename="gr00t-stats-report.json",
                    kind="gr00t_stats_report",
                    expected_state=ExportState.CORE_STRUCTURAL_VALIDATED,
                    target_state=ExportState.GROOT_STATS_VALIDATED,
                    validator=lambda report: validate_gr00t_stats_report(
                        report, staging_path=staging, isaac_root=self.isaac_root
                    ),
                ),
            )
            if not adopted:
                report = self.stats_validator(
                    staging_path=staging,
                    isaac_root=self.isaac_root,
                    runner=self.stats_runner,
                    repository_probe=self.repository_probe,
                )
                try:
                    validate_gr00t_stats_report(report, staging_path=staging, isaac_root=self.isaac_root)
                except Exception as error:
                    self._persist_failed_report(export_id, "gr00t-stats-report.json", report)
                    self._terminal_gate_failure(export, "export_gr00t_stats_failed", error)
                self._artifact_operation(
                    export,
                    lambda: persist_validation_report(
                        workspace=self.workspace,
                        export_id=export_id,
                        staging_path=staging,
                        filename="gr00t-stats-report.json",
                        kind="gr00t_stats_report",
                        report=report,
                    ),
                )
                self._advance_with_report(
                    export=export,
                    expected=ExportState.CORE_STRUCTURAL_VALIDATED,
                    target=ExportState.GROOT_STATS_VALIDATED,
                    kind="gr00t_stats_report",
                    filename="gr00t-stats-report.json",
                )
            state = ExportState.GROOT_STATS_VALIDATED
            export = self._require_export(export_id)

        if state is ExportState.GROOT_STATS_VALIDATED:
            self._artifact_operation(
                export,
                lambda: self._verify_recorded_gate_artifacts(export, staging, state),
            )
            adopted = self._artifact_operation(
                export,
                lambda: self._adopt_orphan_validation_report(
                    export=export,
                    staging=staging,
                    filename="gr00t-loader-report.json",
                    kind="gr00t_loader_report",
                    expected_state=ExportState.GROOT_STATS_VALIDATED,
                    target_state=ExportState.GROOT_LOADER_VALIDATED,
                    validator=lambda report: validate_gr00t_loader_outer_report(
                        report, staging_path=staging, isaac_root=self.isaac_root
                    ),
                ),
            )
            if not adopted:
                report = self.loader_validator(
                    staging_path=staging,
                    isaac_root=self.isaac_root,
                    runner=self.loader_runner,
                    repository_probe=self.repository_probe,
                )
                try:
                    validate_gr00t_loader_outer_report(report, staging_path=staging, isaac_root=self.isaac_root)
                except Exception as error:
                    self._persist_failed_report(export_id, "gr00t-loader-report.json", report)
                    self._terminal_gate_failure(export, "export_gr00t_loader_failed", error)
                self._artifact_operation(
                    export,
                    lambda: persist_validation_report(
                        workspace=self.workspace,
                        export_id=export_id,
                        staging_path=staging,
                        filename="gr00t-loader-report.json",
                        kind="gr00t_loader_report",
                        report=report,
                    ),
                )
                self._advance_with_report(
                    export=export,
                    expected=ExportState.GROOT_STATS_VALIDATED,
                    target=ExportState.GROOT_LOADER_VALIDATED,
                    kind="gr00t_loader_report",
                    filename="gr00t-loader-report.json",
                )
            state = ExportState.GROOT_LOADER_VALIDATED
            export = self._require_export(export_id)

        if state is ExportState.GROOT_LOADER_VALIDATED:
            self._artifact_operation(
                export,
                lambda: self._verify_recorded_gate_artifacts(export, staging, state),
            )
            artifacts = self._validation_artifacts(staging)
            artifacts.extend(
                self._artifact_operation(
                    export,
                    lambda: self._copy_selected_contact_sheets(export, episodes, staging),
                )
            )
            provenance = self._provenance_document(
                export=export,
                source=source,
                episodes=episodes,
                artifacts=artifacts,
                staging=staging,
            )
            self._artifact_operation(export, lambda: write_provenance(staging, provenance))
            self.database.set_export_state(
                export_id=export_id,
                expected_state=ExportState.GROOT_LOADER_VALIDATED,
                state=ExportState.PROVENANCE_WRITTEN,
            )
            state = ExportState.PROVENANCE_WRITTEN
            export = self._require_export(export_id)

        if state is ExportState.PROVENANCE_WRITTEN:
            self._artifact_operation(
                export,
                lambda: self._verify_recorded_gate_artifacts(export, staging, state),
            )
            artifacts = self._validation_artifacts(staging)
            artifacts.extend(
                self._artifact_operation(
                    export,
                    lambda: self._copy_selected_contact_sheets(export, episodes, staging),
                )
            )
            expected_provenance = self._provenance_document(
                export=export,
                source=source,
                episodes=episodes,
                artifacts=artifacts,
                staging=staging,
            )
            self._artifact_operation(export, lambda: write_provenance(staging, expected_provenance))
            self._artifact_operation(export, lambda: write_checksum_manifest(staging))
            seal_staging_tree(staging)
            provenance = _read_json(staging / "meta/curation_provenance.json")
            expectations = self._final_expectations(
                export=export,
                source=source,
                episodes=episodes,
                staging=staging,
            )
            try:
                final_report = validate_final_consistency(
                    staging_path=staging,
                    provenance=provenance,
                    source_roots=[source.root],
                    stats_report_validator=lambda root: self._verify_recorded_gate_artifacts(
                        export, root, ExportState.PROVENANCE_WRITTEN
                    ),
                    structural_validator=lambda: self._structural_report(
                        root=staging,
                        source=source,
                        export=export,
                        episodes=episodes,
                    ),
                    expectations=expectations,
                )
            except Exception as error:
                self._terminal_gate_failure(export, "export_final_consistency_failed", error)
            report_path = self.workspace / "exports" / export_id / "final-consistency-report.json"
            self._artifact_operation(
                export,
                lambda: _write_new_file(
                    report_path,
                    (canonical_json(final_report) + "\n").encode("utf-8"),
                    mode=0o600,
                ),
            )
            self._advance_with_report(
                export=export,
                expected=ExportState.PROVENANCE_WRITTEN,
                target=ExportState.FINAL_CONSISTENCY_VALIDATED,
                kind="final_consistency_report",
                filename="final-consistency-report.json",
            )
            state = ExportState.FINAL_CONSISTENCY_VALIDATED
            export = self._require_export(export_id)

        if state is ExportState.FINAL_CONSISTENCY_VALIDATED:
            try:
                recorded_final_report = self._artifact_operation(
                    export,
                    lambda: self._verify_recorded_gate_artifacts(export, staging, state),
                )
                identity = pin_publication_source(staging, Path(export["final_path"]))
                verify_publication_identity(staging, identity)
                provenance = _read_json(staging / "meta/curation_provenance.json")
                repeated_final_report = validate_final_consistency(
                    staging_path=staging,
                    provenance=provenance,
                    source_roots=[source.root],
                    stats_report_validator=lambda root: self._verify_recorded_gate_artifacts(
                        export, root, ExportState.FINAL_CONSISTENCY_VALIDATED
                    ),
                    structural_validator=lambda: self._structural_report(
                        root=staging,
                        source=source,
                        export=export,
                        episodes=episodes,
                    ),
                    expectations=self._final_expectations(
                        export=export,
                        source=source,
                        episodes=episodes,
                        staging=staging,
                    ),
                )
                if repeated_final_report != recorded_final_report:
                    raise FinalConsistencyError(
                        "repeated final-consistency result differs from its recorded report"
                    )
                verify_publication_identity(staging, identity)
                self._artifact_operation(
                    export,
                    lambda: _write_new_file(
                        self._publication_identity_path(export_id),
                        (canonical_json(identity.to_dict()) + "\n").encode("utf-8"),
                        mode=0o600,
                    ),
                )
            except PublicationError as error:
                self.database.record_export_failure(
                    export_id=export_id,
                    expected_state=ExportState.FINAL_CONSISTENCY_VALIDATED,
                    failure_summary=error.code,
                    terminal=True,
                )
                raise ExportError("publication identity changed", {"error": error.code}) from error
            except FinalConsistencyError as error:
                self._terminal_gate_failure(export, "export_final_consistency_failed", error)
            self.database.set_export_state(
                export_id=export_id,
                expected_state=ExportState.FINAL_CONSISTENCY_VALIDATED,
                state=ExportState.PUBLISHING,
            )
            try:
                publish_no_clobber(
                    staging,
                    Path(export["final_path"]),
                    expected_identity=identity,
                    parent_fsync=self.parent_fsync,
                    commit_published=lambda: self.database.set_export_state(
                        export_id=export_id,
                        expected_state=ExportState.PUBLISHING,
                        state=ExportState.PUBLISHED,
                    ),
                )
            except PublicationError as error:
                retryable = error.code == "publish_parent_fsync_failed"
                self.database.record_export_failure(
                    export_id=export_id,
                    expected_state=ExportState.PUBLISHING,
                    failure_summary=error.code,
                    terminal=not retryable,
                )
                raise ExportError("publication failed", {"error": error.code}) from error
        return self._result(self._require_export(export_id))

    def _adopt_orphan_validation_report(
        self,
        *,
        export: Mapping[str, Any],
        staging: Path,
        filename: str,
        kind: str,
        expected_state: ExportState,
        target_state: ExportState,
        validator: Callable[[Mapping[str, Any]], None],
    ) -> bool:
        workspace_path = self.workspace / "exports" / str(export["id"]) / filename
        staged_path = staging / "meta/curation_artifacts" / filename
        workspace_final, workspace_temporary = canonical_file_presence(workspace_path)
        staged_final, staged_temporary = canonical_file_presence(staged_path)
        if not workspace_final:
            if workspace_temporary or staged_final or staged_temporary:
                raise ArtifactInstallConflict(f"orphan {filename} lacks its canonical workspace final")
            return False
        contents = read_canonical_file(workspace_path)
        try:
            document = json.loads(contents)
            if not isinstance(document, dict) or contents != (canonical_json(document) + "\n").encode("utf-8"):
                raise ValueError("report bytes are not canonical JSON")
            validator(document)
        except Exception as error:
            raise ArtifactInstallConflict(f"orphan {filename} is invalid") from error
        install_canonical_file(workspace_path, contents, mode=0o600)
        install_canonical_file(staged_path, contents)
        self._advance_with_report(
            export=export,
            expected=expected_state,
            target=target_state,
            kind=kind,
            filename=filename,
        )
        return True

    def _advance_with_report(
        self,
        *,
        export: Mapping[str, Any],
        expected: ExportState,
        target: ExportState,
        kind: str,
        filename: str,
    ) -> None:
        path = self.workspace / "exports" / str(export["id"]) / filename
        self.database.advance_export_with_artifact(
            export_id=str(export["id"]),
            expected_state=expected,
            state=target,
            kind=kind,
            relative_path=path.relative_to(self.workspace).as_posix(),
            media_type="application/json",
            byte_size=path.stat().st_size,
            sha256=_hash_file(path),
        )

    def _verify_recorded_gate_artifacts(
        self,
        export: Mapping[str, Any],
        staging: Path,
        state: ExportState,
    ) -> dict[str, Any] | None:
        contracts = [
            (
                ExportState.CORE_STRUCTURAL_VALIDATED,
                "structural_artifact_id",
                "structural_report",
                "structural-report.json",
                True,
            ),
            (
                ExportState.GROOT_STATS_VALIDATED,
                "gr00t_stats_artifact_id",
                "gr00t_stats_report",
                "gr00t-stats-report.json",
                True,
            ),
            (
                ExportState.GROOT_LOADER_VALIDATED,
                "gr00t_loader_artifact_id",
                "gr00t_loader_report",
                "gr00t-loader-report.json",
                True,
            ),
            (
                ExportState.FINAL_CONSISTENCY_VALIDATED,
                "final_consistency_artifact_id",
                "final_consistency_report",
                "final-consistency-report.json",
                False,
            ),
        ]
        order = list(ExportState)
        recorded_final_report: dict[str, Any] | None = None
        for minimum_state, pointer, kind, filename, has_staged_copy in contracts:
            if order.index(state) < order.index(minimum_state):
                continue
            artifact_id = export.get(pointer)
            if not isinstance(artifact_id, str):
                raise ArtifactInstallConflict(f"missing database artifact pointer: {pointer}")
            with self.database.open_connection() as connection:
                row = connection.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
            if row is None:
                raise ArtifactInstallConflict(f"missing database artifact row: {pointer}")
            artifact = dict(row)
            workspace_path = self.workspace / "exports" / str(export["id"]) / filename
            expected_relative = workspace_path.relative_to(self.workspace).as_posix()
            workspace_contents = read_canonical_file(workspace_path)
            if (
                artifact["export_id"] != export["id"]
                or artifact["kind"] != kind
                or artifact["relative_path"] != expected_relative
                or artifact["media_type"] != "application/json"
                or artifact["byte_size"] != len(workspace_contents)
                or artifact["sha256"] != hashlib.sha256(workspace_contents).hexdigest()
            ):
                raise ArtifactInstallConflict(f"database artifact evidence disagrees: {filename}")
            if has_staged_copy:
                staged_path = staging / "meta/curation_artifacts" / filename
                staged_contents = read_canonical_file(staged_path)
                if staged_contents != workspace_contents:
                    raise ArtifactInstallConflict(f"staged artifact evidence disagrees: {filename}")
            if filename == "gr00t-stats-report.json":
                try:
                    document = json.loads(workspace_contents)
                    if not isinstance(document, dict) or workspace_contents != (
                        canonical_json(document) + "\n"
                    ).encode("utf-8"):
                        raise ValueError("stats report copies are not exact canonical JSON")
                    validate_gr00t_stats_report(
                        document,
                        staging_path=staging,
                        isaac_root=self.isaac_root,
                        command_staging_path=Path(str(export["staging_path"])),
                    )
                except Exception as error:
                    raise ArtifactInstallConflict("recorded GR00T stats report or outputs disagree") from error
            elif filename == "final-consistency-report.json":
                try:
                    document = json.loads(workspace_contents)
                    if not isinstance(document, dict) or workspace_contents != (
                        canonical_json(document) + "\n"
                    ).encode("utf-8"):
                        raise ValueError("final report is not exact canonical JSON")
                    recorded_final_report = document
                except Exception as error:
                    raise ArtifactInstallConflict("recorded final-consistency report is invalid") from error
        return recorded_final_report

    def _persist_failed_report(self, export_id: str, filename: str, report: Mapping[str, Any]) -> None:
        path = self.workspace / "exports" / export_id / filename
        try:
            _write_new_file(path, (canonical_json(dict(report)) + "\n").encode("utf-8"), mode=0o600)
        except ArtifactInstallConflict as error:
            self._terminal_gate_failure(
                self._require_export(export_id), "export_artifact_reconciliation_conflict", error
            )

    def _artifact_operation(
        self,
        export: Mapping[str, Any],
        operation: Callable[[], Any],
    ) -> Any:
        try:
            return operation()
        except ArtifactInstallConflict as error:
            self._terminal_gate_failure(export, "export_artifact_reconciliation_conflict", error)

    def _terminal_gate_failure(
        self,
        export: Mapping[str, Any],
        code: str,
        cause: BaseException | None = None,
    ) -> None:
        state = ExportState(export["state"])
        self.database.record_export_failure(
            export_id=str(export["id"]),
            expected_state=state,
            failure_summary=code,
            terminal=True,
        )
        failure = ExportError("export validation failed", {"error": code})
        if cause is None:
            raise failure
        raise failure from cause

    def _validation_artifacts(self, staging: Path) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for filename, kind in (
            ("structural-report.json", "structural_report"),
            ("gr00t-stats-report.json", "gr00t_stats_report"),
            ("gr00t-loader-report.json", "gr00t_loader_report"),
        ):
            path = staging / "meta/curation_artifacts" / filename
            descriptor = _artifact_descriptor(path, root=staging, kind=kind)
            artifacts.append(descriptor)
        return artifacts

    def _authoritative_provenance_artifacts(
        self,
        export: Mapping[str, Any],
        episodes: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        contracts = (
            ("structural_artifact_id", "structural_report", "structural-report.json"),
            ("gr00t_stats_artifact_id", "gr00t_stats_report", "gr00t-stats-report.json"),
            ("gr00t_loader_artifact_id", "gr00t_loader_report", "gr00t-loader-report.json"),
        )
        with self.database.open_connection() as connection:
            for pointer, kind, filename in contracts:
                artifact_id = export.get(pointer)
                row = (
                    None
                    if not isinstance(artifact_id, str)
                    else connection.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
                )
                if row is None:
                    raise ArtifactInstallConflict(f"authoritative artifact pointer is missing: {pointer}")
                evidence = dict(row)
                workspace_path = self.workspace / "exports" / str(export["id"]) / filename
                contents = read_canonical_file(workspace_path)
                digest = hashlib.sha256(contents).hexdigest()
                expected_relative = workspace_path.relative_to(self.workspace).as_posix()
                if (
                    evidence["export_id"] != export["id"]
                    or evidence["kind"] != kind
                    or evidence["relative_path"] != expected_relative
                    or evidence["media_type"] != "application/json"
                    or evidence["byte_size"] != len(contents)
                    or evidence["sha256"] != digest
                ):
                    raise ArtifactInstallConflict(f"authoritative database artifact disagrees: {filename}")
                artifacts.append(
                    {
                        "bytes": len(contents),
                        "kind": kind,
                        "media_type": "application/json",
                        "path": f"meta/curation_artifacts/{filename}",
                        "sha256": digest,
                    }
                )
        for source_path, kind, filename in self._selected_contact_sheet_sources(export, episodes):
            result = source_path.stat(follow_symlinks=False)
            artifacts.append(
                {
                    "bytes": result.st_size,
                    "kind": kind,
                    "media_type": "image/png",
                    "path": f"meta/curation_artifacts/contact_sheets/{filename}",
                    "sha256": _hash_file(source_path),
                }
            )
        artifacts.sort(key=lambda item: item["path"].encode("utf-8"))
        if len({item["path"] for item in artifacts}) != len(artifacts):
            raise ArtifactInstallConflict("authoritative provenance artifact paths are duplicated")
        return artifacts

    def _copy_selected_contact_sheets(
        self,
        export: Mapping[str, Any],
        episodes: Sequence[Mapping[str, Any]],
        staging: Path,
    ) -> list[dict[str, Any]]:
        selected = self._selected_contact_sheet_sources(export, episodes)
        artifacts: list[dict[str, Any]] = []
        for source_path, kind, filename in selected:
            destination = staging / "meta/curation_artifacts/contact_sheets" / filename
            _copy_independent_regular_file(source_path, destination)
            descriptor = _artifact_descriptor(destination, root=staging, kind=kind, media_type="image/png")
            artifacts.append(descriptor)
        return artifacts

    def _selected_contact_sheet_sources(
        self,
        export: Mapping[str, Any],
        episodes: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Path, str, str]]:
        from .contact_sheets import (
            ContactSheetDatasetIdentity,
            final_contact_sheet_path,
            proposal_contact_sheet_path,
        )

        identity = ContactSheetDatasetIdentity(
            dataset_id=int(export["dataset_id"]),
            dataset_alias=str(export["dataset_alias"]),
            source_manifest_sha256=str(export["source_manifest_sha256"]),
        )
        selected: list[tuple[Path, str, str]] = []
        with self.database.open_connection() as connection:
            for row in episodes:
                if row["review_state"] != ReviewState.APPROVED_KEEP.value:
                    continue
                final_relative = final_contact_sheet_path(
                    identity, row["source_episode_index"], row["approval_revision"]
                )
                selected.append(
                    (self.workspace / final_relative, "final_contact_sheet", Path(final_relative).name)
                )
                audit = connection.execute(
                    """
                    SELECT details_json FROM audit_events
                    WHERE dataset_id=? AND operation='keep_approved'
                        AND new_revision=?
                    ORDER BY created_at DESC, id DESC
                    """,
                    (export["dataset_id"], row["revision"]),
                ).fetchall()
                for event in audit:
                    details = json.loads(event["details_json"])
                    evidence = details.get("contact_sheet_evidence")
                    if (
                        isinstance(evidence, dict)
                        and evidence.get("source_episode_index") == row["source_episode_index"]
                    ):
                        proposal_id = evidence.get("proposal_id")
                        if proposal_id is not None:
                            proposal_relative = proposal_contact_sheet_path(identity, proposal_id)
                            selected.append(
                                (
                                    self.workspace / proposal_relative,
                                    "proposal_contact_sheet",
                                    Path(proposal_relative).name,
                                )
                            )
                        break
        if not self.require_contact_sheets:
            selected = [item for item in selected if item[0].is_file() and not item[0].is_symlink()]
        for source_path, kind, filename in selected:
            if not source_path.is_file() or source_path.is_symlink():
                raise ExportError("contact-sheet evidence is missing", {"error": "export_contact_sheet_missing"})
        return selected

    def _provenance_document(
        self,
        *,
        export: Mapping[str, Any],
        source: SourceRecord,
        episodes: Sequence[Mapping[str, Any]],
        artifacts: Sequence[Mapping[str, Any]],
        staging: Path,
    ) -> dict[str, Any]:
        kept_rows = [row for row in episodes if row["review_state"] == ReviewState.APPROVED_KEEP.value]
        rejected_rows = [row for row in episodes if row["review_state"] == ReviewState.APPROVED_REJECT.value]
        stable = build_stable_tasks(
            [
                expand_prompts(
                    object_name=row["object_name"],
                    hand=row["pickup_hand"],
                    turn=row["turn_direction"],
                )
                for row in kept_rows
            ]
        )
        cosmos = self._cosmos_provenance(export, kept_rows)
        stats_report = _read_json(staging / "meta/curation_artifacts/gr00t-stats-report.json")
        repositories = [
            self.repository_probe(self.visualizer_root, "lerobot-dataset-visualizer"),
            stats_report["repository"],
        ]
        template_lines = PROMPT_TEMPLATE_BYTES.decode("utf-8").splitlines()[1:]
        original_tasks = _read_jsonl(source.root / "meta/tasks.jsonl")
        return build_curation_provenance(
            source={
                "dataset_alias": source.alias,
                "manifest_sha256": source.fingerprint,
                "file_count": len(source.file_hashes),
                "original_tasks": original_tasks,
            },
            approval={
                "snapshot_sha256": export["approval_snapshot_sha256"],
                "prompt_template_version": export["prompt_template_version"],
                "prompt_template_sha256": export["prompt_template_sha256"],
                "templates": [
                    {"step": step, "template": template} for step, template in enumerate(template_lines, start=1)
                ],
            },
            software={"exporter_version": "pnp-trash-curation-v1", "repositories": repositories},
            cosmos=cosmos,
            export={
                "export_id": export["id"],
                "created_at_utc": export["created_at"],
                "source_to_output": [
                    {"source_episode_index": row["source_episode_index"], "output_episode_index": index}
                    for index, row in enumerate(kept_rows)
                ],
            },
            episodes={
                "kept": [
                    {
                        "source_episode_index": row["source_episode_index"],
                        "output_episode_index": index,
                        "object": row["object_name"],
                        "hand": row["pickup_hand"],
                        "turn": row["turn_direction"],
                        "transition_frames": [row[f"step_{step}_start_frame"] for step in range(2, 8)],
                        "reviewer": row["reviewer"],
                        "revision": row["revision"],
                        "approved_at": row["approved_at"],
                    }
                    for index, row in enumerate(kept_rows)
                ],
                "rejected": [
                    {
                        "source_episode_index": row["source_episode_index"],
                        "reason": row["rejection_reason"],
                        "reviewer": row["reviewer"],
                        "revision": row["revision"],
                        "approved_at": row["approved_at"],
                    }
                    for row in rejected_rows
                ],
            },
            tasks=[
                {"task_index": task.task_index, "ordering_step": task.ordering_step, "prompt": task.prompt}
                for task in stable
            ],
            artifacts=artifacts,
        )

    def _final_expectations(
        self,
        *,
        export: Mapping[str, Any],
        source: SourceRecord,
        episodes: Sequence[Mapping[str, Any]],
        staging: Path,
    ) -> dict[str, Any]:
        kept_rows = [row for row in episodes if row["review_state"] == ReviewState.APPROVED_KEEP.value]
        rejected_rows = [row for row in episodes if row["review_state"] == ReviewState.APPROVED_REJECT.value]
        stable = build_stable_tasks(
            [
                expand_prompts(
                    object_name=row["object_name"],
                    hand=row["pickup_hand"],
                    turn=row["turn_direction"],
                )
                for row in kept_rows
            ]
        )
        template_lines = PROMPT_TEMPLATE_BYTES.decode("utf-8").splitlines()[1:]
        return {
            "approval_snapshot_sha256": export["approval_snapshot_sha256"],
            "source_manifest_sha256": source.fingerprint,
            "prompt_template_version": export["prompt_template_version"],
            "prompt_template_sha256": export["prompt_template_sha256"],
            "templates": [
                {"step": step, "template": template} for step, template in enumerate(template_lines, start=1)
            ],
            "source_to_output": [
                {"source_episode_index": row["source_episode_index"], "output_episode_index": index}
                for index, row in enumerate(kept_rows)
            ],
            "episodes": {
                "kept": [
                    {
                        "source_episode_index": row["source_episode_index"],
                        "output_episode_index": index,
                        "object": row["object_name"],
                        "hand": row["pickup_hand"],
                        "turn": row["turn_direction"],
                        "transition_frames": [row[f"step_{step}_start_frame"] for step in range(2, 8)],
                        "reviewer": row["reviewer"],
                        "revision": row["revision"],
                        "approved_at": row["approved_at"],
                    }
                    for index, row in enumerate(kept_rows)
                ],
                "rejected": [
                    {
                        "source_episode_index": row["source_episode_index"],
                        "reason": row["rejection_reason"],
                        "reviewer": row["reviewer"],
                        "revision": row["revision"],
                        "approved_at": row["approved_at"],
                    }
                    for row in rejected_rows
                ],
            },
            "tasks": [
                {"task_index": task.task_index, "ordering_step": task.ordering_step, "prompt": task.prompt}
                for task in stable
            ],
            "structural_report_sha256": _hash_file(staging / "meta/curation_artifacts/structural-report.json"),
            "artifacts": self._authoritative_provenance_artifacts(export, episodes),
        }

    def _publication_identity_path(self, export_id: str) -> Path:
        return self.workspace / "exports" / export_id / "publication-identity.json"

    def _cosmos_provenance(
        self, export: Mapping[str, Any], kept_rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        proposal_ids: set[str] = set()
        with self.database.open_connection() as connection:
            for row in kept_rows:
                events = connection.execute(
                    """
                    SELECT details_json FROM audit_events
                    WHERE dataset_id=? AND operation='keep_approved' AND new_revision=?
                    ORDER BY created_at DESC, id DESC
                    """,
                    (export["dataset_id"], row["revision"]),
                ).fetchall()
                for event in events:
                    details = json.loads(event["details_json"])
                    evidence = details.get("contact_sheet_evidence")
                    if (
                        isinstance(evidence, dict)
                        and evidence.get("source_episode_index") == row["source_episode_index"]
                    ):
                        if isinstance(evidence.get("proposal_id"), str):
                            proposal_ids.add(evidence["proposal_id"])
                        break
            attempt_rows: list[Mapping[str, Any]] = []
            for proposal_id in sorted(proposal_ids):
                row = connection.execute(
                    """
                    SELECT attempt.id AS attempt_id, attempt.job_id
                    FROM cosmos_proposals AS proposal
                    JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                    WHERE proposal.id=?
                    """,
                    (proposal_id,),
                ).fetchone()
                if row is None:
                    raise ExportError("proposal provenance is missing", {"error": "export_provenance_invalid"})
                attempt_rows.append(dict(row))
            attempt_ids = sorted({row["attempt_id"] for row in attempt_rows})
            job_ids = sorted({row["job_id"] for row in attempt_rows})
            workspace_artifacts: list[dict[str, Any]] = []
            for attempt_id in attempt_ids:
                for artifact in connection.execute(
                    "SELECT id, sha256 FROM artifacts WHERE attempt_id=? ORDER BY id", (attempt_id,)
                ):
                    workspace_artifacts.append({"artifact_id": artifact["id"], "sha256": artifact["sha256"]})
        return {
            "model": self.cosmos_model,
            "endpoint_identity": self.cosmos_endpoint_identity,
            "contract_version": CONTRACT_VERSION,
            "sampling": {
                "target_fps": TARGET_SAMPLING_FPS,
                "resize_max_long_edge": RESIZE_MAX_LONG_EDGE,
                "jpeg_quality": JPEG_QUALITY,
            },
            "limits": {
                "max_duration_s": MAX_DURATION_SECONDS,
                "max_frames": MAX_SAMPLED_FRAMES,
                "max_payload_bytes": MAX_PAYLOAD_BYTES,
            },
            "job_ids": job_ids,
            "attempt_ids": attempt_ids,
            "workspace_artifacts": sorted(workspace_artifacts, key=lambda row: row["artifact_id"]),
        }

    def _reconcile_publishing(self, export: Mapping[str, Any]) -> None:
        source = self._source_authority(export)
        episodes = self.database.list_export_episodes(export_id=export["id"])
        try:
            identity = PublicationIdentity.from_dict(
                _read_json(self._publication_identity_path(str(export["id"])))
            )
        except Exception as error:
            self.database.record_export_failure(
                export_id=export["id"],
                expected_state=ExportState.PUBLISHING,
                failure_summary="publish_identity_invalid",
                terminal=True,
            )
            raise ExportError(
                "publication identity requires operator inspection",
                {"error": "publish_identity_invalid"},
            ) from error

        def final_gate(path: Path) -> None:
            recorded_final_report = self._verify_recorded_gate_artifacts(
                export,
                path,
                ExportState.PUBLISHING,
            )
            provenance = _read_json(path / "meta/curation_provenance.json")
            repeated_final_report = validate_final_consistency(
                staging_path=path,
                provenance=provenance,
                source_roots=[source.root],
                stats_report_validator=lambda root: self._verify_recorded_gate_artifacts(
                    export, root, ExportState.PUBLISHING
                ),
                structural_validator=lambda: self._structural_report(
                    root=path, source=source, export=export, episodes=episodes
                ),
                expectations=self._final_expectations(
                    export=export,
                    source=source,
                    episodes=episodes,
                    staging=path,
                ),
            )
            if repeated_final_report != recorded_final_report:
                raise FinalConsistencyError("reconciled final-consistency result differs from its recorded report")

        try:
            result = reconcile_publication_paths(
                Path(export["staging_path"]),
                Path(export["final_path"]),
                final_gate=final_gate,
                parent_fsync=self.parent_fsync,
                commit_published=lambda: self.database.set_export_state(
                    export_id=export["id"],
                    expected_state=ExportState.PUBLISHING,
                    state=ExportState.PUBLISHED,
                ),
                return_to_validated=lambda: self.database.set_export_state(
                    export_id=export["id"],
                    expected_state=ExportState.PUBLISHING,
                    state=ExportState.FINAL_CONSISTENCY_VALIDATED,
                ),
                fail_operator=lambda code: self.database.record_export_failure(
                    export_id=export["id"],
                    expected_state=ExportState.PUBLISHING,
                    failure_summary=code,
                    terminal=True,
                ),
                expected_identity=identity,
            )
        except (ArtifactInstallConflict, FinalConsistencyError) as error:
            code = "publish_artifact_reconciliation_conflict"
            self.database.record_export_failure(
                export_id=export["id"],
                expected_state=ExportState.PUBLISHING,
                failure_summary=code,
                terminal=True,
            )
            raise ExportError("publication reconciliation failed", {"error": code}) from error
        except PublicationError as error:
            if error.code == "publish_parent_fsync_failed":
                self.database.record_export_failure(
                    export_id=export["id"],
                    expected_state=ExportState.PUBLISHING,
                    failure_summary=error.code,
                    terminal=False,
                )
            else:
                self.database.record_export_failure(
                    export_id=export["id"],
                    expected_state=ExportState.PUBLISHING,
                    failure_summary=error.code,
                    terminal=True,
                )
            raise ExportError("publication reconciliation failed", {"error": error.code}) from error
        if result == "failed":
            raise ExportError(
                "publication paths require operator inspection", {"error": "publish_reconciliation_failed"}
            )

    @staticmethod
    def _result(export: Mapping[str, Any]) -> dict[str, object]:
        return {
            "approval_snapshot_sha256": export["approval_snapshot_sha256"],
            "export_id": export["id"],
            "final_path": export["final_path"],
            "state": export["state"],
        }


def _copy_independent_regular_file(source: Path, destination: Path) -> None:
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("artifact source is not a regular file")
        contents = bytearray()
        offset = 0
        while chunk := os.pread(source_fd, 1024 * 1024, offset):
            contents.extend(chunk)
            offset += len(chunk)
        after = os.fstat(source_fd)
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ArtifactInstallConflict("contact-sheet source changed while copying")
    finally:
        os.close(source_fd)
    install_canonical_file(destination, bytes(contents))
    destination_stat = destination.stat(follow_symlinks=False)
    if (source_stat.st_dev, source_stat.st_ino) == (destination_stat.st_dev, destination_stat.st_ino):
        raise ValueError("artifact copy is a hardlink")


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
        "meta/stats.json",
        "meta/relative_stats.json",
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
        except ExportError:
            raise
        except OSError as error:
            raise ExportError("export executor lock is unsafe", {"error": "export_path_invalid"}) from error

        yield directory_fd
        try:
            current_directory_fd = _open_absolute_directory(lock_directory, create=False)
            try:
                current_stat = os.fstat(current_directory_fd)
            finally:
                os.close(current_directory_fd)
        except OSError as error:
            raise ExportError("export executor lock is unsafe", {"error": "export_path_invalid"}) from error
        if (current_stat.st_dev, current_stat.st_ino) != (
            directory_stat.st_dev,
            directory_stat.st_ino,
        ):
            raise ExportError("export control directory changed", {"error": "export_path_invalid"})
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
