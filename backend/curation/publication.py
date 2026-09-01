"""Linux-only atomic no-clobber publication and crash reconciliation."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import stat
import sys
from typing import Any
from uuid import uuid4

AT_FDCWD = -100
RENAME_NOREPLACE = 1


class PublicationError(RuntimeError):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class PublicationIdentity:
    parent_device: int
    parent_inode: int
    root_device: int
    root_inode: int

    def to_dict(self) -> dict[str, int]:
        return {
            "parent_device": self.parent_device,
            "parent_inode": self.parent_inode,
            "root_device": self.root_device,
            "root_inode": self.root_inode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PublicationIdentity:
        if set(value) != {"parent_device", "parent_inode", "root_device", "root_inode"} or any(
            type(value[key]) is not int or value[key] < 0 for key in value
        ):
            raise PublicationError("publish_identity_invalid")
        return cls(**value)


def _pin_child_directory(parent: Path, child_name: str) -> PublicationIdentity:
    parent_fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_result = os.fstat(parent_fd)
        child_fd = os.open(
            child_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            child_result = os.fstat(child_fd)
        finally:
            os.close(child_fd)
    except OSError as error:
        raise PublicationError("publish_source_identity_changed") from error
    finally:
        os.close(parent_fd)
    return PublicationIdentity(
        parent_device=parent_result.st_dev,
        parent_inode=parent_result.st_ino,
        root_device=child_result.st_dev,
        root_inode=child_result.st_ino,
    )


def pin_publication_source(staging_path: Path, final_path: Path) -> PublicationIdentity:
    staging = Path(os.path.abspath(staging_path))
    final = Path(os.path.abspath(final_path))
    if staging.parent != final.parent:
        raise PublicationError("publish_paths_not_same_parent")
    return _pin_child_directory(staging.parent, staging.name)


def _verify_identity(path: Path, expected: PublicationIdentity) -> None:
    actual = _pin_child_directory(path.parent, path.name)
    if actual != expected:
        raise PublicationError("publish_source_identity_changed")


def verify_publication_identity(path: Path, expected: PublicationIdentity) -> None:
    """Securely reopen both parent and child and compare the pinned identity."""

    _verify_identity(Path(os.path.abspath(path)), expected)


def _libc_renameat2(source: bytes, destination: bytes) -> None:
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "renameat2 is Linux-only")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.renameat2
    except AttributeError:
        raise OSError(errno.ENOSYS, "libc has no renameat2") from None
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(AT_FDCWD, source, AT_FDCWD, destination, RENAME_NOREPLACE) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))


def rename_noreplace(
    staging_path: Path,
    final_path: Path,
    *,
    syscall: Callable[[bytes, bytes], Any] = _libc_renameat2,
) -> None:
    source = os.fsencode(os.path.abspath(staging_path))
    destination = os.fsencode(os.path.abspath(final_path))
    try:
        syscall(source, destination)
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise PublicationError("publish_destination_exists") from error
        if error.errno in {
            errno.ENOSYS,
            errno.EINVAL,
            errno.EXDEV,
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            errno.EOPNOTSUPP,
        }:
            raise PublicationError("publish_noreplace_unsupported") from error
        raise PublicationError("publish_rename_failed") from error


def preflight_rename_noreplace(
    parent: Path,
    *,
    syscall: Callable[[bytes, bytes], Any] = _libc_renameat2,
) -> None:
    """Prove this parent filesystem supports true renameat2 NOREPLACE."""

    directory = Path(parent).resolve(strict=True)
    parent_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    source_name = f".curation-noreplace-source-{uuid4().hex}"
    destination_name = f".curation-noreplace-destination-{uuid4().hex}"
    source = directory / source_name
    destination = directory / destination_name
    try:
        os.mkdir(source_name, mode=0o700, dir_fd=parent_fd)
        os.mkdir(destination_name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            rename_noreplace(source, destination, syscall=syscall)
        except PublicationError as error:
            if error.code != "publish_destination_exists":
                raise
        else:
            raise PublicationError("publish_noreplace_unsupported")
        os.rmdir(destination_name, dir_fd=parent_fd)
        rename_noreplace(source, destination, syscall=syscall)
        if source.exists() or not destination.is_dir() or destination.is_symlink():
            raise PublicationError("publish_noreplace_unsupported")
    finally:
        for name in (source_name, destination_name):
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.fsync(parent_fd)
        os.close(parent_fd)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.ENOTDIR, "not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_no_clobber(
    staging_path: Path,
    final_path: Path,
    *,
    expected_identity: PublicationIdentity,
    parent_fsync: Callable[[Path], None] = fsync_directory,
    commit_published: Callable[[], None] = lambda: None,
    before_rename: Callable[[], None] = lambda: None,
    after_rename: Callable[[], None] = lambda: None,
    renamer: Callable[[Path, Path], None] = rename_noreplace,
) -> None:
    staging = Path(staging_path)
    final = Path(final_path)
    if staging.parent != final.parent:
        raise PublicationError("publish_paths_not_same_parent")
    before_rename()
    _verify_identity(staging, expected_identity)
    renamer(staging, final)
    after_rename()
    _verify_identity(final, expected_identity)
    try:
        parent_fsync(final.parent)
    except OSError as error:
        raise PublicationError("publish_parent_fsync_failed") from error
    _verify_identity(final, expected_identity)
    commit_published()


def _present_directory(path: Path) -> bool:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
        raise PublicationError("publish_path_unsafe")
    return True


def reconcile_publication_paths(
    staging_path: Path,
    final_path: Path,
    *,
    final_gate: Callable[[Path], Any],
    parent_fsync: Callable[[Path], None],
    commit_published: Callable[[], None],
    return_to_validated: Callable[[], None],
    fail_operator: Callable[[str], None],
    expected_identity: PublicationIdentity | None = None,
) -> str:
    staging = Path(staging_path)
    final = Path(final_path)
    staging_present = _present_directory(staging)
    final_present = _present_directory(final)
    if staging_present and not final_present:
        if expected_identity is not None:
            _verify_identity(staging, expected_identity)
        return_to_validated()
        return "final_consistency_validated"
    if not staging_present and final_present:
        if expected_identity is not None:
            _verify_identity(final, expected_identity)
        final_gate(final)
        if expected_identity is not None:
            _verify_identity(final, expected_identity)
        try:
            parent_fsync(final.parent)
        except OSError as error:
            raise PublicationError("publish_parent_fsync_failed") from error
        if expected_identity is not None:
            _verify_identity(final, expected_identity)
        commit_published()
        return "published"
    code = "publish_paths_both_present" if staging_present else "publish_paths_both_absent"
    fail_operator(code)
    return "failed"
