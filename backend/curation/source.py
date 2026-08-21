from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Iterator, Mapping

from .config import CurationConfigurationError
from .security import OpenedAsset, SourceFileIdentity, open_regular_file_beneath, safe_relative_asset_path


@dataclass(frozen=True)
class SourceRecord:
    alias: str
    root: Path
    manifest_path: Path
    fingerprint: str
    file_hashes: Mapping[str, str]
    file_identities: Mapping[str, SourceFileIdentity]

    def open_asset(self, asset_path: str) -> OpenedAsset | None:
        relative = safe_relative_asset_path(asset_path)
        if relative is None:
            return None
        key = relative.as_posix()
        identity = self.file_identities.get(key)
        if identity is None:
            return None
        return open_regular_file_beneath(self.root, relative, identity)


def _manifest_bytes(root: Path) -> tuple[bytes, dict[str, str], dict[str, SourceFileIdentity]]:
    entries: list[tuple[bytes, str, Path]] = []
    for directory, _, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if "\n" in relative or "\r" in relative:
                raise CurationConfigurationError("source file paths must not contain CR or LF")
            try:
                encoded = relative.encode("utf-8")
            except UnicodeEncodeError as error:
                raise CurationConfigurationError("source paths must be UTF-8") from error
            entries.append((encoded, relative, path))
    lines: list[bytes] = []
    hashes: dict[str, str] = {}
    identities: dict[str, SourceFileIdentity] = {}
    for _, relative, path in sorted(entries, key=lambda item: item[0]):
        digest, identity = _sha256_file(path)
        hashes[relative] = digest
        identities[relative] = identity
        lines.append(f"{digest}  {relative}\n".encode("utf-8"))
    return b"".join(lines), hashes, identities


def _sha256_file(path: Path) -> tuple[str, SourceFileIdentity]:
    digest = hashlib.sha256()
    fd = -1
    try:
        fd = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CurationConfigurationError(f"source file is not regular: {path}")
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        if SourceFileIdentity.from_stat(before) != SourceFileIdentity.from_stat(after):
            raise CurationConfigurationError(f"source file changed while hashing: {path}")
        return digest.hexdigest(), SourceFileIdentity.from_stat(after)
    finally:
        if fd >= 0:
            os.close(fd)


@contextmanager
def _manifest_lock(manifest_path: Path) -> Iterator[None]:
    parent_fd = -1
    lock_fd = -1
    try:
        try:
            parent_fd = os.open(
                manifest_path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
                raise CurationConfigurationError("manifest parent must be a directory")
            lock_fd = os.open(
                f"{manifest_path.name}.lock",
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                0o600,
                dir_fd=parent_fd,
            )
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise CurationConfigurationError("manifest lock path must be a regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError as error:
            raise CurationConfigurationError("manifest lock path must be a regular file") from error
        yield
    finally:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _install_manifest_no_clobber(temporary_path: Path, manifest_path: Path) -> None:
    os.link(temporary_path, manifest_path)


def _read_existing_manifest(manifest_path: Path) -> bytes:
    try:
        descriptor = os.open(
            manifest_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise CurationConfigurationError("manifest path must be a regular file") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CurationConfigurationError("manifest path must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _manifest_exists(manifest_path: Path) -> bool:
    try:
        info = os.lstat(manifest_path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise CurationConfigurationError("manifest path must be a regular file")
    return True


def _persist_manifest(manifest_path: Path, contents: bytes) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock(manifest_path):
        for stale in manifest_path.parent.glob(f".{manifest_path.name}.*.tmp"):
            stale.unlink(missing_ok=True)
        if _manifest_exists(manifest_path):
            if _read_existing_manifest(manifest_path) != contents:
                raise CurationConfigurationError("source manifest changed for registered alias")
            _fsync_directory(manifest_path.parent)
            return
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(temporary_fd, "wb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                _install_manifest_no_clobber(temporary_path, manifest_path)
            except FileExistsError:
                if _read_existing_manifest(manifest_path) != contents:
                    raise CurationConfigurationError("source manifest changed for registered alias")
                _fsync_directory(manifest_path.parent)
            else:
                _fsync_directory(manifest_path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)


class SourceRegistry:
    """Immutable startup registry; browser aliases never carry source paths."""

    def __init__(self, records: Mapping[str, SourceRecord]):
        self._records = MappingProxyType(dict(records))

    @property
    def records(self) -> Mapping[str, SourceRecord]:
        return self._records

    @classmethod
    def from_paths(cls, aliases: Mapping[str, Path], *, workspace: Path) -> "SourceRegistry":
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        records: dict[str, SourceRecord] = {}
        for alias, supplied_root in aliases.items():
            root = supplied_root.resolve(strict=True)
            if not root.is_dir():
                raise CurationConfigurationError(f"source alias {alias} is not a directory")
            manifest, hashes, identities = _manifest_bytes(root)
            if len(aliases) == 1:
                manifest_path = workspace / "source-files.sha256"
            else:
                manifest_path = (
                    workspace
                    / "source-manifests"
                    / hashlib.sha256(alias.encode()).hexdigest()
                    / "source-files.sha256"
                )
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
            _persist_manifest(manifest_path, manifest)
            records[alias] = SourceRecord(
                alias=alias,
                root=root,
                manifest_path=manifest_path,
                fingerprint=hashlib.sha256(manifest).hexdigest(),
                file_hashes=MappingProxyType(hashes),
                file_identities=MappingProxyType(identities),
            )
        return cls(records)

    def resolve_alias(self, org: str, dataset: str) -> SourceRecord | None:
        return self._records.get(f"{org}/{dataset}")
