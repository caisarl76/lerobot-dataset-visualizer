from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import stat
from typing import Any, Awaitable, Callable
from urllib.parse import unquote

from starlette.responses import Response


@dataclass
class OpenedAsset:
    """A regular source file pinned by an open descriptor until response completion."""

    fd: int
    stat_result: os.stat_result
    relative_path: Path
    closed: bool = False

    @property
    def size(self) -> int:
        return self.stat_result.st_size

    def close(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


@dataclass(frozen=True)
class SourceFileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "SourceFileIdentity":
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def matches(self, value: os.stat_result) -> bool:
        return self == self.from_stat(value)


def safe_relative_asset_path(asset_path: str) -> Path | None:
    """Return a normalized relative path, rejecting encoded and raw traversal."""
    decoded = unquote(asset_path)
    candidate = Path(decoded)
    if not decoded or candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _open_flags(*, directory: bool, nonblocking: bool = False) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    if nonblocking:
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def open_regular_file_beneath(
    root: Path, relative: Path, expected_identity: SourceFileIdentity
) -> OpenedAsset | None:
    """Open an immutable registered file without following any path symlink.

    Every component is resolved by directory descriptor, then the final file
    identity is compared with the one measured while building the manifest.
    The caller owns the returned descriptor.
    """
    directory_fd = -1
    file_fd = -1
    try:
        directory_fd = os.open(root, _open_flags(directory=True))
        parts = relative.parts
        if not parts:
            return None
        for component in parts[:-1]:
            next_fd = os.open(component, _open_flags(directory=True), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], _open_flags(directory=False, nonblocking=True), dir_fd=directory_fd)
        result = os.fstat(file_fd)
        if not stat.S_ISREG(result.st_mode) or not expected_identity.matches(result):
            os.close(file_fd)
            file_fd = -1
            return None
        opened = OpenedAsset(file_fd, result, relative)
        file_fd = -1
        return opened
    except OSError:
        return None
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if file_fd >= 0:
            os.close(file_fd)


def _is_loopback_host(host: Any) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class CurationLoopbackGuard:
    """Reject curation paths if the ASGI server is not bound to loopback."""

    _PREFIXES = ("/api/local-datasets/", "/api/curation/")

    def __init__(self, app: Callable[..., Awaitable[None]]):
        self.app = app

    async def __call__(
        self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[None]]
    ) -> None:
        if scope.get("type") == "http" and scope.get("path", "").startswith(self._PREFIXES):
            server = scope.get("server")
            host = server[0] if isinstance(server, (list, tuple)) and server else None
            if not _is_loopback_host(host):
                await Response(status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
