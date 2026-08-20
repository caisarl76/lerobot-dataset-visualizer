from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .config import CurationConfigurationError
from .security import contained_regular_file, safe_relative_asset_path


@dataclass(frozen=True)
class SourceRecord:
    alias: str
    root: Path
    manifest_path: Path
    fingerprint: str
    file_hashes: Mapping[str, str]

    def resolve_asset(self, asset_path: str) -> tuple[Path, str] | None:
        relative = safe_relative_asset_path(asset_path)
        if relative is None:
            return None
        candidate = contained_regular_file(self.root, relative)
        if candidate is None:
            return None
        key = relative.as_posix()
        digest = self.file_hashes.get(key)
        if digest is None:
            return None
        return candidate, digest


def _manifest_bytes(root: Path) -> tuple[bytes, dict[str, str]]:
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
    for _, relative, path in sorted(entries, key=lambda item: item[0]):
        digest = _sha256_file(path)
        hashes[relative] = digest
        lines.append(f"{digest}  {relative}\n".encode("utf-8"))
    return b"".join(lines), hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
            manifest, hashes = _manifest_bytes(root)
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
            if manifest_path.exists() and manifest_path.read_bytes() != manifest:
                raise CurationConfigurationError(f"source manifest changed for registered alias {alias}")
            if not manifest_path.exists():
                manifest_path.write_bytes(manifest)
            records[alias] = SourceRecord(
                alias=alias,
                root=root,
                manifest_path=manifest_path,
                fingerprint=hashlib.sha256(manifest).hexdigest(),
                file_hashes=MappingProxyType(hashes),
            )
        return cls(records)

    def resolve_alias(self, org: str, dataset: str) -> SourceRecord | None:
        return self._records.get(f"{org}/{dataset}")
