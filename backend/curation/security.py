from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


def safe_relative_asset_path(asset_path: str) -> Path | None:
    """Return a normalized relative path, rejecting encoded and raw traversal."""
    decoded = unquote(asset_path)
    candidate = Path(decoded)
    if not decoded or candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def contained_regular_file(root: Path, relative: Path) -> Path | None:
    """Resolve a registered asset only when it is a regular file under root."""
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return candidate
