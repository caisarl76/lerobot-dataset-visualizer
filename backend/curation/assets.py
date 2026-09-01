from __future__ import annotations

from collections.abc import Iterable, Mapping
import mimetypes
import os
from pathlib import Path

from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse

from .security import OpenedAsset
from .source import SourceRegistry

_CHUNK_SIZE = 64 * 1024


def _mime_type(path: Path) -> str:
    if path.suffix == ".parquet":
        return "application/vnd.apache.parquet"
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _range_interval(value: str, size: int) -> tuple[int, int] | None:
    if size <= 0 or not value.startswith("bytes=") or "," in value:
        return None
    specification = value[6:]
    if specification.count("-") != 1:
        return None
    first, last = specification.split("-", 1)
    if not first and not last:
        return None
    if first and (not first.isascii() or not first.isdecimal()):
        return None
    if last and (not last.isascii() or not last.isdecimal()):
        return None
    try:
        if not first:
            suffix = int(last)
            if suffix <= 0:
                return None
            return max(0, size - suffix), size - 1
        start = int(first)
        if start < 0 or start >= size:
            return None
        if not last:
            return start, size - 1
        end = int(last)
        if end < start:
            return None
        return start, min(end, size - 1)
    except ValueError:
        return None


def _read_interval(asset: OpenedAsset, start: int, length: int) -> Iterable[bytes]:
    try:
        remaining = length
        while remaining:
            chunk = os.pread(asset.fd, min(_CHUNK_SIZE, remaining), start + length - remaining)
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        asset.close()


class LocalAssetService:
    def __init__(self, registry: SourceRegistry):
        self.registry = registry

    def serve(
        self,
        org: str,
        dataset: str,
        revision: str,
        asset_path: str,
        *,
        method: str,
        headers: Mapping[str, str],
    ) -> Response:
        normalized_headers = {key.lower(): value for key, value in headers.items()}
        if "authorization" in normalized_headers:
            return Response(status_code=400)
        if method not in {"GET", "HEAD"} or revision != "main":
            return Response(status_code=404)
        record = self.registry.resolve_alias(org, dataset)
        asset = record.open_asset(asset_path) if record else None
        if asset is None:
            return Response(status_code=404)
        size = asset.size
        digest = record.file_hashes[asset.relative_path.as_posix()]
        etag = f'"{digest}"'
        relative_parts = asset.relative_path.parts
        cache_control = "no-store" if relative_parts[0] == "meta" else "private, max-age=0, must-revalidate"
        base_headers = {
            "Accept-Ranges": "bytes",
            "ETag": etag,
            "Content-Type": _mime_type(asset.relative_path),
            "Cache-Control": cache_control,
        }
        range_value = normalized_headers.get("range")
        if range_value is None and _if_none_match(normalized_headers.get("if-none-match"), etag):
            asset.close()
            return Response(status_code=304, headers=base_headers)
        if range_value is not None and normalized_headers.get("if-range") not in {None, etag}:
            range_value = None
        if range_value is not None:
            interval = _range_interval(range_value, size)
            if interval is None:
                asset.close()
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            start, end = interval
            length = end - start + 1
            range_headers = {
                **base_headers,
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(length),
            }
            if method == "HEAD":
                asset.close()
                return Response(status_code=206, headers=range_headers)
            return StreamingResponse(
                _read_interval(asset, start, length),
                status_code=206,
                headers=range_headers,
                background=BackgroundTask(asset.close),
            )
        full_headers = {**base_headers, "Content-Length": str(size)}
        if method == "HEAD":
            asset.close()
            return Response(status_code=200, headers=full_headers)
        return StreamingResponse(
            _read_interval(asset, 0, size),
            status_code=200,
            headers=full_headers,
            background=BackgroundTask(asset.close),
        )


def _if_none_match(value: str | None, current_etag: str) -> bool:
    if value is None:
        return False
    for candidate in value.split(","):
        candidate = candidate.strip()
        if candidate == "*":
            return True
        if candidate.startswith("W/"):
            candidate = candidate[2:].strip()
        if candidate == current_etag:
            return True
    return False
