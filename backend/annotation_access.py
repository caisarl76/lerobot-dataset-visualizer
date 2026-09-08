"""Authenticated service boundary for private-Space annotation deployments."""

import hmac
import json
import os
from pathlib import Path
import re
from string import Formatter

from starlette.responses import JSONResponse


def validate_dataset_paths(root: Path) -> None:
    """Check untrusted Hub metadata before upstream readers/converters use it."""
    root = root.resolve()

    def contained(value: str) -> None:
        path = Path(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or "\x00" in value
            or not (root / path).resolve().is_relative_to(root)
        ):
            raise ValueError("Dataset path escapes its root")

    # copytree follows links, including metadata unrelated to the video template.
    for directory in ("meta", "data", "videos"):
        contained(directory)
        for parent, dirs, files in os.walk(root / directory, followlinks=False):
            for name in dirs + files:
                path = Path(parent) / name
                contained(str(path.relative_to(root)))
                if path.is_symlink() and path.is_dir():
                    raise ValueError("Dataset path must not contain directory symlinks")

    info = json.loads((root / "meta/info.json").read_text())
    cameras = [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]
    for key in cameras:
        if not key or key in {".", ".."} or any(char in key for char in "/\\{}\x00"):
            raise ValueError("Camera key must be a single safe path component")

    fields = {"episode_index", "episode_chunk", "chunk_index", "file_index", "video_key"}
    templates = {key: info[key] for key in ("data_path", "video_path") if info.get(key) is not None}
    for template in templates.values():
        if not isinstance(template, str) or not template:
            raise ValueError("Dataset path template must be a nonempty string")
        contained(template)
        for _, field, spec, conversion in Formatter().parse(template):
            if field is not None and (
                field not in fields or conversion or (spec and not re.fullmatch(r"0?[1-9][0-9]?d|d", spec))
            ):
                raise ValueError("Unsupported dataset path template")

    def check_record(row: dict) -> None:
        def number(key: str, default: int = 0) -> int:
            value = row.get(key, default)
            try:
                result = int(value)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("Dataset path index must be an integer") from exc
            if result < 0 or result != value:
                raise ValueError("Dataset path index must be a nonnegative integer")
            return result

        episode = number("episode_index")
        chunk_size = int(info.get("chunks_size", 1000))
        if chunk_size <= 0:
            raise ValueError("Dataset path chunk size must be positive")
        values = {"episode_index": episode, "episode_chunk": episode // chunk_size, "video_key": "camera"}
        for kind, template in templates.items():
            for camera in (cameras or ["camera"]) if kind == "video_path" else [None]:
                prefix = f"videos/{camera}" if camera is not None else "data"
                values.update(
                    chunk_index=number(prefix + "/chunk_index"), file_index=number(prefix + "/file_index")
                )
                if camera is not None:
                    values["video_key"] = camera
                contained(template.format(**values))

    check_record({})
    legacy = root / "meta/episodes.jsonl"
    if legacy.exists():
        with legacy.open() as stream:
            for line in stream:
                if line.strip():
                    check_record(json.loads(line))
    import pyarrow.parquet as pq

    for path in (root / "meta/episodes").rglob("*.parquet"):
        names = pq.read_schema(path).names
        columns = [
            name for name in names if name == "episode_index" or name.endswith(("/chunk_index", "/file_index"))
        ]
        for row in pq.read_table(path, columns=columns).to_pylist():
            check_record(row)


def forbidden(value):
    if isinstance(value, dict):
        return any(
            (
                key in {"local_path", "output_dir", "api_key", "hf_token", "api_base", "serve_command"}
                and val is not None
            )
            or forbidden(val)
            for key, val in value.items()
        )
    return isinstance(value, list) and any(forbidden(v) for v in value)


class AnnotationAccess:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        token = os.environ.get("ANNOTATION_BACKEND_TOKEN")
        if scope["type"] != "http" or not token:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if not hmac.compare_digest(headers.get(b"authorization", b""), ("Bearer " + token).encode()):
            return await JSONResponse({"detail": "Unauthorized"}, 401)(scope, receive, send)
        path, method = scope["path"], scope["method"]
        allowed = (
            (
                method in {"GET", "HEAD"}
                and re.fullmatch(r"/datasets/local/[A-Za-z0-9_-]+/resolve/main/(meta|data|videos)/.+", path)
            )
            or (
                method == "GET"
                and (
                    path in {"/api/health", "/api/annotation/config"}
                    or re.fullmatch(r"/api/annotation/jobs/[a-f0-9]{32}", path)
                )
            )
            or (
                method == "POST"
                and path
                in {
                    "/api/dataset/load",
                    "/api/annotation/prepare",
                    "/api/annotation/jobs",
                    "/api/annotation/validate",
                }
            )
            or (method in {"GET", "POST"} and re.fullmatch(r"/api/episodes/\d+/(atoms|review)", path))
            or (method == "GET" and re.fullmatch(r"/api/episodes/\d+/frame_timestamps", path))
            or (method == "GET" and re.fullmatch(r"/api/workflow/[A-Za-z0-9_-]+", path))
            or (method == "POST" and re.fullmatch(r"/api/workflow/[A-Za-z0-9_-]+/(decision|export|publish)", path))
        )
        if not allowed:
            return await JSONResponse({"detail": "Route unavailable in hosted mode"}, 403)(scope, receive, send)
        from urllib.parse import parse_qs

        if parse_qs(scope.get("query_string", b"").decode()).get("local_path"):
            return await JSONResponse({"detail": "Local paths unavailable"}, 403)(scope, receive, send)
        if method == "POST":
            body = b""
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                body += event.get("body", b"")
                if len(body) > 1048576:
                    return await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)
                if not event.get("more_body"):
                    break
            try:
                payload = json.loads(body)
            except ValueError:
                return await JSONResponse({"detail": "Invalid JSON"}, 422)(scope, receive, send)
            if forbidden(payload):
                return await JSONResponse({"detail": "Server-only configuration"}, 403)(scope, receive, send)
            sent = False

            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)
