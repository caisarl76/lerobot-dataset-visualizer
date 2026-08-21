from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from curation.assets import LocalAssetService, _read_interval
from curation.config import _REQUIRED, legacy_browser_origin
from curation.security import CurationLoopbackGuard
import curation.source as source_module
from curation.source import SourceRegistry
from fastapi.testclient import TestClient
import pytest

_ASSET_ROUTE = "/api/local-datasets/{org}/{dataset}/resolve/{revision}/{path}"


def _route_path(path: str, *, org: str = "local", dataset: str = "pnp_trash", revision: str = "main") -> str:
    return _ASSET_ROUTE.format(org=org, dataset=dataset, revision=revision, path=path)


@pytest.fixture
def local_assets(tmp_path: Path) -> tuple[LocalAssetService, Path, Path]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "episodes").mkdir()
    (source / "data").mkdir()
    (source / "videos").mkdir()
    (source / "meta" / "info.json").write_bytes(b'{"fps": 50}')
    (source / "meta" / "episodes" / "file-000.parquet").write_bytes(b"episode metadata")
    (source / "data" / "episode.parquet").write_bytes(b"abcdefghij")
    (source / "videos" / "episode.mp4").write_bytes(b"0123456789")
    (source / "not-a-file").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (source / "escaping-link").symlink_to(outside)
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "workspace")
    return LocalAssetService(registry), source, outside


def test_manifest_is_deterministic_and_lives_only_in_workspace(
    local_assets: tuple[LocalAssetService, Path, Path],
) -> None:
    service, source, _ = local_assets
    record = service.registry.resolve_alias("local", "pnp_trash")

    assert record.manifest_path.parent != source
    assert record.manifest_path.name == "source-files.sha256"
    assert record.fingerprint == __import__("hashlib").sha256(record.manifest_path.read_bytes()).hexdigest()
    assert "meta/info.json" in record.manifest_path.read_text()


def test_manifest_rejects_source_paths_with_line_breaks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "unsafe\nname").write_text("no")

    with pytest.raises(ValueError, match="CR or LF"):
        SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "workspace")


def test_asset_full_head_ranges_and_revalidation(local_assets: tuple[LocalAssetService, Path, Path]) -> None:
    service, _, _ = local_assets
    full = service.serve("local", "pnp_trash", "main", "data/episode.parquet", method="HEAD", headers={})
    assert full.status_code == 200
    assert full.headers["content-length"] == "10"
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["cache-control"] == "private, max-age=0, must-revalidate"
    assert full.headers["content-type"].startswith("application/vnd.apache.parquet")
    assert "content-encoding" not in full.headers
    etag = full.headers["etag"]

    head = service.serve("local", "pnp_trash", "main", "data/episode.parquet", method="HEAD", headers={})
    assert head.status_code == 200
    assert head.headers["content-length"] == "10"

    for header, expected, content_range in [
        ("bytes=2-4", b"cde", "bytes 2-4/10"),
        ("bytes=7-", b"hij", "bytes 7-9/10"),
        ("bytes=-2", b"ij", "bytes 8-9/10"),
    ]:
        ranged = service.serve(
            "local",
            "pnp_trash",
            "main",
            "data/episode.parquet",
            method="HEAD",
            headers={"range": header},
        )
        assert ranged.status_code == 206
        assert ranged.headers["content-range"] == content_range
        assert ranged.headers["content-length"] == str(len(expected))

    record = service.registry.resolve_alias("local", "pnp_trash")
    full_asset = record.open_asset("data/episode.parquet")
    ranged_asset = record.open_asset("data/episode.parquet")
    assert full_asset is not None and ranged_asset is not None
    assert b"".join(_read_interval(full_asset, 0, 10)) == b"abcdefghij"
    assert b"".join(_read_interval(ranged_asset, 2, 3)) == b"cde"

    not_modified = service.serve(
        "local",
        "pnp_trash",
        "main",
        "data/episode.parquet",
        method="HEAD",
        headers={"if-none-match": etag},
    )
    assert not_modified.status_code == 304
    fallback = service.serve(
        "local",
        "pnp_trash",
        "main",
        "data/episode.parquet",
        method="HEAD",
        headers={"range": "bytes=0-1", "if-range": '"different"'},
    )
    assert fallback.status_code == 200
    matching = service.serve(
        "local",
        "pnp_trash",
        "main",
        "data/episode.parquet",
        method="HEAD",
        headers={"range": "bytes=0-1", "if-range": etag},
    )
    assert matching.status_code == 206

    video = service.serve("local", "pnp_trash", "main", "videos/episode.mp4", method="HEAD", headers={})
    assert video.headers["cache-control"] == "private, max-age=0, must-revalidate"
    assert video.headers["content-type"].startswith("video/mp4")


@pytest.mark.parametrize(
    "revision,path,headers,status",
    [
        ("other", "data/episode.parquet", {}, 404),
        ("main", "/etc/passwd", {}, 404),
        ("main", "../outside.txt", {}, 404),
        ("main", "%2e%2e/outside.txt", {}, 404),
        ("main", "escaping-link", {}, 404),
        ("main", "not-a-file", {}, 404),
        ("main", "data/episode.parquet", {"authorization": "Bearer never"}, 400),
        ("main", "data/episode.parquet", {"range": "bytes=0-1,3-4"}, 416),
        ("main", "data/episode.parquet", {"range": "letters=0-1"}, 416),
        ("main", "data/episode.parquet", {"range": "bytes=20-30"}, 416),
        ("main", "data/episode.parquet", {"range": "bytes=+1-2"}, 416),
        ("main", "data/episode.parquet", {"range": "bytes= 1-2"}, 416),
        ("main", "data/episode.parquet", {"range": "bytes=1-2 "}, 416),
        ("main", "data/episode.parquet", {"range": "bytes=١-2"}, 416),
    ],
)
def test_asset_rejects_unsafe_requests(
    local_assets: tuple[LocalAssetService, Path, Path],
    revision: str,
    path: str,
    headers: dict[str, str],
    status: int,
) -> None:
    service, _, _ = local_assets
    response = service.serve("local", "pnp_trash", revision, path, method="GET", headers=headers)
    assert response.status_code == status
    if status == 416:
        assert response.headers["content-range"] == "bytes */10"


def test_metadata_is_no_store_and_unknown_alias_is_not_exposed(
    local_assets: tuple[LocalAssetService, Path, Path],
) -> None:
    service, _, _ = local_assets
    metadata = service.serve("local", "pnp_trash", "main", "meta/info.json", method="GET", headers={})
    assert metadata.status_code == 200
    assert metadata.headers["cache-control"] == "no-store"
    assert metadata.headers["content-type"].startswith("application/json")
    episode_metadata = service.serve(
        "local", "pnp_trash", "main", "meta/episodes/file-000.parquet", method="GET", headers={}
    )
    assert episode_metadata.headers["cache-control"] == "no-store"
    unknown = service.serve("local", "not_registered", "main", "meta/info.json", method="GET", headers={})
    assert unknown.status_code == 404


@pytest.fixture
def configured_curation_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Load an isolated app module so curation config cannot touch legacy state."""
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "videos").mkdir()
    (source / "meta" / "info.json").write_text(json.dumps({"fps": 50}))
    (source / "meta" / "tasks.jsonl").write_bytes(b'{"task_index": 0}\n')
    (source / "data" / "episode.parquet").write_bytes(b"abcdefghij")
    (source / "videos" / "episode.mp4").write_bytes(b"0123456789")
    (source / "not-a-file").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (source / "escaping-link").symlink_to(outside)
    isaac = tmp_path / "isaac"
    isaac.mkdir()
    environment = {
        "CURATION_DATASET_ALIASES_JSON": json.dumps({"local/pnp_trash": str(source)}),
        "CURATION_WORKSPACE": str(tmp_path / "workspace"),
        "CURATION_OUTPUT": str(tmp_path / "output"),
        "CURATION_BROWSER_ORIGIN": "http://127.0.0.1:3000",
        "CURATION_BEARER_TOKEN": "test-token",
        "COSMOS_BASE_URL": "http://127.0.0.1:8001/v1",
        "COSMOS_MODEL": "cosmos3-nano",
        "COSMOS_API_KEY_ENV": "COSMOS_API_KEY",
        "COSMOS_ENDPOINT_IDENTITY": "h100-cosmos",
        "ISAAC_GROOT_ROOT": str(isaac),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    module_name = f"curation_route_test_{uuid4().hex}"
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            yield client
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    "asset_path,expected_body,mime_type,cache_control",
    [
        ("meta/info.json", b'{"fps": 50}', "application/json", "no-store"),
        ("meta/tasks.jsonl", b'{"task_index": 0}\n', "application/x-ndjson", "no-store"),
        (
            "data/episode.parquet",
            b"abcdefghij",
            "application/vnd.apache.parquet",
            "private, max-age=0, must-revalidate",
        ),
        ("videos/episode.mp4", b"0123456789", "video/mp4", "private, max-age=0, must-revalidate"),
    ],
)
def test_local_asset_route_get_and_head_headers_for_every_browser_asset_type(
    configured_curation_client: TestClient,
    asset_path: str,
    expected_body: bytes,
    mime_type: str,
    cache_control: str,
) -> None:
    response = configured_curation_client.get(_route_path(asset_path))
    assert response.status_code == 200
    assert response.content == expected_body
    assert response.headers["content-length"] == str(len(expected_body))
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-type"].startswith(mime_type)
    assert response.headers["cache-control"] == cache_control
    assert response.headers["etag"]
    assert "content-encoding" not in response.headers

    head = configured_curation_client.head(_route_path(asset_path))
    assert head.status_code == 200
    assert head.content == b""
    for name in ("content-length", "accept-ranges", "content-type", "cache-control", "etag"):
        assert head.headers[name] == response.headers[name]
    assert "content-encoding" not in head.headers


@pytest.mark.parametrize(
    "range_value,expected_body,content_range",
    [
        ("bytes=2-4", b"cde", "bytes 2-4/10"),
        ("bytes=7-", b"hij", "bytes 7-9/10"),
        ("bytes=-2", b"ij", "bytes 8-9/10"),
    ],
)
def test_local_asset_route_supports_all_single_range_forms(
    configured_curation_client: TestClient,
    range_value: str,
    expected_body: bytes,
    content_range: str,
) -> None:
    response = configured_curation_client.get(_route_path("data/episode.parquet"), headers={"Range": range_value})
    assert response.status_code == 206
    assert response.content == expected_body
    assert response.headers["content-range"] == content_range
    assert response.headers["content-length"] == str(len(expected_body))


def test_local_asset_route_revalidates_etags_and_honors_if_range(
    configured_curation_client: TestClient,
) -> None:
    path = _route_path("data/episode.parquet")
    etag = configured_curation_client.get(path).headers["etag"]

    for if_none_match in (etag, "*", f"W/{etag}", f'"other", {etag}'):
        not_modified = configured_curation_client.get(path, headers={"If-None-Match": if_none_match})
        assert not_modified.status_code == 304
        assert not_modified.content == b""
    matching_range = configured_curation_client.get(path, headers={"Range": "bytes=0-1", "If-Range": etag})
    assert matching_range.status_code == 206
    assert matching_range.content == b"ab"
    mismatched_range = configured_curation_client.get(
        path, headers={"Range": "bytes=0-1", "If-Range": '"not-the-etag"'}
    )
    assert mismatched_range.status_code == 200
    assert mismatched_range.content == b"abcdefghij"


@pytest.mark.parametrize(
    "range_value",
    [
        "bytes=0-1,3-4",
        "letters=0-1",
        "bytes=20-30",
        "bytes=+1-2",
        "bytes= 1-2",
        "bytes=1-2 ",
        b"bytes=\xd9\xa1-2",  # Raw UTF-8: httpx refuses Unicode header strings before ASGI.
    ],
)
def test_local_asset_route_rejects_malformed_or_multiple_ranges(
    configured_curation_client: TestClient, range_value: str | bytes
) -> None:
    response = configured_curation_client.get(_route_path("data/episode.parquet"), headers={"Range": range_value})
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */10"
    assert response.content == b""


def test_local_asset_route_decoding_auth_and_cors_are_fail_closed(
    configured_curation_client: TestClient,
) -> None:
    metadata_path = _route_path("meta/info.json")
    allowed = configured_curation_client.get(metadata_path, headers={"Origin": "http://127.0.0.1:3000"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"
    exposed = allowed.headers["access-control-expose-headers"].lower()
    assert {"accept-ranges", "content-range", "content-length", "etag"} <= set(
        header.strip() for header in exposed.split(",")
    )
    assert allowed.headers["cache-control"] == "no-store"
    encoded_metadata = configured_curation_client.get(_route_path("meta%2finfo.json"))
    assert encoded_metadata.status_code == 200
    assert encoded_metadata.headers["cache-control"] == "no-store"
    assert (
        configured_curation_client.get(metadata_path, headers={"Origin": "http://evil.test"}).headers.get(
            "access-control-allow-origin"
        )
        is None
    )
    assert (
        configured_curation_client.get(metadata_path, headers={"Authorization": "Bearer hf-token"}).status_code
        == 400
    )
    assert configured_curation_client.get(_route_path("%2e%2e/outside.txt")).status_code == 404


@pytest.mark.parametrize(
    "org,dataset,revision,asset_path",
    [
        ("not_registered", "pnp_trash", "main", "meta/info.json"),
        ("local", "pnp_trash", "other", "meta/info.json"),
        ("local", "pnp_trash", "main", "/etc/passwd"),
        ("local", "pnp_trash", "main", "../outside.txt"),
        ("local", "pnp_trash", "main", "%2e%2e/outside.txt"),
        ("local", "pnp_trash", "main", "escaping-link"),
        ("local", "pnp_trash", "main", "not-a-file"),
    ],
)
def test_local_asset_route_rejects_unregistered_or_unsafe_targets(
    configured_curation_client: TestClient,
    org: str,
    dataset: str,
    revision: str,
    asset_path: str,
) -> None:
    response = configured_curation_client.get(_route_path(asset_path, org=org, dataset=dataset, revision=revision))
    assert response.status_code == 404


@pytest.fixture
def legacy_cors_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Load an isolated legacy app, with all curation configuration absent."""
    for name in _REQUIRED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LEROBOT_ANNOTATE_BROWSER_ORIGIN", "http://127.0.0.1:3000")
    module_name = f"legacy_cors_test_{uuid4().hex}"
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            yield client
    finally:
        sys.modules.pop(module_name, None)


def test_legacy_backend_uses_one_explicit_origin_for_requests_and_preflights(
    legacy_cors_client: TestClient,
) -> None:
    origin = "http://127.0.0.1:3000"
    allowed = legacy_cors_client.get("/api/health", headers={"Origin": origin})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == origin
    preflight = legacy_cors_client.options(
        "/api/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == origin
    denied = legacy_cors_client.get("/api/health", headers={"Origin": "http://evil.test"})
    assert denied.headers.get("access-control-allow-origin") is None


def test_loopback_guard_rejects_a_nonloopback_server_scope() -> None:
    invoked = False

    async def inner(scope: object, receive: object, send: object) -> None:
        nonlocal invoked
        invoked = True

    async def invoke() -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        guard = CurationLoopbackGuard(inner)
        await guard(
            {
                "type": "http",
                "path": "/api/local-datasets/local/pnp_trash/resolve/main/meta/info.json",
                "method": "GET",
                "headers": [],
                "server": ("0.0.0.0", 8000),
            },
            receive,
            send,
        )
        return messages

    messages = asyncio.run(invoke())
    assert invoked is False
    assert messages[0]["status"] == 403


def test_loopback_guard_accepts_the_asgi_list_server_representation() -> None:
    invoked = False

    async def inner(scope: object, receive: object, send: object) -> None:
        nonlocal invoked
        invoked = True

    async def invoke() -> None:
        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            return None

        guard = CurationLoopbackGuard(inner)
        await guard(
            {
                "type": "http",
                "path": "/api/local-datasets/local/pnp_trash/resolve/main/meta/info.json",
                "method": "GET",
                "headers": [],
                "server": ["127.0.0.1", 8000],
            },
            receive,
            send,
        )

    asyncio.run(invoke())
    assert invoked is True


def test_legacy_browser_origin_default_matches_the_documented_nextjs_url() -> None:
    assert legacy_browser_origin({}) == "http://localhost:3000"


def test_opened_asset_uses_one_descriptor_and_closes_after_streaming(
    local_assets: tuple[LocalAssetService, Path, Path],
) -> None:
    service, _, _ = local_assets
    record = service.registry.resolve_alias("local", "pnp_trash")
    opened = record.open_asset("data/episode.parquet")
    assert opened is not None
    stream = _read_interval(opened, 2, 3)
    assert next(stream) == b"cde"
    stream.close()
    assert opened.closed is True
    with pytest.raises(OSError):
        os.fstat(opened.fd)


def test_head_and_nonbody_outcomes_close_opened_descriptors(
    local_assets: tuple[LocalAssetService, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = local_assets
    opened = []
    real_open = source_module.open_regular_file_beneath

    def track_open(*args: object):
        asset = real_open(*args)
        if asset is not None:
            opened.append(asset)
        return asset

    monkeypatch.setattr(source_module, "open_regular_file_beneath", track_open)
    head = service.serve("local", "pnp_trash", "main", "data/episode.parquet", method="HEAD", headers={})
    assert head.status_code == 200
    ranged_head = service.serve(
        "local", "pnp_trash", "main", "data/episode.parquet", method="HEAD", headers={"Range": "bytes=2-4"}
    )
    assert ranged_head.status_code == 206
    assert ranged_head.headers["content-range"] == "bytes 2-4/10"
    not_modified = service.serve(
        "local", "pnp_trash", "main", "data/episode.parquet", method="GET", headers={"If-None-Match": "*"}
    )
    assert not_modified.status_code == 304
    malformed = service.serve(
        "local", "pnp_trash", "main", "data/episode.parquet", method="GET", headers={"Range": "bytes=1-2,4-5"}
    )
    assert malformed.status_code == 416
    assert opened and all(asset.closed for asset in opened)


def test_streaming_response_cleanup_closes_an_unconsumed_descriptor(
    local_assets: tuple[LocalAssetService, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = local_assets
    opened = []
    real_open = source_module.open_regular_file_beneath

    def track_open(*args: object):
        asset = real_open(*args)
        if asset is not None:
            opened.append(asset)
        return asset

    monkeypatch.setattr(source_module, "open_regular_file_beneath", track_open)
    response = service.serve("local", "pnp_trash", "main", "data/episode.parquet", method="GET", headers={})
    assert response.background is not None
    response.background.func(*response.background.args, **response.background.kwargs)
    assert opened and all(asset.closed for asset in opened)


@pytest.mark.parametrize("if_none_match", ["*", 'W/"etag"', '"other", W/"etag"'])
def test_if_none_match_accepts_wildcards_weak_tags_and_lists(
    local_assets: tuple[LocalAssetService, Path, Path], if_none_match: str
) -> None:
    service, _, _ = local_assets
    etag = service.serve("local", "pnp_trash", "main", "data/episode.parquet", method="HEAD", headers={}).headers[
        "etag"
    ]
    value = if_none_match.replace('"etag"', etag)
    response = service.serve(
        "local", "pnp_trash", "main", "data/episode.parquet", method="GET", headers={"If-None-Match": value}
    )
    assert response.status_code == 304


def test_open_asset_rejects_a_regular_file_swapped_to_an_in_root_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    target = source / "data" / "episode.parquet"
    target.write_bytes(b"same bytes")
    (source / "data" / "replacement.parquet").write_bytes(b"same bytes")
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "workspace")
    target.unlink()
    target.symlink_to("replacement.parquet")

    response = LocalAssetService(registry).serve(
        "local", "pnp_trash", "main", "data/episode.parquet", method="GET", headers={}
    )
    assert response.status_code == 404


def test_open_asset_rejects_same_size_mutation_with_restored_mtime(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    target = source / "data" / "episode.parquet"
    target.write_bytes(b"original")
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "workspace")
    original_stat = target.stat()
    target.write_bytes(b"changed!")
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert registry.resolve_alias("local", "pnp_trash").open_asset("data/episode.parquet") is None


def test_manifest_rejects_a_source_mutated_while_its_hash_is_being_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "asset"
    target.write_bytes(b"initial")
    real_read = source_module.os.read
    changed = False

    def mutate_after_read(fd: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(fd, size)
        if not changed:
            changed = True
            target.write_bytes(b"updated")
            current = target.stat()
            os.utime(target, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000))
        return chunk

    monkeypatch.setattr(source_module.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed while hashing"):
        SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "workspace")


def test_open_asset_rejects_a_fifo_swap_without_blocking(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    target = source / "data" / "episode.parquet"
    target.write_bytes(b"regular")
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "workspace")
    target.unlink()
    os.mkfifo(target)

    assert registry.resolve_alias("local", "pnp_trash").open_asset("data/episode.parquet") is None


def test_manifest_rejects_an_existing_workspace_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "asset").write_bytes(b"asset")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest, _, _ = source_module._manifest_bytes(source)
    outside = tmp_path / "outside-manifest"
    outside.write_bytes(manifest)
    (workspace / "source-files.sha256").symlink_to(outside)

    with pytest.raises(ValueError, match="manifest path must be a regular file"):
        SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)


def test_manifest_bytes_are_utf8_path_byte_sorted_and_restart_rejects_source_change(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "z").write_bytes(b"z")
    (source / "é").write_bytes(b"accent")
    (source / "a").write_bytes(b"a")
    workspace = tmp_path / "workspace"
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
    manifest = (workspace / "source-files.sha256").read_bytes()
    expected = b"".join(
        f"{hashlib.sha256(contents).hexdigest()}  {name}\n".encode("utf-8")
        for name, contents in [("a", b"a"), ("z", b"z"), ("é", b"accent")]
    )
    assert manifest == expected
    assert registry.resolve_alias("local", "pnp_trash").fingerprint == hashlib.sha256(manifest).hexdigest()

    (source / "a").write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest changed"):
        SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)


def test_manifest_creation_is_serialized_and_recovers_after_failed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "asset").write_bytes(b"asset")
    workspace = tmp_path / "workspace"

    def register() -> str:
        return (
            SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
            .resolve_alias("local", "pnp_trash")
            .fingerprint
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        fingerprints = list(executor.map(lambda _: register(), range(2)))
    assert len(set(fingerprints)) == 1

    (workspace / "source-files.sha256").unlink()
    original_install = source_module._install_manifest_no_clobber
    monkeypatch.setattr(
        source_module,
        "_install_manifest_no_clobber",
        lambda *_: (_ for _ in ()).throw(OSError("crash")),
    )
    with pytest.raises(OSError, match="crash"):
        register()
    assert not list(workspace.glob(".source-files.sha256.*.tmp"))
    monkeypatch.setattr(source_module, "_install_manifest_no_clobber", original_install)
    assert register()
