from __future__ import annotations

from pathlib import Path

from curation.assets import LocalAssetService, _read_interval
from curation.source import SourceRegistry
import pytest


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
    full = service.serve("local", "pnp_trash", "main", "data/episode.parquet", method="GET", headers={})
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
            method="GET",
            headers={"range": header},
        )
        assert ranged.status_code == 206
        assert ranged.headers["content-range"] == content_range
        assert ranged.headers["content-length"] == str(len(expected))

    asset = service.registry.resolve_alias("local", "pnp_trash").root / "data" / "episode.parquet"
    assert b"".join(_read_interval(asset, 0, 10)) == b"abcdefghij"
    assert b"".join(_read_interval(asset, 2, 3)) == b"cde"

    not_modified = service.serve(
        "local",
        "pnp_trash",
        "main",
        "data/episode.parquet",
        method="GET",
        headers={"if-none-match": etag},
    )
    assert not_modified.status_code == 304
    fallback = service.serve(
        "local",
        "pnp_trash",
        "main",
        "data/episode.parquet",
        method="GET",
        headers={"range": "bytes=0-1", "if-range": '"different"'},
    )
    assert fallback.status_code == 200
    matching = service.serve(
        "local",
        "pnp_trash",
        "main",
        "data/episode.parquet",
        method="GET",
        headers={"range": "bytes=0-1", "if-range": etag},
    )
    assert matching.status_code == 206

    video = service.serve("local", "pnp_trash", "main", "videos/episode.mp4", method="GET", headers={})
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
