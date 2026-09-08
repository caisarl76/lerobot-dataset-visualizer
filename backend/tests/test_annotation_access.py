import asyncio
import json

from fastapi import FastAPI
import httpx
import pytest


def test_hosted_access_requires_secret_and_disallows_paths(monkeypatch):
    from annotation_access import AnnotationAccess

    monkeypatch.setenv("ANNOTATION_BACKEND_TOKEN", "test-secret")
    app = FastAPI()
    app.add_middleware(AnnotationAccess)

    @app.post("/api/annotation/prepare")
    def prepare():
        return {"ok": True}

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.post("/api/annotation/prepare", json={})).status_code == 401
            headers = {"Authorization": "Bearer test-secret"}
            assert (
                await c.post("/api/annotation/prepare", headers=headers, json={"local_path": "/etc"})
            ).status_code == 403
            assert (
                await c.post("/api/annotation/prepare", headers=headers, json={"repo_id": "org/data"})
            ).status_code == 200
            assert (await c.get("/api/export", headers=headers)).status_code == 403

    asyncio.run(scenario())


@pytest.mark.parametrize("version", ["v2.1", "v3.1"])
@pytest.mark.parametrize("field", ["data_path", "video_path"])
@pytest.mark.parametrize(
    "path", ["/tmp/private.mp4", "videos/../../private.mp4", "videos/{video_key.__class__}/file.mp4"]
)
def test_dataset_paths_reject_host_metadata_escape(tmp_path, version, field, path):
    from annotation_access import validate_dataset_paths

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"codebase_version": version, field: path}))
    with pytest.raises(ValueError, match="[Pp]ath|template"):
        validate_dataset_paths(tmp_path)


def test_dataset_paths_reject_camera_key_traversal(tmp_path):
    from annotation_access import validate_dataset_paths

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {"observation.images.cam/../../../../outside": {"dtype": "video"}},
            }
        )
    )
    with pytest.raises(ValueError, match="[Cc]amera|[Pp]ath"):
        validate_dataset_paths(tmp_path)


def test_dataset_paths_reject_symlink_escape_without_touching_target(tmp_path):
    from annotation_access import validate_dataset_paths

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"private video")
    (root / "videos").mkdir()
    (root / "videos/episode.mp4").symlink_to(outside)
    with pytest.raises(ValueError, match="[Pp]ath"):
        validate_dataset_paths(root)
    assert outside.read_bytes() == b"private video"


def test_dataset_paths_validate_expanded_episode_indices(tmp_path):
    from annotation_access import validate_dataset_paths
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "data_path": "data/chunk-{chunk_index}/file-{file_index}.parquet",
            }
        )
    )
    pq.write_table(
        pa.table({"episode_index": [0], "data/chunk_index": ["../../../private"]}),
        tmp_path / "meta/episodes/chunk-000/file-000.parquet",
    )
    with pytest.raises(ValueError, match="path index"):
        validate_dataset_paths(tmp_path)


@pytest.mark.parametrize("version", ["v2.1", "v3.1"])
def test_dataset_paths_accept_official_templates(tmp_path, version):
    from annotation_access import validate_dataset_paths

    (tmp_path / "meta").mkdir()
    shard = (
        "chunk-{episode_chunk:03d}/episode_{episode_index:06d}"
        if version == "v2.1"
        else "chunk-{chunk_index:03d}/file-{file_index:03d}"
    )
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "data_path": "data/" + shard + ".parquet",
                "video_path": "videos/{video_key}/" + shard + ".mp4",
                "features": {"observation.images.cam": {"dtype": "video"}},
            }
        )
    )
    validate_dataset_paths(tmp_path)
