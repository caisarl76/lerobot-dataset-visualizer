"""Small on-disk fixtures using the production metadata and run schemas."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def write_v21(root: Path, lengths: list[int]) -> Path:
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {},
    }
    write_json(root / "meta/info.json", info)
    rows = [{"episode_index": ep, "length": n, "tasks": ["pick"]} for ep, n in enumerate(lengths)]
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for ep, n in enumerate(lengths):
        path = root / info["data_path"].format(episode_chunk=0, episode_index=ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {"episode_index": [ep] * n, "frame_index": list(range(n)), "timestamp": [i / 10 for i in range(n)]}
            ),
            path,
        )
    return root


def write_v3(root, lengths):
    write_v21(root, lengths)
    (root / "meta/episodes.jsonl").unlink()
    info = json.loads((root / "meta/info.json").read_text())
    info.update(codebase_version="v3.0", data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    write_json(root / "meta/info.json", info)
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    offsets = [sum(lengths[:ep]) for ep in range(len(lengths))]
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array(range(len(lengths)), type=pa.int64()),
                "length": pa.array(lengths, type=pa.int64()),
                "data/chunk_index": [0] * len(lengths),
                "data/file_index": list(range(len(lengths))),
                "dataset_from_index": offsets,
                "dataset_to_index": [start + n for start, n in zip(offsets, lengths)],
            }
        ),
        path,
    )
    for ep in range(len(lengths)):
        (root / f"data/chunk-000/episode_{ep:06d}.parquet").rename(root / f"data/chunk-000/file-{ep:03d}.parquet")
    return root


def write_run(
    workspace: Path,
    source: Path,
    checkpoint: Path | None = None,
    episodes: list[int] | None = None,
    run_id: str = "a" * 32,
) -> Path:
    checkpoint = checkpoint or workspace / "drafts" / run_id
    payload = {
        "version": 1,
        "run_id": run_id,
        "source_root": str(source.resolve()),
        "root": str(checkpoint.resolve()),
        "repo_id": None,
        "source_commit": None,
        "source_format": "v2.1",
        "source_file_hashes": {},
        "revision": 1,
        "current_job_id": None,
        "publication_state": "draft",
        "task_prompt": "",
        "subtask_prompts": [],
        "example_episode_indices": [],
        "episodes": {
            str(ep): {
                "original_episode_index": ep,
                "generation_status": "pending",
                "issues": [],
                "decision": "pending",
                "decision_reason": None,
            }
            for ep in (episodes or [])
        },
    }
    return write_json(workspace / "runs" / run_id / "run.json", payload)


@pytest.fixture
def review_fixture(tmp_path):
    from annotation_history import annotation_hash, exclusions_hash
    from annotation_monitor import discover_datasets, read_runs

    workspace = tmp_path / "workspace"
    source = write_v21(tmp_path / "collection/source", [10, 20, 30, 40])
    checkpoint = write_v21(workspace / "drafts/current", [10, 20, 30])
    path = write_run(workspace, source, checkpoint, [0, 1, 2])
    run = json.loads(path.read_text())
    annotations, reviews = {}, {}
    for ep, decision in enumerate(("keep", "pending", "delete")):
        run["episodes"][str(ep)]["decision"] = decision
        atoms = [{"timestamp": 0.0, "role": "assistant", "style": "subtask", "content": f"pick {ep}"}]
        annotations[str(ep)] = {"atoms": atoms}
        reviews[str(ep)] = {
            "annotation_sha256": annotation_hash(atoms),
            "exclusions_sha256": exclusions_hash(),
            "reviewed_at": "2026-09-16T00:00:00Z",
        }
    write_json(path, run)
    write_json(checkpoint / "meta/lerobot_annotations.json", {"version": 2, "episodes": annotations})
    write_json(checkpoint / "meta/annotation_reviews.json", reviews)
    return discover_datasets(source.parent, workspace)[0], read_runs(workspace)[0], workspace
