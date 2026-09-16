import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_v21(root: Path, lengths: list[int]) -> Path:
    (root / "meta").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "fps": 10, "total_episodes": len(lengths),
            "total_frames": sum(lengths), "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {}}
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = [{"episode_index": ep, "length": n, "tasks": ["pick"]} for ep, n in enumerate(lengths)]
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for ep, n in enumerate(lengths):
        path = root / info["data_path"].format(episode_chunk=0, episode_index=ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"episode_index": [ep] * n, "frame_index": list(range(n)),
                                 "timestamp": [i / 10 for i in range(n)]}), path)
    return root


def write_run(workspace: Path, source: Path, checkpoint: Path | None = None,
              episodes: list[dict] | None = None, run_id: str = "run-1") -> Path:
    run_dir = workspace / "runs" / run_id
    run_dir.mkdir(parents=True)
    payload = {"run_id": run_id, "source_root": str(source), "root": str(source),
               "checkpoint": str(checkpoint) if checkpoint else None,
               "episodes": episodes or []}
    (run_dir / "run.json").write_text(json.dumps(payload))
    return run_dir / "run.json"
