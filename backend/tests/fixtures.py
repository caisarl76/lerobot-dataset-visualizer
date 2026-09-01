from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def legacy_v31_dataset(tmp_path: Path) -> Path:
    """A three-frame v3.1 dataset fixture with no video dependency."""
    root = tmp_path / "legacy-v31"
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    info = {
        "codebase_version": "v3.1",
        "fps": 10,
        "total_episodes": 1,
        "total_frames": 3,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    episodes = pa.table(
        {
            "episode_index": pa.array([0], type=pa.int64()),
            "data/chunk_index": pa.array([0], type=pa.int64()),
            "data/file_index": pa.array([0], type=pa.int64()),
            "length": pa.array([3], type=pa.int64()),
        }
    )
    pq.write_table(episodes, episodes_dir / "file-000.parquet")

    frames = pa.table(
        {
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float64()),
            "observation.state": pa.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(frames, data_dir / "file-000.parquet")

    return root
