import json

from backend import official_annotations as engine
import datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import pytest


def test_official_deletion_reindexes_and_preserves_saved_labels_and_data(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    source = tmp_path / "source"
    dataset = LeRobotDataset.create(
        repo_id="local/test",
        fps=10,
        root=source,
        use_videos=False,
        features={"observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}},
    )
    for ep in range(3):
        for frame in range(4):
            dataset.add_frame(
                {"observation.state": np.array([ep, frame], dtype=np.float32), "task": "pick up trash"}
            )
        dataset.save_episode()
    dataset.finalize()
    (source / "meta/modality.json").write_text('{"test":"retained"}')
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    manual = {
        "style": "subtask",
        "content": "human reviewed grasp",
        "role": "assistant",
        "timestamp": 0.0,
        "camera": None,
        "tool_calls": None,
    }
    output = tmp_path / "cleaned"
    result = engine.delete_dataset_episodes(source, output, {2: [manual]}, [0, 1])
    assert result["old_to_new"] == {2: 0}
    assert result["validation"]["ok"]
    records = list(engine.iter_episodes(output))
    assert len(records) == 1 and records[0].episode_index == 0
    assert engine.read_atoms(records[0]) == [manual]
    rows = pq.read_table(records[0].data_path).to_pylist()
    assert [r["episode_index"] for r in rows] == [0] * 4
    assert [r["index"] for r in rows] == list(range(4))
    assert [r["observation.state"] for r in rows] == [[2.0, float(i)] for i in range(4)]
    assert (output / "meta/modality.json").read_bytes() == (source / "meta/modality.json").read_bytes()
    assert json.loads((output / "meta/info.json").read_text())["total_episodes"] == 1
    assert {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before

    for ids in ([], [0, 0], [99], [0, 1, 2]):
        with pytest.raises(ValueError):
            engine.delete_dataset_episodes(source, tmp_path / "invalid", {}, ids)
    assert not (tmp_path / "invalid").exists()
