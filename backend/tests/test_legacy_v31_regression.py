from __future__ import annotations

import json
from pathlib import Path

from app import SAY_TOOL_SCHEMA, app
from fastapi.testclient import TestClient
import jsonschema
import pyarrow.parquet as pq

LOAD_KEYS = {
    "repo_id",
    "local_path",
    "revision",
    "root",
    "fps",
    "num_episodes",
    "persistent_styles",
    "event_styles",
}
GET_ATOMS_KEYS = {"episode_index", "atoms"}
SET_ATOMS_KEYS = {"ok", "saved", "path"}
TIMESTAMPS_KEYS = {"episode_index", "timestamps"}
EXPORT_KEYS = {"output_dir", "persistent_rows", "event_rows"}


def _legacy_atoms() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Sort the table", "style": "task_aug", "timestamp": 0.0},
        {"role": "assistant", "content": "Reach for the can", "style": "subtask", "timestamp": 0.01},
        {"role": "assistant", "content": "Then place it in the bin", "style": "plan", "timestamp": 0.02},
        {"role": "assistant", "content": "The bin is on the right", "style": "memory", "timestamp": 0.03},
        {"role": "user", "content": "Be careful", "style": "interjection", "timestamp": 0.04},
        {
            "role": "user",
            "content": "Where is the can?",
            "style": "vqa",
            "timestamp": 0.14,
            "camera": "observation.images.front",
        },
        {
            "role": "assistant",
            "content": None,
            "style": None,
            "timestamp": 0.18,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "say", "arguments": {"text": "I found it."}},
                }
            ],
        },
    ]


def test_legacy_v31_atoms_routes_and_export_contract(legacy_v31_dataset: Path, tmp_path: Path) -> None:
    source_path = legacy_v31_dataset / "data" / "chunk-000" / "file-000.parquet"
    source_table = pq.read_table(source_path)
    client = TestClient(app)
    local_path = str(legacy_v31_dataset)

    load = client.post("/api/dataset/load", json={"local_path": local_path})
    assert load.status_code == 200
    assert set(load.json()) == LOAD_KEYS
    assert load.json() | {"root": "ignored"} == {
        "repo_id": None,
        "local_path": local_path,
        "revision": None,
        "root": "ignored",
        "fps": 10.0,
        "num_episodes": 1,
        "persistent_styles": ["memory", "plan", "subtask", "task_aug"],
        "event_styles": ["interjection", "vqa"],
    }

    empty_atoms = client.get(f"/api/episodes/0/atoms?local_path={local_path}")
    assert empty_atoms.status_code == 200
    assert set(empty_atoms.json()) == GET_ATOMS_KEYS
    assert empty_atoms.json() == {"episode_index": 0, "atoms": []}

    saved = client.post(
        "/api/episodes/0/atoms",
        json={"local_path": local_path, "episode_index": 0, "atoms": _legacy_atoms()},
    )
    assert saved.status_code == 200
    assert set(saved.json()) == SET_ATOMS_KEYS
    assert saved.json()["ok"] is True
    assert saved.json()["saved"] == 7
    assert saved.json()["path"] == str(legacy_v31_dataset / "meta" / "lerobot_annotations.json")

    atoms = client.get(f"/api/episodes/0/atoms?local_path={local_path}")
    assert atoms.status_code == 200
    assert set(atoms.json()) == GET_ATOMS_KEYS
    assert atoms.json()["episode_index"] == 0
    assert all(
        set(atom) == {"role", "content", "style", "timestamp", "camera", "tool_calls"}
        for atom in atoms.json()["atoms"]
    )
    assert [atom["style"] for atom in atoms.json()["atoms"]] == [
        "task_aug",
        "subtask",
        "plan",
        "memory",
        "interjection",
        "vqa",
        None,
    ]
    assert [atom["timestamp"] for atom in atoms.json()["atoms"]] == [
        0.0,
        0.01,
        0.02,
        0.03,
        0.0,
        0.1,
        0.2,
    ]
    assert atoms.json()["atoms"][-1]["tool_calls"][0]["function"]["name"] == "say"

    timestamps = client.get(f"/api/episodes/0/frame_timestamps?local_path={local_path}")
    assert timestamps.status_code == 200
    assert set(timestamps.json()) == TIMESTAMPS_KEYS
    assert timestamps.json() == {"episode_index": 0, "timestamps": [0.0, 0.1, 0.2]}

    output_dir = tmp_path / "exported-legacy-v31"
    exported = client.post("/api/export", json={"local_path": local_path, "output_dir": str(output_dir)})
    assert exported.status_code == 200
    assert set(exported.json()) == EXPORT_KEYS
    assert exported.json() == {
        "output_dir": str(output_dir),
        "persistent_rows": 4,
        "event_rows": 3,
    }

    export_table = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet")
    assert export_table.column_names == [
        *source_table.column_names,
        "language_persistent",
        "language_events",
    ]
    for name in source_table.column_names:
        assert export_table.schema.field(name).type == source_table.schema.field(name).type
        assert export_table.column(name).to_pylist() == source_table.column(name).to_pylist()

    persistent = export_table.column("language_persistent").to_pylist()
    events = export_table.column("language_events").to_pylist()
    assert [atom["style"] for atom in persistent[0]] == ["task_aug", "subtask", "plan", "memory"]
    assert all(rows == persistent[0] for rows in persistent[1:])
    assert [atom["style"] for atom in events[0]] == ["interjection"]
    assert [atom["style"] for atom in events[1]] == ["vqa"]
    assert events[2][0]["style"] is None
    assert events[2][0]["tool_calls"][0]["function"]["name"] == "say"
    assert set(persistent[0][0]) == {"role", "content", "style", "timestamp", "camera", "tool_calls"}
    assert set(events[0][0]) == {"role", "content", "style", "camera", "tool_calls"}

    exported_info = json.loads((output_dir / "meta" / "info.json").read_text())
    assert "tools" not in exported_info["features"]
    assert exported_info["tools"] == [SAY_TOOL_SCHEMA]
    jsonschema.Draft202012Validator.check_schema(exported_info["tools"][0]["function"]["parameters"])
