from __future__ import annotations

import json
from pathlib import Path

import app as backend_app
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
EXPECTED_SAY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak a short utterance to the user via the TTS executor.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The verbatim text to speak."}},
            "required": ["text"],
        },
    },
}


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


def test_legacy_v31_atoms_routes_and_export_contract(
    clear_dataset_state: None, legacy_v31_dataset: Path, tmp_path: Path
) -> None:
    source_path = legacy_v31_dataset / "data" / "chunk-000" / "file-000.parquet"
    source_table = pq.read_table(source_path)
    client = TestClient(backend_app.app)
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

    annotation_payload = json.loads((legacy_v31_dataset / "meta" / "lerobot_annotations.json").read_text())
    assert annotation_payload["version"] == 2
    assert [atom["style"] for atom in annotation_payload["episodes"]["0"]["atoms"]] == [
        "task_aug",
        "subtask",
        "plan",
        "memory",
        "interjection",
        "vqa",
        None,
    ]
    backend_app._states.clear()

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
    assert [atom["role"] for atom in atoms.json()["atoms"]] == [
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
        "user",
        "assistant",
    ]
    assert [atom["content"] for atom in atoms.json()["atoms"]] == [
        "Sort the table",
        "Reach for the can",
        "Then place it in the bin",
        "The bin is on the right",
        "Be careful",
        "Where is the can?",
        None,
    ]
    assert atoms.json()["atoms"][5]["camera"] == "observation.images.front"
    assert atoms.json()["atoms"][6]["tool_calls"] == [
        {
            "type": "function",
            "function": {"name": "say", "arguments": {"text": "I found it."}},
        }
    ]

    timestamps = client.get(f"/api/episodes/0/frame_timestamps?local_path={local_path}")
    assert timestamps.status_code == 200
    assert set(timestamps.json()) == TIMESTAMPS_KEYS
    assert timestamps.json() == {"episode_index": 0, "timestamps": [0.0, 0.1, 0.2]}

    output_dir = tmp_path / "exported-legacy-v31"
    backend_app._states.clear()
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
    assert persistent[0] == [
        {
            "role": "user",
            "content": "Sort the table",
            "style": "task_aug",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": "Reach for the can",
            "style": "subtask",
            "timestamp": 0.01,
            "camera": None,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": "Then place it in the bin",
            "style": "plan",
            "timestamp": 0.02,
            "camera": None,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": "The bin is on the right",
            "style": "memory",
            "timestamp": 0.03,
            "camera": None,
            "tool_calls": None,
        },
    ]
    assert all(rows == persistent[0] for rows in persistent[1:])
    assert events == [
        [
            {
                "role": "user",
                "content": "Be careful",
                "style": "interjection",
                "camera": None,
                "tool_calls": None,
            }
        ],
        [
            {
                "role": "user",
                "content": "Where is the can?",
                "style": "vqa",
                "camera": "observation.images.front",
                "tool_calls": None,
            }
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "style": None,
                "camera": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "say", "arguments": {"text": "I found it."}},
                    }
                ],
            }
        ],
    ]
    assert set(persistent[0][0]) == {"role", "content", "style", "timestamp", "camera", "tool_calls"}
    assert set(events[0][0]) == {"role", "content", "style", "camera", "tool_calls"}

    exported_info = json.loads((output_dir / "meta" / "info.json").read_text())
    assert set(exported_info) == {
        "codebase_version",
        "fps",
        "total_episodes",
        "total_frames",
        "data_path",
        "features",
        "tools",
    }
    assert set(exported_info["features"]) == {
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
        "language_persistent",
        "language_events",
    }
    assert "task_index" not in exported_info["features"]
    assert "tools" not in exported_info["features"]
    assert exported_info["tools"] == [EXPECTED_SAY_TOOL_SCHEMA]
    jsonschema.Draft202012Validator.check_schema(exported_info["tools"][0]["function"]["parameters"])
