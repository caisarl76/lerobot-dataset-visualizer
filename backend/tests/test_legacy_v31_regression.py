"""Editor drafts remain editable; publication now uses the official contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import app as backend_app
import httpx
import pyarrow.parquet as pq


def test_editor_drafts_and_official_export(clear_dataset_state, legacy_v31_dataset: Path, tmp_path: Path):
    root = legacy_v31_dataset
    data_path = root / "data/chunk-000/file-000.parquet"
    original_bytes = data_path.read_bytes()
    original_table = pq.read_table(data_path)
    atoms = [
        {"role": "user", "content": "Sort the table", "style": "task_aug", "timestamp": 0.0},
        {"role": "assistant", "content": "Reach for the can", "style": "subtask", "timestamp": 0.0},
        {"role": "assistant", "content": "Place it in the bin", "style": "plan", "timestamp": 0.0},
        {"role": "assistant", "content": "The bin is on the right", "style": "memory", "timestamp": 0.0},
        {"role": "user", "content": "Be careful", "style": "interjection", "timestamp": 0.04},
    ]

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=backend_app.app), base_url="http://test"
        ) as client:
            identity = {"local_path": str(root)}
            loaded = await client.post("/api/dataset/load", json=identity)
            assert loaded.status_code == 200
            assert loaded.json()["num_episodes"] == 1
            payload = {**identity, "episode_index": 0, "atoms": atoms}
            saved = await client.post("/api/episodes/0/atoms", json=payload)
            assert saved.json() == {"ok": True, "saved": 5, "path": str(root / "meta/lerobot_annotations.json")}
            backend_app._states.clear()
            reread = await client.get("/api/episodes/0/atoms", params=identity)
            assert reread.json()["atoms"][-1]["timestamp"] == 0.0
            # The editor can persist an incomplete draft; official export rejects it.
            invalid = await client.post("/api/export", json={**identity, "output_dir": str(tmp_path / "invalid")})
            assert invalid.status_code == 422
            assert "speech" in invalid.json()["detail"]
            atoms.append(
                {
                    "role": "assistant",
                    "content": None,
                    "style": None,
                    "timestamp": 0.04,
                    "tool_calls": [
                        {"type": "function", "function": {"name": "say", "arguments": {"text": "I will."}}}
                    ],
                }
            )
            payload["atoms"] = atoms
            assert (await client.post("/api/episodes/0/atoms", json=payload)).status_code == 200
            report = await client.post(
                "/api/annotation/validate",
                json={
                    **payload,
                    "atoms": (await client.get("/api/episodes/0/atoms", params=identity)).json()["atoms"],
                },
            )
            assert report.json()["ok"]
            exported = await client.post("/api/export", json={**identity, "output_dir": str(tmp_path / "valid")})
            assert exported.status_code == 200, exported.text
            assert exported.json() == {
                "output_dir": str(tmp_path / "valid"),
                "persistent_rows": 4,
                "event_rows": 2,
            }
            assert (
                await client.post("/api/export", json={**identity, "output_dir": str(root)})
            ).status_code == 422
            assert (
                await client.post("/api/episodes/99/atoms", json={**identity, "episode_index": 99, "atoms": []})
            ).status_code == 404

    asyncio.run(scenario())
    assert data_path.read_bytes() == original_bytes
    exported = pq.read_table(tmp_path / "valid/data/chunk-000/file-000.parquet")
    for name in original_table.column_names:
        assert exported[name].equals(original_table[name])
    assert len(exported["language_persistent"][0].as_py()) == 4
    events = exported["language_events"].to_pylist()
    assert len(events[0]) == 2 and events[1:] == [[], []]
    assert all("timestamp" not in atom for atom in events[0])
    info = json.loads((tmp_path / "valid/meta/info.json").read_text())
    assert info["tools"][0]["function"]["name"] == "say"
    assert "tools" not in info["features"]


def test_preparation_job_and_local_assets(
    clear_dataset_state, legacy_v31_dataset: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(backend_app, "EXPORT_ROOT", tmp_path / "workspace")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=backend_app.app), base_url="http://test"
        ) as client:
            defaults = await client.get("/api/annotation/config")
            assert defaults.json()["config"]["plan"]["enabled"] is True
            assert "api_key" not in defaults.json()["config"]["vlm"]
            bad = await client.post(
                "/api/annotation/jobs",
                json={"local_path": str(legacy_v31_dataset), "config": {"plan": {"bogus": True}}},
            )
            assert bad.status_code == 422
            response = await client.post("/api/annotation/prepare", json={"local_path": str(legacy_v31_dataset)})
            job_id = response.json()["job_id"]
            for _ in range(200):
                job = (await client.get(f"/api/annotation/jobs/{job_id}")).json()
                if job["status"] not in {"queued", "running"}:
                    break
                await asyncio.sleep(0.01)
            assert job["status"] == "completed", job
            alias = job["result"]["repo_id"]
            assert alias.startswith("local/annotation-")
            loaded = await client.post("/api/dataset/load", json={"repo_id": alias})
            assert loaded.status_code == 200
            assert Path(loaded.json()["root"]) != legacy_v31_dataset
            asset = f"/datasets/{alias}/resolve/main/meta/info.json"
            ranged = await client.get(asset, headers={"Range": "bytes=0-9"})
            assert ranged.status_code == 206 and len(ranged.content) == 10
            assert (await client.head(asset)).status_code == 200
            assert (
                await client.get(f"/datasets/{alias}/resolve/main/meta/%2E%2E/%2E%2E/jobs/{job_id}.json")
            ).status_code == 404
            assert (await client.get(f"/datasets/{alias}/resolve/main/.env")).status_code == 404

    asyncio.run(scenario())
