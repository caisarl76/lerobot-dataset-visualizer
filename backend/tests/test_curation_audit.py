from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import curation.audit as audit_module
from curation.audit import build_audit_report
from curation.contact_sheets import (
    ContactSheetDatasetIdentity,
    proposal_contact_sheet_path,
    receipt_path,
)
from curation.cosmos_transport import AtomicArtifactStore
from curation.db import CurationDatabase
from curation.grip import HAND_FEATURE_NAMES
from curation.review import ReviewService
from curation.router import build_curation_router
from curation.source import SourceRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROPOSAL_IDS = (
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002",
)


def _signal(frame_count: int = 61) -> np.ndarray:
    envelope = np.zeros(frame_count, dtype=np.float64)
    envelope[12:23] = np.linspace(0.0, 1.0, 11)
    envelope[23:39] = 1.0
    envelope[39:50] = np.linspace(1.0, 0.0, 11)
    values = np.stack([envelope * scale for scale in np.linspace(0.7, 1.3, 7)], axis=1)
    state = np.zeros((frame_count, 43), dtype=np.float64)
    state[:, 22:29] = values
    state[:, 36:43] = values
    return state


def _audit_source(tmp_path: Path) -> tuple[SourceRegistry, Path]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True)
    names = [f"body_{index}" for index in range(43)]
    names[22:29] = HAND_FEATURE_NAMES["left"]
    names[36:43] = HAND_FEATURE_NAMES["right"]
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "total_episodes": 3,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
                "features": {"observation.state": {"shape": [43], "names": names}},
            }
        )
    )
    with (source / "meta" / "episodes.jsonl").open("w") as handle:
        for episode_index in range(3):
            handle.write(json.dumps({"episode_index": episode_index, "length": 61, "tasks": ["task"]}) + "\n")
    for episode_index in (0, 1):
        timestamps = np.arange(61, dtype=np.float64) / 10.0
        state = _signal()
        pq.write_table(
            pa.table(
                {
                    "episode_index": [episode_index] * 61,
                    "frame_index": list(range(61)),
                    "timestamp": timestamps,
                    "observation.state": pa.FixedSizeListArray.from_arrays(
                        pa.array(state.reshape(-1), type=pa.float64()), 43
                    ),
                }
            ),
            source / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
        )
        video = (
            source / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{episode_index:06d}.mp4"
        )
        video.write_bytes(b"registered video placeholder")
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "manifest")
    return registry, source


def _service_with_states(tmp_path: Path) -> ReviewService:
    registry, _ = _audit_source(tmp_path)
    database = CurationDatabase(tmp_path / "workspace" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)
    service.open_workspace("local/pnp_trash", actor="curator")

    keep = service.save_draft(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=0,
        actor="curator",
        object_name="can",
        pickup_hand="left",
        turn_direction="right",
        transition_frames=[40, 45, 50, 54, 57, 59],
    )
    service.approve_keep(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=keep["revision"],
        actor="curator",
        reviewer="reviewer",
    )
    service.approve_reject(
        dataset_alias="local/pnp_trash",
        source_episode_index=1,
        expected_revision=0,
        actor="curator",
        reviewer="reviewer",
        reason="wrong object",
    )
    # Model one pending episode with incomplete coverage and one completed episode.
    dataset = database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    job = database.create_cosmos_job(dataset_id=dataset["id"], configuration={"model": "cosmos3-nano"})
    with database.open_connection() as connection:
        connection.execute("UPDATE cosmos_jobs SET state='running', total_attempts=3 WHERE id=?", (job["id"],))
        attempts = [
            ("attempt-0", 0, "succeeded", None),
            ("attempt-1", 1, "succeeded", None),
            ("attempt-2", 2, "manual_only", "alignment_unproven"),
        ]
        for attempt_id, episode_index, state, error in attempts:
            connection.execute(
                """
                INSERT INTO cosmos_attempts(
                    id, job_id, source_episode_index, attempt_number, state,
                    error_class, error_summary, created_at, updated_at
                ) VALUES (?, ?, ?, 0, 'requesting', NULL, NULL,
                    '2026-08-20T00:00:00Z', '2026-08-20T00:00:00Z')
                """,
                (attempt_id, job["id"], episode_index),
            )
        model = {
            "schema_version": 2,
            "episode_complete": True,
            "reasoning": "SECRET RAW REASONING MUST NOT LEAK",
            "segments": [{"step": step, "status": "completed"} for step in range(1, 8)],
        }
        connection.execute(
            """
            INSERT INTO cosmos_proposals(
                id, attempt_id, model_response_json,
                step_2_start_frame, step_3_start_frame, step_4_start_frame,
                step_5_start_frame, step_6_start_frame, step_7_start_frame,
                validation_warnings_json, state, created_at
            ) VALUES (?, 'attempt-0', ?, 40, 45, 50, 54, 57, 59, '[]', 'active', 'now')
            """,
            (PROPOSAL_IDS[0], json.dumps(model)),
        )
        incomplete = {"schema_version": 2, "episode_complete": False, "missing_steps": [4]}
        connection.execute(
            """
            INSERT INTO cosmos_proposals(
                id, attempt_id, model_response_json,
                step_2_start_frame, step_3_start_frame, step_4_start_frame,
                step_5_start_frame, step_6_start_frame, step_7_start_frame,
                validation_warnings_json, state, created_at
            ) VALUES (?, 'attempt-1', ?, 5, 10, NULL, 20, 25, 30, '["incomplete"]', 'active', 'now')
            """,
            (PROPOSAL_IDS[1], json.dumps(incomplete)),
        )
        for attempt_id, _episode_index, state, error in attempts:
            connection.execute(
                """
                UPDATE cosmos_attempts SET state=?, error_class=?, error_summary=? WHERE id=?
                """,
                (state, "ManualOnly" if error else None, error, attempt_id),
            )
        connection.execute(
            "UPDATE cosmos_jobs SET state='completed_with_failures', succeeded_attempts=2, "
            "manual_only_attempts=1 WHERE id=?",
            (job["id"],),
        )
        # A corrupted draft-like row proves the audit detects rather than trusts persisted boundaries.
        connection.execute(
            """
            UPDATE episodes SET review_state='draft', step_2_start_frame=4,
                step_3_start_frame=4, step_4_start_frame=NULL WHERE dataset_id=? AND source_episode_index=2
            """,
            (dataset["id"],),
        )
    return service


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return any(key == forbidden or _contains_key(child, forbidden) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def test_audit_reports_counts_distributions_errors_grip_unreadable_and_fingerprint(tmp_path: Path) -> None:
    service = _service_with_states(tmp_path)

    report = service.audit("local/pnp_trash")

    assert report["dataset_alias"] == "local/pnp_trash"
    assert report["review_state_counts"] == {
        "pending": 0,
        "draft": 1,
        "approved_keep": 1,
        "approved_reject": 1,
    }
    assert report["cosmos"]["job_state_counts"] == {"completed_with_failures": 1}
    assert report["cosmos"]["attempt_state_counts"] == {"manual_only": 1, "succeeded": 2}
    assert report["cosmos"]["proposal_result_counts"] == {"complete": 1, "incomplete": 1}
    assert report["transition_time_distributions"]["step_2_start"]["count"] == 1
    assert report["transition_time_distributions"]["step_2_start"]["median_s"] == 4.0
    assert report["phase_duration_distributions"]["step_1"]["median_s"] == 4.0
    error_codes = {error["code"] for error in report["boundary_errors"]}
    assert {"zero_length_phase", "incomplete_coverage", "non_increasing_transitions"} <= error_codes
    assert report["grip_disagreements"] == [
        {
            "source_episode_index": 0,
            "warnings": ["grip_grasp_disagrees_with_step_2_start"],
            "grasp_delta_s": report["grip_disagreements"][0]["grasp_delta_s"],
            "release_delta_s": report["grip_disagreements"][0]["release_delta_s"],
        }
    ]
    assert any(item["source_episode_index"] == 2 for item in report["unreadable_files"])
    assert report["source_fingerprint"] == {
        "expected_sha256": report["source_fingerprint"]["expected_sha256"],
        "current_matches": True,
    }
    serialized = json.dumps(report)
    assert "SECRET RAW REASONING" not in serialized
    assert not _contains_key(report, "reasoning")
    assert "model_response" not in serialized


def test_audit_reports_missing_and_conflicting_contact_sheet_evidence(tmp_path: Path) -> None:
    service = _service_with_states(tmp_path)
    workspace = service.database.path.parent
    dataset = service.database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    identity = ContactSheetDatasetIdentity(
        dataset_id=dataset["id"],
        dataset_alias=dataset["alias"],
        source_manifest_sha256=dataset["source_manifest_sha256"],
    )
    proposal_path = proposal_contact_sheet_path(identity, PROPOSAL_IDS[0])
    sheet = workspace / proposal_path
    receipt = workspace / receipt_path(proposal_path)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_bytes(b"conflicting evidence")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "proposal",
                "dataset_id": identity.dataset_id,
                "dataset_alias": identity.dataset_alias,
                "source_manifest_sha256": identity.source_manifest_sha256,
                "source_episode_index": 0,
                "proposal_id": PROPOSAL_IDS[0],
                "approval_revision": None,
                "final_transition_frames": None,
                "proposal_transition_frames": [40, 45, 50, 54, 57, 59],
                "relative_path": proposal_path,
                "sha256": "0" * 64,
                "byte_size": 20,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    report = service.audit("local/pnp_trash")

    assert {
        "kind": "proposal",
        "proposal_id": PROPOSAL_IDS[0],
        "source_episode_index": 0,
        "reason": "conflict",
    } in report["contact_sheet_issues"]
    assert {
        "kind": "proposal",
        "proposal_id": PROPOSAL_IDS[1],
        "source_episode_index": 1,
        "reason": "missing",
    } in report["contact_sheet_issues"]
    final_issues = [item for item in report["contact_sheet_issues"] if item["kind"] == "final"]
    assert final_issues == [
        {
            "kind": "final",
            "source_episode_index": 0,
            "approval_revision": 2,
            "reason": "missing",
        }
    ]


def test_audit_reports_source_inventory_mismatch_without_disclosing_local_path(tmp_path: Path) -> None:
    service = _service_with_states(tmp_path)
    record = service.source_registry.records["local/pnp_trash"]
    info = record.root / "meta" / "info.json"
    replacement = info.with_name("replacement.json")
    replacement.write_bytes(info.read_bytes())
    os.replace(replacement, info)

    report = service.audit("local/pnp_trash")

    assert report["source_fingerprint"]["current_matches"] is False
    assert str(record.root) not in json.dumps(report)


def test_audit_keeps_historical_final_evidence_visible_after_prompt_invalidation(
    tmp_path: Path,
) -> None:
    service = _service_with_states(tmp_path)
    dataset = service.database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    service.database.migrate_prompt_template(
        dataset_id=dataset["id"],
        expected_prompt_template_version=dataset["prompt_template_version"],
        expected_prompt_template_sha256=dataset["prompt_template_sha256"],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="d" * 64,
        actor="migrator",
    )

    report = service.audit("local/pnp_trash")

    assert {
        "kind": "final",
        "source_episode_index": 0,
        "approval_revision": 2,
        "reason": "missing",
    } in report["contact_sheet_issues"]


def test_audit_uses_one_sqlite_snapshot_across_concurrent_review_and_cosmos_mutation(tmp_path: Path) -> None:
    service = _service_with_states(tmp_path)
    dataset = service.database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    hook_calls = 0

    def mutate_after_episode_rows_are_read() -> None:
        nonlocal hook_calls
        hook_calls += 1
        with service.database.open_connection() as connection:
            connection.execute(
                "UPDATE episodes SET review_state='draft' WHERE dataset_id=? AND source_episode_index=0",
                (dataset["id"],),
            )
        service.database.create_cosmos_job(dataset_id=dataset["id"], configuration={"concurrent": True})

    report = build_audit_report(
        database=service.database,
        source_registry=service.source_registry,
        dataset_alias="local/pnp_trash",
        snapshot_hook=mutate_after_episode_rows_are_read,
    )

    assert hook_calls == 1
    assert report["review_state_counts"]["approved_keep"] == 1
    assert report["review_state_counts"]["draft"] == 1
    assert report["cosmos"]["job_state_counts"] == {"completed_with_failures": 1}
    persisted_episode = service.database.get_episode(dataset_id=dataset["id"], source_episode_index=0)
    assert persisted_episode["review_state"] == "draft"
    with service.database.open_connection() as connection:
        persisted_job_count = connection.execute(
            "SELECT count(*) FROM cosmos_jobs WHERE dataset_id=?", (dataset["id"],)
        ).fetchone()[0]
    assert persisted_job_count == 2


def test_audit_replacement_after_initial_inventory_check_can_never_report_current_match(
    tmp_path: Path,
) -> None:
    service = _service_with_states(tmp_path)
    record = service.source_registry.records["local/pnp_trash"]
    hook_calls = 0

    def replace_registered_metadata_after_initial_check() -> None:
        nonlocal hook_calls
        hook_calls += 1
        info = record.root / "meta" / "info.json"
        replacement = info.with_name("replacement.json")
        replacement.write_bytes(info.read_bytes())
        os.replace(replacement, info)

    report = build_audit_report(
        database=service.database,
        source_registry=service.source_registry,
        dataset_alias="local/pnp_trash",
        source_scan_hook=replace_registered_metadata_after_initial_check,
    )

    assert hook_calls == 1
    assert report["source_fingerprint"]["current_matches"] is False
    assert {
        "source_episode_index": None,
        "asset_kind": "source_inventory",
        "reason": "fingerprint_mismatch",
    } in report["unreadable_files"]


def test_audit_reuses_one_noncleaning_contact_sheet_inspector_for_all_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_with_states(tmp_path)
    constructor_calls = 0
    cleanup_calls = 0
    original_init = AtomicArtifactStore.__init__
    original_cleanup = AtomicArtifactStore._cleanup_locked

    def recording_init(self: AtomicArtifactStore, *args: object, **kwargs: object) -> None:
        nonlocal constructor_calls
        constructor_calls += 1
        original_init(self, *args, **kwargs)

    def recording_cleanup(self: AtomicArtifactStore, *args: object, **kwargs: object) -> list[dict[str, Any]]:
        nonlocal cleanup_calls
        cleanup_calls += 1
        return original_cleanup(self, *args, **kwargs)

    monkeypatch.setattr(AtomicArtifactStore, "__init__", recording_init)
    monkeypatch.setattr(AtomicArtifactStore, "_cleanup_locked", recording_cleanup)

    service.audit("local/pnp_trash")

    assert constructor_calls == 1
    assert cleanup_calls == 0


def test_audit_source_cache_avoids_rehash_decode_and_grip_until_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service_with_states(tmp_path)
    audit_module._SOURCE_EVIDENCE_CACHE.clear()
    calls = {"hash": 0, "video": 0, "grip": 0}
    original_hash = audit_module._sha256_fd
    original_video = audit_module._registered_video_readable
    original_grip = audit_module.read_grip_diagnostic

    def recording_hash(descriptor: int) -> str:
        calls["hash"] += 1
        return original_hash(descriptor)

    def recording_video(record: object, source_episode_index: int) -> bool:
        calls["video"] += 1
        return original_video(record, source_episode_index)

    def recording_grip(record: object, *, source_episode_index: int, side: str) -> object:
        calls["grip"] += 1
        return original_grip(record, source_episode_index=source_episode_index, side=side)

    monkeypatch.setattr(audit_module, "_sha256_fd", recording_hash)
    monkeypatch.setattr(audit_module, "_registered_video_readable", recording_video)
    monkeypatch.setattr(audit_module, "read_grip_diagnostic", recording_grip)

    first = service.audit("local/pnp_trash")
    assert first["source_fingerprint"]["current_matches"] is True
    assert all(count > 0 for count in calls.values())
    calls = {"hash": 0, "video": 0, "grip": 0}

    second = service.audit("local/pnp_trash")

    assert second["source_fingerprint"]["current_matches"] is True
    assert calls == {"hash": 0, "video": 0, "grip": 0}

    record = service.source_registry.records["local/pnp_trash"]
    info = record.root / "meta" / "info.json"
    replacement = info.with_name("replacement.json")
    replacement.write_bytes(info.read_bytes())
    os.replace(replacement, info)
    mutated = service.audit("local/pnp_trash")
    assert mutated["source_fingerprint"]["current_matches"] is False


@pytest.fixture
def audit_client(tmp_path: Path) -> tuple[TestClient, ReviewService]:
    service = _service_with_states(tmp_path)
    app = FastAPI()
    app.include_router(build_curation_router(review_service=service, bearer_token="audit-secret"))
    return TestClient(app, base_url="http://127.0.0.1"), service


@pytest.mark.parametrize(
    "path",
    [
        "/api/curation/episodes/0/grip?dataset_alias=local/pnp_trash",
        "/api/curation/audit?dataset_alias=local/pnp_trash",
    ],
)
def test_grip_and_audit_endpoints_require_exact_bearer(
    audit_client: tuple[TestClient, ReviewService], path: str
) -> None:
    client, _ = audit_client
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get(path, headers={"Authorization": "Bearer audit-secret"})
    assert response.status_code == 200


def test_grip_endpoint_returns_advisory_without_review_state_mutation(
    audit_client: tuple[TestClient, ReviewService],
) -> None:
    client, service = audit_client
    dataset = service.database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    before = service.database.get_episode(dataset_id=dataset["id"], source_episode_index=0)

    response = client.get(
        "/api/curation/episodes/0/grip",
        params={"dataset_alias": "local/pnp_trash"},
        headers={"Authorization": "Bearer audit-secret"},
    )

    after = service.database.get_episode(dataset_id=dataset["id"], source_episode_index=0)
    assert response.status_code == 200
    assert response.json()["advisories"] == ["grip_grasp_disagrees_with_step_2_start"]
    assert before == after
