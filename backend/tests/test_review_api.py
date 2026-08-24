from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
from typing import Any

from curation.assets import LocalAssetService
import curation.db as db_module
from curation.db import CurationDatabase
from curation.prompts import PROMPT_TEMPLATE_SHA256, PROMPT_TEMPLATE_VERSION
from curation.review import ReviewConflict, ReviewNotFound, ReviewService, ReviewValidation
from curation.router import build_curation_router
from curation.source import SourceRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def review_source(tmp_path: Path) -> tuple[SourceRegistry, Path]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "videos").mkdir(parents=True)
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "total_episodes": 2,
                "total_frames": 16,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        )
    )
    with (source / "meta" / "episodes.jsonl").open("w") as handle:
        for episode_index in range(2):
            handle.write(json.dumps({"episode_index": episode_index, "length": 8, "tasks": ["whole task"]}))
            handle.write("\n")
    for episode_index in range(2):
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([episode_index] * 8, type=pa.int64()),
                    "frame_index": pa.array(range(8), type=pa.int64()),
                    "timestamp": pa.array([frame / 10 for frame in range(8)], type=pa.float64()),
                }
            ),
            source / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
        )
    (source / "videos" / "episode_000000.mp4").write_bytes(b"registered video bytes")
    (source / "unknown.bin").write_bytes(b"registered unknown bytes")
    return SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "manifest"), source


@pytest.fixture
def review_service(tmp_path: Path, review_source: tuple[SourceRegistry, Path]) -> ReviewService:
    registry, _ = review_source
    database = CurationDatabase(tmp_path / "workspace" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)
    service.open_workspace("local/pnp_trash", actor="curator-a")
    return service


def _complete_draft(service: ReviewService, episode_index: int = 0) -> dict[str, Any]:
    return service.save_draft(
        dataset_alias="local/pnp_trash",
        source_episode_index=episode_index,
        expected_revision=0,
        actor="curator-a",
        object_name="  Crumpled   CAN ",
        pickup_hand="left",
        turn_direction="right",
        transition_frames=[1, 2, 3, 4, 5, 6],
    )


def _reject(service: ReviewService, episode_index: int = 0, *, revision: int = 0) -> dict[str, Any]:
    return service.approve_reject(
        dataset_alias="local/pnp_trash",
        source_episode_index=episode_index,
        expected_revision=revision,
        actor="curator-a",
        reviewer="reviewer-a",
        reason=None,
    )


def _insert_proposal(
    service: ReviewService,
    *,
    episode_index: int = 0,
    transitions: list[int | None] | None = None,
    warnings: list[str] | None = None,
) -> str:
    database = service.database
    dataset = database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    job = database.create_cosmos_job(dataset_id=dataset["id"], configuration={})
    proposal_id = f"proposal-{episode_index}"
    with database.open_connection() as connection:
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) VALUES (?, ?, ?, 0, 'requesting', 'now', 'now')
            """,
            (f"attempt-{episode_index}", job["id"], episode_index),
        )
        values = transitions if transitions is not None else [1, 2, 3, 4, 5, 6]
        connection.execute(
            """
            INSERT INTO cosmos_proposals(
                id, attempt_id, model_response_json,
                step_2_start_frame, step_3_start_frame, step_4_start_frame,
                step_5_start_frame, step_6_start_frame, step_7_start_frame,
                validation_warnings_json, state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 'now')
            """,
            (
                proposal_id,
                f"attempt-{episode_index}",
                json.dumps({"schema_version": 2, "episode_complete": all(v is not None for v in values)}),
                *values,
                json.dumps(warnings or []),
            ),
        )
    return proposal_id


def test_workspace_registration_is_one_atomic_transaction(
    tmp_path: Path,
    review_source: tuple[SourceRegistry, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, source = review_source
    record = registry.records["local/pnp_trash"]
    database = CurationDatabase(tmp_path / "atomic-workspace" / "curation.sqlite3")
    database.initialize()
    original_insert = db_module._insert_workspace_episode
    calls = 0

    def fail_mid_insert(connection: Any, **episode: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected episode insert failure")
        return original_insert(connection, **episode)

    monkeypatch.setattr(db_module, "_insert_workspace_episode", fail_mid_insert)
    with pytest.raises(RuntimeError, match="injected"):
        database.open_review_workspace(
            alias="local/pnp_trash",
            source_path=str(source.resolve()),
            source_manifest_sha256=record.fingerprint,
            episode_lengths={0: 8, 1: 8},
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
            actor="curator",
        )

    assert database.get_dataset(alias="local/pnp_trash") is None
    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0


def test_concurrent_workspace_opens_converge_on_one_exact_registration(
    tmp_path: Path, review_source: tuple[SourceRegistry, Path]
) -> None:
    registry, source = review_source
    record = registry.records["local/pnp_trash"]
    database = CurationDatabase(tmp_path / "concurrent-workspace" / "curation.sqlite3")
    database.initialize()

    def open_once(actor: str) -> dict[str, Any]:
        return database.open_review_workspace(
            alias="local/pnp_trash",
            source_path=str(source.resolve()),
            source_manifest_sha256=record.fingerprint,
            episode_lengths={0: 8, 1: 8},
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
            actor=actor,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(open_once, ["curator-a", "curator-b"]))

    assert results[0]["dataset"] == results[1]["dataset"]
    assert [row["source_episode_index"] for row in results[0]["episodes"]] == [0, 1]
    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM datasets").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM episodes").fetchone()[0] == 2


def test_workspace_and_episode_responses_are_alias_only_and_include_review_context(
    review_service: ReviewService,
) -> None:
    workspace = review_service.open_workspace("local/pnp_trash", actor="curator-a")
    episode = review_service.get_episode("local/pnp_trash", 0)

    assert workspace == {
        "dataset_alias": "local/pnp_trash",
        "source_fingerprint": workspace["source_fingerprint"],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "episode_count": 2,
    }
    assert "/" not in workspace["source_fingerprint"]
    assert episode["source_length"] == 8
    assert episode["timestamps"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert episode["decision"]["review_state"] == "pending"
    assert episode["active_proposal"] is None
    assert episode["prompt_preview"] is None
    assert episode["warnings"] == []
    assert episode["revision"] == 0
    assert episode["approval_locked"] is False
    assert "source_path" not in json.dumps(workspace)
    assert set(review_service.summary("local/pnp_trash")["counts"]) == {
        "pending",
        "draft",
        "approved_keep",
        "approved_reject",
    }


def test_swapped_registered_metadata_fails_closed(
    tmp_path: Path, review_source: tuple[SourceRegistry, Path]
) -> None:
    registry, source = review_source
    metadata = source / "meta" / "episodes.jsonl"
    replacement = source / "meta" / "replacement.jsonl"
    replacement.write_bytes(metadata.read_bytes())
    os.replace(replacement, metadata)
    database = CurationDatabase(tmp_path / "swapped-metadata" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)

    with pytest.raises(ReviewConflict) as metadata_error:
        service.open_workspace("local/pnp_trash", actor="curator")
    assert metadata_error.value.payload["error"] == "source_fingerprint_mismatch"


def test_swapped_registered_parquet_fails_closed(
    tmp_path: Path, review_source: tuple[SourceRegistry, Path]
) -> None:
    registry, source = review_source
    database = CurationDatabase(tmp_path / "swapped-parquet" / "curation.sqlite3")
    database.initialize()
    service = ReviewService(database=database, source_registry=registry)
    service.open_workspace("local/pnp_trash", actor="curator")
    parquet = source / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_replacement = parquet.with_name("replacement.parquet")
    parquet_replacement.write_bytes(parquet.read_bytes())
    os.replace(parquet_replacement, parquet)

    with pytest.raises(ReviewConflict) as parquet_error:
        service.get_episode("local/pnp_trash", 0)
    assert parquet_error.value.payload["error"] == "source_fingerprint_mismatch"


@pytest.mark.parametrize(
    "relative_path",
    [
        "meta/episodes.jsonl",
        "data/chunk-000/episode_000000.parquet",
        "videos/episode_000000.mp4",
        "unknown.bin",
    ],
)
def test_every_review_path_revalidates_complete_inventory_after_timestamps_are_cached(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
    relative_path: str,
) -> None:
    _, source = review_source
    review_service.get_episode("local/pnp_trash", 0)
    target = source / relative_path
    replacement = target.with_name(f"replacement-{target.name}")
    replacement.write_bytes(target.read_bytes())
    os.replace(replacement, target)

    operations = [
        lambda: review_service.open_workspace("local/pnp_trash", actor="curator"),
        lambda: review_service.get_episode("local/pnp_trash", 0),
        lambda: review_service.save_draft(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=0,
            actor="curator",
        ),
    ]
    for operation in operations:
        with pytest.raises(ReviewConflict) as raised:
            operation()
        assert raised.value.payload["error"] == "source_fingerprint_mismatch"


@pytest.mark.parametrize("change", ["added", "deleted"])
def test_cached_review_paths_reject_added_or_deleted_regular_files(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
    change: str,
) -> None:
    _, source = review_source
    review_service.get_episode("local/pnp_trash", 0)
    if change == "added":
        (source / "added-after-registration.bin").write_bytes(b"new inventory entry")
    else:
        (source / "unknown.bin").unlink()

    operations = [
        lambda: review_service.open_workspace("local/pnp_trash", actor="curator"),
        lambda: review_service.get_episode("local/pnp_trash", 0),
        lambda: review_service.save_draft(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=0,
            actor="curator",
        ),
    ]
    for operation in operations:
        with pytest.raises(ReviewConflict) as raised:
            operation()
        assert raised.value.payload["error"] == "source_fingerprint_mismatch"


def test_draft_normalizes_object_previews_prompts_and_validates_boundaries(review_service: ReviewService) -> None:
    draft = _complete_draft(review_service)

    assert draft["decision"]["object_name"] == "Crumpled CAN"
    assert draft["decision"]["transition_frames"] == [1, 2, 3, 4, 5, 6]
    assert draft["prompt_preview"][1] == "pick up the Crumpled CAN from the table with the left hand"
    assert draft["revision"] == 1
    assert draft["decision"]["review_state"] == "draft"

    for invalid in ([0, 2, 3, 4, 5, 6], [1, 2, 2, 4, 5, 6], [1, 2, 3, 4, 5, 8], [1, 2, 3]):
        with pytest.raises(ReviewValidation):
            review_service.save_draft(
                dataset_alias="local/pnp_trash",
                source_episode_index=1,
                expected_revision=0,
                actor="curator",
                transition_frames=list(invalid),
            )


def test_whitespace_only_draft_object_is_validation_error_without_revision_change(
    review_service: ReviewService,
) -> None:
    with pytest.raises(ReviewValidation, match="object_name"):
        review_service.save_draft(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=0,
            actor="curator",
            object_name=" \t  ",
        )

    current = review_service.get_episode("local/pnp_trash", 0)
    assert current["revision"] == 0
    assert current["decision"]["review_state"] == "pending"
    assert current["decision"]["object_name"] is None


def test_pending_and_draft_allowed_edges_and_approval_invariants(review_service: ReviewService) -> None:
    draft = _complete_draft(review_service)
    approved = review_service.approve_keep(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=draft["revision"],
        actor="curator-a",
        reviewer="reviewer-a",
    )

    assert approved["decision"]["review_state"] == "approved_keep"
    assert approved["approval_locked"] is True
    assert approved["reviewer"] == "reviewer-a"
    assert approved["approval_revision"] == approved["revision"] == 2
    assert approved["approved_at"]

    reopened = review_service.reopen(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=approved["revision"],
        actor="curator-a",
    )
    assert reopened["decision"]["review_state"] == "draft"
    assert reopened["reviewer"] is None
    assert reopened["approval_revision"] is None
    assert reopened["approved_at"] is None

    rejected = _reject(review_service, 1)
    assert rejected["decision"]["review_state"] == "approved_reject"
    assert rejected["decision"]["rejection_reason"] is None
    assert rejected["approval_revision"] == rejected["revision"]
    assert (
        review_service.reopen(
            dataset_alias="local/pnp_trash",
            source_episode_index=1,
            expected_revision=rejected["revision"],
            actor="curator-a",
        )["decision"]["review_state"]
        == "draft"
    )


@pytest.mark.parametrize(
    "state,operation",
    [
        ("pending", "approve_keep"),
        ("pending", "reopen"),
        ("draft", "reopen"),
        ("approved_keep", "save_draft"),
        ("approved_keep", "apply_proposal"),
        ("approved_keep", "approve_keep"),
        ("approved_keep", "approve_reject"),
        ("approved_reject", "save_draft"),
        ("approved_reject", "apply_proposal"),
        ("approved_reject", "approve_keep"),
        ("approved_reject", "approve_reject"),
    ],
)
def test_forbidden_review_edges_are_rejected_without_mutation(
    review_service: ReviewService, state: str, operation: str
) -> None:
    if operation == "apply_proposal":
        _insert_proposal(review_service)
    if state == "draft":
        current = _complete_draft(review_service)
    elif state == "approved_keep":
        draft = _complete_draft(review_service)
        current = review_service.approve_keep(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=draft["revision"],
            actor="curator",
            reviewer="reviewer",
        )
    elif state == "approved_reject":
        current = _reject(review_service)
    else:
        current = review_service.get_episode("local/pnp_trash", 0)

    arguments: dict[str, Any] = {
        "dataset_alias": "local/pnp_trash",
        "source_episode_index": 0,
        "expected_revision": current["revision"],
        "actor": "curator",
    }
    if operation in {"approve_keep", "approve_reject"}:
        arguments["reviewer"] = "reviewer"
    if operation == "approve_reject":
        arguments["reason"] = "bad"
    before = review_service.get_episode("local/pnp_trash", 0)

    with pytest.raises(ReviewConflict):
        getattr(review_service, operation)(**arguments)

    assert review_service.get_episode("local/pnp_trash", 0) == before


def test_apply_complete_and_incomplete_proposals_only_create_drafts(review_service: ReviewService) -> None:
    proposal_id = _insert_proposal(
        review_service,
        transitions=[1, 2, None, 4, None, 6],
        warnings=["steps 4 and 6 were not observed"],
    )
    applied = review_service.apply_proposal(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=0,
        actor="curator",
    )

    assert applied["decision"]["review_state"] == "draft"
    assert applied["decision"]["transition_frames"] == [1, 2, None, 4, None, 6]
    assert applied["active_proposal"]["id"] == proposal_id
    assert applied["warnings"] == [
        "draft_object_name_invalid",
        "draft_pickup_hand_invalid",
        "draft_transition_frames_invalid",
        "draft_turn_direction_invalid",
        "steps 4 and 6 were not observed",
    ]
    with pytest.raises(ReviewValidation):
        review_service.approve_keep(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=applied["revision"],
            actor="curator",
            reviewer="reviewer",
        )


def test_draft_can_apply_a_proposal_or_be_approved_reject(review_service: ReviewService) -> None:
    draft = _complete_draft(review_service)
    _insert_proposal(review_service, transitions=[1, 2, 3, 4, 5, 7])
    applied = review_service.apply_proposal(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=draft["revision"],
        actor="curator",
    )
    assert applied["decision"]["review_state"] == "draft"
    assert applied["decision"]["transition_frames"] == [1, 2, 3, 4, 5, 7]
    rejected = review_service.approve_reject(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=applied["revision"],
        actor="curator",
        reviewer="reviewer",
        reason="object missed the bin",
    )
    assert rejected["decision"]["review_state"] == "approved_reject"
    assert rejected["decision"]["rejection_reason"] == "object missed the bin"


def test_approved_rows_are_unchanged_when_a_new_proposal_appears(review_service: ReviewService) -> None:
    approved = _reject(review_service)
    _insert_proposal(review_service)

    current = review_service.get_episode("local/pnp_trash", 0)
    assert current["decision"] == approved["decision"]
    assert current["revision"] == approved["revision"]
    assert current["approval_locked"] is True
    assert current["active_proposal"] is not None


def test_approved_lock_is_checked_before_looking_for_a_proposal(review_service: ReviewService) -> None:
    approved = _reject(review_service)

    with pytest.raises(ReviewConflict) as raised:
        review_service.apply_proposal(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=approved["revision"],
            actor="curator",
        )

    assert raised.value.payload["error"] == "approval_locked"


def test_prompt_change_invalidates_only_keeps_in_one_transaction(
    review_service: ReviewService, review_source: tuple[SourceRegistry, Path]
) -> None:
    draft = _complete_draft(review_service)
    keep = review_service.approve_keep(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=draft["revision"],
        actor="curator",
        reviewer="reviewer",
    )
    reject = _reject(review_service, 1)
    changed = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="c" * 64,
    )

    changed.migrate_prompt_contract(
        "local/pnp_trash",
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        actor="template-migrator",
    )
    invalidated = changed.get_episode("local/pnp_trash", 0)
    unchanged_reject = changed.get_episode("local/pnp_trash", 1)

    assert invalidated["decision"]["review_state"] == "draft"
    assert invalidated["revision"] == keep["revision"] + 1
    assert invalidated["reviewer"] is None
    assert invalidated["approval_revision"] is None
    assert invalidated["approved_at"] is None
    assert unchanged_reject["decision"] == reject["decision"]
    assert unchanged_reject["revision"] == reject["revision"]
    with review_service.database.open_connection() as connection:
        events = connection.execute(
            "SELECT operation, episode_id, actor FROM audit_events WHERE operation='prompt_template_invalidated'"
        ).fetchall()
        assert len(events) == 1
        assert events[0]["actor"] == "template-migrator"


def test_approval_wins_then_prompt_invalidation_reopens_the_keep(review_service: ReviewService) -> None:
    draft = _complete_draft(review_service)
    approved = review_service.approve_keep(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=draft["revision"],
        actor="curator",
        reviewer="reviewer",
    )
    dataset = review_service.database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None

    invalidated = review_service.database.migrate_prompt_template(
        dataset_id=dataset["id"],
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="1" * 64,
        actor="template-migrator",
    )

    assert len(invalidated) == 1
    assert invalidated[0]["review_state"] == "draft"
    assert invalidated[0]["revision"] == approved["revision"] + 1
    assert invalidated[0]["approval_revision"] is None


def test_prompt_invalidation_between_preflight_and_approval_blocks_stale_keep(
    review_service: ReviewService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = _complete_draft(review_service)
    database = review_service.database
    dataset = database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    original_transition = database.transition_review_episode

    def invalidate_then_transition(**arguments: Any) -> dict[str, Any]:
        database.migrate_prompt_template(
            dataset_id=dataset["id"],
            expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
            expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
            prompt_template_version="pnp-trash-prompts-v2",
            prompt_template_sha256="2" * 64,
            actor="template-migrator",
        )
        return original_transition(**arguments)

    monkeypatch.setattr(database, "transition_review_episode", invalidate_then_transition)
    with pytest.raises(ReviewConflict) as raised:
        review_service.approve_keep(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=draft["revision"],
            actor="curator",
            reviewer="reviewer",
        )

    assert raised.value.payload["error"] == "prompt_contract_conflict"
    assert raised.value.payload["required_prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert raised.value.payload["required_prompt_template_sha256"] == PROMPT_TEMPLATE_SHA256
    assert raised.value.payload["dataset_prompt_template_version"] == "pnp-trash-prompts-v2"
    assert raised.value.payload["dataset_prompt_template_sha256"] == "2" * 64
    assert raised.value.payload["episode_prompt_template_sha256"] == PROMPT_TEMPLATE_SHA256
    current = database.get_episode(dataset_id=dataset["id"], source_episode_index=0)
    assert current is not None
    assert current["review_state"] == "draft"
    assert current["revision"] == draft["revision"]
    assert current["approval_revision"] is None


def test_stale_service_is_blocked_after_another_service_changes_the_template(
    review_service: ReviewService, review_source: tuple[SourceRegistry, Path]
) -> None:
    newer = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="f" * 64,
    )
    newer.migrate_prompt_contract(
        "local/pnp_trash",
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        actor="template-migrator",
    )

    with pytest.raises(ReviewConflict) as raised:
        review_service.summary("local/pnp_trash")

    assert raised.value.payload == {
        "error": "stale_review_service",
        "dataset_alias": "local/pnp_trash",
        "current_prompt_template_version": "pnp-trash-prompts-v2",
        "current_prompt_template_sha256": "f" * 64,
    }
    with pytest.raises(ReviewConflict) as reopened:
        review_service.open_workspace("local/pnp_trash", actor="old-service")
    assert reopened.value.payload["error"] == "stale_review_service"


def test_fresh_stale_service_after_upgrade_cannot_downgrade_prompt_contract(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
) -> None:
    newer = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="a" * 64,
    )
    newer.migrate_prompt_contract(
        "local/pnp_trash",
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        actor="template-migrator",
    )
    fresh_stale = ReviewService(database=review_service.database, source_registry=review_source[0])

    with pytest.raises(ReviewConflict) as raised:
        fresh_stale.open_workspace("local/pnp_trash", actor="stale-process")

    assert raised.value.payload["error"] == "stale_review_service"
    persisted = review_service.database.get_dataset(alias="local/pnp_trash")
    assert persisted is not None
    assert persisted["prompt_template_version"] == "pnp-trash-prompts-v2"
    assert persisted["prompt_template_sha256"] == "a" * 64


def test_normal_open_never_implicitly_upgrades_prompt_contract(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
) -> None:
    newer = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="9" * 64,
    )

    with pytest.raises(ReviewConflict) as raised:
        newer.open_workspace("local/pnp_trash", actor="normal-open")

    assert raised.value.payload["error"] == "stale_review_service"
    persisted = review_service.database.get_dataset(alias="local/pnp_trash")
    assert persisted is not None
    assert persisted["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert persisted["prompt_template_sha256"] == PROMPT_TEMPLATE_SHA256


def test_cached_old_and_new_concurrent_opens_cannot_revert_upgraded_prompt_contract(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
) -> None:
    newer = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="b" * 64,
    )
    newer.migrate_prompt_contract(
        "local/pnp_trash",
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        actor="template-migrator",
    )
    old_process = review_service
    new_process = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="b" * 64,
    )
    barrier = threading.Barrier(2)

    def open_after_barrier(service: ReviewService, actor: str) -> str:
        barrier.wait()
        try:
            service.open_workspace("local/pnp_trash", actor=actor)
        except ReviewConflict as error:
            return error.payload["error"]
        return "opened"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(open_after_barrier, old_process, "old-process"),
            executor.submit(open_after_barrier, new_process, "new-process"),
        ]
        results = {future.result() for future in futures}

    assert results == {"stale_review_service", "opened"}
    persisted = review_service.database.get_dataset(alias="local/pnp_trash")
    assert persisted is not None
    assert persisted["prompt_template_version"] == "pnp-trash-prompts-v2"
    assert persisted["prompt_template_sha256"] == "b" * 64


def test_prompt_migration_compare_and_swap_conflict_and_downgrade_rejection(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
) -> None:
    version_two = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="c" * 64,
    )
    version_three = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_template_version="pnp-trash-prompts-v3",
        prompt_template_sha256="d" * 64,
    )
    version_two.migrate_prompt_contract(
        "local/pnp_trash",
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        actor="v2-migrator",
    )

    with pytest.raises(ReviewConflict) as cas_conflict:
        version_three.migrate_prompt_contract(
            "local/pnp_trash",
            expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
            expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
            actor="v3-migrator",
        )
    assert cas_conflict.value.payload["error"] == "prompt_migration_conflict"
    assert cas_conflict.value.payload["current_prompt_template_version"] == "pnp-trash-prompts-v2"

    with pytest.raises(ReviewConflict) as downgrade:
        review_service.migrate_prompt_contract(
            "local/pnp_trash",
            expected_prompt_template_version="pnp-trash-prompts-v2",
            expected_prompt_template_sha256="c" * 64,
            actor="old-migrator",
        )
    assert downgrade.value.payload["error"] == "prompt_migration_downgrade"
    persisted = review_service.database.get_dataset(alias="local/pnp_trash")
    assert persisted is not None
    assert persisted["prompt_template_version"] == "pnp-trash-prompts-v2"
    assert persisted["prompt_template_sha256"] == "c" * 64


def test_new_service_reconstructs_persisted_workspace_without_process_cache(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
) -> None:
    draft = _complete_draft(review_service)
    restarted = ReviewService(database=review_service.database, source_registry=review_source[0])

    summary = restarted.summary("local/pnp_trash")
    current = restarted.get_episode("local/pnp_trash", 0)

    assert summary["counts"]["draft"] == 1
    assert current["revision"] == draft["revision"]
    assert current["decision"] == draft["decision"]


def test_response_warnings_report_draft_and_corrupt_approval_invariants(
    review_service: ReviewService,
) -> None:
    draft = review_service.save_draft(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=0,
        actor="curator",
        transition_frames=[1, 2, None, 4, 5, 6],
    )
    assert draft["warnings"] == [
        "draft_object_name_invalid",
        "draft_pickup_hand_invalid",
        "draft_transition_frames_invalid",
        "draft_turn_direction_invalid",
    ]

    complete = _complete_draft(review_service, 1)
    review_service.approve_keep(
        dataset_alias="local/pnp_trash",
        source_episode_index=1,
        expected_revision=complete["revision"],
        actor="curator",
        reviewer="reviewer",
    )
    dataset = review_service.database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    episode = review_service.database.get_episode(dataset_id=dataset["id"], source_episode_index=1)
    assert episode is not None
    with review_service.database.open_connection() as connection:
        connection.execute(
            "UPDATE episodes SET approval_revision=NULL, reviewer=NULL WHERE id=?",
            (episode["id"],),
        )
    corrupt = review_service.get_episode("local/pnp_trash", 1)
    assert corrupt["warnings"] == ["approval_reviewer_missing", "approval_revision_mismatch"]


def test_prompt_invalidation_rolls_back_dataset_rows_and_audits_together(
    review_service: ReviewService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for episode_index in range(2):
        draft = _complete_draft(review_service, episode_index)
        review_service.approve_keep(
            dataset_alias="local/pnp_trash",
            source_episode_index=episode_index,
            expected_revision=draft["revision"],
            actor="curator",
            reviewer="reviewer",
        )
    database = review_service.database
    dataset_before = database.get_dataset(alias="local/pnp_trash")
    episodes_before = database.list_episodes(dataset_id=dataset_before["id"])
    original_append = database._append_audit_event
    calls = 0

    def fail_second_audit(connection: Any, **event: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected audit failure")
        return original_append(connection, **event)

    monkeypatch.setattr(database, "_append_audit_event", fail_second_audit)
    with pytest.raises(RuntimeError, match="injected"):
        database.migrate_prompt_template(
            dataset_id=dataset_before["id"],
            expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
            expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
            prompt_template_version="pnp-trash-prompts-v2",
            prompt_template_sha256="e" * 64,
            actor="template-migrator",
        )

    assert database.get_dataset(alias="local/pnp_trash") == dataset_before
    assert database.list_episodes(dataset_id=dataset_before["id"]) == episodes_before
    with database.open_connection() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM audit_events WHERE operation='prompt_template_invalidated'"
            ).fetchone()[0]
            == 0
        )


def test_stale_prompt_hash_and_empty_reviewer_cannot_be_approved(review_service: ReviewService) -> None:
    draft = _complete_draft(review_service)
    database = review_service.database
    dataset = database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    database.update_episode(
        dataset_id=dataset["id"],
        source_episode_index=0,
        expected_revision=draft["revision"],
        changes={"prompt_template_sha256": "d" * 64},
        actor="test-corruptor",
    )
    current = review_service.get_episode("local/pnp_trash", 0)

    with pytest.raises(ReviewValidation, match="stale"):
        review_service.approve_keep(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=current["revision"],
            actor="curator",
            reviewer="reviewer",
        )

    for reviewer in ("", "   "):
        with pytest.raises(ReviewValidation):
            review_service.approve_keep(
                dataset_alias="local/pnp_trash",
                source_episode_index=0,
                expected_revision=current["revision"],
                actor="curator",
                reviewer=reviewer,
            )


def test_unnormalized_stored_object_cannot_be_approved(review_service: ReviewService) -> None:
    draft = _complete_draft(review_service)
    database = review_service.database
    dataset = database.get_dataset(alias="local/pnp_trash")
    assert dataset is not None
    corrupted = database.update_episode(
        dataset_id=dataset["id"],
        source_episode_index=0,
        expected_revision=draft["revision"],
        changes={"object_name": "  Crumpled CAN  "},
        actor="test-corruptor",
    )

    with pytest.raises(ReviewValidation, match="already normalized"):
        review_service.approve_keep(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=corrupted["revision"],
            actor="curator",
            reviewer="reviewer",
        )


@pytest.mark.parametrize("mode", ["raises", "six", "empty", "wrong"])
def test_invalid_or_failing_prompt_expander_cannot_be_approved(
    review_service: ReviewService,
    review_source: tuple[SourceRegistry, Path],
    mode: str,
) -> None:
    draft = _complete_draft(review_service)
    calls: list[dict[str, str]] = []

    def invalid_expander(**arguments: str) -> list[str]:
        calls.append(arguments)
        if mode == "raises":
            raise RuntimeError("prompt expansion failed")
        if mode == "six":
            return ["prompt"] * 6
        if mode == "empty":
            return ["prompt"] * 6 + [""]
        return ["plausible but wrong prompt"] * 7

    validating = ReviewService(
        database=review_service.database,
        source_registry=review_source[0],
        prompt_expander=invalid_expander,
    )
    validating.open_workspace("local/pnp_trash", actor="curator")

    with pytest.raises(ReviewValidation, match="seven nonempty"):
        validating.approve_keep(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=draft["revision"],
            actor="curator",
            reviewer="reviewer",
        )

    assert calls == [{"object_name": "Crumpled CAN", "hand": "left", "turn": "right"}]


def test_optimistic_conflict_payload_returns_the_current_episode_in_both_directions(
    review_service: ReviewService,
) -> None:
    first = _complete_draft(review_service)
    with pytest.raises(ReviewConflict) as older:
        review_service.save_draft(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=0,
            actor="other",
            object_name="cup",
        )
    assert older.value.payload["error"] == "revision_conflict"
    assert older.value.payload["episode"]["revision"] == first["revision"]

    second = review_service.save_draft(
        dataset_alias="local/pnp_trash",
        source_episode_index=0,
        expected_revision=first["revision"],
        actor="other",
        object_name="cup",
    )
    with pytest.raises(ReviewConflict) as newer:
        review_service.save_draft(
            dataset_alias="local/pnp_trash",
            source_episode_index=0,
            expected_revision=first["revision"],
            actor="first-client",
            object_name="bottle",
        )
    assert newer.value.payload["episode"]["revision"] == second["revision"]
    assert newer.value.payload["episode"]["decision"]["object_name"] == "cup"


def test_unknown_or_changed_source_never_reuses_registered_decisions(
    tmp_path: Path, review_service: ReviewService, review_source: tuple[SourceRegistry, Path]
) -> None:
    with pytest.raises(ReviewNotFound):
        review_service.open_workspace("/absolute/source", actor="curator")

    _, source = review_source
    (source / "meta" / "info.json").write_text(json.dumps({"fps": 11}))
    changed_registry = SourceRegistry.from_paths(
        {"local/pnp_trash": source}, workspace=tmp_path / "changed-manifest"
    )
    changed_service = ReviewService(database=review_service.database, source_registry=changed_registry)
    with pytest.raises(ReviewConflict, match="fingerprint"):
        changed_service.open_workspace("local/pnp_trash", actor="curator")


def test_workspace_open_requires_an_actor(review_service: ReviewService) -> None:
    with pytest.raises(ReviewValidation, match="actor"):
        review_service.open_workspace("local/pnp_trash", actor="   ")


@pytest.fixture
def review_client(review_service: ReviewService) -> TestClient:
    app = FastAPI()
    app.include_router(
        build_curation_router(
            asset_service=LocalAssetService(review_service.source_registry),
            review_service=review_service,
            bearer_token="secret-token",
        )
    )
    return TestClient(app, base_url="http://127.0.0.1")


def test_review_api_requires_exact_bearer_and_rejects_paths(review_client: TestClient) -> None:
    route = "/api/curation/workspaces/open"
    body = {"dataset_alias": "local/pnp_trash", "actor": "curator"}
    assert review_client.post(route, json=body).status_code == 401
    assert (
        review_client.post(
            route,
            headers={"Authorization": "Bearer wrong"},
            json=body,
        ).status_code
        == 401
    )
    good = review_client.post(
        route,
        headers={"Authorization": "Bearer secret-token"},
        json=body,
    )
    assert good.status_code == 200
    assert good.json()["dataset_alias"] == "local/pnp_trash"

    path_leak = review_client.post(
        route,
        headers={"Authorization": "Bearer secret-token"},
        json={**body, "local_path": "/tmp/source"},
    )
    assert path_leak.status_code == 422
    absolute_alias = review_client.post(
        route,
        headers={"Authorization": "Bearer secret-token"},
        json={"dataset_alias": "/tmp/source", "actor": "curator"},
    )
    assert absolute_alias.status_code == 404
    missing_actor = review_client.post(
        route,
        headers={"Authorization": "Bearer secret-token"},
        json={"dataset_alias": "local/pnp_trash"},
    )
    assert missing_actor.status_code == 422


def test_review_api_shapes_mutations_and_conflicts(review_client: TestClient) -> None:
    headers = {"Authorization": "Bearer secret-token"}
    episode = review_client.get(
        "/api/curation/episodes/0", params={"dataset_alias": "local/pnp_trash"}, headers=headers
    )
    assert episode.status_code == 200
    assert {
        "dataset_alias",
        "source_episode_index",
        "source_length",
        "timestamps",
        "decision",
        "active_proposal",
        "prompt_preview",
        "warnings",
        "revision",
        "approval_locked",
        "reviewer",
        "approval_revision",
        "approved_at",
    } == set(episode.json())

    body = {
        "dataset_alias": "local/pnp_trash",
        "expected_revision": 0,
        "actor": "curator",
        "object_name": "can",
        "pickup_hand": "left",
        "turn_direction": "right",
        "transition_frames": [1, 2, 3, 4, 5, 6],
    }
    saved = review_client.patch("/api/curation/episodes/0/draft", headers=headers, json=body)
    assert saved.status_code == 200
    conflict = review_client.patch(
        "/api/curation/episodes/0/draft", headers=headers, json={**body, "object_name": "cup"}
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "revision_conflict"
    assert conflict.json()["episode"]["revision"] == 1

    approval = review_client.post(
        "/api/curation/episodes/0/approve-keep",
        headers=headers,
        json={
            "dataset_alias": "local/pnp_trash",
            "expected_revision": 1,
            "actor": "curator",
            "reviewer": "reviewer",
        },
    )
    assert approval.status_code == 200
    assert approval.json()["approval_revision"] == approval.json()["revision"]


def test_review_api_rejects_whitespace_object_without_mutation(review_client: TestClient) -> None:
    headers = {"Authorization": "Bearer secret-token"}
    response = review_client.patch(
        "/api/curation/episodes/0/draft",
        headers=headers,
        json={
            "dataset_alias": "local/pnp_trash",
            "expected_revision": 0,
            "actor": "curator",
            "object_name": "   ",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_review"
    current = review_client.get(
        "/api/curation/episodes/0",
        params={"dataset_alias": "local/pnp_trash"},
        headers=headers,
    )
    assert current.status_code == 200
    assert current.json()["revision"] == 0
    assert current.json()["decision"]["review_state"] == "pending"


@pytest.mark.parametrize(
    "payload",
    [
        {"expected_revision": True, "transition_frames": [1, 2, 3, 4, 5, 6]},
        {"expected_revision": "0", "transition_frames": [1, 2, 3, 4, 5, 6]},
        {"expected_revision": 0, "transition_frames": [1, 2, 3, 4, 5, True]},
        {"expected_revision": 0, "transition_frames": [1, 2, 3, 4, 5, "6"]},
    ],
)
def test_review_api_rejects_coerced_revision_and_frame_types(
    review_client: TestClient, payload: dict[str, Any]
) -> None:
    response = review_client.patch(
        "/api/curation/episodes/0/draft",
        headers={"Authorization": "Bearer secret-token"},
        json={"dataset_alias": "local/pnp_trash", "actor": "curator", **payload},
    )
    assert response.status_code == 422


def test_all_review_routes_are_wired_authenticated_and_assets_are_not_bearer_protected(
    review_client: TestClient, review_service: ReviewService
) -> None:
    mutation = {
        "dataset_alias": "local/pnp_trash",
        "expected_revision": 0,
        "actor": "curator",
    }
    approval = {**mutation, "reviewer": "reviewer"}
    routes = [
        ("POST", "/api/curation/workspaces/open", {"dataset_alias": "local/pnp_trash", "actor": "curator"}),
        ("GET", "/api/curation/summary?dataset_alias=local/pnp_trash", None),
        ("GET", "/api/curation/episodes/0?dataset_alias=local/pnp_trash", None),
        ("PATCH", "/api/curation/episodes/0/draft", mutation),
        ("POST", "/api/curation/episodes/0/apply-proposal", mutation),
        ("POST", "/api/curation/episodes/0/approve-keep", approval),
        ("POST", "/api/curation/episodes/0/approve-reject", approval),
        ("POST", "/api/curation/episodes/0/reopen", mutation),
    ]
    for method, path, body in routes:
        response = review_client.request(method, path, json=body)
        assert response.status_code == 401, (method, path, response.text)

    headers = {"Authorization": "Bearer secret-token"}
    summary = review_client.get(
        "/api/curation/summary", params={"dataset_alias": "local/pnp_trash"}, headers=headers
    )
    assert summary.status_code == 200
    assert summary.json()["counts"]["pending"] == 2

    _insert_proposal(review_service)
    applied = review_client.post("/api/curation/episodes/0/apply-proposal", headers=headers, json=mutation)
    assert applied.status_code == 200
    assert applied.json()["decision"]["review_state"] == "draft"
    rejected = review_client.post(
        "/api/curation/episodes/0/approve-reject",
        headers=headers,
        json={**approval, "expected_revision": 1, "reason": None},
    )
    assert rejected.status_code == 200
    reopened = review_client.post(
        "/api/curation/episodes/0/reopen",
        headers=headers,
        json={**mutation, "expected_revision": 2},
    )
    assert reopened.status_code == 200
    assert reopened.json()["decision"]["review_state"] == "draft"

    asset_path = "/api/local-datasets/local/pnp_trash/resolve/main/meta/info.json"
    assert review_client.get(asset_path).status_code == 200
    assert review_client.get(asset_path, headers=headers).status_code == 400
