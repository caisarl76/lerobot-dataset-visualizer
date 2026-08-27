from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import curation.contact_sheets as contact_sheet_module
from curation.contact_sheets import (
    FINAL_TRANSITION_NAMES,
    ContactSheetCoordinator,
    ContactSheetDatasetIdentity,
    ContactSheetReconcileStatus,
    ContactSheetService,
    ContactSheetUnavailable,
    final_contact_sheet_path,
    proposal_contact_sheet_path,
    receipt_path,
)
from curation.cosmos_transport import ArtifactConflict
from curation.db import CurationDatabase
from curation.models import ReviewState
from curation.prompts import PROMPT_TEMPLATE_SHA256, PROMPT_TEMPLATE_VERSION
from curation.source import SourceRegistry
from PIL import Image
import pytest


def _identity(dataset_id: int = 1, alias: str = "local/pnp_trash") -> ContactSheetDatasetIdentity:
    return ContactSheetDatasetIdentity(
        dataset_id=dataset_id,
        dataset_alias=alias,
        source_manifest_sha256=(f"{dataset_id:x}" * 64)[:64],
    )


def _decoder(calls: list[tuple[int, ...]]):
    def decode(indices: tuple[int, ...]) -> dict[int, Image.Image]:
        calls.append(indices)
        return {index: Image.new("RGB", (48, 32), color=(index % 255, 20, 30)) for index in indices}

    return decode


def test_proposal_sheet_is_six_fixed_cells_and_nulls_never_decode_or_fabricate_evidence(
    tmp_path: Path,
) -> None:
    service = ContactSheetService(tmp_path / "workspace", identity=_identity())
    proposal_id = str(uuid4())
    calls: list[tuple[int, ...]] = []

    result = service.write_proposal(
        proposal_id=proposal_id,
        transition_frames=[3, None, 9, None, 15, 18],
        timestamps=[index / 10 for index in range(20)],
        model_statuses=["completed", "not_observed", "partial", "not_observed", "completed", "completed"],
        decode_frames=_decoder(calls),
    )

    assert result.relative_path == proposal_contact_sheet_path(_identity(), proposal_id)
    assert (tmp_path / "workspace" / result.relative_path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(result.cells) == 6
    assert tuple(cell.transition_name for cell in result.cells) == FINAL_TRANSITION_NAMES
    assert calls == [(3, 9, 15, 18)]
    for position in (1, 3):
        cell = result.cells[position]
        assert cell.frame is None
        assert cell.timestamp_s is None
        assert cell.proposal_delta_s is None
        assert cell.image_decoded is False
        assert cell.placeholder == "NOT OBSERVED"
        assert "timestamp" not in cell.display_lines
        assert all("delta" not in line.lower() for line in cell.display_lines)
    integer = result.cells[0]
    assert integer.frame == 3
    assert integer.timestamp_s == 0.3
    assert integer.status == "completed"
    assert integer.image_decoded is True
    assert integer.placeholder is None


def test_approved_keep_final_sheet_uses_human_frames_and_nullable_proposal_has_no_delta(
    tmp_path: Path,
) -> None:
    service = ContactSheetService(tmp_path / "workspace", identity=_identity())
    calls: list[tuple[int, ...]] = []

    result = service.write_final(
        review_state="approved_keep",
        source_episode_index=7,
        approval_revision=12,
        final_transition_frames=[2, 5, 8, 11, 14, 17],
        proposal_transition_frames=[1, None, 10, None, 13, 18],
        timestamps=[index / 10 for index in range(20)],
        decode_frames=_decoder(calls),
    )

    assert result is not None
    assert result.relative_path == final_contact_sheet_path(_identity(), 7, 12)
    assert calls == [(2, 5, 8, 11, 14, 17)]
    assert [cell.frame for cell in result.cells] == [2, 5, 8, 11, 14, 17]
    assert result.cells[0].proposal_delta_s == pytest.approx(0.1)
    assert result.cells[1].proposal_delta_s is None
    assert "proposal: NOT OBSERVED" in result.cells[1].display_lines
    assert all("delta" not in line.lower() for line in result.cells[1].display_lines)


def test_approved_reject_emits_no_final_sheet_and_keep_rejects_incomplete_human_frames(
    tmp_path: Path,
) -> None:
    service = ContactSheetService(tmp_path / "workspace", identity=_identity())
    calls: list[tuple[int, ...]] = []
    decoder = _decoder(calls)

    rejected = service.write_final(
        review_state="approved_reject",
        source_episode_index=3,
        approval_revision=4,
        final_transition_frames=[None] * 6,
        proposal_transition_frames=[1, 2, 3, 4, 5, 6],
        timestamps=[index / 10 for index in range(10)],
        decode_frames=decoder,
    )

    assert rejected is None
    assert calls == []
    assert not (tmp_path / "workspace" / "contact_sheets" / "finals").exists()
    with pytest.raises(ContactSheetUnavailable, match="six non-null"):
        service.write_final(
            review_state="approved_keep",
            source_episode_index=3,
            approval_revision=4,
            final_transition_frames=[1, 2, None, 4, 5, 6],
            proposal_transition_frames=[1, 2, 3, 4, 5, 6],
            timestamps=[index / 10 for index in range(10)],
            decode_frames=decoder,
        )


def test_immutable_names_are_restart_safe_and_never_overwrite_prior_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    proposal_id = str(uuid4())
    arguments = {
        "proposal_id": proposal_id,
        "transition_frames": [1, 2, 3, 4, 5, 6],
        "timestamps": [index / 10 for index in range(10)],
        "model_statuses": ["completed"] * 6,
    }
    first = ContactSheetService(workspace, identity=_identity()).write_proposal(
        **arguments,
        decode_frames=lambda indices: {index: Image.new("RGB", (8, 8), "red") for index in indices},
    )
    replay = ContactSheetService(workspace, identity=_identity()).write_proposal(
        **arguments,
        decode_frames=lambda indices: {index: Image.new("RGB", (8, 8), "red") for index in indices},
    )
    assert replay.sha256 == first.sha256
    assert replay.relative_path == first.relative_path

    with pytest.raises(ArtifactConflict):
        ContactSheetService(workspace, identity=_identity()).write_proposal(
            **arguments,
            decode_frames=lambda indices: {index: Image.new("RGB", (8, 8), "blue") for index in indices},
        )


@pytest.mark.parametrize("size", [(4097, 1), (1, 4097), (3001, 3000)])
def test_injected_giant_image_is_rejected_before_conversion_without_partial_sheet(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    workspace = tmp_path / "workspace"
    proposal_id = str(uuid4())
    giant = Image.new("1", size, 1)
    converted = False
    original_convert = giant.convert

    def record_convert(*args: object, **kwargs: object) -> Image.Image:
        nonlocal converted
        converted = True
        return original_convert(*args, **kwargs)

    giant.convert = record_convert  # type: ignore[method-assign]
    relative_path = proposal_contact_sheet_path(_identity(), proposal_id)

    with pytest.raises(ContactSheetUnavailable, match="dimensions"):
        ContactSheetService(workspace, identity=_identity()).write_proposal(
            proposal_id=proposal_id,
            transition_frames=[1, None, None, None, None, None],
            timestamps=[0.0, 0.1],
            model_statuses=["completed", *("not_observed" for _ in range(5))],
            decode_frames=lambda indices: {1: giant},
        )

    assert converted is False
    assert not (workspace / relative_path).exists()
    assert not (workspace / receipt_path(relative_path)).exists()


@pytest.mark.parametrize(
    "oversize_kind",
    ["stream_width", "stream_height", "stream_pixels", "decoded_frame"],
)
def test_registered_decoder_rejects_giant_video_before_ndarray(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    oversize_kind: str,
) -> None:
    source = tmp_path / "source"
    video = source / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"registered fake video")
    record = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "manifest").records[
        "local/pnp_trash"
    ]
    ndarray_called = False

    class FakeFrame:
        width = 4097 if oversize_kind == "decoded_frame" else 16
        height = 1 if oversize_kind == "decoded_frame" else 16

        def to_ndarray(self, *, format: str) -> object:
            nonlocal ndarray_called
            ndarray_called = True
            raise AssertionError("oversize frame must be rejected before conversion")

    class FakeStream:
        width = 4097 if oversize_kind == "stream_width" else 3001 if oversize_kind == "stream_pixels" else 16
        height = 4097 if oversize_kind == "stream_height" else 3000 if oversize_kind == "stream_pixels" else 16

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def __enter__(self) -> FakeContainer:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def decode(self, stream: object):
            yield FakeFrame()

    monkeypatch.setattr(contact_sheet_module.av, "open", lambda *args, **kwargs: FakeContainer())
    decoder = contact_sheet_module.registered_frame_decoder(
        record,
        "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
    )

    with pytest.raises(ContactSheetUnavailable, match="dimensions"):
        decoder((0,))

    assert ndarray_called is False


def test_registered_decoder_rejects_excessive_frame_index_before_opening_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    video = source / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"registered fake video")
    record = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=tmp_path / "manifest").records[
        "local/pnp_trash"
    ]
    monkeypatch.setattr(
        contact_sheet_module.av,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("video must not be opened")),
    )
    decoder = contact_sheet_module.registered_frame_decoder(
        record,
        "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
    )

    with pytest.raises(ContactSheetUnavailable, match="decode work limit"):
        decoder((10_000,))


@pytest.mark.parametrize(
    "proposal_id",
    ["../escape", "/absolute", "proposal with spaces", "00000000-0000-0000-0000-00000000000z"],
)
def test_proposal_filenames_accept_only_canonical_uuid_identifiers(tmp_path: Path, proposal_id: str) -> None:
    service = ContactSheetService(tmp_path / "workspace", identity=_identity())
    with pytest.raises(ValueError, match="UUID"):
        service.write_proposal(
            proposal_id=proposal_id,
            transition_frames=[None] * 6,
            timestamps=[0.0],
            model_statuses=["not_observed"] * 6,
            decode_frames=lambda indices: {},
        )


def _coordinator_case(
    tmp_path: Path,
) -> tuple[ContactSheetCoordinator, CurationDatabase, str, list[tuple[int, ...]]]:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True)
    (source / "meta" / "info.json").write_text(json.dumps({"fps": 10}))
    (source / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 20}) + "\n")
    (source / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"registered parquet")
    video = source / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    video.write_bytes(b"registered video")
    workspace = tmp_path / "workspace"
    registry = SourceRegistry.from_paths({"local/pnp_trash": source}, workspace=workspace)
    record = registry.records["local/pnp_trash"]
    database = CurationDatabase(workspace / "curation.sqlite3")
    database.initialize()
    dataset = database.register_dataset(
        alias="local/pnp_trash",
        source_path=str(record.root),
        source_manifest_sha256=record.fingerprint,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
    )
    database.create_episode(dataset_id=dataset["id"], source_episode_index=0, source_length=20)
    job = database.create_cosmos_job(dataset_id=dataset["id"], configuration={})
    proposal_id = str(uuid4())
    attempt_id = str(uuid4())
    response = {
        "schema_version": 2,
        "episode_complete": False,
        "segments": [
            {"step": step, "status": "not_observed" if step in {3, 5} else "completed"} for step in range(1, 8)
        ],
    }
    with database.open_connection() as connection:
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) VALUES (?, ?, 0, 0, 'requesting', 'now', 'now')
            """,
            (attempt_id, job["id"]),
        )
        connection.execute(
            """
            INSERT INTO cosmos_proposals(
                id, attempt_id, model_response_json,
                step_2_start_frame, step_3_start_frame, step_4_start_frame,
                step_5_start_frame, step_6_start_frame, step_7_start_frame,
                validation_warnings_json, state, created_at
            ) VALUES (?, ?, ?, 3, NULL, 9, NULL, 15, 18, '[]', 'active', 'now')
            """,
            (proposal_id, attempt_id, json.dumps(response)),
        )
        connection.execute("UPDATE cosmos_attempts SET state='succeeded' WHERE id=?", (attempt_id,))
    calls: list[tuple[int, ...]] = []
    coordinator = ContactSheetCoordinator(
        database=database,
        dataset_id=dataset["id"],
        source_record=record,
        workspace=workspace,
        timestamp_loader=lambda episode_index: [index / 10 for index in range(20)],
        decoder_factory=lambda record, path: _decoder(calls),
    )
    return coordinator, database, proposal_id, calls


def test_coordinator_backfills_postcommit_proposal_once_without_db_artifact_rows(tmp_path: Path) -> None:
    coordinator, database, proposal_id, calls = _coordinator_case(tmp_path)

    results = coordinator.reconcile_all()

    assert [result.relative_path for result in results] == [
        proposal_contact_sheet_path(coordinator.identity, proposal_id)
    ]
    assert [result.status for result in results] == ["created"]
    assert calls == [(3, 9, 15, 18)]
    with database.open_connection() as connection:
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0

    coordinator.decoder_factory = lambda record, path: (_ for _ in ()).throw(AssertionError("must not decode"))
    replay = coordinator.reconcile_all()
    assert replay[0].status == "available"
    assert calls == [(3, 9, 15, 18)]


def test_coordinator_conflicting_preexisting_sheet_fails_closed_without_overwrite(tmp_path: Path) -> None:
    coordinator, _database, proposal_id, _calls = _coordinator_case(tmp_path)
    path = coordinator.workspace / proposal_contact_sheet_path(coordinator.identity, proposal_id)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"hostile existing evidence")

    with pytest.raises(ArtifactConflict):
        coordinator.ensure_proposal(proposal_id)

    assert path.read_bytes() == b"hostile existing evidence"


def test_final_reconciliation_uses_proposal_snapshotted_atomically_at_approval(tmp_path: Path) -> None:
    coordinator, database, proposal_id, calls = _coordinator_case(tmp_path)
    dataset_id = coordinator.dataset_id
    draft = database.transition_review_episode(
        dataset_id=dataset_id,
        source_episode_index=0,
        expected_revision=0,
        allowed_states=frozenset({ReviewState.PENDING}),
        changes={
            "review_state": ReviewState.DRAFT,
            "step_2_start_frame": 2,
            "step_3_start_frame": 5,
            "step_4_start_frame": 8,
            "step_5_start_frame": 11,
            "step_6_start_frame": 14,
            "step_7_start_frame": 17,
        },
        actor="curator",
        operation="draft_saved",
    )
    approved = database.transition_review_episode(
        dataset_id=dataset_id,
        source_episode_index=0,
        expected_revision=draft["revision"],
        allowed_states=frozenset({ReviewState.DRAFT}),
        changes={
            "review_state": ReviewState.APPROVED_KEEP,
            "approval_revision": draft["revision"] + 1,
            "reviewer": "reviewer",
            "approved_at": "2026-08-26T00:00:00Z",
        },
        actor="curator",
        operation="keep_approved",
        snapshot_active_proposal_for_contact_sheet=True,
    )
    with database.open_connection() as connection:
        event = connection.execute(
            "SELECT details_json FROM audit_events WHERE operation='keep_approved'"
        ).fetchone()
        assert json.loads(event["details_json"])["contact_sheet_proposal_id"] == proposal_id
        connection.execute("UPDATE cosmos_proposals SET state='superseded' WHERE id=?", (proposal_id,))
        attempt_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) SELECT ?, job_id, 0, 1, 'requesting', 'now', 'now'
              FROM cosmos_attempts WHERE attempt_number=0
            """,
            (attempt_id,),
        )
        connection.execute(
            """
            INSERT INTO cosmos_proposals(
                id, attempt_id, model_response_json,
                step_2_start_frame, step_3_start_frame, step_4_start_frame,
                step_5_start_frame, step_6_start_frame, step_7_start_frame,
                validation_warnings_json, state, created_at
            ) VALUES (?, ?, ?, 1, 4, 7, 10, 13, 16, '[]', 'active', 'now')
            """,
            (
                str(uuid4()),
                attempt_id,
                json.dumps(
                    {
                        "episode_complete": True,
                        "segments": [{"step": step, "status": "completed"} for step in range(1, 8)],
                    }
                ),
            ),
        )
        connection.execute("UPDATE cosmos_attempts SET state='succeeded' WHERE id=?", (attempt_id,))

    result = coordinator.ensure_final(0, approved["approval_revision"])

    assert result is not None
    assert result.cells[0].proposal_delta_s == pytest.approx(-0.1)
    assert result.cells[1].proposal_delta_s is None
    assert calls == [(2, 5, 8, 11, 14, 17)]

    statuses = coordinator.reconcile_all()
    proposal_statuses = [status for status in statuses if status.kind == "proposal"]
    assert len(proposal_statuses) == 2
    assert all(status.status == "created" for status in proposal_statuses)


def test_approved_reject_never_accepts_or_generates_a_final_sheet(tmp_path: Path) -> None:
    coordinator, database, _proposal_id, calls = _coordinator_case(tmp_path)
    rejected = database.transition_review_episode(
        dataset_id=coordinator.dataset_id,
        source_episode_index=0,
        expected_revision=0,
        allowed_states=frozenset({ReviewState.PENDING}),
        changes={
            "review_state": ReviewState.APPROVED_REJECT,
            "approval_revision": 1,
            "reviewer": "reviewer",
            "approved_at": "2026-08-26T00:00:00Z",
            "rejection_reason": "wrong task",
        },
        actor="curator",
        operation="reject_approved",
    )
    relative_path = final_contact_sheet_path(coordinator.identity, 0, 1)
    coordinator.service.store.write_bytes(relative_path, b"hostile", media_type="image/png")

    assert coordinator.ensure_final(0, rejected["approval_revision"]) is None
    assert calls == []


def test_dataset_identity_namespaces_both_sheet_and_receipt_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = _identity(1, "local/first")
    second = _identity(2, "local/second")
    proposal_id = str(uuid4())

    first_proposal = ContactSheetService(workspace, identity=first).write_proposal(
        proposal_id=proposal_id,
        transition_frames=[None] * 6,
        timestamps=[0.0],
        model_statuses=["not_observed"] * 6,
        decode_frames=lambda indices: {},
    )
    second_proposal = ContactSheetService(workspace, identity=second).write_proposal(
        proposal_id=proposal_id,
        transition_frames=[None] * 6,
        timestamps=[0.0],
        model_statuses=["not_observed"] * 6,
        decode_frames=lambda indices: {},
    )
    first_final = final_contact_sheet_path(first, 0, 2)
    second_final = final_contact_sheet_path(second, 0, 2)

    assert first_proposal.relative_path != second_proposal.relative_path
    assert first_final != second_final
    assert receipt_path(first_proposal.relative_path) != receipt_path(second_proposal.relative_path)
    assert receipt_path(first_final) != receipt_path(second_final)


def test_coordinator_rejects_cross_dataset_receipt_binding_before_decode(tmp_path: Path) -> None:
    coordinator, _database, proposal_id, calls = _coordinator_case(tmp_path)
    relative_path = proposal_contact_sheet_path(coordinator.identity, proposal_id)
    coordinator.service.store.write_json(
        receipt_path(relative_path),
        {
            "schema_version": 1,
            "kind": "proposal",
            "dataset_id": coordinator.dataset_id + 1,
            "dataset_alias": "local/other",
            "source_manifest_sha256": "0" * 64,
            "source_episode_index": 0,
            "proposal_id": proposal_id,
            "approval_revision": None,
            "final_transition_frames": None,
            "proposal_transition_frames": [3, None, 9, None, 15, 18],
            "relative_path": relative_path,
            "sha256": "0" * 64,
            "byte_size": 0,
        },
    )

    with pytest.raises(ArtifactConflict, match="binding"):
        coordinator.ensure_proposal(proposal_id)

    assert calls == []


def test_approval_event_and_receipt_bind_the_exact_immutable_evidence_snapshot(tmp_path: Path) -> None:
    coordinator, database, proposal_id, _calls = _coordinator_case(tmp_path)
    draft = database.transition_review_episode(
        dataset_id=coordinator.dataset_id,
        source_episode_index=0,
        expected_revision=0,
        allowed_states=frozenset({ReviewState.PENDING}),
        changes={
            "review_state": ReviewState.DRAFT,
            **{f"step_{step}_start_frame": frame for step, frame in zip(range(2, 8), [2, 5, 8, 11, 14, 17])},
        },
        actor="curator",
        operation="draft_saved",
    )
    approved = database.transition_review_episode(
        dataset_id=coordinator.dataset_id,
        source_episode_index=0,
        expected_revision=draft["revision"],
        allowed_states=frozenset({ReviewState.DRAFT}),
        changes={
            "review_state": ReviewState.APPROVED_KEEP,
            "approval_revision": draft["revision"] + 1,
            "reviewer": "reviewer",
            "approved_at": "2026-08-26T00:00:00Z",
        },
        actor="curator",
        operation="keep_approved",
        snapshot_active_proposal_for_contact_sheet=True,
    )
    result = coordinator.ensure_final(0, approved["approval_revision"])
    assert result is not None

    with database.open_connection() as connection:
        raw = connection.execute(
            "SELECT details_json FROM audit_events WHERE operation='keep_approved'"
        ).fetchone()["details_json"]
    snapshot = json.loads(raw)["contact_sheet_evidence"]
    assert snapshot == {
        "schema_version": 1,
        "dataset_id": coordinator.dataset_id,
        "dataset_alias": "local/pnp_trash",
        "source_manifest_sha256": coordinator.source_record.fingerprint,
        "source_episode_index": 0,
        "approval_revision": approved["approval_revision"],
        "final_transition_frames": [2, 5, 8, 11, 14, 17],
        "proposal_id": proposal_id,
        "proposal_transition_frames": [3, None, 9, None, 15, 18],
    }
    receipt = json.loads((coordinator.workspace / receipt_path(result.relative_path)).read_text())
    assert receipt["kind"] == "final"
    assert receipt["dataset_id"] == coordinator.dataset_id
    assert receipt["dataset_alias"] == "local/pnp_trash"
    assert receipt["source_manifest_sha256"] == coordinator.source_record.fingerprint
    assert receipt["source_episode_index"] == 0
    assert receipt["approval_revision"] == approved["approval_revision"]
    assert receipt["final_transition_frames"] == [2, 5, 8, 11, 14, 17]
    assert receipt["proposal_id"] == proposal_id
    assert receipt["proposal_transition_frames"] == [3, None, 9, None, 15, 18]
    assert receipt["relative_path"] == result.relative_path
    assert receipt["sha256"] == result.sha256
    assert receipt["byte_size"] == result.byte_size


def test_reconcile_all_continues_per_record_and_backfills_historical_keep_after_invalidation(
    tmp_path: Path,
) -> None:
    coordinator, database, _proposal_id, calls = _coordinator_case(tmp_path)
    draft = database.transition_review_episode(
        dataset_id=coordinator.dataset_id,
        source_episode_index=0,
        expected_revision=0,
        allowed_states=frozenset({ReviewState.PENDING}),
        changes={
            "review_state": ReviewState.DRAFT,
            **{f"step_{step}_start_frame": frame for step, frame in zip(range(2, 8), [2, 5, 8, 11, 14, 17])},
        },
        actor="curator",
        operation="draft_saved",
    )
    approved = database.transition_review_episode(
        dataset_id=coordinator.dataset_id,
        source_episode_index=0,
        expected_revision=draft["revision"],
        allowed_states=frozenset({ReviewState.DRAFT}),
        changes={
            "review_state": ReviewState.APPROVED_KEEP,
            "approval_revision": draft["revision"] + 1,
            "reviewer": "reviewer",
            "approved_at": "2026-08-26T00:00:00Z",
        },
        actor="curator",
        operation="keep_approved",
        snapshot_active_proposal_for_contact_sheet=True,
    )
    database.migrate_prompt_template(
        dataset_id=coordinator.dataset_id,
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="f" * 64,
        actor="migrator",
    )
    original_factory = coordinator.decoder_factory
    failed_once = False

    def fail_first(record: object, path: str):
        decoder = original_factory(record, path)

        def decode(indices: tuple[int, ...]):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise ContactSheetUnavailable("injected unavailable proposal")
            return decoder(indices)

        return decode

    coordinator.decoder_factory = fail_first
    statuses = coordinator.reconcile_all()

    assert all(isinstance(status, ContactSheetReconcileStatus) for status in statuses)
    assert statuses[0].status == "pending"
    assert any(
        status.kind == "final"
        and status.approval_revision == approved["approval_revision"]
        and status.status == "created"
        for status in statuses
    )
    assert calls[-1] == (2, 5, 8, 11, 14, 17)

    coordinator.decoder_factory = original_factory
    replay = coordinator.reconcile_all()
    assert all(status.status in {"available", "created"} for status in replay)


def test_crashed_final_generation_reopens_and_backfills_after_prompt_invalidation(
    tmp_path: Path,
) -> None:
    coordinator, database, _proposal_id, _calls = _coordinator_case(tmp_path)
    transitions = [2, 5, 8, 11, 14, 17]
    draft = database.transition_review_episode(
        dataset_id=coordinator.dataset_id,
        source_episode_index=0,
        expected_revision=0,
        allowed_states=frozenset({ReviewState.PENDING}),
        changes={
            "review_state": ReviewState.DRAFT,
            **{f"step_{step}_start_frame": frame for step, frame in zip(range(2, 8), transitions)},
        },
        actor="curator",
        operation="draft_saved",
    )
    approved = database.transition_review_episode(
        dataset_id=coordinator.dataset_id,
        source_episode_index=0,
        expected_revision=draft["revision"],
        allowed_states=frozenset({ReviewState.DRAFT}),
        changes={
            "review_state": ReviewState.APPROVED_KEEP,
            "approval_revision": draft["revision"] + 1,
            "reviewer": "reviewer",
            "approved_at": "2026-08-26T00:00:00Z",
        },
        actor="curator",
        operation="keep_approved",
        snapshot_active_proposal_for_contact_sheet=True,
    )
    coordinator.decoder_factory = lambda record, path: (
        lambda indices: (_ for _ in ()).throw(ContactSheetUnavailable("injected crash window"))
    )
    with pytest.raises(ContactSheetUnavailable):
        coordinator.ensure_final(0, approved["approval_revision"])

    database.migrate_prompt_template(
        dataset_id=coordinator.dataset_id,
        expected_prompt_template_version=PROMPT_TEMPLATE_VERSION,
        expected_prompt_template_sha256=PROMPT_TEMPLATE_SHA256,
        prompt_template_version="pnp-trash-prompts-v2",
        prompt_template_sha256="e" * 64,
        actor="migrator",
    )
    reopened_calls: list[tuple[int, ...]] = []
    reopened = ContactSheetCoordinator(
        database=database,
        dataset_id=coordinator.dataset_id,
        source_record=coordinator.source_record,
        workspace=coordinator.workspace,
        timestamp_loader=lambda episode_index: [index / 10 for index in range(20)],
        decoder_factory=lambda record, path: _decoder(reopened_calls),
    )

    statuses = reopened.reconcile_all()

    assert any(
        status.kind == "final"
        and status.approval_revision == approved["approval_revision"]
        and status.status == "created"
        for status in statuses
    )
    assert reopened_calls[-1] == tuple(transitions)
