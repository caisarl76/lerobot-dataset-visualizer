from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import threading
import time

from curation.cosmos_contract import CosmosContractError
from curation.cosmos_transport import (
    ArtifactConflict,
    ArtifactSecurityError,
    AtomicArtifactStore,
    PreparedSample,
    build_canonical_prompt,
    build_parsed_artifact,
    build_request_artifact,
    prepare_initial_request,
)
from curation.db import (
    ARTIFACT_KIND_COSMOS_PARSED,
    ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE,
    ARTIFACT_KIND_COSMOS_REQUEST,
    ARTIFACT_KIND_COSMOS_RESPONSE,
    CurationDatabase,
)
import numpy as np
from PIL import Image
import pytest
from test_cosmos_contract import COMPLETE_RESPONSE


def _jpeg(red: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), (red, 0, 0)).save(
        output,
        format="JPEG",
        quality=85,
        optimize=False,
        progressive=False,
    )
    return output.getvalue()


def _sample() -> PreparedSample:
    timestamps = tuple(index / 50 for index in range(101))
    indices = (0, 25, 50, 75, 100)
    return PreparedSample(
        source_fps=50.0,
        total_num_frames=101,
        duration_s=2.02,
        frame_indices=indices,
        parquet_timestamps=tuple(timestamps[index] for index in indices),
        all_parquet_timestamps=timestamps,
        jpeg_frames=tuple(_jpeg(index) for index in indices),
        decoder_name="PyAV",
        decoder_version="17.1.0",
        source_video_sha256="a" * 64,
    )


def test_request_artifact_is_derived_canonically_and_redacts_only_base64_payload() -> None:
    sample = _sample()
    prompt = build_canonical_prompt()
    prepared = prepare_initial_request(model="model", prompt=prompt, sample=sample)
    artifact = build_request_artifact(
        attempt_id="00000000-0000-0000-0000-000000000001",
        source_episode_index=7,
        prepared_request=prepared,
    )

    assert artifact.keys() == {
        "schema_version",
        "contract_version",
        "attempt_id",
        "source_episode_index",
        "source_video_sha256",
        "sampled_payload_sha256",
        "sampling",
        "request_body",
    }
    assert artifact["request_body"] == prepared.redacted_body
    assert artifact["sampling"] == {
        "original_fps": 50.0,
        "original_frame_count": 101,
        "original_duration_s": 2.02,
        "target_fps": 2,
        "selected_frame_indices": [0, 25, 50, 75, 100],
        "selected_parquet_timestamps_s": [0.0, 0.5, 1.0, 1.5, 2.0],
        "decoder": {"name": "PyAV", "version": "17.1.0"},
        "color_space": "RGB",
        "resize": {"allow_upscale": False, "max_long_edge": 640, "resampling": "LANCZOS"},
        "jpeg": {"quality": 85, "optimize": False, "progressive": False},
    }
    assert "data:video" not in json.dumps(artifact)
    with pytest.raises(TypeError):
        build_request_artifact(
            attempt_id="00000000-0000-0000-0000-000000000001",
            source_episode_index=7,
            prepared_request=prepared,
            request_body={"tampered": True},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_fps": 0.0},
        {"source_fps": float("nan")},
        {"source_fps": True},
        {"source_fps": 50},
        {"source_fps": np.float64(50)},
        {"total_num_frames": 0},
        {"total_num_frames": True},
        {"total_num_frames": np.int64(101)},
        {"duration_s": 2.021},
        {"duration_s": 2},
        {"duration_s": float("inf")},
        {"duration_s": np.float64(2.02)},
        {"frame_indices": [0, 25, 50, 75, 100]},
        {"frame_indices": (0,)},
        {"frame_indices": (0, 25, 50, 75, 75)},
        {"frame_indices": (0, 25, 50, 75, 101)},
        {"frame_indices": (0, 25, 50, 75, True)},
        {"frame_indices": (0, 25, 50, 75, np.int64(100))},
        {"parquet_timestamps": (0.0, 0.5, 1.0, 1.5, 1.99)},
        {"parquet_timestamps": (0.0, 0.5, 1.0, 1.5, float("nan"))},
        {"parquet_timestamps": (0.0, 0.5, 1.0, 1.5, 2)},
        {"parquet_timestamps": (0.0, 0.5, 1.0, 1.5, np.float64(2.0))},
        {"all_parquet_timestamps": tuple(index / 50 for index in range(100))},
        {"all_parquet_timestamps": (0.0,) * 101},
        {"all_parquet_timestamps": (0.0, float("nan")) + tuple(index / 50 for index in range(2, 101))},
        {"all_parquet_timestamps": (np.float64(0),) + tuple(index / 50 for index in range(1, 101))},
        {"jpeg_frames": (_jpeg(0), _jpeg(25), _jpeg(50), _jpeg(75), b"")},
        {"jpeg_frames": (_jpeg(0), _jpeg(25), _jpeg(50), _jpeg(75), b"not-jpeg")},
        {"decoder_name": ""},
        {"decoder_version": ""},
        {"source_video_sha256": "A" * 64},
        {"source_video_sha256": "a" * 63},
    ],
)
def test_prepared_sample_rejects_hostile_or_inconsistent_state(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_sample(), **changes)


def test_parsed_artifact_reparses_and_resnaps_the_exact_raw_response() -> None:
    raw_response = json.dumps(COMPLETE_RESPONSE, ensure_ascii=False)
    timeline = tuple(index / 50 for index in range(2_060))

    artifact = build_parsed_artifact(
        raw_response=raw_response,
        duration_s=41.2,
        parquet_timestamps=timeline,
        validation_warnings=(),
    )

    assert artifact["model_response"] == COMPLETE_RESPONSE
    assert artifact["snapped_transition_frames"] == [430, 755, 920, 1411, 1644, 1786]
    assert artifact["raw_response_sha256"] == hashlib.sha256(raw_response.encode()).hexdigest()
    with pytest.raises(CosmosContractError):
        build_parsed_artifact(
            raw_response="not the accepted response",
            duration_s=41.2,
            parquet_timestamps=timeline,
            validation_warnings=(),
        )
    with pytest.raises(TypeError):
        build_parsed_artifact(
            raw_response=raw_response,
            duration_s=41.2,
            parquet_timestamps=timeline,
            validation_warnings=(),
            model_response={"unrelated": True},
        )


def test_atomic_write_fsyncs_complete_bytes_hashes_then_registers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = AtomicArtifactStore(workspace)
    observed: list[tuple[str, bytes]] = []

    def register(record: object) -> str:
        relative_path = getattr(record, "relative_path")
        observed.append((relative_path, (workspace / relative_path).read_bytes()))
        return "db-artifact-id"

    result = store.write_bytes(
        "artifacts/cosmos/attempt/response.txt",
        "exact UTF-8 ✓".encode(),
        media_type="text/plain; charset=utf-8",
        register=register,
    )

    expected = "exact UTF-8 ✓".encode()
    assert result.record.byte_size == len(expected)
    assert result.record.sha256 == hashlib.sha256(expected).hexdigest()
    assert result.database_reference == "db-artifact-id"
    assert observed == [("artifacts/cosmos/attempt/response.txt", expected)]
    assert not list(workspace.rglob("*.tmp"))


def test_installed_bytes_are_rechecked_after_link_before_hash_and_database_callback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    callbacks: list[str] = []

    def mutate_after_link(event: str, relative_path: str) -> None:
        if event == "artifact_linked":
            (workspace / relative_path).write_bytes(b"tampered")

    store = AtomicArtifactStore(workspace, event_hook=mutate_after_link)
    with pytest.raises(ArtifactConflict, match="changed before hashing"):
        store.write_bytes(
            "artifacts/cosmos/id/response.txt",
            b"installed",
            media_type="text/plain",
            register=lambda _: callbacks.append("database") or "id",
        )

    assert callbacks == []


def test_parent_replacement_after_link_is_detected_before_database_callback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    destination = workspace / "artifacts/cosmos/id/response.txt"
    displaced = workspace / "displaced-id"
    callbacks: list[str] = []

    def replace_parent_after_link(event: str, _: str) -> None:
        if event == "artifact_linked":
            destination.parent.rename(displaced)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"complete")

    store = AtomicArtifactStore(workspace, event_hook=replace_parent_after_link)
    with pytest.raises(ArtifactSecurityError, match="changed"):
        store.write_bytes(
            "artifacts/cosmos/id/response.txt",
            b"complete",
            media_type="text/plain",
            register=lambda _: callbacks.append("database") or "id",
        )

    assert callbacks == []
    assert (displaced / "response.txt").read_bytes() == b"complete"


def test_workspace_root_replacement_after_link_is_detected_before_database_callback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    displaced = tmp_path / "displaced-workspace"
    relative = "artifacts/cosmos/id/response.txt"
    callbacks: list[str] = []

    def replace_workspace_after_link(event: str, _: str) -> None:
        if event == "artifact_linked":
            workspace.rename(displaced)
            replacement = workspace / relative
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(b"complete")

    store = AtomicArtifactStore(workspace, event_hook=replace_workspace_after_link)
    with pytest.raises(ArtifactSecurityError, match="workspace"):
        store.write_bytes(
            relative,
            b"complete",
            media_type="text/plain",
            register=lambda _: callbacks.append("database") or "id",
        )

    assert callbacks == []
    assert (displaced / relative).read_bytes() == b"complete"


def test_crash_then_identical_replay_reconciles_complete_evidence_before_db(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = AtomicArtifactStore(workspace)
    callbacks: list[str] = []

    def crash(_: object) -> None:
        raise RuntimeError("database crashed")

    with pytest.raises(RuntimeError, match="database crashed"):
        store.write_bytes(
            "artifacts/cosmos/attempt/response.txt",
            b"complete",
            media_type="text/plain",
            register=crash,
        )
    replay = store.write_bytes(
        "artifacts/cosmos/attempt/response.txt",
        b"complete",
        media_type="text/plain",
        register=lambda _: callbacks.append("registered") or "id",
    )

    assert replay.database_reference == "id"
    assert callbacks == ["registered"]
    assert (workspace / replay.record.relative_path).read_bytes() == b"complete"


def test_fresh_store_adopts_complete_crash_evidence_without_original_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    relative = "artifacts/cosmos/attempt/response.txt"
    contents = b"complete response evidence"
    first = AtomicArtifactStore(workspace)

    with pytest.raises(RuntimeError, match="database crashed"):
        first.write_bytes(
            relative,
            contents,
            media_type="text/plain; charset=utf-8",
            register=lambda _: (_ for _ in ()).throw(RuntimeError("database crashed")),
        )

    registered: list[object] = []
    fresh = AtomicArtifactStore(workspace)
    inspected = fresh.inspect_existing(
        relative,
        media_type="text/plain; charset=utf-8",
        expected_sha256=hashlib.sha256(contents).hexdigest(),
        expected_byte_size=len(contents),
    )
    adopted = fresh.adopt_existing(
        relative,
        media_type="text/plain; charset=utf-8",
        expected_sha256=inspected.sha256,
        expected_byte_size=inspected.byte_size,
        register=lambda record: registered.append(record) or "artifact-id",
    )

    assert adopted.record == inspected
    assert adopted.database_reference == "artifact-id"
    assert registered == [inspected]


def test_fresh_store_recovers_exact_response_bytes_for_parsing_without_http(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    relative = "artifacts/cosmos/attempt/response.txt"
    raw_response = json.dumps(COMPLETE_RESPONSE, ensure_ascii=False)
    first = AtomicArtifactStore(workspace)

    with pytest.raises(RuntimeError, match="database crashed"):
        first.write_text(
            relative,
            raw_response,
            register=lambda _: (_ for _ in ()).throw(RuntimeError("database crashed")),
        )

    fresh = AtomicArtifactStore(workspace)
    recovered = fresh.read_existing(
        relative,
        media_type="text/plain; charset=utf-8",
        expected_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        expected_byte_size=len(raw_response.encode("utf-8")),
    )
    parsed = build_parsed_artifact(
        raw_response=recovered.contents.decode("utf-8"),
        duration_s=41.2,
        parquet_timestamps=tuple(index / 50 for index in range(2_060)),
        validation_warnings=(),
    )

    assert recovered.record.relative_path == relative
    assert recovered.contents == raw_response.encode("utf-8")
    assert parsed["model_response"] == COMPLETE_RESPONSE


def test_adoption_reconciles_all_response_evidence_without_replacing_canonical_pointer(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database = CurationDatabase(tmp_path / "curation.sqlite3")
    database.initialize()
    dataset = database.register_dataset(
        alias="local/pnp-trash",
        source_path="/source",
        source_manifest_sha256="a" * 64,
        prompt_template_version="pnp-trash-v1",
        prompt_template_sha256="b" * 64,
    )
    database.create_episode(dataset_id=dataset["id"], source_episode_index=0, source_length=10)
    job = database.create_cosmos_job(dataset_id=dataset["id"], configuration={})
    with database.open_connection() as connection:
        connection.execute(
            """
            INSERT INTO cosmos_attempts(
                id, job_id, source_episode_index, attempt_number, state, created_at, updated_at
            ) VALUES ('attempt-adopt', ?, 0, 0, 'queued', 'now', 'now')
            """,
            (job["id"],),
        )
    evidence = (
        (
            ARTIFACT_KIND_COSMOS_REQUEST,
            "request.json",
            "application/json",
            b'{"schema_version":1}',
            True,
        ),
        (
            ARTIFACT_KIND_COSMOS_RESPONSE,
            "response.txt",
            "text/plain; charset=utf-8",
            b"complete response",
            True,
        ),
        (
            ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE,
            "repair-response.txt",
            "text/plain; charset=utf-8",
            b"repaired response",
            False,
        ),
        (ARTIFACT_KIND_COSMOS_PARSED, "parsed.json", "application/json", b'{"schema_version":1}', False),
    )
    for _, name, _, contents, _ in evidence:
        destination = workspace / "artifacts/cosmos/attempt" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
    store = AtomicArtifactStore(workspace)
    adopted: list[dict[str, object]] = []
    for kind, name, media_type, _, update_pointer in evidence:
        relative = f"artifacts/cosmos/attempt/{name}"

        def reconcile(record: object) -> dict[str, object]:
            return database.reconcile_attempt_artifact(
                dataset_id=dataset["id"],
                attempt_id="attempt-adopt",
                kind=kind,
                relative_path=getattr(record, "relative_path"),
                media_type=getattr(record, "media_type"),
                byte_size=getattr(record, "byte_size"),
                sha256=getattr(record, "sha256"),
                update_attempt_pointer=update_pointer,
            )["artifact"]

        def commit_then_raise(record: object) -> None:
            reconcile(record)
            raise ConnectionError("commit acknowledgement was lost")

        with pytest.raises(ConnectionError, match="acknowledgement"):
            store.adopt_existing(relative, media_type=media_type, register=commit_then_raise)
        retried = store.adopt_existing(relative, media_type=media_type, register=reconcile)
        adopted.append(retried.database_reference)

    with database.open_connection() as connection:
        attempt = connection.execute(
            "SELECT request_artifact_id, response_artifact_id FROM cosmos_attempts WHERE id='attempt-adopt'"
        ).fetchone()
        artifact_rows = list(connection.execute("SELECT id, kind FROM artifacts ORDER BY kind"))

    assert len({artifact["id"] for artifact in adopted}) == 4
    assert {row["kind"] for row in artifact_rows} == {
        ARTIFACT_KIND_COSMOS_REQUEST,
        ARTIFACT_KIND_COSMOS_RESPONSE,
        ARTIFACT_KIND_COSMOS_REPAIR_RESPONSE,
        ARTIFACT_KIND_COSMOS_PARSED,
    }
    assert attempt["request_artifact_id"] == adopted[0]["id"]
    assert attempt["response_artifact_id"] == adopted[1]["id"]


@pytest.mark.parametrize(
    ("expected_sha256", "expected_byte_size"),
    [("0" * 64, None), (None, 999)],
)
def test_adoption_rejects_expected_hash_or_size_mismatch_before_database_callback(
    tmp_path: Path,
    expected_sha256: str | None,
    expected_byte_size: int | None,
) -> None:
    workspace = tmp_path / "workspace"
    destination = workspace / "artifacts/response.txt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"evidence")
    callbacks: list[str] = []
    store = AtomicArtifactStore(workspace)

    with pytest.raises(ArtifactConflict, match="expected"):
        store.adopt_existing(
            "artifacts/response.txt",
            media_type="text/plain",
            expected_sha256=expected_sha256,
            expected_byte_size=expected_byte_size,
            register=lambda _: callbacks.append("database") or "id",
        )

    assert callbacks == []


def test_adoption_rejects_symlink_and_parent_swap_before_database_callback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "response.txt").write_bytes(b"outside")
    destination = workspace / "artifacts/cosmos/id/response.txt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"evidence")
    callbacks: list[str] = []

    destination.unlink()
    destination.symlink_to(outside / "response.txt")
    with pytest.raises(ArtifactSecurityError):
        AtomicArtifactStore(workspace).adopt_existing(
            "artifacts/cosmos/id/response.txt",
            media_type="text/plain",
            register=lambda _: callbacks.append("symlink") or "id",
        )
    destination.unlink()
    destination.write_bytes(b"evidence")

    displaced = workspace / "displaced"

    def swap_parent(event: str, _: str) -> None:
        if event == "artifact_inspected":
            destination.parent.rename(displaced)
            destination.parent.symlink_to(outside, target_is_directory=True)

    store = AtomicArtifactStore(workspace, event_hook=swap_parent)
    with pytest.raises(ArtifactSecurityError):
        store.adopt_existing(
            "artifacts/cosmos/id/response.txt",
            media_type="text/plain",
            register=lambda _: callbacks.append("swapped") or "id",
        )

    assert callbacks == []


def test_workspace_root_replacement_during_adoption_blocks_database_callback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    displaced = tmp_path / "displaced-workspace"
    relative = "artifacts/cosmos/id/response.txt"
    destination = workspace / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"evidence")
    callbacks: list[str] = []

    def replace_workspace(event: str, _: str) -> None:
        if event == "artifact_inspected":
            workspace.rename(displaced)
            replacement = workspace / relative
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(b"evidence")

    store = AtomicArtifactStore(workspace, event_hook=replace_workspace)
    with pytest.raises(ArtifactSecurityError, match="workspace"):
        store.adopt_existing(
            relative,
            media_type="text/plain",
            register=lambda _: callbacks.append("database") or "id",
        )

    assert callbacks == []


def test_adoption_callback_failure_can_be_retried_idempotently(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    destination = workspace / "artifacts/response.txt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"complete")
    store = AtomicArtifactStore(workspace)
    calls = 0

    def register(_: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("database unavailable")
        return "artifact-id"

    with pytest.raises(RuntimeError, match="database unavailable"):
        store.adopt_existing("artifacts/response.txt", media_type="text/plain", register=register)
    result = store.adopt_existing(
        "artifacts/response.txt",
        media_type="text/plain",
        register=register,
    )

    assert calls == 2
    assert result.database_reference == "artifact-id"
    assert result.record.sha256 == hashlib.sha256(b"complete").hexdigest()


def test_adoption_never_accepts_owned_temporary_file_namespace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = AtomicArtifactStore(workspace)
    temporary = workspace / f".curation-artifact-v1-{secrets.token_hex(16)}.tmp"
    temporary.write_bytes(b"partial")

    with pytest.raises(ValueError, match="temporary"):
        store.adopt_existing(
            temporary.name,
            media_type="application/octet-stream",
            register=lambda _: "id",
        )


def test_differing_replay_conflicts_without_overwriting_complete_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = AtomicArtifactStore(workspace)
    relative = "artifacts/cosmos/attempt/response.txt"
    store.write_bytes(relative, b"first", media_type="text/plain")

    with pytest.raises(ArtifactConflict):
        store.write_bytes(relative, b"second", media_type="text/plain")

    assert (workspace / relative).read_bytes() == b"first"


def test_concurrent_differing_writers_install_exactly_one_without_overwrite(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    stores = (AtomicArtifactStore(workspace), AtomicArtifactStore(workspace))
    barrier = threading.Barrier(2)
    successes: list[bytes] = []
    conflicts: list[Exception] = []

    def write(store: AtomicArtifactStore, contents: bytes) -> None:
        barrier.wait()
        try:
            store.write_bytes("artifacts/cosmos/id/response.txt", contents, media_type="text/plain")
            successes.append(contents)
        except ArtifactConflict as error:
            conflicts.append(error)

    threads = [
        threading.Thread(target=write, args=(stores[0], b"first")),
        threading.Thread(target=write, args=(stores[1], b"second")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(successes) == 1
    assert len(conflicts) == 1
    assert (workspace / "artifacts/cosmos/id/response.txt").read_bytes() == successes[0]


def test_symlink_and_directory_destinations_cannot_escape_or_be_overwritten(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = AtomicArtifactStore(workspace)
    (workspace / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactSecurityError):
        store.write_bytes("artifacts/cosmos/id/response.txt", b"attack", media_type="text/plain")
    assert not list(outside.iterdir())

    (workspace / "artifacts").unlink()
    destination = workspace / "safe/response.txt"
    destination.mkdir(parents=True)
    with pytest.raises(ArtifactConflict):
        store.write_bytes("safe/response.txt", b"attack", media_type="text/plain")

    destination.rmdir()
    outside_file = outside / "evidence.txt"
    outside_file.write_bytes(b"outside")
    destination.symlink_to(outside_file)
    with pytest.raises(ArtifactSecurityError):
        store.write_bytes("safe/response.txt", b"attack", media_type="text/plain")
    assert outside_file.read_bytes() == b"outside"


def test_new_ancestor_and_final_link_durability_are_instrumented(tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    store = AtomicArtifactStore(
        tmp_path / "workspace", event_hook=lambda event, path: events.append((event, path))
    )
    events.clear()

    store.write_bytes("artifacts/cosmos/id/response.txt", b"durable", media_type="text/plain")

    assert events == [
        ("directory_created", "artifacts"),
        ("directory_fsync", "."),
        ("directory_created", "artifacts/cosmos"),
        ("directory_fsync", "artifacts"),
        ("directory_created", "artifacts/cosmos/id"),
        ("directory_fsync", "artifacts/cosmos"),
        ("temporary_fsynced", "artifacts/cosmos/id"),
        ("artifact_linked", "artifacts/cosmos/id/response.txt"),
        ("directory_fsync", "artifacts/cosmos/id"),
    ]


def test_constructor_performs_locked_startup_cleanup_and_preserves_complete_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    directory = workspace / "artifacts/cosmos/id"
    directory.mkdir(parents=True)
    complete = directory / "response.txt"
    temporary = directory / f".curation-artifact-v1-{secrets.token_hex(16)}.tmp"
    unrelated = directory / ".application-cache.tmp"
    source_manifest_temporary = directory / f".source-files.sha256.{secrets.token_hex(16)}.tmp"
    outside_artifact_tree = workspace / f".curation-artifact-v1-{secrets.token_hex(16)}.tmp"
    complete.write_bytes(b"evidence")
    temporary.write_bytes(b"partial")
    unrelated.write_bytes(b"application-owned")
    source_manifest_temporary.write_bytes(b"manifest-owned")
    outside_artifact_tree.write_bytes(b"wrong-subtree")

    store = AtomicArtifactStore(workspace)

    assert store.startup_cleanup_report == [
        {
            "relative_path": outside_artifact_tree.name,
            "byte_size": 13,
            "removed": True,
        },
        {
            "relative_path": f"artifacts/cosmos/id/{temporary.name}",
            "byte_size": 7,
            "removed": True,
        },
    ]
    assert complete.read_bytes() == b"evidence"
    assert not temporary.exists()
    assert unrelated.read_bytes() == b"application-owned"
    assert source_manifest_temporary.read_bytes() == b"manifest-owned"
    assert not outside_artifact_tree.exists()


def test_read_only_artifact_inspector_performs_zero_filesystem_mutations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    writer = AtomicArtifactStore(workspace)
    written = writer.write_bytes("evidence/sheet.png", b"immutable", media_type="image/png")
    stale = workspace / "evidence" / ".curation-artifact-v1-0123456789abcdef0123456789abcdef.tmp"
    stale.write_bytes(b"preserve in read-only mode")

    def snapshot() -> dict[str, bytes | None]:
        return {
            path.relative_to(workspace).as_posix(): (path.read_bytes() if path.is_file() else None)
            for path in sorted(workspace.rglob("*"))
        }

    before = snapshot()
    inspector = AtomicArtifactStore(workspace, cleanup_on_start=False)
    observed = inspector.read_existing(
        "evidence/sheet.png",
        media_type="image/png",
        expected_sha256=written.record.sha256,
        expected_byte_size=written.record.byte_size,
    )

    assert observed.contents == b"immutable"
    assert inspector.startup_cleanup_report == []
    assert snapshot() == before
    with pytest.raises(ArtifactSecurityError, match="read-only"):
        inspector.write_bytes("evidence/new.png", b"forbidden", media_type="image/png")
    with pytest.raises(ArtifactSecurityError, match="read-only"):
        inspector.cleanup_temporary_files()
    assert snapshot() == before


def test_artifact_store_cleanup_mode_requires_exact_boolean_without_creating_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "must-not-exist"

    with pytest.raises(TypeError, match="boolean"):
        AtomicArtifactStore(workspace, cleanup_on_start=1)  # type: ignore[arg-type]

    assert not workspace.exists()
    with pytest.raises(ArtifactSecurityError, match="unavailable"):
        AtomicArtifactStore(workspace, cleanup_on_start=False)
    assert not workspace.exists()


def test_complete_artifact_cannot_be_named_inside_the_cleanup_namespace(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path / "workspace")
    reserved = f"artifacts/cosmos/.curation-artifact-v1-{secrets.token_hex(16)}.tmp"

    with pytest.raises(ValueError, match="reserved"):
        store.write_bytes(reserved, b"complete", media_type="text/plain")


def test_startup_cleanup_finds_exact_owned_temps_across_the_workspace_tree(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    contact_sheets = workspace / "contact_sheets"
    exports = workspace / "exports/nested"
    contact_sheets.mkdir(parents=True)
    exports.mkdir(parents=True)
    contact_temp = contact_sheets / f".curation-artifact-v1-{secrets.token_hex(16)}.tmp"
    export_temp = exports / f".curation-artifact-v1-{secrets.token_hex(16)}.tmp"
    source_temp = exports / f".source-files.sha256.{secrets.token_hex(16)}.tmp"
    unrelated_temp = contact_sheets / ".thumbnail-cache.tmp"
    contact_temp.write_bytes(b"contact partial")
    export_temp.write_bytes(b"export partial")
    source_temp.write_bytes(b"manifest partial")
    unrelated_temp.write_bytes(b"unrelated partial")

    store = AtomicArtifactStore(workspace)

    assert store.startup_cleanup_report == [
        {
            "relative_path": f"contact_sheets/{contact_temp.name}",
            "byte_size": len(b"contact partial"),
            "removed": True,
        },
        {
            "relative_path": f"exports/nested/{export_temp.name}",
            "byte_size": len(b"export partial"),
            "removed": True,
        },
    ]
    assert not contact_temp.exists()
    assert not export_temp.exists()
    assert source_temp.read_bytes() == b"manifest partial"
    assert unrelated_temp.read_bytes() == b"unrelated partial"


def test_startup_cleanup_cannot_delete_live_source_manifest_temp_before_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / f".source-files.sha256.{secrets.token_hex(16)}.tmp"
    manifest = workspace / "source-files.sha256"
    source_fsynced = threading.Event()
    release_source = threading.Event()
    writer_errors: list[BaseException] = []

    def write_manifest() -> None:
        descriptor = -1
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(descriptor, b"complete manifest\n")
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            source_fsynced.set()
            assert release_source.wait(timeout=5)
            os.link(temporary, manifest)
            directory_fd = os.open(workspace, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            temporary.unlink()
        except BaseException as error:
            writer_errors.append(error)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    writer = threading.Thread(target=write_manifest)
    writer.start()
    assert source_fsynced.wait(timeout=5)

    store = AtomicArtifactStore(workspace)

    assert store.startup_cleanup_report == []
    assert temporary.read_bytes() == b"complete manifest\n"
    release_source.set()
    writer.join(timeout=5)
    assert writer_errors == []
    assert manifest.read_bytes() == b"complete manifest\n"


def test_cleanup_lock_cannot_delete_a_live_writer_temporary_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    temporary_fsynced = threading.Event()
    release_writer = threading.Event()
    cleanup_finished = threading.Event()
    cleanup_reports: list[list[dict[str, object]]] = []

    def hook(event: str, _: str) -> None:
        if event == "temporary_fsynced":
            temporary_fsynced.set()
            assert release_writer.wait(timeout=5)

    writer_store = AtomicArtifactStore(workspace, event_hook=hook)

    def write() -> None:
        writer_store.write_bytes("artifacts/cosmos/id/response.txt", b"complete", media_type="text/plain")

    def cleanup() -> None:
        cleanup_reports.append(AtomicArtifactStore(workspace).startup_cleanup_report)
        cleanup_finished.set()

    writer_thread = threading.Thread(target=write)
    writer_thread.start()
    assert temporary_fsynced.wait(timeout=5)
    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.start()
    time.sleep(0.05)
    assert not cleanup_finished.is_set()
    release_writer.set()
    writer_thread.join(timeout=5)
    cleanup_thread.join(timeout=5)

    assert cleanup_finished.is_set()
    assert cleanup_reports == [[]]
    assert (workspace / "artifacts/cosmos/id/response.txt").read_bytes() == b"complete"
    assert not list(workspace.rglob("*.tmp"))
