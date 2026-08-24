from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time

from curation.cosmos_contract import CosmosContractError
from curation.cosmos_transport import (
    ArtifactConflict,
    ArtifactSecurityError,
    AtomicArtifactStore,
    PreparedSample,
    build_canonical_prompt,
    build_initial_request_body,
    build_parsed_artifact,
    build_request_artifact,
)
import pytest
from test_cosmos_contract import COMPLETE_RESPONSE


def _sample() -> PreparedSample:
    return PreparedSample(
        source_fps=50.0,
        total_num_frames=3,
        duration_s=0.06,
        frame_indices=(0, 2),
        parquet_timestamps=(0.0, 0.04),
        all_parquet_timestamps=(0.0, 0.02, 0.04),
        jpeg_frames=(b"one", b"two"),
        decoder_name="PyAV",
        decoder_version="17.1.0",
        source_video_sha256="a" * 64,
    )


def test_request_artifact_is_derived_canonically_and_redacts_only_base64_payload() -> None:
    sample = _sample()
    prompt = build_canonical_prompt()
    wire_body = build_initial_request_body(model="model", prompt=prompt, sample=sample)
    artifact = build_request_artifact(
        attempt_id="00000000-0000-0000-0000-000000000001",
        source_episode_index=7,
        sample=sample,
        model="model",
        prompt=prompt,
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
    expected_request = json.loads(json.dumps(wire_body))
    expected_request["messages"][0]["content"][0]["video_url"]["url"] = {
        "redacted": "base64",
        "sha256": artifact["sampled_payload_sha256"],
        "bytes": len("b25l,dHdv"),
    }
    assert artifact["request_body"] == expected_request
    assert artifact["sampling"] == {
        "original_fps": 50.0,
        "original_frame_count": 3,
        "original_duration_s": 0.06,
        "target_fps": 2,
        "selected_frame_indices": [0, 2],
        "selected_parquet_timestamps_s": [0.0, 0.04],
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
            sample=sample,
            model="model",
            prompt=prompt,
            request_body={"tampered": True},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_fps": 0.0},
        {"source_fps": float("nan")},
        {"source_fps": True},
        {"total_num_frames": 0},
        {"total_num_frames": True},
        {"duration_s": 0.061},
        {"duration_s": float("inf")},
        {"frame_indices": [0, 2]},
        {"frame_indices": (0,)},
        {"frame_indices": (0, 0)},
        {"frame_indices": (0, 3)},
        {"frame_indices": (0, True)},
        {"parquet_timestamps": (0.0, 0.02)},
        {"parquet_timestamps": (0.0, float("nan"))},
        {"all_parquet_timestamps": (0.0, 0.02)},
        {"all_parquet_timestamps": (0.0, float("nan"), 0.04)},
        {"jpeg_frames": (b"one", b"")},
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
    temporary = directory / ".response.txt.deadbeef.tmp"
    complete.write_bytes(b"evidence")
    temporary.write_bytes(b"partial")

    store = AtomicArtifactStore(workspace)

    assert store.startup_cleanup_report == [
        {
            "relative_path": "artifacts/cosmos/id/.response.txt.deadbeef.tmp",
            "byte_size": 7,
            "removed": True,
        }
    ]
    assert complete.read_bytes() == b"evidence"
    assert not temporary.exists()


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
