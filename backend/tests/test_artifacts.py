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
