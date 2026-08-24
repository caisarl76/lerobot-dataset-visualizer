from __future__ import annotations

import hashlib
import json
from pathlib import Path

from curation.cosmos_transport import (
    AtomicArtifactStore,
    PreparedSample,
    build_initial_request_body,
    build_parsed_artifact,
    build_request_artifact,
)
import pytest


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


def test_request_artifact_has_explicit_shape_and_redacts_the_base64_payload() -> None:
    sample = _sample()
    wire_body = build_initial_request_body(model="model", prompt="prompt", sample=sample)
    artifact = build_request_artifact(
        attempt_id="00000000-0000-0000-0000-000000000001",
        source_episode_index=7,
        sample=sample,
        request_body=wire_body,
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
    assert artifact["schema_version"] == 1
    assert artifact["contract_version"] == "pnp-trash-cosmos-v2"
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
    descriptor = artifact["request_body"]["messages"][0]["content"][0]["video_url"]["url"]
    assert descriptor == {
        "redacted": "base64",
        "sha256": artifact["sampled_payload_sha256"],
        "bytes": len("b25l,dHdv"),
    }
    assert "data:video" not in json.dumps(artifact)


def test_parsed_artifact_has_six_transitions_warnings_and_validated_raw_hash() -> None:
    model_response = {"schema_version": 2}
    artifact = build_parsed_artifact(
        model_response=model_response,
        snapped_transition_frames=(1, 2, None, None, None, None),
        validation_warnings=("human edits required",),
        raw_response="exact response",
    )
    assert artifact == {
        "schema_version": 1,
        "contract_version": "pnp-trash-cosmos-v2",
        "model_response": model_response,
        "snapped_transition_frames": [1, 2, None, None, None, None],
        "validation_warnings": ["human edits required"],
        "raw_response_sha256": hashlib.sha256(b"exact response").hexdigest(),
    }
    with pytest.raises(ValueError, match="six"):
        build_parsed_artifact(
            model_response=model_response,
            snapped_transition_frames=(1,),
            validation_warnings=(),
            raw_response="exact response",
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


def test_database_callback_failure_leaves_only_complete_unreferenced_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = AtomicArtifactStore(workspace)
    database_rows: list[object] = []

    def crash_before_commit(record: object) -> None:
        path = workspace / getattr(record, "relative_path")
        assert path.read_bytes() == b"complete"
        assert getattr(record, "sha256") == hashlib.sha256(b"complete").hexdigest()
        raise RuntimeError("database crashed")

    with pytest.raises(RuntimeError, match="database crashed"):
        store.write_bytes(
            "artifacts/cosmos/attempt/response.txt",
            b"complete",
            media_type="text/plain; charset=utf-8",
            register=lambda record: database_rows.append(crash_before_commit(record)),
        )

    assert database_rows == []
    assert (workspace / "artifacts/cosmos/attempt/response.txt").read_bytes() == b"complete"


def test_startup_cleanup_reports_and_removes_only_temporary_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    complete = workspace / "artifacts/cosmos/id/response.txt"
    temporary = workspace / "artifacts/cosmos/id/.response.txt.deadbeef.tmp"
    referenced_temporary = workspace / "artifacts/cosmos/id/.request.json.referenced.tmp"
    complete.parent.mkdir(parents=True)
    complete.write_bytes(b"evidence")
    temporary.write_bytes(b"partial")
    referenced_temporary.write_bytes(b"referenced")

    report = AtomicArtifactStore(workspace).cleanup_temporary_files(
        referenced_relative_paths={referenced_temporary.relative_to(workspace).as_posix()}
    )

    assert report == [
        {
            "relative_path": "artifacts/cosmos/id/.response.txt.deadbeef.tmp",
            "byte_size": 7,
            "removed": True,
        },
    ]
    assert complete.read_bytes() == b"evidence"
    assert not temporary.exists()
    assert referenced_temporary.read_bytes() == b"referenced"
