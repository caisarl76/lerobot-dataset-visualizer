from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

from curation.cosmos_transport import (
    CONTRACT_VERSION,
    build_canonical_prompt,
    build_parsed_artifact,
    prove_alignment_and_select,
)
from curation.prompts import PROMPT_TEMPLATE_SHA256, PROMPT_TEMPLATE_VERSION
from curation.runbook_validation import (
    RunbookValidationError,
    build_smoke_authority,
    validate_full_batch_authority,
    validate_representative_episode,
)
from curation.source import _manifest_bytes
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_cosmos_contract import COMPLETE_RESPONSE

ATTEMPT_ID = "11111111-1111-1111-1111-111111111111"
PROPOSAL_ID = "22222222-2222-2222-2222-222222222222"
SMOKE_EPISODE_INDEX = 4


def _configuration(
    tmp_path: Path,
    episode_indices: list[int],
    *,
    manifest_sha256: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_alias": "local/pnp_trash",
        "dataset_id": 1,
        "source_path": str(tmp_path / "source"),
        "source_manifest_sha256": manifest_sha256,
        "source_fps": 50.0,
        "episode_indices": episode_indices,
        "prompt": {"version": PROMPT_TEMPLATE_VERSION, "sha256": PROMPT_TEMPLATE_SHA256},
        "cosmos": {
            "base_url": "http://127.0.0.1:8001/v1",
            "model": "cosmos3-nano",
            "api_key_env": "PNP_TRASH_H100_KEY",
            "endpoint_identity": "h100-cosmos",
        },
        "sampling": {"target_fps": 2},
        "worker": {"concurrency": 1, "lease_seconds": 180, "heartbeat_seconds": 15},
        "transport": {"timeout_seconds": 120, "initial_attempts": 2, "repair_attempts": 1},
        "limits": {
            "maximum_duration_seconds": 120,
            "maximum_sampled_frames": 240,
            "maximum_payload_bytes": 67_108_864,
        },
    }


def _live_shape_timestamps(frame_count: int = 2_060) -> list[float]:
    return [float(np.float32(index / 50)) for index in range(frame_count)]


def _request(*, all_timestamps: list[float], source_video_sha256: str) -> dict[str, object]:
    frame_count = len(all_timestamps)
    proof = prove_alignment_and_select(
        frame_indices=range(frame_count),
        timestamps=all_timestamps,
        source_fps=50.0,
        video_rate=50.0,
        video_frame_count=frame_count,
    )
    indices = list(proof.frame_indices)
    timestamps = list(proof.parquet_timestamps)
    payload_sha256 = "b" * 64
    duration = frame_count / 50
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "attempt_id": ATTEMPT_ID,
        "source_episode_index": SMOKE_EPISODE_INDEX,
        "source_video_sha256": source_video_sha256,
        "sampled_payload_sha256": payload_sha256,
        "sampling": {
            "original_fps": 50.0,
            "original_frame_count": frame_count,
            "original_duration_s": duration,
            "target_fps": 2,
            "selected_frame_indices": indices,
            "selected_parquet_timestamps_s": timestamps,
            "decoder": {"name": "PyAV", "version": "17.1.0"},
            "color_space": "RGB",
            "resize": {"allow_upscale": False, "max_long_edge": 640, "resampling": "LANCZOS"},
            "jpeg": {"quality": 85, "optimize": False, "progressive": False},
        },
        "request_body": {
            "model": "cosmos3-nano",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": {"redacted": "base64", "sha256": payload_sha256, "bytes": 1234}},
                        },
                        {"type": "text", "text": build_canonical_prompt()},
                    ],
                }
            ],
            "temperature": 0,
            "seed": 0,
            "max_completion_tokens": 4096,
            "stream": False,
            "media_io_kwargs": {
                "video": {
                    "fps": 50.0,
                    "frames_indices": indices,
                    "total_num_frames": frame_count,
                    "duration": duration,
                    "do_sample_frames": False,
                }
            },
        },
    }


def _smoke_fixture(tmp_path: Path, *, repair_authoritative: bool = False) -> tuple[dict, dict, Path]:
    source = tmp_path / "source"
    parquet_path = source / "data" / "chunk-000" / "episode_000004.parquet"
    video_path = source / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000004.mp4"
    parquet_path.parent.mkdir(parents=True)
    video_path.parent.mkdir(parents=True)
    (source / "meta").mkdir()
    all_timestamps = _live_shape_timestamps()
    (source / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "total_episodes": 92,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
                "features": {"observation.images.ego_view": {"dtype": "video"}},
            }
        )
    )
    (source / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": SMOKE_EPISODE_INDEX, "length": 2_060}) + "\n"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [SMOKE_EPISODE_INDEX] * 2_060,
                "frame_index": list(range(2_060)),
                "timestamp": all_timestamps,
            }
        ),
        parquet_path,
    )
    video_path.write_bytes(b"authenticated video fixture")
    video_sha256 = hashlib.sha256(video_path.read_bytes()).hexdigest()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest, _hashes, _identities = _manifest_bytes(source)
    (workspace / "source-files.sha256").write_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    attempt_root = workspace / "artifacts" / "cosmos" / ATTEMPT_ID
    attempt_root.mkdir(parents=True)
    request = _request(all_timestamps=all_timestamps, source_video_sha256=video_sha256)
    (attempt_root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    raw = json.dumps(COMPLETE_RESPONSE, ensure_ascii=False)
    initial_raw = "initial response rejected" if repair_authoritative else raw
    (attempt_root / "response.txt").write_text(initial_raw, encoding="utf-8")
    if repair_authoritative:
        (attempt_root / "repair-response.txt").write_text(raw, encoding="utf-8")
    parsed = build_parsed_artifact(
        raw_response=raw,
        duration_s=41.2,
        parquet_timestamps=all_timestamps,
        validation_warnings=("repair_used",) if repair_authoritative else (),
    )
    (attempt_root / "parsed.json").write_text(json.dumps(parsed), encoding="utf-8")

    relative_png = f"contact_sheets/datasets/dataset_1_{manifest_sha256}/proposals/proposal_{PROPOSAL_ID}.png"
    png_path = workspace / relative_png
    png_path.parent.mkdir(parents=True)
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (20, 30, 40)).save(buffer, format="PNG")
    png = buffer.getvalue()
    png_path.write_bytes(png)
    receipt_path = (
        workspace / f"contact_sheets/datasets/dataset_1_{manifest_sha256}/receipts/proposals/"
        f"proposal_{PROPOSAL_ID}.png.receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "kind": "proposal",
        "dataset_id": 1,
        "dataset_alias": "local/pnp_trash",
        "source_manifest_sha256": manifest_sha256,
        "source_episode_index": SMOKE_EPISODE_INDEX,
        "proposal_id": PROPOSAL_ID,
        "approval_revision": None,
        "final_transition_frames": None,
        "proposal_transition_frames": parsed["snapped_transition_frames"],
        "relative_path": relative_png,
        "sha256": hashlib.sha256(png).hexdigest(),
        "byte_size": len(png),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    status = {
        "job_id": "job-smoke",
        "state": "completed",
        "counts": {"succeeded": 1},
        "episodes": [
            {
                "attempt_id": ATTEMPT_ID,
                "source_episode_index": SMOKE_EPISODE_INDEX,
                "state": "succeeded",
            }
        ],
        "active_proposal_coverage": 1,
        "configuration": _configuration(
            tmp_path,
            [SMOKE_EPISODE_INDEX],
            manifest_sha256=manifest_sha256,
        ),
    }
    episode = {"active_proposal": {"id": PROPOSAL_ID, "attempt_id": ATTEMPT_ID}}
    return status, episode, workspace


def test_representative_preflight_pins_episode_four_and_proves_bounded_video(
    tmp_path: Path,
) -> None:
    status, _episode, workspace = _smoke_fixture(tmp_path)
    configuration = status["configuration"]

    result = validate_representative_episode(
        workspace=workspace,
        source_path=Path(configuration["source_path"]),
        source_manifest_sha256=configuration["source_manifest_sha256"],
        source_episode_index=SMOKE_EPISODE_INDEX,
        video_probe=lambda _asset: (50.0, 2_060),
    )

    assert result == {
        "source_episode_index": 4,
        "frame_count": 2_060,
        "duration_s": 41.2,
        "sampled_frame_count": 83,
    }


def test_smoke_authority_accepts_exact_initial_and_repair_response_evidence(tmp_path: Path) -> None:
    assert _live_shape_timestamps()[1] != 1 / 50
    for repair in (False, True):
        status, episode, workspace = _smoke_fixture(tmp_path / str(repair), repair_authoritative=repair)
        authority = build_smoke_authority(
            workspace=workspace,
            status=status,
            episode=episode,
            expected_smoke_job_id="job-smoke",
        )

        assert authority["job_id"] == "job-smoke"
        assert authority["attempt_id"] == ATTEMPT_ID
        expected_name = "repair-response.txt" if repair else "response.txt"
        assert authority["artifacts"]["authoritative_response"]["relative_path"].endswith(expected_name)
        assert authority["configuration"] == status["configuration"]


def test_smoke_authority_rejects_status_from_a_different_posted_job(tmp_path: Path) -> None:
    status, episode, workspace = _smoke_fixture(tmp_path)
    status["job_id"] = "stale-job"

    with pytest.raises(RunbookValidationError, match="smoke status job identity"):
        build_smoke_authority(
            workspace=workspace,
            status=status,
            episode=episode,
            expected_smoke_job_id="job-smoke",
        )


def test_smoke_authority_reparses_raw_response_against_actual_float_parquet_timeline(
    tmp_path: Path,
) -> None:
    status, episode, workspace = _smoke_fixture(tmp_path)
    attempt_root = workspace / "artifacts" / "cosmos" / ATTEMPT_ID
    response = deepcopy(COMPLETE_RESPONSE)
    response["segments"][0]["end_s"] = 0.03
    response["segments"][1]["start_s"] = 0.03
    raw = json.dumps(response, ensure_ascii=False)
    (attempt_root / "response.txt").write_text(raw, encoding="utf-8")
    ideal_timestamps = [index / 50 for index in range(2_060)]
    ideal_parsed = build_parsed_artifact(
        raw_response=raw,
        duration_s=41.2,
        parquet_timestamps=ideal_timestamps,
        validation_warnings=(),
    )
    (attempt_root / "parsed.json").write_text(json.dumps(ideal_parsed), encoding="utf-8")

    with pytest.raises(RunbookValidationError, match="derive exactly"):
        build_smoke_authority(
            workspace=workspace,
            status=status,
            episode=episode,
            expected_smoke_job_id="job-smoke",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("sampling", "deterministic 50-to-2 sampling"),
        ("timestamp", "deterministic 50-to-2 sampling"),
        ("video_hash", "source video"),
        ("request_extra", "request artifact schema"),
        ("parsed_extra", "parsed artifact schema"),
        ("raw_hash", "authoritative raw response"),
        ("receipt_dataset", "contact-sheet receipt binding"),
        ("receipt_frames", "contact-sheet receipt binding"),
        ("receipt_hash", "contact-sheet receipt binding"),
        ("png_bytes", "contact-sheet receipt binding"),
    ],
)
def test_smoke_authority_rejects_hostile_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    status, episode, workspace = _smoke_fixture(tmp_path)
    attempt_root = workspace / "artifacts" / "cosmos" / ATTEMPT_ID
    request_path = attempt_root / "request.json"
    parsed_path = attempt_root / "parsed.json"
    receipt_path = next(workspace.glob("contact_sheets/**/receipts/proposals/*.json"))
    png_path = next(workspace.glob("contact_sheets/**/proposals/*.png"))

    if mutation in {"sampling", "timestamp", "video_hash", "request_extra"}:
        document = json.loads(request_path.read_text())
        if mutation == "sampling":
            document["sampling"]["selected_frame_indices"][1] = 24
        elif mutation == "timestamp":
            document["sampling"]["selected_parquet_timestamps_s"][1] += 0.0001
        elif mutation == "video_hash":
            document["source_video_sha256"] = "d" * 64
        else:
            document["unexpected"] = True
        request_path.write_text(json.dumps(document))
    elif mutation in {"parsed_extra", "raw_hash"}:
        document = json.loads(parsed_path.read_text())
        if mutation == "parsed_extra":
            document["unexpected"] = True
        else:
            document["raw_response_sha256"] = "d" * 64
        parsed_path.write_text(json.dumps(document))
    elif mutation in {"receipt_dataset", "receipt_frames", "receipt_hash"}:
        document = json.loads(receipt_path.read_text())
        if mutation == "receipt_dataset":
            document["dataset_id"] = 2
        elif mutation == "receipt_frames":
            document["proposal_transition_frames"][0] += 1
        else:
            document["sha256"] = "e" * 64
        receipt_path.write_text(json.dumps(document))
    else:
        png_path.write_bytes(b"not a png")

    with pytest.raises(RunbookValidationError, match=message):
        build_smoke_authority(
            workspace=workspace,
            status=status,
            episode=episode,
            expected_smoke_job_id="job-smoke",
        )


def test_full_batch_authority_revalidates_smoke_and_exact_new_configuration(tmp_path: Path) -> None:
    status, episode, workspace = _smoke_fixture(tmp_path)
    authority = build_smoke_authority(
        workspace=workspace,
        status=status,
        episode=episode,
        expected_smoke_job_id="job-smoke",
    )
    authority_bytes = json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    full_status = {
        "job_id": "job-full",
        "state": "queued",
        "configuration": _configuration(
            tmp_path,
            list(range(92)),
            manifest_sha256=status["configuration"]["source_manifest_sha256"],
        ),
        "episodes": [
            {
                "attempt_id": f"attempt-{index}",
                "attempt_number": 0,
                "source_episode_index": index,
                "state": "queued",
            }
            for index in range(92)
        ],
    }

    validate_full_batch_authority(
        workspace=workspace,
        authority=authority,
        authority_sha256=authority_sha256,
        smoke_status=status,
        smoke_episode=episode,
        full_status=full_status,
        expected_full_job_id="job-full",
        expected_smoke_job_id="job-smoke",
    )

    stale_status = deepcopy(status)
    stale_status["episodes"][0]["attempt_id"] = "33333333-3333-3333-3333-333333333333"
    with pytest.raises(RunbookValidationError, match="active proposal"):
        validate_full_batch_authority(
            workspace=workspace,
            authority=authority,
            authority_sha256=authority_sha256,
            smoke_status=stale_status,
            smoke_episode=episode,
            full_status=full_status,
            expected_full_job_id="job-full",
            expected_smoke_job_id="job-smoke",
        )

    mismatched = deepcopy(full_status)
    mismatched["configuration"]["transport"]["timeout_seconds"] = 119
    with pytest.raises(RunbookValidationError, match="full-batch configuration"):
        validate_full_batch_authority(
            workspace=workspace,
            authority=authority,
            authority_sha256=authority_sha256,
            smoke_status=status,
            smoke_episode=episode,
            full_status=mismatched,
            expected_full_job_id="job-full",
            expected_smoke_job_id="job-smoke",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_job_id",
        "empty_job_id",
        "non_object_episode",
        "missing_episode",
        "extra_episode",
        "missing_episode_key",
        "extra_episode_key",
    ],
)
def test_full_batch_authority_rejects_wrong_job_or_open_episode_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    status, episode, workspace = _smoke_fixture(tmp_path)
    authority = build_smoke_authority(
        workspace=workspace,
        status=status,
        episode=episode,
        expected_smoke_job_id="job-smoke",
    )
    authority_sha256 = hashlib.sha256(
        json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    full_status = {
        "job_id": "job-full",
        "state": "queued",
        "configuration": _configuration(
            tmp_path,
            list(range(92)),
            manifest_sha256=status["configuration"]["source_manifest_sha256"],
        ),
        "episodes": [
            {
                "attempt_id": f"attempt-{index}",
                "attempt_number": 0,
                "source_episode_index": index,
                "state": "queued",
            }
            for index in range(92)
        ],
    }
    if mutation == "wrong_job_id":
        full_status["job_id"] = "stale-job"
    elif mutation == "empty_job_id":
        full_status["job_id"] = ""
    elif mutation == "non_object_episode":
        full_status["episodes"][4] = 4
    elif mutation == "missing_episode":
        full_status["episodes"].pop()
    elif mutation == "extra_episode":
        full_status["episodes"].append(deepcopy(full_status["episodes"][-1]))
    elif mutation == "missing_episode_key":
        del full_status["episodes"][4]["attempt_number"]
    else:
        full_status["episodes"][4]["unexpected"] = True

    with pytest.raises(RunbookValidationError, match="full-batch status"):
        validate_full_batch_authority(
            workspace=workspace,
            authority=authority,
            authority_sha256=authority_sha256,
            smoke_status=status,
            smoke_episode=episode,
            full_status=full_status,
            expected_full_job_id="job-full",
            expected_smoke_job_id="job-smoke",
        )
