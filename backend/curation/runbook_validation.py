"""Fail-closed evidence checks used by the pnp-trash operator runbook."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any

import av
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .cosmos_contract import CosmosContractError, build_cosmos_proposal
from .cosmos_transport import (
    CONTRACT_VERSION,
    JPEG_QUALITY,
    RESIZE_MAX_LONG_EDGE,
    build_canonical_prompt,
    prove_alignment_and_select,
)
from .security import OpenedAsset, SourceFileIdentity
from .source import SourceRecord, _manifest_eligible_files
from .worker import InvalidPersistedConfiguration, validate_frozen_job_configuration


class RunbookValidationError(ValueError):
    """A runbook trust gate could not authenticate its evidence."""


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUEST_KEYS = {
    "schema_version",
    "contract_version",
    "attempt_id",
    "source_episode_index",
    "source_video_sha256",
    "sampled_payload_sha256",
    "sampling",
    "request_body",
}
_SAMPLING_KEYS = {
    "original_fps",
    "original_frame_count",
    "original_duration_s",
    "target_fps",
    "selected_frame_indices",
    "selected_parquet_timestamps_s",
    "decoder",
    "color_space",
    "resize",
    "jpeg",
}
_PARSED_KEYS = {
    "schema_version",
    "contract_version",
    "model_response",
    "snapped_transition_frames",
    "validation_warnings",
    "raw_response_sha256",
}
_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "dataset_id",
    "dataset_alias",
    "source_manifest_sha256",
    "source_episode_index",
    "proposal_id",
    "approval_revision",
    "final_transition_frames",
    "proposal_transition_frames",
    "relative_path",
    "sha256",
    "byte_size",
}
_AUTHORITY_KEYS = {
    "schema_version",
    "job_id",
    "attempt_id",
    "proposal_id",
    "configuration",
    "artifacts",
}
_SMOKE_SOURCE_EPISODE_INDEX = 4
_SMOKE_SOURCE_FRAME_COUNT = 2_060
_SMOKE_SOURCE_FPS = 50.0
_SMOKE_DURATION_SECONDS = 41.2
_SMOKE_SAMPLED_FRAME_COUNT = 83


@dataclass(frozen=True)
class _RegisteredTimeline:
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    selected_indices: tuple[int, ...]
    selected_timestamps: tuple[float, ...]
    duration_s: float
    video_sha256: str


def _fail(message: str) -> None:
    raise RunbookValidationError(message)


def _exact_object(value: Any, keys: set[str], message: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _fail(message)
    return value


def _strict_json_bytes(contents: bytes, message: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError
            document[key] = value
        return document

    try:
        return json.loads(
            contents,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: _fail(message),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail(message)


def _canonical_bytes(document: Any) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _regular_file(workspace: Path, relative_path: str, label: str) -> tuple[Path, bytes]:
    root = workspace.resolve(strict=True)
    relative = Path(relative_path)
    if relative.is_absolute() or relative.parts in {(), (".",)} or ".." in relative.parts:
        _fail(f"{label} artifact path is invalid")
    candidate = root
    try:
        for part in relative.parts:
            candidate = candidate / part
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                _fail(f"{label} artifact is unavailable")
        if not stat.S_ISREG(candidate.lstat().st_mode):
            _fail(f"{label} artifact is unavailable")
        if candidate.resolve(strict=True).relative_to(root) != relative:
            _fail(f"{label} artifact path is invalid")
        return candidate, candidate.read_bytes()
    except (FileNotFoundError, OSError, ValueError):
        _fail(f"{label} artifact is unavailable")


def _record(relative_path: str, contents: bytes) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "byte_size": len(contents),
    }


def _parse_source_manifest(contents: bytes) -> dict[str, str]:
    hashes: dict[str, str] = {}
    previous: bytes | None = None
    for line in contents.splitlines(keepends=True):
        if not line.endswith(b"\n") or len(line) < 68 or line[64:66] != b"  ":
            _fail("registered source manifest is invalid")
        digest_bytes = line[:64]
        relative_bytes = line[66:-1]
        try:
            digest = digest_bytes.decode("ascii")
            relative = relative_bytes.decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            _fail("registered source manifest is invalid")
        if (
            not _SHA256.fullmatch(digest)
            or not relative
            or relative in hashes
            or "\r" in relative
            or "\n" in relative
            or (previous is not None and relative_bytes <= previous)
        ):
            _fail("registered source manifest is invalid")
        hashes[relative] = digest
        previous = relative_bytes
    if not hashes:
        _fail("registered source manifest is invalid")
    return hashes


def _current_identity(path: Path) -> SourceFileIdentity:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            _fail("registered source inventory is invalid")
        return SourceFileIdentity.from_stat(information)
    except OSError:
        _fail("registered source inventory is invalid")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _registered_source(
    *,
    workspace: Path,
    source_path: Path,
    source_manifest_sha256: str,
) -> SourceRecord:
    try:
        supplied_root = Path(source_path)
        root = supplied_root.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("registered source path is invalid")
    if not supplied_root.is_absolute() or root != supplied_root or not root.is_dir():
        _fail("registered source path is invalid")
    _, manifest = _regular_file(workspace, "source-files.sha256", "registered-source-manifest")
    if (
        not _SHA256.fullmatch(source_manifest_sha256)
        or hashlib.sha256(manifest).hexdigest() != source_manifest_sha256
    ):
        _fail("registered source manifest identity does not match")
    hashes = _parse_source_manifest(manifest)
    try:
        eligible = _manifest_eligible_files(root)
    except (OSError, ValueError):
        _fail("registered source inventory is invalid")
    paths = {relative: path for _encoded, relative, path in eligible}
    if set(paths) != set(hashes):
        _fail("registered source inventory does not match its manifest")
    identities = {relative: _current_identity(path) for relative, path in paths.items()}
    return SourceRecord(
        alias="local/pnp_trash",
        root=root,
        manifest_path=workspace.resolve(strict=True) / "source-files.sha256",
        fingerprint=source_manifest_sha256,
        file_hashes=MappingProxyType(hashes),
        file_identities=MappingProxyType(identities),
    )


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _read_registered_bytes(record: SourceRecord, relative_path: str, label: str) -> bytes:
    asset = record.open_asset(relative_path)
    if asset is None:
        _fail(f"registered {label} is unavailable")
    try:
        digest = _hash_descriptor(asset.fd)
        os.lseek(asset.fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(asset.fd, 1024 * 1024):
            chunks.append(chunk)
        if not record.verify_pinned_asset(asset, sha256=digest):
            _fail(f"registered {label} identity does not match")
        return b"".join(chunks)
    finally:
        asset.close()


def _probe_registered_video(asset: OpenedAsset) -> tuple[float, int]:
    os.lseek(asset.fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(asset.fd), "rb") as handle:
        with av.open(handle, mode="r") as container:
            streams = container.streams.video
            if len(streams) != 1 or streams[0].average_rate is None:
                _fail("registered smoke video stream is invalid")
            rate = float(streams[0].average_rate)
            count = sum(1 for _frame in container.decode(streams[0]))
    return rate, count


def _registered_timeline(
    *,
    workspace: Path,
    source_path: Path,
    source_manifest_sha256: str,
    source_episode_index: int,
    video_probe: Callable[[OpenedAsset], tuple[float, int]] | None = None,
) -> _RegisteredTimeline:
    if source_episode_index != _SMOKE_SOURCE_EPISODE_INDEX:
        _fail("smoke source episode must be the pinned eligible representative 4")
    record = _registered_source(
        workspace=workspace,
        source_path=source_path,
        source_manifest_sha256=source_manifest_sha256,
    )
    info = _strict_json_bytes(
        _read_registered_bytes(record, "meta/info.json", "source info"),
        "registered source info is not strict JSON",
    )
    if (
        type(info) is not dict
        or info.get("fps") != 50
        or info.get("total_episodes") != 92
        or info.get("data_path") != "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        or info.get("video_path") != "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        or info.get("features", {}).get("observation.images.ego_view", {}).get("dtype") != "video"
    ):
        _fail("registered source metadata does not match the smoke contract")
    episodes_bytes = _read_registered_bytes(record, "meta/episodes.jsonl", "episode metadata")
    try:
        episode_rows = [json.loads(line) for line in episodes_bytes.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("registered episode metadata is invalid")
    matches = [
        row for row in episode_rows if type(row) is dict and row.get("episode_index") == source_episode_index
    ]
    if len(matches) != 1 or matches[0].get("length") != _SMOKE_SOURCE_FRAME_COUNT:
        _fail("registered episode metadata does not match pinned episode 4")

    parquet_relative = "data/chunk-000/episode_000004.parquet"
    parquet = record.open_asset(parquet_relative)
    if parquet is None:
        _fail("registered smoke parquet is unavailable")
    try:
        with os.fdopen(os.dup(parquet.fd), "rb") as handle:
            table = pq.read_table(handle, columns=["episode_index", "frame_index", "timestamp"])
        parquet_digest = _hash_descriptor(parquet.fd)
        if not record.verify_pinned_asset(parquet, sha256=parquet_digest):
            _fail("registered smoke parquet identity does not match")
    except (OSError, pa.ArrowException, TypeError, ValueError):
        _fail("registered smoke parquet is invalid")
    finally:
        parquet.close()
    episode_indices = table.column("episode_index").to_pylist()
    frame_indices = table.column("frame_index").to_pylist()
    timestamps = table.column("timestamp").to_pylist()
    if len(frame_indices) != _SMOKE_SOURCE_FRAME_COUNT or any(
        index != source_episode_index for index in episode_indices
    ):
        _fail("registered smoke parquet episode identity does not match")

    video_relative = "videos/chunk-000/observation.images.ego_view/episode_000004.mp4"
    video = record.open_asset(video_relative)
    if video is None:
        _fail("registered smoke video is unavailable")
    try:
        video_digest = _hash_descriptor(video.fd)
        if video_probe is None:
            video_rate = _SMOKE_SOURCE_FPS
            video_frame_count = _SMOKE_SOURCE_FRAME_COUNT
        else:
            video_rate, video_frame_count = video_probe(video)
        if not record.verify_pinned_asset(video, sha256=video_digest):
            _fail("registered smoke video identity does not match")
    except (OSError, TypeError, ValueError, av.error.FFmpegError):
        _fail("registered smoke video is invalid")
    finally:
        video.close()
    try:
        proof = prove_alignment_and_select(
            frame_indices=frame_indices,
            timestamps=timestamps,
            source_fps=_SMOKE_SOURCE_FPS,
            video_rate=video_rate,
            video_frame_count=video_frame_count,
        )
    except (TypeError, ValueError):
        _fail("registered smoke parquet/video alignment is invalid")
    return _RegisteredTimeline(
        frame_indices=tuple(int(index) for index in frame_indices),
        timestamps=tuple(float(timestamp) for timestamp in timestamps),
        selected_indices=proof.frame_indices,
        selected_timestamps=proof.parquet_timestamps,
        duration_s=proof.duration_s,
        video_sha256=video_digest,
    )


def validate_representative_episode(
    *,
    workspace: Path,
    source_path: Path,
    source_manifest_sha256: str,
    source_episode_index: int,
    video_probe: Callable[[OpenedAsset], tuple[float, int]] = _probe_registered_video,
) -> dict[str, object]:
    """Read-only proof that the pinned smoke episode fits every model limit."""

    timeline = _registered_timeline(
        workspace=workspace,
        source_path=source_path,
        source_manifest_sha256=source_manifest_sha256,
        source_episode_index=source_episode_index,
        video_probe=video_probe,
    )
    if (
        len(timeline.frame_indices) != _SMOKE_SOURCE_FRAME_COUNT
        or timeline.duration_s != _SMOKE_DURATION_SECONDS
        or len(timeline.selected_indices) != _SMOKE_SAMPLED_FRAME_COUNT
        or timeline.duration_s > 120
        or len(timeline.selected_indices) > 240
    ):
        _fail("pinned smoke episode exceeds the Cosmos sampling limits")
    return {
        "source_episode_index": source_episode_index,
        "frame_count": len(timeline.frame_indices),
        "duration_s": timeline.duration_s,
        "sampled_frame_count": len(timeline.selected_indices),
    }


def _validate_smoke_status(
    status: Mapping[str, Any], episode: Mapping[str, Any], *, expected_smoke_job_id: str
) -> tuple[str, str, str, dict[str, Any], int]:
    job_id = status.get("job_id")
    if not isinstance(expected_smoke_job_id, str) or not expected_smoke_job_id or job_id != expected_smoke_job_id:
        _fail("smoke status job identity does not match the posted job")
    if not isinstance(job_id, str) or not job_id or status.get("state") != "completed":
        _fail("one-episode smoke status is not exactly completed")
    counts = status.get("counts")
    episodes = status.get("episodes")
    if (
        type(counts) is not dict
        or counts.get("succeeded", 0) != 1
        or counts.get("manual_only", 0) != 0
        or counts.get("retryable", 0) != 0
        or status.get("active_proposal_coverage") != 1
        or type(episodes) is not list
        or len(episodes) != 1
    ):
        _fail("one-episode smoke status is not exactly one succeeded attempt")
    attempt = episodes[0]
    if (
        type(attempt) is not dict
        or attempt.get("source_episode_index") != _SMOKE_SOURCE_EPISODE_INDEX
        or attempt.get("state") != "succeeded"
        or not isinstance(attempt.get("attempt_id"), str)
        or not attempt["attempt_id"]
    ):
        _fail("one-episode smoke status is not exactly one succeeded attempt")
    active = episode.get("active_proposal")
    if (
        type(active) is not dict
        or not isinstance(active.get("id"), str)
        or not active["id"]
        or active.get("attempt_id") != attempt["attempt_id"]
    ):
        _fail("active proposal does not bind the exact smoke attempt")
    configuration = status.get("configuration")
    try:
        validated = validate_frozen_job_configuration(_canonical_bytes(configuration))
    except (InvalidPersistedConfiguration, TypeError, ValueError):
        _fail("smoke configuration is invalid")
    if validated.episode_indices != (_SMOKE_SOURCE_EPISODE_INDEX,):
        _fail("smoke configuration does not select exactly the pinned representative episode 4")
    return job_id, attempt["attempt_id"], active["id"], configuration, _SMOKE_SOURCE_EPISODE_INDEX


def _validate_request(
    document: Any,
    *,
    attempt_id: str,
    model: str,
    source_episode_index: int,
    timeline: _RegisteredTimeline,
) -> None:
    request = _exact_object(document, _REQUEST_KEYS, "request artifact schema is not closed")
    if (
        request["schema_version"] != 1
        or request["contract_version"] != CONTRACT_VERSION
        or request["attempt_id"] != attempt_id
        or request["source_episode_index"] != source_episode_index
        or not _SHA256.fullmatch(str(request["sampled_payload_sha256"]))
    ):
        _fail("request artifact identity is invalid")
    if request["source_video_sha256"] != timeline.video_sha256:
        _fail("request source video identity does not match the registered source")

    sampling = _exact_object(request["sampling"], _SAMPLING_KEYS, "request sampling schema is not closed")
    frame_count = sampling["original_frame_count"]
    duration = sampling["original_duration_s"]
    indices = sampling["selected_frame_indices"]
    timestamps = sampling["selected_parquet_timestamps_s"]
    if (
        type(sampling["original_fps"]) is not float
        or sampling["original_fps"] != 50.0
        or frame_count != len(timeline.frame_indices)
        or type(duration) is not float
        or duration != timeline.duration_s
        or sampling["target_fps"] != 2
        or indices != list(timeline.selected_indices)
        or timestamps != list(timeline.selected_timestamps)
        or len(indices) > 240
    ):
        _fail("deterministic 50-to-2 sampling formula or parquet timestamps do not match")
    if (
        _exact_object(sampling["decoder"], {"name", "version"}, "request decoder schema is not closed")
        != sampling["decoder"]
        or not all(isinstance(value, str) and value for value in sampling["decoder"].values())
        or sampling["color_space"] != "RGB"
        or sampling["resize"]
        != {"allow_upscale": False, "max_long_edge": RESIZE_MAX_LONG_EDGE, "resampling": "LANCZOS"}
        or sampling["jpeg"] != {"quality": JPEG_QUALITY, "optimize": False, "progressive": False}
    ):
        _fail("request sampling constants are invalid")

    body = _exact_object(
        request["request_body"],
        {"model", "messages", "temperature", "seed", "max_completion_tokens", "stream", "media_io_kwargs"},
        "request body schema is not closed",
    )
    messages = body["messages"]
    if type(messages) is not list or len(messages) != 1:
        _fail("request body schema is not closed")
    message = _exact_object(messages[0], {"role", "content"}, "request body schema is not closed")
    content = message["content"]
    if type(content) is not list or len(content) != 2:
        _fail("request body schema is not closed")
    video_content = _exact_object(content[0], {"type", "video_url"}, "request body schema is not closed")
    text_content = _exact_object(content[1], {"type", "text"}, "request body schema is not closed")
    video_url = _exact_object(video_content["video_url"], {"url"}, "request body schema is not closed")
    descriptor = _exact_object(
        video_url["url"], {"redacted", "sha256", "bytes"}, "request payload descriptor schema is not closed"
    )
    media = _exact_object(body["media_io_kwargs"], {"video"}, "request body schema is not closed")
    media_video = _exact_object(
        media["video"],
        {"fps", "frames_indices", "total_num_frames", "duration", "do_sample_frames"},
        "request body schema is not closed",
    )
    if (
        body["model"] != model
        or message["role"] != "user"
        or video_content["type"] != "video_url"
        or text_content != {"type": "text", "text": build_canonical_prompt()}
        or body["temperature"] != 0
        or body["seed"] != 0
        or body["max_completion_tokens"] != 4096
        or body["stream"] is not False
        or descriptor.get("redacted") != "base64"
        or descriptor.get("sha256") != request["sampled_payload_sha256"]
        or type(descriptor.get("bytes")) is not int
        or descriptor["bytes"] <= 0
        or media_video
        != {
            "fps": 50.0,
            "frames_indices": indices,
            "total_num_frames": frame_count,
            "duration": duration,
            "do_sample_frames": False,
        }
    ):
        _fail("request body does not match the frozen Cosmos transport contract")


def _validate_parsed_and_raw(
    workspace: Path,
    attempt_root: str,
    parsed_bytes: bytes,
    *,
    timeline: _RegisteredTimeline,
) -> tuple[dict[str, Any], str, bytes, dict[str, object], dict[str, object] | None]:
    parsed = _exact_object(
        _strict_json_bytes(parsed_bytes, "parsed artifact is not strict JSON"),
        _PARSED_KEYS,
        "parsed artifact schema is not closed",
    )
    if (
        parsed["schema_version"] != 1
        or parsed["contract_version"] != CONTRACT_VERSION
        or not _SHA256.fullmatch(str(parsed["raw_response_sha256"]))
        or type(parsed["validation_warnings"]) is not list
        or any(not isinstance(item, str) for item in parsed["validation_warnings"])
    ):
        _fail("parsed artifact identity is invalid")
    candidates: list[tuple[str, bytes]] = []
    response_relative = f"{attempt_root}/response.txt"
    _, response_bytes = _regular_file(workspace, response_relative, "response")
    if hashlib.sha256(response_bytes).hexdigest() == parsed["raw_response_sha256"]:
        candidates.append((response_relative, response_bytes))
    repair_relative = f"{attempt_root}/repair-response.txt"
    repair_record: dict[str, object] | None = None
    repair_path = workspace.resolve() / repair_relative
    if repair_path.exists() or repair_path.is_symlink():
        _, repair_bytes = _regular_file(workspace, repair_relative, "repair-response")
        repair_record = _record(repair_relative, repair_bytes)
        if hashlib.sha256(repair_bytes).hexdigest() == parsed["raw_response_sha256"]:
            candidates.append((repair_relative, repair_bytes))
    if not candidates:
        _fail("parsed artifact does not bind an authoritative raw response")
    authoritative_relative, authoritative_bytes = candidates[-1]
    try:
        proposal = build_cosmos_proposal(
            authoritative_bytes.decode("utf-8"),
            duration_s=timeline.duration_s,
            parquet_timestamps=timeline.timestamps,
        )
    except (CosmosContractError, UnicodeDecodeError, TypeError, ValueError):
        _fail("authoritative raw response does not satisfy the Cosmos v2 contract")
    if parsed["model_response"] != proposal.model_response or parsed["snapped_transition_frames"] != list(
        proposal.snapped_transition_frames
    ):
        _fail("parsed artifact does not derive exactly from the authoritative raw response")
    return (
        parsed,
        authoritative_relative,
        authoritative_bytes,
        _record(response_relative, response_bytes),
        repair_record,
    )


def build_smoke_authority(
    *,
    workspace: Path,
    status: Mapping[str, Any],
    episode: Mapping[str, Any],
    expected_smoke_job_id: str,
) -> dict[str, Any]:
    """Authenticate the exact successful smoke and return its canonical authority."""

    job_id, attempt_id, proposal_id, configuration, source_episode_index = _validate_smoke_status(
        status,
        episode,
        expected_smoke_job_id=expected_smoke_job_id,
    )
    timeline = _registered_timeline(
        workspace=workspace,
        source_path=Path(configuration["source_path"]),
        source_manifest_sha256=configuration["source_manifest_sha256"],
        source_episode_index=source_episode_index,
    )
    attempt_root = f"artifacts/cosmos/{attempt_id}"
    request_relative = f"{attempt_root}/request.json"
    parsed_relative = f"{attempt_root}/parsed.json"
    _, request_bytes = _regular_file(workspace, request_relative, "request")
    _, parsed_bytes = _regular_file(workspace, parsed_relative, "parsed")
    request = _strict_json_bytes(request_bytes, "request artifact is not strict JSON")
    _validate_request(
        request,
        attempt_id=attempt_id,
        model=configuration["cosmos"]["model"],
        source_episode_index=source_episode_index,
        timeline=timeline,
    )
    parsed, raw_relative, raw_bytes, response_record, repair_record = _validate_parsed_and_raw(
        workspace,
        attempt_root,
        parsed_bytes,
        timeline=timeline,
    )

    namespace = (
        f"contact_sheets/datasets/dataset_{configuration['dataset_id']}_{configuration['source_manifest_sha256']}"
    )
    png_relative = f"{namespace}/proposals/proposal_{proposal_id}.png"
    receipt_relative = f"{namespace}/receipts/proposals/proposal_{proposal_id}.png.receipt.json"
    _, png_bytes = _regular_file(workspace, png_relative, "contact-sheet")
    _, receipt_bytes = _regular_file(workspace, receipt_relative, "contact-sheet-receipt")
    receipt = _exact_object(
        _strict_json_bytes(receipt_bytes, "contact-sheet receipt is not strict JSON"),
        _RECEIPT_KEYS,
        "contact-sheet receipt schema is not closed",
    )
    expected_receipt = {
        "schema_version": 1,
        "kind": "proposal",
        "dataset_id": configuration["dataset_id"],
        "dataset_alias": configuration["dataset_alias"],
        "source_manifest_sha256": configuration["source_manifest_sha256"],
        "source_episode_index": source_episode_index,
        "proposal_id": proposal_id,
        "approval_revision": None,
        "final_transition_frames": None,
        "proposal_transition_frames": parsed["snapped_transition_frames"],
        "relative_path": png_relative,
        "sha256": hashlib.sha256(png_bytes).hexdigest(),
        "byte_size": len(png_bytes),
    }
    if receipt != expected_receipt:
        _fail("contact-sheet receipt binding does not match the exact smoke evidence")
    try:
        with Image.open(BytesIO(png_bytes)) as image:
            image.verify()
            if image.format != "PNG":
                raise ValueError
    except (OSError, SyntaxError, ValueError):
        _fail("contact-sheet PNG bytes are invalid")

    return {
        "schema_version": 1,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "proposal_id": proposal_id,
        "configuration": configuration,
        "artifacts": {
            "request": _record(request_relative, request_bytes),
            "initial_response": response_record,
            "repair_response": repair_record,
            "authoritative_response": _record(raw_relative, raw_bytes),
            "parsed": _record(parsed_relative, parsed_bytes),
            "contact_sheet": _record(png_relative, png_bytes),
            "contact_sheet_receipt": _record(receipt_relative, receipt_bytes),
        },
    }


def validate_full_batch_authority(
    *,
    workspace: Path,
    authority: Mapping[str, Any],
    authority_sha256: str,
    smoke_status: Mapping[str, Any],
    smoke_episode: Mapping[str, Any],
    full_status: Mapping[str, Any],
    expected_full_job_id: str,
    expected_smoke_job_id: str,
) -> None:
    """Reauthenticate a smoke authority and fence a fresh 92-episode job."""

    if (
        type(authority) is not dict
        or set(authority) != _AUTHORITY_KEYS
        or not _SHA256.fullmatch(authority_sha256)
        or hashlib.sha256(_canonical_bytes(authority)).hexdigest() != authority_sha256
    ):
        _fail("frozen smoke authority hash is invalid")
    current = build_smoke_authority(
        workspace=workspace,
        status=smoke_status,
        episode=smoke_episode,
        expected_smoke_job_id=expected_smoke_job_id,
    )
    if current != authority:
        _fail("current smoke authority does not match the confirmed evidence")

    if (
        not isinstance(expected_full_job_id, str)
        or not expected_full_job_id
        or full_status.get("job_id") != expected_full_job_id
    ):
        _fail("full-batch status job identity does not match the newly created job")
    full_configuration = full_status.get("configuration")
    try:
        validated = validate_frozen_job_configuration(_canonical_bytes(full_configuration))
    except (InvalidPersistedConfiguration, TypeError, ValueError):
        _fail("full-batch configuration is invalid")
    if validated.episode_indices != tuple(range(92)):
        _fail("full-batch configuration does not select exactly episodes 0 through 91")
    smoke_configuration = authority["configuration"]
    comparable_keys = {
        "schema_version",
        "dataset_alias",
        "dataset_id",
        "source_path",
        "source_manifest_sha256",
        "source_fps",
        "prompt",
        "cosmos",
        "sampling",
        "worker",
        "transport",
        "limits",
    }
    if any(full_configuration.get(key) != smoke_configuration.get(key) for key in comparable_keys):
        _fail("full-batch configuration diverges from the confirmed smoke authority")
    episodes = full_status.get("episodes")
    episode_keys = {"attempt_id", "attempt_number", "source_episode_index", "state"}
    if (
        full_status.get("state") != "queued"
        or type(episodes) is not list
        or len(episodes) != 92
        or any(type(item) is not dict or set(item) != episode_keys for item in episodes)
        or [item["source_episode_index"] for item in episodes] != list(range(92))
        or any(
            not isinstance(item["attempt_id"], str)
            or not item["attempt_id"]
            or item["attempt_number"] != 0
            or item["state"] != "queued"
            for item in episodes
        )
    ):
        _fail("full-batch status does not contain exactly 92 queued episode attempts")


def _load_argument(value: str, label: str) -> dict[str, Any]:
    document = _strict_json_bytes(value.encode("utf-8"), f"{label} is not strict JSON")
    if type(document) is not dict:
        _fail(f"{label} is not a JSON object")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--workspace", type=Path, required=True)
    smoke.add_argument("--status-json", required=True)
    smoke.add_argument("--episode-json", required=True)
    smoke.add_argument("--expected-smoke-job-id", required=True)
    representative = subparsers.add_parser("representative")
    representative.add_argument("--workspace", type=Path, required=True)
    representative.add_argument("--source-path", type=Path, required=True)
    representative.add_argument("--source-manifest-sha256", required=True)
    representative.add_argument("--source-episode-index", type=int, required=True)
    full = subparsers.add_parser("full-batch")
    full.add_argument("--workspace", type=Path, required=True)
    full.add_argument("--authority-json", required=True)
    full.add_argument("--authority-sha256", required=True)
    full.add_argument("--smoke-status-json", required=True)
    full.add_argument("--smoke-episode-json", required=True)
    full.add_argument("--full-status-json", required=True)
    full.add_argument("--expected-full-job-id", required=True)
    full.add_argument("--expected-smoke-job-id", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.operation == "representative":
            result = validate_representative_episode(
                workspace=arguments.workspace,
                source_path=arguments.source_path,
                source_manifest_sha256=arguments.source_manifest_sha256,
                source_episode_index=arguments.source_episode_index,
            )
            print(_canonical_bytes(result).decode("utf-8"))
        elif arguments.operation == "smoke":
            authority = build_smoke_authority(
                workspace=arguments.workspace,
                status=_load_argument(arguments.status_json, "smoke status"),
                episode=_load_argument(arguments.episode_json, "smoke episode"),
                expected_smoke_job_id=arguments.expected_smoke_job_id,
            )
            print(_canonical_bytes(authority).decode("utf-8"))
        else:
            validate_full_batch_authority(
                workspace=arguments.workspace,
                authority=_load_argument(arguments.authority_json, "smoke authority"),
                authority_sha256=arguments.authority_sha256,
                smoke_status=_load_argument(arguments.smoke_status_json, "smoke status"),
                smoke_episode=_load_argument(arguments.smoke_episode_json, "smoke episode"),
                full_status=_load_argument(arguments.full_status_json, "full-batch status"),
                expected_full_job_id=arguments.expected_full_job_id,
                expected_smoke_job_id=arguments.expected_smoke_job_id,
            )
    except RunbookValidationError as error:
        parser.exit(1, f"FAIL: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
