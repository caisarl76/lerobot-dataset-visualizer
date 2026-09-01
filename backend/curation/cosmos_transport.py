"""Deterministic Cosmos sampling, bounded HTTP transport, and durable artifacts.

The worker that owns lifecycle state is intentionally separate from this
module.  This layer returns neutral HTTP exchange envelopes and performs a
database callback only after an artifact is complete, durable, and hashed.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import time
from typing import Any, Generic, Literal, TypeVar
from urllib.parse import urlsplit
from uuid import UUID

import av
import httpx
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .cosmos_contract import (
    COSMOS_RESPONSE_V2_SCHEMA,
    CosmosContractError,
    CosmosProposal,
    build_cosmos_proposal,
)
from .security import OpenedAsset
from .source import SourceRecord

CONTRACT_VERSION = "pnp-trash-cosmos-v2"
TARGET_SAMPLING_FPS = 2
MAX_DURATION_SECONDS = 120
MAX_SAMPLED_FRAMES = 240
MAX_PAYLOAD_BYTES = 67_108_864
PROMPT_MAX_BYTES = 32 * 1024
RESPONSE_MAX_BYTES = 2 * 1024 * 1024
REPAIR_INVALID_RESPONSE_MAX_BYTES = 64 * 1024
MAX_OUTER_JSON_DEPTH = 128
HTTP_TIMEOUT_SECONDS = 120.0
JPEG_QUALITY = 85
RESIZE_MAX_LONG_EDGE = 640
DATA_URL_PREFIX = "data:video/jpeg;base64,"

CANONICAL_PROMPT_PREFIX = "\n".join(
    (
        "You annotate one egocentric robot episode for temporal subtask supervision.",
        (
            "Time 0.0 is the first supplied video frame. All start_s and end_s values are seconds on "
            "the original video timeline described by media_io_kwargs."
        ),
        "Identify these phases exactly once and in this order:",
        "1 approach_brown_table: approach the brown table until locomotion stops at the table.",
        "2 pick_up_object: reach, grasp, and lift the object; include failed regrasp attempts in this phase.",
        "3 turn_to_find_black_trash_bin: turn until the black trash bin is found.",
        "4 approach_black_trash_bin: approach the bin while retaining the object.",
        "5 lean_down_to_black_trash_bin: lean down and position over the bin.",
        "6 drop_object_into_black_trash_bin: release the object into the bin.",
        "7 stand_straight: return to and hold a standing-straight pose.",
        "Use visible evidence only. Do not infer a completed phase when it is absent or ambiguous.",
        (
            "For each phase set status to completed, partial, or not_observed. Use completed only when "
            "the described subtask succeeds. Use partial when the phase is attempted but does not "
            "successfully complete, including interrupted or failed attempts. Use not_observed when "
            "there is no visible evidence for an attempt."
        ),
        (
            "For completed or partial phases provide visible start/end times, confidence, and evidence. "
            "For not_observed phases set start_s, end_s, confidence, and evidence to null."
        ),
        (
            "Set missing_steps to exactly the step numbers whose status is not completed. Set "
            "episode_complete true if and only if missing_steps is empty."
        ),
        (
            "Only a complete episode must have segments covering the episode in order with adjacent "
            "boundaries differing by at most 0.5 seconds. In an incomplete episode, do not invent "
            "timing to fill gaps."
        ),
        (
            "You may first emit one <think>...</think> block. After it, emit exactly one JSON object "
            "matching this schema, with no Markdown fence or trailing prose:"
        ),
    )
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strict_json_document(raw: bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object member: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON constant: {value}")

    decoded = raw.decode("utf-8")
    document = json.loads(
        decoded,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    pending: list[tuple[Any, int]] = [(document, 1)]
    while pending:
        value, depth = pending.pop()
        if depth > MAX_OUTER_JSON_DEPTH:
            raise ValueError("JSON nesting limit exceeded")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("JSON number is outside the finite range")
    return document


def build_canonical_prompt() -> str:
    """Build the one frozen prompt and enforce its UTF-8 wire-size limit."""

    prompt = CANONICAL_PROMPT_PREFIX + "\n" + _canonical_json(COSMOS_RESPONSE_V2_SCHEMA)
    if len(prompt.encode("utf-8")) > PROMPT_MAX_BYTES:
        raise ValueError("canonical Cosmos prompt exceeds the 32 KiB UTF-8 limit")
    return prompt


@dataclass(frozen=True)
class SamplingLimits:
    max_duration_seconds: float = MAX_DURATION_SECONDS
    max_sampled_frames: int = MAX_SAMPLED_FRAMES
    max_payload_bytes: int = MAX_PAYLOAD_BYTES

    def __post_init__(self) -> None:
        if type(self.max_duration_seconds) not in {int, float} or isinstance(self.max_duration_seconds, bool):
            raise ValueError("max_duration_seconds must be a built-in finite and positive number")
        try:
            duration_is_finite = math.isfinite(self.max_duration_seconds)
        except (OverflowError, TypeError, ValueError):
            duration_is_finite = False
        if not duration_is_finite or self.max_duration_seconds <= 0:
            raise ValueError("sampling limits must be finite and positive")
        if type(self.max_sampled_frames) is not int or self.max_sampled_frames <= 0:
            raise ValueError("max_sampled_frames must be a built-in positive integer")
        if type(self.max_payload_bytes) is not int or self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be a built-in positive integer")


@dataclass(frozen=True)
class AlignmentProof:
    source_fps: float
    total_num_frames: int
    duration_s: float
    frame_indices: tuple[int, ...]
    parquet_timestamps: tuple[float, ...]


@dataclass(frozen=True)
class PreparedSample:
    source_fps: float
    total_num_frames: int
    duration_s: float
    frame_indices: tuple[int, ...]
    parquet_timestamps: tuple[float, ...]
    all_parquet_timestamps: tuple[float, ...]
    jpeg_frames: tuple[bytes, ...]
    decoder_name: str
    decoder_version: str
    source_video_sha256: str

    def __post_init__(self) -> None:
        if type(self.source_fps) is not float or not math.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source_fps must be a canonical finite positive float")
        if type(self.total_num_frames) is not int or self.total_num_frames <= 0:
            raise ValueError("total_num_frames must be a canonical positive integer")
        expected_duration = self.total_num_frames / self.source_fps
        if (
            type(self.duration_s) is not float
            or not math.isfinite(self.duration_s)
            or self.duration_s != expected_duration
        ):
            raise ValueError("duration_s must be a canonical float equal to total_num_frames/source_fps")
        if not all(
            type(value) is tuple
            for value in (
                self.frame_indices,
                self.parquet_timestamps,
                self.all_parquet_timestamps,
                self.jpeg_frames,
            )
        ):
            raise ValueError("prepared sample sequences must be immutable tuples")
        cardinality = len(self.frame_indices)
        if cardinality == 0 or len(self.parquet_timestamps) != cardinality or len(self.jpeg_frames) != cardinality:
            raise ValueError("sample indices, timestamps, and JPEGs must have equal nonzero cardinality")
        if any(type(index) is not int or not 0 <= index < self.total_num_frames for index in self.frame_indices):
            raise ValueError("sample frame indices must be canonical in-range integers")
        if any(left >= right for left, right in zip(self.frame_indices, self.frame_indices[1:])):
            raise ValueError("sample frame indices must be strictly increasing")
        selected_timestamps = _canonical_float_tuple(
            self.parquet_timestamps, expected_length=cardinality, name="sample timestamps"
        )
        if any(left >= right for left, right in zip(selected_timestamps, selected_timestamps[1:])):
            raise ValueError("sample timestamps must be strictly increasing")
        all_timestamps = _canonical_float_tuple(
            self.all_parquet_timestamps,
            expected_length=self.total_num_frames,
            name="all parquet timestamps",
        )
        if any(left >= right for left, right in zip(all_timestamps, all_timestamps[1:])):
            raise ValueError("all parquet timestamps must be strictly increasing")
        if all_timestamps[0] < 0 or all_timestamps[-1] >= self.duration_s:
            raise ValueError("all parquet timestamps must be within the episode duration")
        if any(
            selected_timestamps[position] != all_timestamps[frame_index]
            for position, frame_index in enumerate(self.frame_indices)
        ):
            raise ValueError("sample timestamps must match their source-frame timestamps")
        for frame in self.jpeg_frames:
            _validate_sample_jpeg(frame)
        if not _valid_utf8_nonempty(self.decoder_name) or not _valid_utf8_nonempty(self.decoder_version):
            raise ValueError("decoder name and version must be nonempty UTF-8")
        if not isinstance(self.source_video_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_video_sha256
        ):
            raise ValueError("source video hash must be lowercase SHA-256")

    @property
    def payload_ascii(self) -> bytes:
        return b",".join(base64.b64encode(frame) for frame in self.jpeg_frames)

    @property
    def data_url(self) -> str:
        return DATA_URL_PREFIX + self.payload_ascii.decode("ascii")


@dataclass(frozen=True)
class SamplingOutcome:
    status: Literal["ready", "manual_only"]
    reason: str | None
    sample: PreparedSample | None
    reject_episode: bool = False

    @classmethod
    def ready(cls, sample: PreparedSample) -> "SamplingOutcome":
        return cls(status="ready", reason=None, sample=sample)

    @classmethod
    def manual_only(cls, reason: str) -> "SamplingOutcome":
        return cls(status="manual_only", reason=reason, sample=None)


def _float64_vector(values: Iterable[Any], *, name: str) -> np.ndarray:
    raw_values = list(values)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
        for value in raw_values
    ):
        raise ValueError(f"{name} must be a one-dimensional numeric sequence")
    try:
        vector = np.asarray(raw_values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from error
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return vector


def _validated_finite_values(values: Iterable[Any], *, expected_length: int, name: str) -> tuple[float, ...]:
    vector = _float64_vector(values, name=name)
    if vector.size != expected_length or not np.isfinite(vector).all():
        raise ValueError(f"{name} must have the expected length and contain only finite values")
    return tuple(float(value) for value in vector)


def _canonical_float_tuple(values: tuple[Any, ...], *, expected_length: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_length or any(type(value) is not float for value in values):
        raise ValueError(f"{name} must be a canonical tuple of built-in floats with the expected length")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_sample_jpeg(frame: Any) -> None:
    if type(frame) is not bytes or not frame:
        raise ValueError("sample JPEGs must be nonempty bytes")
    try:
        with Image.open(BytesIO(frame)) as image:
            image.load()
            if (
                image.format != "JPEG"
                or image.mode != "RGB"
                or min(image.size) <= 0
                or max(image.size) > RESIZE_MAX_LONG_EDGE
            ):
                raise ValueError("sample JPEGs must decode as bounded RGB JPEG images")
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError("sample JPEGs must decode as bounded RGB JPEG images") from error


def _valid_utf8_nonempty(value: Any) -> bool:
    return bool(value) and _valid_utf8_string(value)


def _valid_utf8_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _json_is_utf8_serializable(value: Any) -> bool:
    try:
        _canonical_json(value).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError):
        return False
    return True


def _safe_error_strings(errors: Iterable[str]) -> tuple[str, ...]:
    return tuple(error.encode("utf-8", "backslashreplace").decode("utf-8") for error in errors)


def prove_alignment_and_select(
    *,
    frame_indices: Iterable[Any],
    timestamps: Iterable[Any],
    source_fps: float,
    video_rate: float,
    video_frame_count: int,
) -> AlignmentProof:
    """Prove parquet/video identity and select canonical float64 2 fps targets."""

    if isinstance(source_fps, bool) or not isinstance(source_fps, (int, float, np.floating)):
        raise ValueError("source FPS must be a finite positive number")
    if isinstance(video_rate, bool) or not isinstance(video_rate, (int, float, np.floating)):
        raise ValueError("video stream rate must be a finite positive number")
    try:
        fps = float(source_fps)
        rate = float(video_rate)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("source and video rate must be finite positive numbers") from error
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source FPS must be a finite positive number")
    if not math.isfinite(rate) or rate <= 0 or abs(rate - fps) > 1e-6:
        raise ValueError("video stream rate does not agree with source FPS")

    raw_indices = list(frame_indices)
    if not raw_indices:
        raise ValueError("frame_index must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in raw_indices):
        raise ValueError("frame_index values must be integers")
    try:
        indices = np.asarray(raw_indices, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("frame_index values must be integers") from error
    expected = np.arange(indices.size, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError("frame_index identity could not be proven")

    parquet_timestamps = _float64_vector(timestamps, name="timestamp")
    if parquet_timestamps.size != indices.size or not np.isfinite(parquet_timestamps).all():
        raise ValueError("timestamp values must be finite and match frame count")
    ideal = expected.astype(np.float64) / np.float64(fps)
    if np.any(np.abs(parquet_timestamps - ideal) > np.float64(1.0 / (2.0 * fps))):
        raise ValueError("timestamp alignment tolerance was exceeded")
    if (
        isinstance(video_frame_count, bool)
        or not isinstance(video_frame_count, (int, np.integer))
        or int(video_frame_count) != indices.size
    ):
        raise ValueError("video frame count does not equal parquet row count")

    last_timestamp = parquet_timestamps[-1]
    selected: list[int] = []
    selected_timestamps: list[float] = []
    target_number = 0
    while True:
        target = np.float64(target_number) * np.float64(0.5)
        if target > last_timestamp:
            break
        # np.argmin returns the first minimum, which is the lower frame index.
        index = int(np.argmin(np.abs(parquet_timestamps - target)))
        target_number += 1
        if selected and index == selected[-1]:
            continue
        selected.append(index)
        selected_timestamps.append(float(parquet_timestamps[index]))
    return AlignmentProof(
        source_fps=fps,
        total_num_frames=int(indices.size),
        duration_s=float(np.float64(indices.size) / np.float64(fps)),
        frame_indices=tuple(selected),
        parquet_timestamps=tuple(selected_timestamps),
    )


def _read_parquet_timeline(parquet_asset: OpenedAsset) -> tuple[list[Any], np.ndarray]:
    os.lseek(parquet_asset.fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(parquet_asset.fd), "rb", closefd=True) as handle:
        table = pq.read_table(handle, columns=["frame_index", "timestamp"])
    return table.column("frame_index").to_pylist(), _float64_vector(
        table.column("timestamp").to_pylist(), name="timestamp"
    )


def _jpeg_from_frame(frame: av.VideoFrame) -> bytes:
    image = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
    long_edge = max(image.size)
    if long_edge > RESIZE_MAX_LONG_EDGE:
        scale = RESIZE_MAX_LONG_EDGE / long_edge
        size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(size, resample=Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=False,
        progressive=False,
    )
    return output.getvalue()


def _decode_selected_frames(
    video_asset: OpenedAsset, selected: Sequence[int]
) -> tuple[float, int, tuple[bytes, ...]]:
    selected_set = set(selected)
    encoded: dict[int, bytes] = {}
    os.lseek(video_asset.fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(video_asset.fd), "rb", closefd=True) as handle:
        with av.open(handle, mode="r") as container:
            streams = container.streams.video
            if len(streams) != 1:
                raise ValueError("video must contain exactly one video stream")
            stream = streams[0]
            raw_rate = stream.average_rate
            if raw_rate is None:
                raise ValueError("video stream rate is unavailable")
            video_rate = float(raw_rate)
            decoded_count = 0
            for decoded_count, frame in enumerate(container.decode(stream), start=1):
                frame_index = decoded_count - 1
                if frame_index in selected_set:
                    encoded[frame_index] = _jpeg_from_frame(frame)
    missing = [index for index in selected if index not in encoded]
    if missing:
        raise ValueError(f"selected video frames were not decoded: {missing}")
    return video_rate, decoded_count, tuple(encoded[index] for index in selected)


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def prepare_episode_samples(
    *,
    source_record: SourceRecord,
    parquet_asset_path: str,
    video_asset_path: str,
    source_fps: float,
    limits: SamplingLimits | None = None,
) -> SamplingOutcome:
    """Prepare samples from two immutable registered descriptors held until completion."""

    effective_limits = SamplingLimits() if limits is None else limits
    parquet_asset = source_record.open_asset(parquet_asset_path)
    if parquet_asset is None:
        return SamplingOutcome.manual_only("alignment_unproven")
    video_asset = source_record.open_asset(video_asset_path)
    if video_asset is None:
        parquet_asset.close()
        return SamplingOutcome.manual_only("alignment_unproven")
    try:
        indices, timestamps = _read_parquet_timeline(parquet_asset)
        if isinstance(source_fps, bool) or not isinstance(source_fps, (int, float, np.floating)):
            raise ValueError("source FPS must be finite and positive")
        try:
            fps = float(source_fps)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("source FPS must be finite and positive") from error
        if not math.isfinite(fps) or fps <= 0 or not indices:
            raise ValueError("source FPS and parquet timeline must be nonempty")
        duration = len(indices) / fps
        if duration > effective_limits.max_duration_seconds:
            return SamplingOutcome.manual_only("duration_limit")

        provisional = prove_alignment_and_select(
            frame_indices=indices,
            timestamps=timestamps,
            source_fps=fps,
            video_rate=fps,
            video_frame_count=len(indices),
        )
        if len(provisional.frame_indices) > effective_limits.max_sampled_frames:
            return SamplingOutcome.manual_only("sample_count_limit")
        video_rate, decoded_count, jpeg_frames = _decode_selected_frames(video_asset, provisional.frame_indices)
        proof = prove_alignment_and_select(
            frame_indices=indices,
            timestamps=timestamps,
            source_fps=fps,
            video_rate=video_rate,
            video_frame_count=decoded_count,
        )
        payload = b",".join(base64.b64encode(frame) for frame in jpeg_frames)
        if len(payload) > effective_limits.max_payload_bytes:
            return SamplingOutcome.manual_only("payload_size_limit")
        parquet_sha256 = _sha256_fd(parquet_asset.fd)
        video_sha256 = _sha256_fd(video_asset.fd)
        if not source_record.verify_pinned_asset(
            parquet_asset, sha256=parquet_sha256
        ) or not source_record.verify_pinned_asset(video_asset, sha256=video_sha256):
            return SamplingOutcome.manual_only("source_changed")
        return SamplingOutcome.ready(
            PreparedSample(
                source_fps=proof.source_fps,
                total_num_frames=proof.total_num_frames,
                duration_s=proof.duration_s,
                frame_indices=proof.frame_indices,
                parquet_timestamps=proof.parquet_timestamps,
                all_parquet_timestamps=tuple(float(value) for value in timestamps),
                jpeg_frames=jpeg_frames,
                decoder_name="PyAV",
                decoder_version=av.__version__,
                source_video_sha256=video_sha256,
            )
        )
    except (OSError, TypeError, ValueError, pa.ArrowException, av.error.FFmpegError):
        return SamplingOutcome.manual_only("alignment_unproven")
    finally:
        parquet_asset.close()
        video_asset.close()


def build_initial_request_body(*, model: str, prompt: str, sample: Any) -> dict[str, Any]:
    """Format the approved wire shape; production proof lives in prepare_initial_request."""

    if not _valid_utf8_nonempty(model):
        raise ValueError("Cosmos model identifier is required")
    if not _valid_utf8_nonempty(prompt):
        raise ValueError("Cosmos prompt must be nonempty UTF-8")
    if len(prompt.encode("utf-8")) > PROMPT_MAX_BYTES:
        raise ValueError("canonical Cosmos prompt exceeds the 32 KiB UTF-8 limit")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": sample.data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "seed": 0,
        "max_completion_tokens": 4096,
        "stream": False,
        "media_io_kwargs": {
            "video": {
                "fps": sample.source_fps,
                "num_frames": -1,
                "frames_indices": list(sample.frame_indices),
                "total_num_frames": sample.total_num_frames,
                "duration": sample.duration_s,
                "do_sample_frames": False,
            }
        },
    }


def _validate_prepared_sample_for_request(sample: PreparedSample) -> None:
    if not isinstance(sample, PreparedSample):
        raise ValueError("initial request requires a PreparedSample")
    if sample.duration_s > MAX_DURATION_SECONDS:
        raise ValueError("prepared sampling proof exceeds the duration limit")
    if len(sample.frame_indices) > MAX_SAMPLED_FRAMES:
        raise ValueError("prepared sampling proof exceeds the sample count limit")
    if len(sample.payload_ascii) > MAX_PAYLOAD_BYTES:
        raise ValueError("prepared sampling proof exceeds the payload limit")
    proof = prove_alignment_and_select(
        frame_indices=range(sample.total_num_frames),
        timestamps=sample.all_parquet_timestamps,
        source_fps=sample.source_fps,
        video_rate=sample.source_fps,
        video_frame_count=sample.total_num_frames,
    )
    if proof.frame_indices != sample.frame_indices or proof.parquet_timestamps != sample.parquet_timestamps:
        raise ValueError("prepared sample does not match the complete 2 fps sampling proof")


@dataclass(frozen=True)
class PreparedInitialRequest:
    """One immutable source for initial wire bytes, request hash, and redaction."""

    wire_bytes: bytes
    body_sha256: str
    sample: PreparedSample

    def __post_init__(self) -> None:
        _validate_prepared_sample_for_request(self.sample)
        if type(self.wire_bytes) is not bytes:
            raise ValueError("prepared initial wire must be immutable bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", self.body_sha256):
            raise ValueError("prepared initial request hash must be lowercase SHA-256")
        if hashlib.sha256(self.wire_bytes).hexdigest() != self.body_sha256:
            raise ValueError("prepared initial request hash does not match its wire bytes")
        try:
            document = _strict_json_document(self.wire_bytes)
            canonical_wire = _canonical_json(document).encode("utf-8")
        except (RecursionError, TypeError, UnicodeDecodeError, UnicodeEncodeError, ValueError) as error:
            raise ValueError("prepared initial wire must be canonical strict UTF-8 JSON") from error
        if canonical_wire != self.wire_bytes or not isinstance(document, dict):
            raise ValueError("prepared initial wire must be canonical strict UTF-8 JSON")
        try:
            model = document["model"]
            prompt = document["messages"][0]["content"][1]["text"]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError("prepared initial wire does not have the approved request shape") from error
        if prompt != build_canonical_prompt():
            raise ValueError("prepared initial wire must contain the canonical prompt")
        expected = build_initial_request_body(model=model, prompt=prompt, sample=self.sample)
        if document != expected:
            raise ValueError("prepared initial wire diverges from its sample or approved request shape")

    @property
    def body(self) -> dict[str, Any]:
        document = _strict_json_document(self.wire_bytes)
        if not isinstance(document, dict):  # pragma: no cover - guaranteed by construction
            raise AssertionError("prepared initial request invariant failed")
        return document

    @property
    def redacted_body(self) -> dict[str, Any]:
        document = self.body
        payload = self.sample.payload_ascii
        document["messages"][0]["content"][0]["video_url"]["url"] = {
            "redacted": "base64",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        return document


def prepare_initial_request(*, model: str, prompt: str, sample: PreparedSample) -> PreparedInitialRequest:
    """Prove a production sample once and freeze the exact initial request bytes."""

    _validate_prepared_sample_for_request(sample)
    if prompt != build_canonical_prompt():
        raise ValueError("production Cosmos requests must use the canonical prompt")
    body = build_initial_request_body(model=model, prompt=prompt, sample=sample)
    wire_bytes = _canonical_json(body).encode("utf-8")
    return PreparedInitialRequest(
        wire_bytes=wire_bytes,
        body_sha256=hashlib.sha256(wire_bytes).hexdigest(),
        sample=sample,
    )


def _build_repair_request_body(
    *, model: str, invalid_response: str, validation_errors: Sequence[str]
) -> dict[str, Any]:
    repair_payload = _canonical_json(
        {
            "validation_errors": list(validation_errors),
            "invalid_response": invalid_response,
        }
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return only one corrected JSON object matching pnp-trash-cosmos-v2.\n" + repair_payload
                ),
            }
        ],
        "temperature": 0,
        "seed": 0,
        "max_completion_tokens": 2048,
        "stream": False,
    }


@dataclass(frozen=True)
class CosmosTransportResult:
    status: Literal["succeeded", "manual_only"]
    reason: str | None
    proposal: CosmosProposal | None
    initial_content: str | None
    repair_content: str | None
    validation_errors: tuple[str, ...]
    exchanges: tuple[dict[str, Any], ...]
    prepared_initial_request: PreparedInitialRequest | None
    reject_episode: bool = False


@dataclass(frozen=True)
class _CallResult:
    content: str | None
    reason: str | None
    retryable: bool
    exchange: dict[str, Any]
    observed_content: str | None = None


@dataclass(frozen=True)
class CosmosCallObservation:
    """One immutable, persistable model-call observation for phase-level resume."""

    phase: Literal["initial", "repair"]
    content: str | None
    observed_content: str | None
    reason: str | None
    retryable: bool
    _exchange_json: bytes

    def __post_init__(self) -> None:
        if self.phase not in {"initial", "repair"}:
            raise ValueError("observation phase is invalid")
        if any(
            value is not None and not _valid_utf8_string(value)
            for value in (self.content, self.observed_content, self.reason)
        ):
            raise ValueError("observation strings must be valid UTF-8")
        if type(self.retryable) is not bool or type(self._exchange_json) is not bytes:
            raise ValueError("observation retry and exchange fields are invalid")
        try:
            exchange = _strict_json_document(self._exchange_json)
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("observation exchange must be canonical strict JSON") from error
        if (
            not isinstance(exchange, dict)
            or exchange.get("phase") != self.phase
            or _canonical_json(exchange).encode("utf-8") != self._exchange_json
        ):
            raise ValueError("observation exchange must be canonical and match its phase")

    @classmethod
    def _from_call(cls, phase: Literal["initial", "repair"], result: _CallResult) -> "CosmosCallObservation":
        exchange_json = _canonical_json(result.exchange).encode("utf-8")
        return cls(
            phase=phase,
            content=result.content,
            observed_content=result.observed_content,
            reason=result.reason,
            retryable=result.retryable,
            _exchange_json=exchange_json,
        )

    @property
    def exchange(self) -> dict[str, Any]:
        exchange = _strict_json_document(self._exchange_json)
        if not isinstance(exchange, dict):  # pragma: no cover - guaranteed by construction
            raise AssertionError("observation exchange invariant failed")
        return exchange


Clock = Callable[[], str]
ObservationSink = Callable[[CosmosCallObservation], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validated_cosmos_endpoint(base_url: Any) -> str:
    has_controls = isinstance(base_url, str) and any(
        ord(character) <= 0x20 or ord(character) == 0x7F for character in base_url
    )
    if not _valid_utf8_nonempty(base_url) or has_controls:
        raise ValueError("Cosmos base URL must be an absolute http/https URL without controls")
    if "\\" in base_url:
        raise ValueError("Cosmos base URL must not contain backslashes")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Cosmos base URL has an invalid host or port") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise ValueError("Cosmos base URL must be absolute and contain no credentials, query, or fragment")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    try:
        return str(httpx.Request("POST", endpoint).url)
    except (httpx.InvalidURL, UnicodeError, ValueError):
        raise ValueError("Cosmos base URL is not a valid HTTP request URL") from None


def _validated_bearer_token(api_key: Any) -> str:
    if (
        not isinstance(api_key, str)
        or not api_key
        or any(not 0x21 <= ord(character) <= 0x7E for character in api_key)
    ):
        raise ValueError("Cosmos API key must be a nonempty visible ASCII bearer token")
    return api_key


class CosmosTransport:
    """Synchronous, deterministic, bounded OpenAI-compatible Cosmos client."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Clock = _utc_now,
        timeout_seconds: float = HTTP_TIMEOUT_SECONDS,
        response_max_bytes: int = RESPONSE_MAX_BYTES,
        persist_observation: ObservationSink | None = None,
    ) -> None:
        if not _valid_utf8_nonempty(model):
            raise ValueError("Cosmos model identifier is required")
        if type(timeout_seconds) not in {int, float} or isinstance(timeout_seconds, bool):
            raise ValueError("Cosmos timeout must be a built-in finite positive number")
        try:
            valid_timeout = math.isfinite(timeout_seconds) and timeout_seconds > 0
        except (OverflowError, TypeError, ValueError):
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("Cosmos timeout must be a built-in finite positive number")
        if type(response_max_bytes) is not int or response_max_bytes <= 0:
            raise ValueError("response_max_bytes must be a built-in positive integer")
        self.endpoint = _validated_cosmos_endpoint(base_url)
        self.model = model
        self.api_key = _validated_bearer_token(api_key)
        if persist_observation is not None and not callable(persist_observation):
            raise ValueError("Cosmos observation persistence hook must be callable")
        self.client = httpx.Client(trust_env=False, follow_redirects=False) if client is None else client
        self.sleep = sleep
        self.clock = clock
        self.timeout_seconds = float(timeout_seconds)
        self.response_max_bytes = response_max_bytes
        self.persist_observation = persist_observation

    def observe_initial(self, prepared_request: PreparedInitialRequest) -> CosmosCallObservation:
        """Perform exactly one initial call for an external durable phase orchestrator."""

        if not isinstance(prepared_request, PreparedInitialRequest):
            raise TypeError("prepared_request must be a PreparedInitialRequest")
        if prepared_request.body.get("model") != self.model:
            raise ValueError("prepared initial request model does not match the transport model")
        result = self._call(
            prepared_request.wire_bytes,
            body_sha256=prepared_request.body_sha256,
            phase="initial",
        )
        return CosmosCallObservation._from_call("initial", result)

    def observe_repair(self, *, invalid_response: str, validation_errors: Sequence[str]) -> CosmosCallObservation:
        """Perform exactly one bounded text-only repair call for durable orchestration."""

        if not _valid_utf8_string(invalid_response):
            raise ValueError("repair input must be valid UTF-8 text")
        if len(invalid_response.encode("utf-8")) > REPAIR_INVALID_RESPONSE_MAX_BYTES:
            raise ValueError("repair input exceeds the 64 KiB limit")
        if any(not _valid_utf8_string(error) for error in validation_errors):
            raise ValueError("repair validation errors must be valid UTF-8 text")
        repair_body = _build_repair_request_body(
            model=self.model,
            invalid_response=invalid_response,
            validation_errors=validation_errors,
        )
        repair_wire = _canonical_json(repair_body).encode("utf-8")
        result = self._call(
            repair_wire,
            body_sha256=hashlib.sha256(repair_wire).hexdigest(),
            phase="repair",
        )
        return CosmosCallObservation._from_call("repair", result)

    def annotate(self, sampling: SamplingOutcome) -> CosmosTransportResult:
        if sampling.status == "manual_only":
            return CosmosTransportResult(
                status="manual_only",
                reason=sampling.reason,
                proposal=None,
                initial_content=None,
                repair_content=None,
                validation_errors=(),
                exchanges=(),
                prepared_initial_request=None,
            )
        if sampling.sample is None:
            raise ValueError("ready sampling outcome must contain a prepared sample")
        if self.persist_observation is None:
            raise ValueError("ready Cosmos annotation requires synchronous observation persistence")
        sample = sampling.sample
        prepared_request = prepare_initial_request(
            model=self.model,
            prompt=build_canonical_prompt(),
            sample=sample,
        )
        exchanges: list[dict[str, Any]] = []
        initial: CosmosCallObservation | None = None
        for attempt_number in range(2):
            initial = self.observe_initial(prepared_request)
            self.persist_observation(initial)
            exchanges.append(initial.exchange)
            if initial.content is not None or not initial.retryable:
                break
            if attempt_number == 0:
                self.sleep(1.0)
        assert initial is not None
        if initial.content is None:
            if initial.reason == "finish_reason" and initial.observed_content is not None:
                invalid_content = initial.observed_content
                validation_errors = ("choices[0].finish_reason must be exactly stop",)
            else:
                return self._manual(
                    initial.reason or "transport_failure",
                    exchanges=exchanges,
                    initial_content=initial.observed_content,
                    prepared_initial_request=prepared_request,
                )
        else:
            invalid_content = initial.content
            try:
                proposal = build_cosmos_proposal(
                    initial.content,
                    duration_s=sample.duration_s,
                    parquet_timestamps=sample.all_parquet_timestamps,
                )
            except CosmosContractError as error:
                validation_errors = _safe_error_strings(error.errors)
            else:
                return CosmosTransportResult(
                    status="succeeded",
                    reason=None,
                    proposal=proposal,
                    initial_content=initial.content,
                    repair_content=None,
                    validation_errors=(),
                    exchanges=tuple(exchanges),
                    prepared_initial_request=prepared_request,
                )

        if len(invalid_content.encode("utf-8")) > REPAIR_INVALID_RESPONSE_MAX_BYTES:
            return self._manual(
                "repair_input_too_large",
                exchanges=exchanges,
                initial_content=invalid_content,
                validation_errors=validation_errors,
                prepared_initial_request=prepared_request,
            )
        repair = self.observe_repair(
            invalid_response=invalid_content,
            validation_errors=validation_errors,
        )
        self.persist_observation(repair)
        exchanges.append(repair.exchange)
        if repair.content is None:
            return self._manual(
                "repair_transport",
                exchanges=exchanges,
                initial_content=invalid_content,
                repair_content=repair.observed_content,
                validation_errors=validation_errors,
                prepared_initial_request=prepared_request,
            )
        try:
            proposal = build_cosmos_proposal(
                repair.content,
                duration_s=sample.duration_s,
                parquet_timestamps=sample.all_parquet_timestamps,
            )
        except CosmosContractError as repair_error:
            return self._manual(
                "repair_invalid",
                exchanges=exchanges,
                initial_content=invalid_content,
                repair_content=repair.content,
                validation_errors=_safe_error_strings(repair_error.errors),
                prepared_initial_request=prepared_request,
            )
        return CosmosTransportResult(
            status="succeeded",
            reason=None,
            proposal=proposal,
            initial_content=invalid_content,
            repair_content=repair.content,
            validation_errors=(),
            exchanges=tuple(exchanges),
            prepared_initial_request=prepared_request,
        )

    def _manual(
        self,
        reason: str,
        *,
        exchanges: Sequence[dict[str, Any]],
        initial_content: str | None = None,
        repair_content: str | None = None,
        validation_errors: Sequence[str] = (),
        prepared_initial_request: PreparedInitialRequest | None = None,
    ) -> CosmosTransportResult:
        return CosmosTransportResult(
            status="manual_only",
            reason=reason,
            proposal=None,
            initial_content=initial_content,
            repair_content=repair_content,
            validation_errors=tuple(validation_errors),
            exchanges=tuple(exchanges),
            prepared_initial_request=prepared_initial_request,
        )

    def _call(
        self,
        encoded_body: bytes,
        *,
        body_sha256: str,
        phase: Literal["initial", "repair"],
    ) -> _CallResult:
        if type(encoded_body) is not bytes or hashlib.sha256(encoded_body).hexdigest() != body_sha256:
            raise ValueError("wire bytes and request body hash diverged")
        request_record = {
            "method": "POST",
            "url": self.endpoint,
            "body_sha256": body_sha256,
        }
        started_at = self.clock()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with self.client.stream(
                "POST",
                self.endpoint,
                content=encoded_body,
                headers=headers,
                timeout=self.timeout_seconds,
                follow_redirects=False,
            ) as response:
                status_code = response.status_code
                envelope = {
                    "status_code": status_code,
                    "id": None,
                    "model": None,
                    "created": None,
                    "usage": None,
                    "finish_reason": None,
                }
                if status_code != 200:
                    reason = "http_status"
                    exchange = self._exchange(
                        phase,
                        started_at,
                        request_record,
                        response=envelope,
                        error_class="HttpStatusError",
                        error_summary=f"Cosmos returned HTTP {status_code}",
                    )
                    return _CallResult(
                        content=None,
                        reason=reason,
                        retryable=status_code in {408, 429} or 500 <= status_code <= 599,
                        exchange=exchange,
                    )
                raw = self._read_bounded(response)
        except _ResponseTooLarge:
            exchange = self._exchange(
                phase,
                started_at,
                request_record,
                response={
                    "status_code": 200,
                    "id": None,
                    "model": None,
                    "created": None,
                    "usage": None,
                    "finish_reason": None,
                },
                error_class="ResponseTooLarge",
                error_summary="Cosmos response exceeded 2 MiB",
            )
            return _CallResult(None, "response_too_large", False, exchange)
        except httpx.InvalidURL:
            exchange = self._exchange(
                phase,
                started_at,
                request_record,
                response=None,
                error_class="InvalidURL",
                error_summary="Cosmos request URL was rejected",
            )
            return _CallResult(None, "invalid_url", False, exchange)
        except httpx.TimeoutException as error:
            exchange = self._exchange(
                phase,
                started_at,
                request_record,
                response=None,
                error_class=type(error).__name__,
                error_summary="Cosmos request timed out",
            )
            return _CallResult(None, "timeout", False, exchange)
        except httpx.ConnectError as error:
            exchange = self._exchange(
                phase,
                started_at,
                request_record,
                response=None,
                error_class=type(error).__name__,
                error_summary="Cosmos connection failed",
            )
            return _CallResult(None, "connection", phase == "initial", exchange)
        except httpx.RequestError as error:
            exchange = self._exchange(
                phase,
                started_at,
                request_record,
                response=None,
                error_class=type(error).__name__,
                error_summary="Cosmos transport failed",
            )
            return _CallResult(None, "transport", False, exchange)

        try:
            document = _strict_json_document(raw)
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return self._envelope_failure(
                phase,
                started_at,
                request_record,
                "invalid_json",
                "response body must be strict UTF-8 JSON without duplicate members or nonfinite numbers",
            )
        if not isinstance(document, dict):
            return self._envelope_failure(
                phase, started_at, request_record, "invalid_json", "response JSON must be an object"
            )
        envelope = self._response_envelope(document, status_code=200)
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            return self._envelope_failure(
                phase, started_at, request_record, "missing_choice", "choices must be a nonempty array", envelope
            )
        first = choices[0]
        if not isinstance(first, dict):
            return self._envelope_failure(
                phase, started_at, request_record, "missing_content", "choices[0] must be an object", envelope
            )
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            return self._envelope_failure(
                phase, started_at, request_record, "missing_content", "message.content must be a string", envelope
            )
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            return self._envelope_failure(
                phase,
                started_at,
                request_record,
                "invalid_utf8",
                "message.content must be valid UTF-8",
                envelope,
            )
        finish_reason = first.get("finish_reason")
        envelope["finish_reason"] = finish_reason if _valid_utf8_string(finish_reason) else None
        if finish_reason != "stop":
            return self._envelope_failure(
                phase,
                started_at,
                request_record,
                "finish_reason",
                "finish_reason must be exactly stop",
                envelope,
                observed_content=content,
            )
        exchange = self._exchange(phase, started_at, request_record, response=envelope)
        return _CallResult(content, None, False, exchange, observed_content=content)

    def _read_bounded(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.response_max_bytes:
                    raise _ResponseTooLarge
            except ValueError:
                pass
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > self.response_max_bytes:
                raise _ResponseTooLarge
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _response_envelope(document: Mapping[str, Any], *, status_code: int) -> dict[str, Any]:
        identifier = document.get("id")
        model = document.get("model")
        usage = document.get("usage")
        return {
            "status_code": status_code,
            "id": identifier if _valid_utf8_string(identifier) else None,
            "model": model if _valid_utf8_string(model) else None,
            "created": document.get("created") if type(document.get("created")) is int else None,
            "usage": usage if isinstance(usage, dict) and _json_is_utf8_serializable(usage) else None,
            "finish_reason": None,
        }

    def _envelope_failure(
        self,
        phase: Literal["initial", "repair"],
        started_at: str,
        request_record: dict[str, Any],
        reason: str,
        summary: str,
        envelope: dict[str, Any] | None = None,
        observed_content: str | None = None,
    ) -> _CallResult:
        response = envelope or {
            "status_code": 200,
            "id": None,
            "model": None,
            "created": None,
            "usage": None,
            "finish_reason": None,
        }
        exchange = self._exchange(
            phase,
            started_at,
            request_record,
            response=response,
            error_class="CosmosEnvelopeError",
            error_summary=summary,
        )
        return _CallResult(None, reason, False, exchange, observed_content=observed_content)

    def _exchange(
        self,
        phase: Literal["initial", "repair"],
        started_at: str,
        request: dict[str, Any],
        *,
        response: dict[str, Any] | None,
        error_class: str | None = None,
        error_summary: str | None = None,
    ) -> dict[str, Any]:
        error = None
        if error_class is not None:
            safe_summary = error_summary if _valid_utf8_nonempty(error_summary) else error_class
            error = {"class": error_class, "summary": safe_summary}
        return {
            "phase": phase,
            "started_at": started_at,
            "finished_at": self.clock(),
            "request": request,
            "response": response,
            "error": error,
        }


class _ResponseTooLarge(Exception):
    pass


def build_request_artifact(
    *,
    attempt_id: str,
    source_episode_index: int,
    prepared_request: PreparedInitialRequest,
) -> dict[str, Any]:
    try:
        UUID(attempt_id)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("attempt_id must be a UUID") from error
    if (
        isinstance(source_episode_index, bool)
        or not isinstance(source_episode_index, int)
        or source_episode_index < 0
    ):
        raise ValueError("source_episode_index must be a nonnegative integer")
    if not isinstance(prepared_request, PreparedInitialRequest):
        raise TypeError("prepared_request must be a PreparedInitialRequest")
    sample = prepared_request.sample
    payload = sample.payload_ascii
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "attempt_id": attempt_id,
        "source_episode_index": source_episode_index,
        "source_video_sha256": sample.source_video_sha256,
        "sampled_payload_sha256": payload_sha256,
        "sampling": {
            "original_fps": sample.source_fps,
            "original_frame_count": sample.total_num_frames,
            "original_duration_s": sample.duration_s,
            "target_fps": TARGET_SAMPLING_FPS,
            "selected_frame_indices": list(sample.frame_indices),
            "selected_parquet_timestamps_s": list(sample.parquet_timestamps),
            "decoder": {"name": sample.decoder_name, "version": sample.decoder_version},
            "color_space": "RGB",
            "resize": {
                "allow_upscale": False,
                "max_long_edge": RESIZE_MAX_LONG_EDGE,
                "resampling": "LANCZOS",
            },
            "jpeg": {"quality": JPEG_QUALITY, "optimize": False, "progressive": False},
        },
        "request_body": prepared_request.redacted_body,
    }


def build_parsed_artifact(
    *,
    raw_response: str,
    duration_s: float,
    parquet_timestamps: Iterable[float],
    validation_warnings: Sequence[str],
) -> dict[str, Any]:
    if any(not _valid_utf8_string(warning) for warning in validation_warnings):
        raise ValueError("validation warnings must be UTF-8 strings")
    proposal = build_cosmos_proposal(
        raw_response,
        duration_s=duration_s,
        parquet_timestamps=parquet_timestamps,
    )
    raw_bytes = raw_response.encode("utf-8")
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "model_response": deepcopy(proposal.model_response),
        "snapped_transition_frames": list(proposal.snapped_transition_frames),
        "validation_warnings": list(validation_warnings),
        "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    media_type: str
    byte_size: int
    sha256: str


ReferenceT = TypeVar("ReferenceT")


@dataclass(frozen=True)
class ArtifactWriteResult(Generic[ReferenceT]):
    record: ArtifactRecord
    database_reference: ReferenceT | None


@dataclass(frozen=True)
class ArtifactReadResult:
    record: ArtifactRecord
    contents: bytes


@dataclass(frozen=True)
class _ArtifactSnapshot:
    record: ArtifactRecord
    contents: bytes
    device: int
    inode: int


class ArtifactConflict(RuntimeError):
    """A complete destination exists with different or non-regular bytes."""


class ArtifactSecurityError(RuntimeError):
    """An artifact path or directory violated the no-symlink contract."""


EventHook = Callable[[str, str], None]


class AtomicArtifactStore:
    """Cross-process-locked, durable, no-clobber artifact installation."""

    _LOCK_NAME = ".artifact-store.lock"
    _TEMP_PREFIX = ".curation-artifact-v1-"
    _TEMP_PATTERN = re.compile(r"\.curation-artifact-v1-[0-9a-f]{32}\.tmp\Z")

    def __init__(
        self,
        workspace: Path,
        *,
        event_hook: EventHook | None = None,
        cleanup_on_start: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self._event_hook = event_hook
        if type(cleanup_on_start) is not bool:
            raise TypeError("cleanup_on_start must be a boolean")
        self._read_only = not cleanup_on_start
        if cleanup_on_start:
            self._ensure_workspace()
        else:
            self._require_existing_workspace()
        root_fd = self._open_workspace_path()
        try:
            root_stat = os.fstat(root_fd)
            self._workspace_identity = (root_stat.st_dev, root_stat.st_ino)
        finally:
            os.close(root_fd)
        self.startup_cleanup_report: list[dict[str, Any]] = []
        if cleanup_on_start:
            with self._locked() as root_fd:
                self.startup_cleanup_report = self._cleanup_locked(root_fd, referenced=frozenset())

    def _require_existing_workspace(self) -> None:
        try:
            workspace_stat = os.lstat(self.workspace)
        except OSError as error:
            raise ArtifactSecurityError("artifact workspace is unavailable") from error
        if not stat.S_ISDIR(workspace_stat.st_mode) or stat.S_ISLNK(workspace_stat.st_mode):
            raise ArtifactSecurityError("artifact workspace must be a real directory")

    def _emit(self, event: str, relative_path: str) -> None:
        if self._event_hook is not None:
            self._event_hook(event, relative_path)

    def _ensure_workspace(self) -> None:
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ArtifactSecurityError("artifact workspace could not be created") from error
        try:
            workspace_stat = os.lstat(self.workspace)
        except OSError as error:
            raise ArtifactSecurityError("artifact workspace is unavailable") from error
        if not stat.S_ISDIR(workspace_stat.st_mode) or stat.S_ISLNK(workspace_stat.st_mode):
            raise ArtifactSecurityError("artifact workspace must be a real directory")

    def _open_workspace_path(self) -> int:
        try:
            descriptor = os.open(
                self.workspace,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise ArtifactSecurityError("artifact workspace must remain a real directory") from error
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ArtifactSecurityError("artifact workspace must remain a directory")
        return descriptor

    def _open_root(self) -> int:
        descriptor = self._open_workspace_path()
        descriptor_stat = os.fstat(descriptor)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != self._workspace_identity:
            os.close(descriptor)
            raise ArtifactSecurityError("configured artifact workspace identity changed")
        return descriptor

    def _open_lock(self, root_fd: int) -> int:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        created = False
        try:
            try:
                lock_fd = os.open(self._LOCK_NAME, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd)
                created = True
            except FileExistsError:
                lock_fd = os.open(self._LOCK_NAME, flags, dir_fd=root_fd)
        except OSError as error:
            raise ArtifactSecurityError("artifact lock must be a regular file") from error
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            os.close(lock_fd)
            raise ArtifactSecurityError("artifact lock must be a regular file")
        if created:
            os.fsync(lock_fd)
            self._fsync_dirfd(root_fd, ".")
        return lock_fd

    @contextmanager
    def _locked(self):
        root_fd = self._open_root()
        if self._read_only:
            try:
                yield root_fd
            finally:
                os.close(root_fd)
            return
        lock_fd = -1
        try:
            lock_fd = self._open_lock(root_fd)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield root_fd
        finally:
            if lock_fd >= 0:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            os.close(root_fd)

    def write_bytes(
        self,
        relative_path: str,
        contents: bytes,
        *,
        media_type: str,
        register: Callable[[ArtifactRecord], ReferenceT] | None = None,
        authorization_guard: Callable[[], Any] | None = None,
    ) -> ArtifactWriteResult[ReferenceT]:
        if self._read_only:
            raise ArtifactSecurityError("read-only artifact inspector cannot write")
        if not isinstance(contents, bytes):
            raise TypeError("artifact contents must be bytes")
        normalized = _safe_artifact_relative_path(relative_path)
        parts = PurePosixPath(normalized).parts
        if self._TEMP_PATTERN.fullmatch(parts[-1]) is not None:
            raise ValueError("artifact destination uses the reserved temporary-file namespace")
        if authorization_guard is not None and not callable(authorization_guard):
            raise TypeError("artifact authorization guard must be callable")
        with self._locked() as root_fd:
            with nullcontext() if authorization_guard is None else authorization_guard():
                parent_fd, parent_relative = self._open_or_create_directories(root_fd, parts[:-1])
                try:
                    installed = self._install_no_clobber(
                        parent_fd,
                        parent_relative=parent_relative,
                        destination_name=parts[-1],
                        destination_relative=normalized,
                        contents=contents,
                        media_type=media_type,
                    )
                finally:
                    os.close(parent_fd)
                named = self._snapshot_from_root(root_fd, normalized, media_type=media_type, fsync_parent=True)
                self._require_same_snapshot(installed, named)
                configured = self._revalidate_configured_snapshot(named, media_type=media_type)
                record = configured.record
                database_reference = None if register is None else register(record)
                return ArtifactWriteResult(record=record, database_reference=database_reference)

    def inspect_existing(
        self,
        relative_path: str,
        *,
        media_type: str,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> ArtifactRecord:
        """Descriptor-safely inspect complete evidence left before a DB reference."""

        normalized = self._validated_existing_target(relative_path)
        self._validate_expected_artifact(expected_sha256, expected_byte_size)
        with self._locked() as root_fd:
            snapshot = self._stable_existing_snapshot(root_fd, normalized, media_type=media_type)
            self._require_expected_artifact(snapshot.record, expected_sha256, expected_byte_size)
            configured = self._revalidate_configured_snapshot(snapshot, media_type=media_type)
            return configured.record

    def read_existing(
        self,
        relative_path: str,
        *,
        media_type: str,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> ArtifactReadResult:
        """Recover exact immutable bytes from complete evidence without a model call."""

        normalized = self._validated_existing_target(relative_path)
        self._validate_expected_artifact(expected_sha256, expected_byte_size)
        with self._locked() as root_fd:
            snapshot = self._stable_existing_snapshot(root_fd, normalized, media_type=media_type)
            self._require_expected_artifact(snapshot.record, expected_sha256, expected_byte_size)
            configured = self._revalidate_configured_snapshot(snapshot, media_type=media_type)
            return ArtifactReadResult(record=configured.record, contents=configured.contents)

    def adopt_existing(
        self,
        relative_path: str,
        *,
        media_type: str,
        register: Callable[[ArtifactRecord], ReferenceT],
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
        authorization_guard: Callable[[], Any] | None = None,
    ) -> ArtifactWriteResult[ReferenceT]:
        """Durably reconcile complete evidence, then idempotently register it."""

        if self._read_only:
            raise ArtifactSecurityError("read-only artifact inspector cannot adopt")
        if not callable(register):
            raise TypeError("artifact adoption requires a database register callback")
        normalized = self._validated_existing_target(relative_path)
        self._validate_expected_artifact(expected_sha256, expected_byte_size)
        if authorization_guard is not None and not callable(authorization_guard):
            raise TypeError("artifact authorization guard must be callable")
        with self._locked() as root_fd:
            with nullcontext() if authorization_guard is None else authorization_guard():
                snapshot = self._stable_existing_snapshot(root_fd, normalized, media_type=media_type)
                self._require_expected_artifact(snapshot.record, expected_sha256, expected_byte_size)
                configured = self._revalidate_configured_snapshot(snapshot, media_type=media_type)
                reference = register(configured.record)
                return ArtifactWriteResult(record=configured.record, database_reference=reference)

    def _validated_existing_target(self, relative_path: str) -> str:
        normalized = _safe_artifact_relative_path(relative_path)
        if self._TEMP_PATTERN.fullmatch(PurePosixPath(normalized).name) is not None:
            raise ValueError("artifact temporary files cannot be inspected or adopted")
        return normalized

    @staticmethod
    def _validate_expected_artifact(expected_sha256: str | None, expected_byte_size: int | None) -> None:
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError("expected artifact SHA-256 must be lowercase hexadecimal")
        if expected_byte_size is not None and (type(expected_byte_size) is not int or expected_byte_size < 0):
            raise ValueError("expected artifact byte size must be a nonnegative built-in integer")

    @staticmethod
    def _require_expected_artifact(
        record: ArtifactRecord,
        expected_sha256: str | None,
        expected_byte_size: int | None,
    ) -> None:
        if expected_sha256 is not None and record.sha256 != expected_sha256:
            raise ArtifactConflict("artifact does not match the expected SHA-256")
        if expected_byte_size is not None and record.byte_size != expected_byte_size:
            raise ArtifactConflict("artifact does not match the expected byte size")

    def _open_or_create_directories(self, root_fd: int, components: Sequence[str]) -> tuple[int, str]:
        current_fd = os.dup(root_fd)
        traversed: list[str] = []
        try:
            for component in components:
                relative = "/".join((*traversed, component))
                created = False
                try:
                    next_fd = os.open(
                        component,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=current_fd,
                    )
                except FileNotFoundError:
                    try:
                        os.mkdir(component, 0o700, dir_fd=current_fd)
                        created = True
                        next_fd = os.open(
                            component,
                            os.O_RDONLY
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=current_fd,
                        )
                    except OSError as error:
                        raise ArtifactSecurityError(f"unsafe artifact directory: {relative}") from error
                except OSError as error:
                    raise ArtifactSecurityError(f"unsafe artifact directory: {relative}") from error
                if created:
                    self._emit("directory_created", relative)
                    self._fsync_dirfd(current_fd, "." if not traversed else "/".join(traversed))
                os.close(current_fd)
                current_fd = next_fd
                traversed.append(component)
            return current_fd, "." if not traversed else "/".join(traversed)
        except Exception:
            os.close(current_fd)
            raise

    @staticmethod
    def _open_existing_directories(root_fd: int, components: Sequence[str]) -> int:
        current_fd = os.dup(root_fd)
        try:
            for component in components:
                try:
                    next_fd = os.open(
                        component,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=current_fd,
                    )
                except FileNotFoundError as error:
                    raise ArtifactConflict("artifact parent directory does not exist") from error
                except OSError as error:
                    raise ArtifactSecurityError("artifact parent path is unsafe") from error
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    def _snapshot_from_root(
        self,
        root_fd: int,
        relative_path: str,
        *,
        media_type: str,
        fsync_parent: bool,
    ) -> _ArtifactSnapshot:
        if not _valid_utf8_nonempty(media_type):
            raise ValueError("artifact media type must be nonempty UTF-8")
        parts = PurePosixPath(relative_path).parts
        parent_fd = self._open_existing_directories(root_fd, parts[:-1])
        try:
            if fsync_parent:
                os.fsync(parent_fd)
            return self._snapshot_at_parent(
                parent_fd,
                parts[-1],
                relative_path=relative_path,
                media_type=media_type,
            )
        finally:
            os.close(parent_fd)

    @staticmethod
    def _snapshot_at_parent(
        parent_fd: int,
        destination_name: str,
        *,
        relative_path: str,
        media_type: str,
    ) -> _ArtifactSnapshot:
        try:
            descriptor = os.open(
                destination_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError as error:
            raise ArtifactConflict("artifact does not exist") from error
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ArtifactSecurityError("artifact destination must not be a symlink") from error
            raise ArtifactConflict("artifact destination is not a readable regular file") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactConflict("artifact destination is not a regular file")
            digest = hashlib.sha256()
            byte_size = 0
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                byte_size += len(chunk)
                chunks.append(chunk)
            after = os.fstat(descriptor)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity or byte_size != after.st_size:
                raise ArtifactSecurityError("artifact changed while it was being inspected")
            return _ArtifactSnapshot(
                record=ArtifactRecord(
                    relative_path=relative_path,
                    media_type=media_type,
                    byte_size=byte_size,
                    sha256=digest.hexdigest(),
                ),
                contents=b"".join(chunks),
                device=after.st_dev,
                inode=after.st_ino,
            )
        finally:
            os.close(descriptor)

    @staticmethod
    def _require_same_snapshot(expected: _ArtifactSnapshot, observed: _ArtifactSnapshot) -> None:
        if (
            expected.device != observed.device
            or expected.inode != observed.inode
            or expected.record != observed.record
            or expected.contents != observed.contents
        ):
            raise ArtifactSecurityError("artifact path changed during durable verification")

    def _revalidate_configured_snapshot(
        self, expected: _ArtifactSnapshot, *, media_type: str
    ) -> _ArtifactSnapshot:
        configured_root_fd = self._open_root()
        try:
            observed = self._snapshot_from_root(
                configured_root_fd,
                expected.record.relative_path,
                media_type=media_type,
                fsync_parent=True,
            )
        finally:
            os.close(configured_root_fd)
        self._require_same_snapshot(expected, observed)
        return observed

    def _stable_existing_snapshot(self, root_fd: int, relative_path: str, *, media_type: str) -> _ArtifactSnapshot:
        first = self._snapshot_from_root(
            root_fd,
            relative_path,
            media_type=media_type,
            fsync_parent=True,
        )
        self._emit("artifact_inspected", relative_path)
        second = self._snapshot_from_root(
            root_fd,
            relative_path,
            media_type=media_type,
            fsync_parent=True,
        )
        self._require_same_snapshot(first, second)
        return second

    def _install_no_clobber(
        self,
        parent_fd: int,
        *,
        parent_relative: str,
        destination_name: str,
        destination_relative: str,
        contents: bytes,
        media_type: str,
    ) -> _ArtifactSnapshot:
        temporary_name = f"{self._TEMP_PREFIX}{secrets.token_hex(16)}.tmp"
        temporary_fd = -1
        temporary_exists = False
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            temporary_exists = True
            _write_all(temporary_fd, contents)
            os.fsync(temporary_fd)
            self._emit("temporary_fsynced", parent_relative)
            os.close(temporary_fd)
            temporary_fd = -1
            try:
                os.link(
                    temporary_name,
                    destination_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = self._read_existing_artifact(parent_fd, destination_name)
                if existing != contents:
                    raise ArtifactConflict(f"artifact already exists with different bytes: {destination_relative}")
                self._emit("artifact_reconciled", destination_relative)
            except OSError as error:
                raise ArtifactSecurityError(f"artifact could not be installed: {destination_relative}") from error
            else:
                self._emit("artifact_linked", destination_relative)
            os.unlink(temporary_name, dir_fd=parent_fd)
            temporary_exists = False
            self._fsync_dirfd(parent_fd, parent_relative)
            installed = self._read_existing_artifact(parent_fd, destination_name)
            if installed != contents:
                raise ArtifactConflict(f"installed artifact bytes changed before hashing: {destination_relative}")
            snapshot = self._snapshot_at_parent(
                parent_fd,
                destination_name,
                relative_path=destination_relative,
                media_type=media_type,
            )
            if (
                snapshot.record.byte_size != len(contents)
                or snapshot.record.sha256 != hashlib.sha256(contents).hexdigest()
            ):
                raise ArtifactConflict(f"installed artifact bytes changed before hashing: {destination_relative}")
            return snapshot
        finally:
            if temporary_fd >= 0:
                os.close(temporary_fd)
            if temporary_exists:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                else:
                    self._fsync_dirfd(parent_fd, parent_relative)

    @staticmethod
    def _read_existing_artifact(parent_fd: int, destination_name: str) -> bytes:
        try:
            descriptor = os.open(
                destination_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ArtifactSecurityError("artifact destination must not be a symlink") from error
            raise ArtifactConflict("artifact destination is not a readable regular file") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ArtifactConflict("artifact destination is not a regular file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _fsync_dirfd(self, descriptor: int, relative: str) -> None:
        os.fsync(descriptor)
        self._emit("directory_fsync", relative)

    def write_json(
        self,
        relative_path: str,
        document: Mapping[str, Any],
        *,
        register: Callable[[ArtifactRecord], ReferenceT] | None = None,
        authorization_guard: Callable[[], Any] | None = None,
    ) -> ArtifactWriteResult[ReferenceT]:
        return self.write_bytes(
            relative_path,
            _canonical_json(dict(document)).encode("utf-8"),
            media_type="application/json",
            register=register,
            authorization_guard=authorization_guard,
        )

    def write_text(
        self,
        relative_path: str,
        content: str,
        *,
        register: Callable[[ArtifactRecord], ReferenceT] | None = None,
        authorization_guard: Callable[[], Any] | None = None,
    ) -> ArtifactWriteResult[ReferenceT]:
        return self.write_bytes(
            relative_path,
            content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            register=register,
            authorization_guard=authorization_guard,
        )

    def cleanup_temporary_files(self, *, referenced_relative_paths: Iterable[str] = ()) -> list[dict[str, Any]]:
        """Remove/report interrupted temporary writes, preserving complete evidence."""

        if self._read_only:
            raise ArtifactSecurityError("read-only artifact inspector cannot clean")
        referenced = {_safe_artifact_relative_path(relative_path) for relative_path in referenced_relative_paths}
        with self._locked() as root_fd:
            return self._cleanup_locked(root_fd, referenced=frozenset(referenced))

    def _cleanup_locked(self, root_fd: int, *, referenced: frozenset[str]) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        self._cleanup_directory_fd(
            root_fd,
            relative_parts=(),
            referenced=referenced,
            report=report,
        )
        return report

    def _cleanup_directory_fd(
        self,
        directory_fd: int,
        *,
        relative_parts: tuple[str, ...],
        referenced: frozenset[str],
        report: list[dict[str, Any]],
    ) -> None:
        removed_here = False
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            try:
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode):
                try:
                    child_fd = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                except OSError:
                    continue
                try:
                    self._cleanup_directory_fd(
                        child_fd,
                        relative_parts=(*relative_parts, name),
                        referenced=referenced,
                        report=report,
                    )
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(entry_stat.st_mode) or self._TEMP_PATTERN.fullmatch(name) is None:
                continue
            relative = "/".join((*relative_parts, name))
            if not _valid_utf8_string(relative) or relative in referenced:
                continue
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            removed_here = True
            report.append(
                {
                    "relative_path": relative,
                    "byte_size": entry_stat.st_size,
                    "removed": True,
                }
            )
        if removed_here:
            self._fsync_dirfd(directory_fd, "." if not relative_parts else "/".join(relative_parts))


def _safe_artifact_relative_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("artifact path must be a nonempty POSIX-relative path")
    raw_parts = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("artifact path must not be absolute or contain traversal components")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("artifact path must not be absolute or contain traversal components")
    return path.as_posix()


def _write_all(descriptor: int, contents: bytes) -> None:
    view = memoryview(contents)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("artifact write made no progress")
        written += count
