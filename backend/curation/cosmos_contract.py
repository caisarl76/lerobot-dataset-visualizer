"""Strict parsing, validation, and source-frame snapping for Cosmos v2 responses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np

_SCHEMA_PATH = Path(__file__).with_name("schemas") / "cosmos_response_v2.schema.json"
with _SCHEMA_PATH.open(encoding="utf-8") as _schema_file:
    COSMOS_RESPONSE_V2_SCHEMA: dict[str, Any] = json.load(_schema_file)
Draft202012Validator.check_schema(COSMOS_RESPONSE_V2_SCHEMA)
_SCHEMA_VALIDATOR = Draft202012Validator(COSMOS_RESPONSE_V2_SCHEMA)


class CosmosContractError(ValueError):
    """A response cannot be represented by the frozen Cosmos v2 contract."""

    def __init__(self, errors: str | Iterable[str]) -> None:
        normalized = (errors,) if isinstance(errors, str) else tuple(errors)
        self.errors = normalized or ("Cosmos response validation failed",)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class CosmosProposal:
    """Validated model response plus its six step-transition frame proposals."""

    model_response: dict[str, Any]
    snapped_transition_frames: tuple[int | None, ...]
    requires_human_edits_before_approval: bool


def parse_cosmos_response(content: str, *, duration_s: float) -> dict[str, Any]:
    """Parse exactly one Cosmos object and enforce its static and dynamic contract."""

    response = _decode_response_object(content)
    schema_errors = sorted(_SCHEMA_VALIDATOR.iter_errors(response), key=_schema_error_key)
    if schema_errors:
        raise CosmosContractError(f"schema {error.json_path}: {error.message}" for error in schema_errors)
    _validate_dynamic_contract(response, duration_s=duration_s)
    return response


def snap_transition_frames(
    model_response: dict[str, Any], parquet_timestamps: Iterable[float]
) -> tuple[int | None, ...]:
    """Snap starts for steps 2--7 to nearest float64 timestamps.

    ``numpy.argmin`` returns the first minimum, which is the required smaller
    source-frame index when two absolute distances tie.
    """

    try:
        timestamps = np.asarray(list(parquet_timestamps), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CosmosContractError("parquet timestamps must be a one-dimensional numeric sequence") from error
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise CosmosContractError("parquet timestamps must be a nonempty one-dimensional sequence")
    if not np.isfinite(timestamps).all():
        raise CosmosContractError("parquet timestamps must all be finite")

    snapped: list[int | None] = []
    for segment in model_response["segments"][1:]:
        start_s = segment["start_s"]
        if start_s is None:
            snapped.append(None)
            continue
        distances = np.abs(timestamps - np.float64(start_s), dtype=np.float64)
        snapped.append(int(np.argmin(distances)))

    frames = tuple(snapped)
    if model_response["episode_complete"]:
        if len(frames) != 6 or any(frame is None for frame in frames):
            raise CosmosContractError("a complete response must yield six non-null transition frames")
        complete_frames = tuple(frame for frame in frames if frame is not None)
        if not _strictly_increasing(complete_frames):
            raise CosmosContractError("a complete response must yield six strictly increasing transition frames")
    return frames


def build_cosmos_proposal(
    content: str,
    *,
    duration_s: float,
    parquet_timestamps: Iterable[float],
) -> CosmosProposal:
    """Validate response text and build its reviewable transition proposal."""

    response = parse_cosmos_response(content, duration_s=duration_s)
    frames = snap_transition_frames(response, parquet_timestamps)
    return CosmosProposal(
        model_response=response,
        snapped_transition_frames=frames,
        requires_human_edits_before_approval=not response["episode_complete"],
    )


def _decode_response_object(content: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise CosmosContractError("response content must be a string")

    candidate = content.strip()
    if candidate.startswith("<think>"):
        think_end = candidate.find("</think>", len("<think>"))
        if think_end < 0:
            raise CosmosContractError("leading <think> block is unterminated")
        candidate = candidate[think_end + len("</think>") :].strip()

    decoder = json.JSONDecoder(
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    try:
        response, end = decoder.raw_decode(candidate)
    except (json.JSONDecodeError, ValueError) as error:
        raise CosmosContractError(f"response must contain exactly one JSON object: {error}") from error
    if candidate[end:].strip():
        raise CosmosContractError("response has trailing content after its JSON object")
    if not isinstance(response, dict):
        raise CosmosContractError("response JSON value must be an object")
    return response


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON number is not allowed: {value}")


def _schema_error_key(error: Any) -> tuple[str, str]:
    return error.json_path, error.message


def _validate_dynamic_contract(response: dict[str, Any], *, duration_s: float) -> None:
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float, np.floating)):
        raise CosmosContractError("video duration must be a finite positive number")
    try:
        duration = float(duration_s)
    except (OverflowError, TypeError, ValueError) as error:
        raise CosmosContractError("video duration must be a finite positive number") from error
    if not math.isfinite(duration) or duration <= 0:
        raise CosmosContractError("video duration must be a finite positive number")

    segments = response["segments"]
    errors: list[str] = []
    timed_segments: list[tuple[dict[str, Any], float, float]] = []
    for segment in segments:
        if not _is_valid_utf8(segment["caption"]):
            errors.append(f"step {segment['step']} caption must be valid UTF-8")
        evidence = segment["evidence"]
        if evidence is not None and not _is_valid_utf8(evidence):
            errors.append(f"step {segment['step']} evidence must be valid UTF-8")
        if segment["status"] == "not_observed":
            continue
        start_s = _finite_float(segment["start_s"])
        end_s = _finite_float(segment["end_s"])
        confidence = _finite_float(segment["confidence"])
        if start_s is None or end_s is None or confidence is None:
            errors.append(f"step {segment['step']} times and confidence must be finite")
            continue
        if not 0 <= start_s < end_s <= duration:
            errors.append(f"step {segment['step']} must satisfy 0 <= start_s < end_s <= duration")
        timed_segments.append((segment, start_s, end_s))

    if any(not _is_valid_utf8(uncertainty) for uncertainty in response["uncertainties"]):
        errors.append("uncertainties must contain valid UTF-8 strings")

    expected_missing = [segment["step"] for segment in segments if segment["status"] != "completed"]
    if response["missing_steps"] != expected_missing:
        errors.append("missing_steps must exactly match partial and not_observed steps in ascending order")
    expected_complete = not expected_missing
    if response["episode_complete"] is not expected_complete:
        errors.append("episode_complete must be true if and only if missing_steps is empty")
    if not expected_complete and not response["uncertainties"]:
        errors.append("an incomplete response must contain at least one uncertainty")

    starts = [start_s for _, start_s, _ in timed_segments]
    if not _strictly_increasing(starts):
        errors.append("non-null timed segments must have strictly increasing starts")

    if expected_complete and len(timed_segments) == len(segments):
        ends = [end_s for _, _, end_s in timed_segments]
        if abs(starts[0]) > 0.25:
            errors.append("a complete response must start at 0 within 0.25 seconds")
        if abs(ends[-1] - duration) > 0.5:
            errors.append("a complete response must end at duration within 0.5 seconds")
        for previous, following in zip(timed_segments, timed_segments[1:]):
            if abs(previous[2] - following[1]) > 0.5:
                errors.append(
                    f"complete steps {previous[0]['step']} and {following[0]['step']} "
                    "must be adjacent within 0.5 seconds"
                )

    if errors:
        raise CosmosContractError(errors)


def _finite_float(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _is_valid_utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _strictly_increasing(values: Iterable[float | int]) -> bool:
    items = tuple(values)
    return all(previous < following for previous, following in zip(items, items[1:]))
