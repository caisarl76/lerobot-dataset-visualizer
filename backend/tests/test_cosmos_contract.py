from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import jsonschema
import pytest

SCHEMA_PATH = Path(__file__).parents[1] / "curation" / "schemas" / "cosmos_response_v2.schema.json"
APPROVED_SCHEMA_SHA256 = "5a2f75eb86e3327e62e73ef4e669f5fe630075f2efe951019004af2587ea4008"

COMPLETE_RESPONSE: dict[str, Any] = {
    "schema_version": 2,
    "episode_complete": True,
    "segments": [
        {
            "step": 1,
            "phase": "approach_brown_table",
            "status": "completed",
            "start_s": 0.0,
            "end_s": 8.6,
            "caption": "approach the brown table",
            "confidence": 0.93,
            "evidence": "the robot stops in front of the table",
        },
        {
            "step": 2,
            "phase": "pick_up_object",
            "status": "completed",
            "start_s": 8.6,
            "end_s": 15.1,
            "caption": "pick up the object",
            "confidence": 0.91,
            "evidence": "the hand closes and raises the object",
        },
        {
            "step": 3,
            "phase": "turn_to_find_black_trash_bin",
            "status": "completed",
            "start_s": 15.1,
            "end_s": 18.4,
            "caption": "turn to find the black trash bin",
            "confidence": 0.88,
            "evidence": "the camera turns until the bin is visible",
        },
        {
            "step": 4,
            "phase": "approach_black_trash_bin",
            "status": "completed",
            "start_s": 18.4,
            "end_s": 28.22,
            "caption": "approach the black trash bin",
            "confidence": 0.94,
            "evidence": "the robot walks toward the bin while holding the object",
        },
        {
            "step": 5,
            "phase": "lean_down_to_black_trash_bin",
            "status": "completed",
            "start_s": 28.22,
            "end_s": 32.88,
            "caption": "lean down to the black trash bin",
            "confidence": 0.9,
            "evidence": "the camera lowers over the bin",
        },
        {
            "step": 6,
            "phase": "drop_object_into_black_trash_bin",
            "status": "completed",
            "start_s": 32.88,
            "end_s": 35.72,
            "caption": "drop the object into the black trash bin",
            "confidence": 0.92,
            "evidence": "the hand opens and the object enters the bin",
        },
        {
            "step": 7,
            "phase": "stand_straight",
            "status": "completed",
            "start_s": 35.72,
            "end_s": 41.2,
            "caption": "go to a standing straight pose",
            "confidence": 0.95,
            "evidence": "the camera rises and stabilizes upright",
        },
    ],
    "missing_steps": [],
    "uncertainties": [],
}

INCOMPLETE_RESPONSE: dict[str, Any] = {
    "schema_version": 2,
    "episode_complete": False,
    "segments": [
        {
            "step": 1,
            "phase": "approach_brown_table",
            "status": "completed",
            "start_s": 0.0,
            "end_s": 8.6,
            "caption": "approach the brown table",
            "confidence": 0.93,
            "evidence": "the robot stops at the table",
        },
        {
            "step": 2,
            "phase": "pick_up_object",
            "status": "completed",
            "start_s": 8.6,
            "end_s": 15.1,
            "caption": "pick up the object",
            "confidence": 0.91,
            "evidence": "the object is lifted",
        },
        {
            "step": 3,
            "phase": "turn_to_find_black_trash_bin",
            "status": "completed",
            "start_s": 15.1,
            "end_s": 18.4,
            "caption": "turn to find the black trash bin",
            "confidence": 0.88,
            "evidence": "the bin becomes visible",
        },
        {
            "step": 4,
            "phase": "approach_black_trash_bin",
            "status": "completed",
            "start_s": 18.4,
            "end_s": 28.2,
            "caption": "approach the black trash bin",
            "confidence": 0.94,
            "evidence": "the robot reaches the bin",
        },
        {
            "step": 5,
            "phase": "lean_down_to_black_trash_bin",
            "status": "partial",
            "start_s": 28.2,
            "end_s": 31.0,
            "caption": "lean down to the black trash bin",
            "confidence": 0.62,
            "evidence": "the camera lowers but the motion is interrupted",
        },
        {
            "step": 6,
            "phase": "drop_object_into_black_trash_bin",
            "status": "not_observed",
            "start_s": None,
            "end_s": None,
            "caption": "drop the object into the black trash bin",
            "confidence": None,
            "evidence": None,
        },
        {
            "step": 7,
            "phase": "stand_straight",
            "status": "not_observed",
            "start_s": None,
            "end_s": None,
            "caption": "go to a standing straight pose",
            "confidence": None,
            "evidence": None,
        },
    ],
    "missing_steps": [5, 6, 7],
    "uncertainties": ["the episode ends during the lean"],
}


def _contract() -> ModuleType:
    module_path = Path(__file__).parents[1] / "curation" / "cosmos_contract.py"
    assert module_path.is_file(), "Cosmos contract implementation has not been created"
    return importlib.import_module("curation.cosmos_contract")


def _text(response: dict[str, Any]) -> str:
    return json.dumps(response, allow_nan=False)


def _parse(response: dict[str, Any], duration_s: float = 41.2) -> dict[str, Any]:
    return _contract().parse_cosmos_response(_text(response), duration_s=duration_s)


def test_schema_is_the_byte_exact_approved_draft_2020_12_contract() -> None:
    assert SCHEMA_PATH.is_file(), "approved response schema has not been created"
    schema_bytes = SCHEMA_PATH.read_bytes()
    assert hashlib.sha256(schema_bytes).hexdigest() == APPROVED_SCHEMA_SHA256

    schema = json.loads(schema_bytes)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"] == {"const": 2}
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["segment"]["additionalProperties"] is False

    ordered_slots = schema["properties"]["segments"]["prefixItems"]
    assert [(slot["allOf"][1]["properties"]["step"]["const"]) for slot in ordered_slots] == list(range(1, 8))
    conditional = schema["$defs"]["segment"]["allOf"][0]
    assert conditional["if"]["properties"]["status"] == {"const": "not_observed"}
    assert all(
        conditional["then"]["properties"][field] == {"type": "null"}
        for field in ("start_s", "end_s", "confidence", "evidence")
    )


@pytest.mark.parametrize("response", [COMPLETE_RESPONSE, INCOMPLETE_RESPONSE])
def test_approved_examples_parse(response: dict[str, Any]) -> None:
    parsed = _parse(response)
    assert parsed == response


def test_one_leading_think_block_is_accepted() -> None:
    content = f"<think>visible evidence checked</think>\n{_text(COMPLETE_RESPONSE)}"
    assert _contract().parse_cosmos_response(content, duration_s=41.2) == COMPLETE_RESPONSE


@pytest.mark.parametrize(
    "content",
    [
        f"```json\n{_text(COMPLETE_RESPONSE)}\n```",
        f"{_text(COMPLETE_RESPONSE)}\nlooks good",
        f"{_text(COMPLETE_RESPONSE)}\n{_text(COMPLETE_RESPONSE)}",
        f"<think>unfinished\n{_text(COMPLETE_RESPONSE)}",
        f"<think>first</think><think>second</think>{_text(COMPLETE_RESPONSE)}",
    ],
    ids=["markdown_fence", "trailing_prose", "multiple_objects", "unterminated_think", "two_thinks"],
)
def test_strict_parser_rejects_noncontract_envelopes(content: str) -> None:
    with pytest.raises(_contract().CosmosContractError):
        _contract().parse_cosmos_response(content, duration_s=41.2)


def test_absent_segments_are_rejected() -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    response.pop("segments")
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


def test_not_observed_cannot_invent_times_or_evidence() -> None:
    response = deepcopy(INCOMPLETE_RESPONSE)
    response["segments"][5].update({"start_s": 31.0, "end_s": 32.0, "confidence": 0.5, "evidence": "guessed"})
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


def test_wrong_step_order_is_rejected() -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    response["segments"][1], response["segments"][2] = (
        response["segments"][2],
        response["segments"][1],
    )
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


@pytest.mark.parametrize(
    ("episode_complete", "missing_steps"),
    [(True, [5, 6, 7]), (False, [])],
)
def test_completion_and_missing_steps_must_agree_bidirectionally(
    episode_complete: bool, missing_steps: list[int]
) -> None:
    source = COMPLETE_RESPONSE if episode_complete is False else INCOMPLETE_RESPONSE
    response = deepcopy(source)
    response["episode_complete"] = episode_complete
    response["missing_steps"] = missing_steps
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


@pytest.mark.parametrize("missing_steps", [[6, 7], [7, 6, 5]])
def test_missing_steps_must_exactly_match_in_ascending_order(missing_steps: list[int]) -> None:
    response = deepcopy(INCOMPLETE_RESPONSE)
    response["missing_steps"] = missing_steps
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


@pytest.mark.parametrize(
    ("field", "value"),
    [("start_s", -0.1), ("end_s", 41.3), ("start_s", 8.6)],
    ids=["negative_start", "past_duration", "empty_segment"],
)
def test_times_must_be_in_range_and_nonempty(field: str, value: float) -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    response["segments"][0][field] = value
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


@pytest.mark.parametrize(
    "content",
    [
        _text(COMPLETE_RESPONSE).replace('"start_s": 0.0', '"start_s": NaN', 1),
        _text(COMPLETE_RESPONSE).replace('"confidence": 0.93', '"confidence": Infinity', 1),
    ],
    ids=["nan_time", "infinite_confidence"],
)
def test_nonfinite_json_numbers_are_rejected(content: str) -> None:
    with pytest.raises(_contract().CosmosContractError):
        _contract().parse_cosmos_response(content, duration_s=41.2)


@pytest.mark.parametrize("field", ["start_s", "end_s", "confidence"])
def test_huge_json_numbers_are_reported_as_contract_errors(field: str) -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    response["segments"][0][field] = 10**1000
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


def test_many_huge_json_numbers_cannot_escape_contract_validation() -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    for segment in response["segments"]:
        segment["start_s"] = 10**1000
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


@pytest.mark.parametrize("field", ["caption", "evidence"])
def test_segment_human_text_must_be_valid_utf8(field: str) -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    response["segments"][0][field] = "\ud800"
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


def test_uncertainties_must_be_valid_utf8() -> None:
    response = deepcopy(INCOMPLETE_RESPONSE)
    response["uncertainties"] = ["\ud800"]
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


def test_incomplete_response_requires_an_uncertainty() -> None:
    response = deepcopy(INCOMPLETE_RESPONSE)
    response["uncertainties"] = []
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


def test_incomplete_timed_slots_keep_strict_start_order_without_filling_gaps() -> None:
    response = deepcopy(INCOMPLETE_RESPONSE)
    response["segments"][4]["start_s"] = 18.0
    response["segments"][4]["end_s"] = 18.2
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


@pytest.mark.parametrize(
    ("segment_index", "field", "value"),
    [
        (0, "start_s", 0.26),
        (6, "end_s", 40.69),
        (2, "start_s", 16.0),
    ],
    ids=["first_segment_coverage", "last_segment_coverage", "adjacency"],
)
def test_complete_response_requires_coverage_and_adjacency(segment_index: int, field: str, value: float) -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    response["segments"][segment_index][field] = value
    with pytest.raises(_contract().CosmosContractError):
        _parse(response)


def test_complete_response_does_not_require_strictly_increasing_end_times() -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    response["segments"][0]["end_s"] = 9.0
    response["segments"][1]["start_s"] = 8.6
    response["segments"][1]["end_s"] = 8.9
    response["segments"][2]["start_s"] = 8.8
    assert _parse(response) == response


def test_complete_proposal_snaps_float64_ties_to_lower_frame_index() -> None:
    response = deepcopy(COMPLETE_RESPONSE)
    starts = [0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    ends = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0]
    for segment, start_s, end_s in zip(response["segments"], starts, ends, strict=True):
        segment["start_s"] = start_s
        segment["end_s"] = end_s

    proposal = _contract().build_cosmos_proposal(_text(response), duration_s=8.0, parquet_timestamps=range(9))
    assert proposal.snapped_transition_frames == (1, 2, 3, 4, 5, 6)
    assert proposal.requires_human_edits_before_approval is False


def test_complete_proposal_rejects_nonincreasing_snapped_transitions() -> None:
    with pytest.raises(_contract().CosmosContractError):
        _contract().build_cosmos_proposal(
            _text(COMPLETE_RESPONSE),
            duration_s=41.2,
            parquet_timestamps=[0.0, 41.2],
        )


def test_partial_proposal_preserves_nulls_and_requires_edits_before_approval() -> None:
    timestamps = [index / 10 for index in range(413)]
    proposal = _contract().build_cosmos_proposal(
        _text(INCOMPLETE_RESPONSE),
        duration_s=41.2,
        parquet_timestamps=timestamps,
    )
    assert proposal.snapped_transition_frames == (86, 151, 184, 282, None, None)
    assert proposal.requires_human_edits_before_approval is True
    assert proposal.model_response == INCOMPLETE_RESPONSE


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, -math.inf])
def test_snapping_rejects_nonfinite_parquet_timestamps(timestamp: float) -> None:
    with pytest.raises(_contract().CosmosContractError):
        _contract().build_cosmos_proposal(
            _text(INCOMPLETE_RESPONSE),
            duration_s=41.2,
            parquet_timestamps=[0.0, timestamp, 41.2],
        )
