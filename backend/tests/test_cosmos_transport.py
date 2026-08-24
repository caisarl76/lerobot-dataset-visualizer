from __future__ import annotations

import hashlib
import json
from typing import Any

from curation.cosmos_contract import COSMOS_RESPONSE_V2_SCHEMA
from curation.cosmos_transport import (
    CANONICAL_PROMPT_PREFIX,
    PROMPT_MAX_BYTES,
    CosmosTransport,
    PreparedSample,
    SamplingOutcome,
    build_canonical_prompt,
)
import httpx
import pytest


def _complete_response() -> dict[str, Any]:
    phases = (
        "approach_brown_table",
        "pick_up_object",
        "turn_to_find_black_trash_bin",
        "approach_black_trash_bin",
        "lean_down_to_black_trash_bin",
        "drop_object_into_black_trash_bin",
        "stand_straight",
    )
    starts = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 35.0)
    ends = (5.0, 10.0, 15.0, 20.0, 30.0, 35.0, 41.2)
    return {
        "schema_version": 2,
        "episode_complete": True,
        "segments": [
            {
                "step": index,
                "phase": phase,
                "status": "completed",
                "start_s": starts[index - 1],
                "end_s": ends[index - 1],
                "caption": phase,
                "confidence": 1.0,
                "evidence": "visible",
            }
            for index, phase in enumerate(phases, start=1)
        ],
        "missing_steps": [],
        "uncertainties": [],
    }


def _incomplete_response() -> dict[str, Any]:
    response = _complete_response()
    response["episode_complete"] = False
    response["segments"][-1].update(
        {"status": "not_observed", "start_s": None, "end_s": None, "confidence": None, "evidence": None}
    )
    response["missing_steps"] = [7]
    response["uncertainties"] = ["recovery is absent"]
    return response


def _sample() -> PreparedSample:
    timestamps = tuple(index / 50 for index in range(2_060))
    return PreparedSample(
        source_fps=50.0,
        total_num_frames=2_060,
        duration_s=41.2,
        frame_indices=(0, 25, 50),
        parquet_timestamps=(0.0, 0.5, 1.0),
        all_parquet_timestamps=timestamps,
        jpeg_frames=(b"abcd", b"efgh", b"ijkl"),
        decoder_name="PyAV",
        decoder_version="test",
        source_video_sha256="1" * 64,
    )


def _response(content: str, *, finish_reason: str = "stop", status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "id": "completion-id",
            "model": "cosmos3-nano-test",
            "created": 123,
            "usage": {"completion_tokens": 10},
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        },
    )


def _transport(
    handler: Any,
    *,
    sleep_calls: list[float] | None = None,
    response_max_bytes: int = 2 * 1024 * 1024,
) -> CosmosTransport:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    calls = sleep_calls if sleep_calls is not None else []
    return CosmosTransport(
        base_url="http://cosmos.test/v1",
        model="cosmos3-nano-test",
        api_key="secret-key",
        client=client,
        sleep=calls.append,
        response_max_bytes=response_max_bytes,
    )


def test_canonical_prompt_freezes_prefix_sorted_minified_schema_and_32kib_limit() -> None:
    assert hashlib.sha256(CANONICAL_PROMPT_PREFIX.encode("utf-8")).hexdigest() == (
        "cc7a3a174d09f28e96c7adb302bc6bbc2bca3c8ee1cd4f6faf5e1eec1dc31478"
    )
    expected = (
        CANONICAL_PROMPT_PREFIX
        + "\n"
        + json.dumps(
            COSMOS_RESPONSE_V2_SCHEMA,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    assert build_canonical_prompt() == expected
    assert len(expected.encode("utf-8")) <= PROMPT_MAX_BYTES
    with pytest.raises(ValueError, match="32 KiB"):
        from curation.cosmos_transport import build_initial_request_body

        build_initial_request_body(model="model", prompt="x" * (PROMPT_MAX_BYTES + 1), sample=_sample())


def test_exact_initial_wire_body_headers_timeout_and_stop_success() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(json.dumps(_complete_response()))

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))

    assert result.status == "succeeded"
    assert result.proposal is not None
    assert len(requests) == 1
    request = requests[0]
    assert request.headers["authorization"] == "Bearer secret-key"
    assert request.headers["content-type"] == "application/json"
    assert set(request.extensions["timeout"].values()) == {120.0}
    assert json.loads(request.content) == {
        "model": "cosmos3-nano-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "data:video/jpeg;base64,YWJjZA==,ZWZnaA==,aWprbA=="},
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
                "frames_indices": [0, 25, 50],
                "total_num_frames": 2060,
                "duration": 41.2,
                "do_sample_frames": False,
            }
        },
    }
    assert "extra_body" not in json.loads(request.content)
    assert result.initial_content == json.dumps(_complete_response())
    assert result.repair_content is None
    assert result.exchanges[0]["phase"] == "initial"
    assert result.exchanges[0]["response"] == {
        "status_code": 200,
        "id": "completion-id",
        "model": "cosmos3-nano-test",
        "created": 123,
        "usage": {"completion_tokens": 10},
        "finish_reason": "stop",
    }


def test_manual_only_sampling_outcome_never_calls_http_or_rejects() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP must not be called")

    result = _transport(handler).annotate(SamplingOutcome.manual_only("duration_limit"))
    assert result.status == "manual_only"
    assert result.reason == "duration_limit"
    assert result.reject_episode is False
    assert calls == 0


def test_connection_and_retryable_statuses_retry_once_with_injected_one_second_delay() -> None:
    for first in (httpx.ConnectError("offline"), 408, 429, 500, 503):
        calls = 0
        sleep_calls: list[float] = []

        def handler(request: httpx.Request, first: Exception | int = first) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                if isinstance(first, Exception):
                    raise first
                return httpx.Response(first, json={"error": "retry"})
            return _response(json.dumps(_complete_response()))

        result = _transport(handler, sleep_calls=sleep_calls).annotate(SamplingOutcome.ready(_sample()))
        assert result.status == "succeeded"
        assert calls == 2
        assert sleep_calls == [1.0]
        assert len(result.exchanges) == 2


@pytest.mark.parametrize("status_code", [400, 401, 404])
def test_nonretryable_4xx_is_manual_only_without_retry(status_code: int) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json={"error": "bad request"})

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    assert (result.status, result.reason, calls) == ("manual_only", "http_status", 1)


def test_timeout_is_manual_only_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    assert (result.status, result.reason, calls) == ("manual_only", "timeout", 1)


@pytest.mark.parametrize(
    ("body", "reason", "observed_content"),
    [
        ({"choices": []}, "missing_choice", None),
        ({"choices": [{}]}, "missing_content", None),
        (
            {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]},
            "finish_reason",
            "{}",
        ),
        (
            {"choices": [{"message": {"content": "{}"}}]},
            "finish_reason",
            "{}",
        ),
    ],
)
def test_invalid_success_envelope_preserves_any_exact_content_for_evidence(
    body: dict[str, Any], reason: str, observed_content: str | None
) -> None:
    result = _transport(lambda _: httpx.Response(200, json=body)).annotate(SamplingOutcome.ready(_sample()))
    assert (result.status, result.reason) == ("manual_only", reason)
    assert result.initial_content == observed_content
    assert result.repair_content is None


def test_response_body_limit_is_enforced_before_json_validation() -> None:
    result = _transport(lambda _: httpx.Response(200, content=b"x" * 101), response_max_bytes=100).annotate(
        SamplingOutcome.ready(_sample())
    )
    assert (result.status, result.reason) == ("manual_only", "response_too_large")


def test_invalid_contract_content_gets_one_text_only_repair_without_retry() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return _response("not json")
        return _response(json.dumps(_complete_response()))

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))

    assert result.status == "succeeded"
    assert result.initial_content == "not json"
    assert result.repair_content == json.dumps(_complete_response())
    assert len(requests) == 2
    repair = requests[1]
    assert repair.keys() == {
        "model",
        "messages",
        "temperature",
        "seed",
        "max_completion_tokens",
        "stream",
    }
    assert repair["max_completion_tokens"] == 2048
    assert "media_io_kwargs" not in repair
    repair_text = repair["messages"][0]["content"]
    prefix = "Return only one corrected JSON object matching pnp-trash-cosmos-v2.\n"
    assert repair_text.startswith(prefix)
    repair_payload = json.loads(repair_text[len(prefix) :])
    assert repair_payload["invalid_response"] == "not json"
    assert isinstance(repair_payload["validation_errors"], list)
    assert result.exchanges[-1]["phase"] == "repair"


def test_semantically_incomplete_valid_content_is_accepted_without_repair() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(json.dumps(_incomplete_response()))

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    assert result.status == "succeeded"
    assert result.proposal is not None
    assert result.proposal.requires_human_edits_before_approval is True
    assert calls == 1


def test_repair_failure_does_not_retry_and_becomes_manual_only() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response("not json")

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    assert (result.status, result.reason, calls) == ("manual_only", "repair_invalid", 2)
    assert result.initial_content == "not json"
    assert result.repair_content == "not json"


def test_repair_envelope_failure_preserves_exact_repair_content_without_retry() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("not json")
        return _response("truncated repair", finish_reason="length")

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    assert (result.status, result.reason, calls) == ("manual_only", "repair_transport", 2)
    assert result.initial_content == "not json"
    assert result.repair_content == "truncated repair"


def test_invalid_response_over_64kib_skips_repair() -> None:
    calls = 0
    invalid = "x" * (64 * 1024 + 1)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(invalid)

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    assert (result.status, result.reason, calls) == ("manual_only", "repair_input_too_large", 1)
