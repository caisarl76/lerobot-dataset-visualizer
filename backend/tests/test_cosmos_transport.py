from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from types import SimpleNamespace
from typing import Any

from curation import cosmos_transport
from curation.cosmos_contract import COSMOS_RESPONSE_V2_SCHEMA
from curation.cosmos_transport import (
    CANONICAL_PROMPT_PREFIX,
    PROMPT_MAX_BYTES,
    CosmosCallObservation,
    CosmosTransport,
    PreparedInitialRequest,
    PreparedSample,
    SamplingOutcome,
    build_canonical_prompt,
    build_initial_request_body,
    build_request_artifact,
    prepare_initial_request,
)
from curation.db import _validate_http_exchange, canonical_json
import httpx
import numpy as np
from PIL import Image
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


def _jpeg(red: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), (red, 0, 0)).save(
        output,
        format="JPEG",
        quality=85,
        optimize=False,
        progressive=False,
    )
    return output.getvalue()


def _sample() -> PreparedSample:
    timestamps = tuple(index / 50 for index in range(2_060))
    indices = tuple(range(0, 2_051, 25))
    return PreparedSample(
        source_fps=50.0,
        total_num_frames=2_060,
        duration_s=41.2,
        frame_indices=indices,
        parquet_timestamps=tuple(timestamps[index] for index in indices),
        all_parquet_timestamps=timestamps,
        jpeg_frames=tuple(_jpeg(index % 256) for index in indices),
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
    persist_observation: Any | None = None,
) -> CosmosTransport:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    calls = sleep_calls if sleep_calls is not None else []
    persisted: list[CosmosCallObservation] = []
    return CosmosTransport(
        base_url="http://cosmos.test/v1",
        model="cosmos3-nano-test",
        api_key="secret-key",
        client=client,
        sleep=calls.append,
        response_max_bytes=response_max_bytes,
        persist_observation=persisted.append if persist_observation is None else persist_observation,
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
        build_initial_request_body(model="model", prompt="x" * (PROMPT_MAX_BYTES + 1), sample=_sample())


def test_approved_abbreviated_plan_wire_fixture_remains_exact_at_formatter_boundary() -> None:
    fixture = SimpleNamespace(
        data_url="data:video/jpeg;base64,YWJjZA==,ZWZnaA==,aWprbA==",
        source_fps=50.0,
        frame_indices=(0, 25, 50),
        total_num_frames=2_060,
        duration_s=41.2,
    )

    body = build_initial_request_body(
        model="cosmos3-nano-test",
        prompt=build_canonical_prompt(),
        sample=fixture,
    )

    assert body == {
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
                "num_frames": -1,
                "frames_indices": [0, 25, 50],
                "total_num_frames": 2_060,
                "duration": 41.2,
                "do_sample_frames": False,
            }
        },
    }


def test_initial_request_explicitly_disables_vllm_video_frame_cap() -> None:
    body = build_initial_request_body(
        model="cosmos3-nano-test",
        prompt=build_canonical_prompt(),
        sample=_sample(),
    )

    assert body["media_io_kwargs"]["video"]["num_frames"] == -1


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
    expected = {
        "model": "cosmos3-nano-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": _sample().data_url},
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
                "num_frames": -1,
                "frames_indices": list(_sample().frame_indices),
                "total_num_frames": 2060,
                "duration": 41.2,
                "do_sample_frames": False,
            }
        },
    }
    assert json.loads(request.content) == expected
    assert result.prepared_initial_request is not None
    assert request.content == result.prepared_initial_request.wire_bytes
    assert result.exchanges[0]["request"]["body_sha256"] == result.prepared_initial_request.body_sha256
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


def test_prepared_initial_request_makes_wire_hash_and_artifact_impossible_to_diverge() -> None:
    sample = _sample()
    prepared = prepare_initial_request(
        model="cosmos3-nano-test",
        prompt=build_canonical_prompt(),
        sample=sample,
    )

    artifact = build_request_artifact(
        attempt_id="00000000-0000-0000-0000-000000000001",
        source_episode_index=7,
        prepared_request=prepared,
    )

    assert prepared.body_sha256 == hashlib.sha256(prepared.wire_bytes).hexdigest()
    assert artifact["request_body"] == prepared.redacted_body
    assert artifact["sampled_payload_sha256"] == hashlib.sha256(sample.payload_ascii).hexdigest()
    with pytest.raises(ValueError, match="hash"):
        PreparedInitialRequest(
            wire_bytes=prepared.wire_bytes,
            body_sha256="0" * 64,
            sample=sample,
        )
    with pytest.raises(ValueError, match="wire"):
        PreparedInitialRequest(
            wire_bytes=prepared.wire_bytes + b" ",
            body_sha256=hashlib.sha256(prepared.wire_bytes + b" ").hexdigest(),
            sample=sample,
        )


def test_production_prepared_request_rejects_abbreviated_or_mismatched_sampling_proof() -> None:
    sample = _sample()
    abbreviated = replace(
        sample,
        frame_indices=sample.frame_indices[:3],
        parquet_timestamps=sample.parquet_timestamps[:3],
        jpeg_frames=sample.jpeg_frames[:3],
    )

    with pytest.raises(ValueError, match="sampling proof"):
        prepare_initial_request(
            model="cosmos3-nano-test",
            prompt=build_canonical_prompt(),
            sample=abbreviated,
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "cosmos.test/v1",
        "ftp://cosmos.test/v1",
        "http:///v1",
        "http://user:password@cosmos.test/v1",
        "http://cosmos.test/v1?token=secret",
        "http://cosmos.test/v1#fragment",
        "http://cosmos.test:99999/v1",
        "http://cosmos.test/\nsecret",
        "http://cosmos.test/\x00secret",
    ],
)
def test_transport_rejects_noncanonical_or_credential_bearing_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        CosmosTransport(base_url=base_url, model="model", api_key="key")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://[v1.foo]/v1",
        "http://😀.example/v1",
        "http://cosmos.\u00a0test/v1",
    ],
)
def test_transport_rejects_authorities_httpx_cannot_construct_without_echoing_them(
    base_url: str,
) -> None:
    with pytest.raises(ValueError) as caught:
        CosmosTransport(base_url=base_url, model="model", api_key="key")

    assert str(caught.value) == "Cosmos base URL is not a valid HTTP request URL"
    assert base_url not in str(caught.value)


@pytest.mark.parametrize(
    "api_key",
    ["space key", " key", "key ", "key\r\nInjected: yes", "key\x00", "key\x1f", "key\x7f"],
)
def test_transport_rejects_non_visible_bearer_tokens(api_key: str) -> None:
    with pytest.raises(ValueError, match="API key"):
        CosmosTransport(base_url="http://cosmos.test/v1", model="model", api_key=api_key)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), "120", np.float64(120)])
def test_transport_rejects_noncanonical_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout"):
        CosmosTransport(
            base_url="http://cosmos.test/v1",
            model="model",
            api_key="key",
            timeout_seconds=timeout,
        )


@pytest.mark.parametrize("limit", [True, 0, -1, 1.0, np.int64(2)])
def test_transport_rejects_noncanonical_response_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="response_max_bytes"):
        CosmosTransport(
            base_url="http://cosmos.test/v1",
            model="model",
            api_key="key",
            response_max_bytes=limit,
        )


def test_default_httpx_client_disables_environment_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_kwargs: list[dict[str, Any]] = []

    class ClientProbe:
        def __init__(self, **kwargs: Any) -> None:
            constructor_kwargs.append(kwargs)

    monkeypatch.setattr(cosmos_transport.httpx, "Client", ClientProbe)

    CosmosTransport(base_url="http://cosmos.test/v1", model="model", api_key="key")

    assert constructor_kwargs == [{"trust_env": False, "follow_redirects": False}]


def test_injected_redirect_following_client_cannot_forward_bearer_or_video_to_second_host() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "cosmos.test":
            return httpx.Response(307, headers={"location": "http://evil.test/steal"})
        return _response(json.dumps(_complete_response()))

    observations: list[CosmosCallObservation] = []
    transport = CosmosTransport(
        base_url="http://cosmos.test/v1",
        model="cosmos3-nano-test",
        api_key="secret-key",
        client=httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ),
        persist_observation=observations.append,
    )

    result = transport.annotate(SamplingOutcome.ready(_sample()))

    assert (result.status, result.reason) == ("manual_only", "http_status")
    assert [request.url.host for request in requests] == ["cosmos.test"]
    assert len(observations) == 1


@pytest.mark.parametrize(
    ("exception_type", "expected_summary"),
    [
        (httpx.ConnectError, "Cosmos connection failed"),
        (httpx.ReadTimeout, "Cosmos request timed out"),
        (httpx.ReadError, "Cosmos transport failed"),
    ],
)
def test_httpx_exception_text_cannot_leak_api_key_into_persisted_exchange(
    exception_type: type[httpx.RequestError], expected_summary: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_type("secret-key appeared in low-level error", request=request)

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    serialized = canonical_json(result.exchanges)

    assert "secret-key" not in serialized
    assert result.exchanges[-1]["error"]["summary"] == expected_summary
    _validate_http_exchange(result.exchanges[-1])


def test_defensive_invalid_url_from_httpx_is_bounded_without_secret_leakage() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.InvalidURL("secret-key appeared in invalid URL details")

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))
    serialized = canonical_json(result.exchanges)

    assert (result.status, result.reason) == ("manual_only", "invalid_url")
    assert "secret-key" not in serialized
    assert result.exchanges[0]["error"] == {
        "class": "InvalidURL",
        "summary": "Cosmos request URL was rejected",
    }
    _validate_http_exchange(result.exchanges[0])


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


def test_retry_observation_is_persisted_before_sleep_and_second_model_call() -> None:
    events: list[str] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        events.append(f"call-{calls}")
        if calls == 1:
            raise httpx.ConnectError("offline", request=request)
        return _response(json.dumps(_complete_response()))

    def persist(observation: CosmosCallObservation) -> None:
        events.append(f"persist-{observation.phase}-{observation.reason}")
        assert observation.exchange["phase"] == observation.phase
        canonical_json(observation.exchange).encode("utf-8")

    transport = _transport(handler, persist_observation=persist)
    transport.sleep = lambda seconds: events.append(f"sleep-{seconds}")

    result = transport.annotate(SamplingOutcome.ready(_sample()))

    assert result.status == "succeeded"
    assert events == [
        "call-1",
        "persist-initial-connection",
        "sleep-1.0",
        "call-2",
        "persist-initial-None",
    ]


def test_initial_invalid_observation_is_persisted_before_repair_call_and_repair_return() -> None:
    events: list[str] = []
    observations: list[CosmosCallObservation] = []
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        events.append(f"call-{calls}")
        if calls == 1:
            return _response("not json")
        return _response(json.dumps(_complete_response()))

    def persist(observation: CosmosCallObservation) -> None:
        observations.append(observation)
        events.append(f"persist-{observation.phase}")

    result = _transport(handler, persist_observation=persist).annotate(SamplingOutcome.ready(_sample()))
    events.append("returned")

    assert result.status == "succeeded"
    assert events == ["call-1", "persist-initial", "call-2", "persist-repair", "returned"]
    assert [(value.phase, value.content, value.observed_content) for value in observations] == [
        ("initial", "not json", "not json"),
        ("repair", json.dumps(_complete_response()), json.dumps(_complete_response())),
    ]


def test_persistence_failure_propagates_and_prevents_sleep_or_any_later_model_call() -> None:
    calls = 0
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    def crash(_: CosmosCallObservation) -> None:
        raise RuntimeError("observation database crashed")

    transport = _transport(handler, persist_observation=crash, sleep_calls=sleep_calls)
    with pytest.raises(RuntimeError, match="observation database crashed"):
        transport.annotate(SamplingOutcome.ready(_sample()))

    assert calls == 1
    assert sleep_calls == []


def test_ready_annotation_requires_synchronous_observation_persistence_before_http() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(json.dumps(_complete_response()))

    transport = CosmosTransport(
        base_url="http://cosmos.test/v1",
        model="cosmos3-nano-test",
        api_key="secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ValueError, match="persistence"):
        transport.annotate(SamplingOutcome.ready(_sample()))
    assert calls == 0


def test_public_phase_observation_api_allows_resume_to_skip_an_already_observed_call() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    transport = _transport(handler)
    prepared = prepare_initial_request(
        model="cosmos3-nano-test",
        prompt=build_canonical_prompt(),
        sample=_sample(),
    )

    observation = transport.observe_initial(prepared)
    persisted = [observation]
    if not persisted:
        transport.observe_initial(prepared)

    assert calls == 1
    assert observation.phase == "initial"
    assert observation.reason == "connection"
    assert observation.retryable is True


def test_initial_observation_rejects_prepared_request_for_a_different_model_without_http() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(json.dumps(_complete_response()))

    transport = _transport(handler)
    wrong_model = prepare_initial_request(
        model="different-cosmos-model",
        prompt=build_canonical_prompt(),
        sample=_sample(),
    )

    with pytest.raises(ValueError, match="model"):
        transport.observe_initial(wrong_model)
    assert calls == 0


def test_public_repair_observation_preserves_the_64kib_preflight_without_http() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(json.dumps(_complete_response()))

    transport = _transport(handler)

    with pytest.raises(ValueError, match="64 KiB"):
        transport.observe_repair(
            invalid_response="x" * (64 * 1024 + 1),
            validation_errors=("invalid",),
        )
    assert calls == 0


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


def test_hostile_transport_error_text_cannot_poison_task4_exchange_serialization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("\ud800", request=request)

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))

    assert (result.status, result.reason) == ("manual_only", "timeout")
    canonical_json(result.exchanges).encode("utf-8")
    _validate_http_exchange(result.exchanges[0])


@pytest.mark.parametrize(
    ("body", "reason"),
    [({"choices": []}, "missing_choice"), ({"choices": [{}]}, "missing_content")],
)
def test_invalid_success_envelope_preserves_any_exact_content_for_evidence(
    body: dict[str, Any], reason: str
) -> None:
    result = _transport(lambda _: httpx.Response(200, json=body)).annotate(SamplingOutcome.ready(_sample()))
    assert (result.status, result.reason) == ("manual_only", reason)
    assert result.initial_content is None
    assert result.repair_content is None


@pytest.mark.parametrize("finish_reason", ["length", None, "content_filter"])
def test_non_stop_initial_with_string_content_gets_exactly_one_text_only_repair(
    finish_reason: str | None,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            document = {
                "choices": [
                    {
                        "message": {"content": "truncated initial"},
                    }
                ]
            }
            if finish_reason is not None:
                document["choices"][0]["finish_reason"] = finish_reason
            return httpx.Response(200, json=document)
        return _response(json.dumps(_complete_response()))

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))

    assert result.status == "succeeded"
    assert result.initial_content == "truncated initial"
    assert result.repair_content == json.dumps(_complete_response())
    assert len(requests) == 2
    assert requests[1]["max_completion_tokens"] == 2048
    assert "media_io_kwargs" not in requests[1]


def test_response_body_limit_is_enforced_before_json_validation() -> None:
    result = _transport(lambda _: httpx.Response(200, content=b"x" * 101), response_max_bytes=100).annotate(
        SamplingOutcome.ready(_sample())
    )
    assert (result.status, result.reason) == ("manual_only", "response_too_large")


@pytest.mark.parametrize(
    "raw_body",
    [
        b'{"choices":[],"choices":[]}',
        b'{"choices":NaN}',
        b'{"choices":Infinity}',
        b'{"choices":1e999}',
        b'{"choices":[{"message":{"content":"ok","content":"duplicate"},"finish_reason":"stop"}]}',
        b'{"\\ud800":1,"\\ud800":2}',
        b'{"choices":[]}' + b"\xff",
    ],
)
def test_hostile_outer_json_is_bounded_manual_only_with_task4_serializable_exchange(raw_body: bytes) -> None:
    result = _transport(lambda _: httpx.Response(200, content=raw_body)).annotate(SamplingOutcome.ready(_sample()))

    assert (result.status, result.reason) == ("manual_only", "invalid_json")
    assert result.initial_content is None
    assert result.exchanges[0]["error"]["class"] == "CosmosEnvelopeError"
    canonical_json(result.exchanges).encode("utf-8")
    _validate_http_exchange(result.exchanges[0])


@pytest.mark.parametrize(
    "raw_body",
    [
        (b'{"deep":' + b"[" * 200 + b"0" + b"]" * 200 + b',"choices":[]}'),
        (
            b'{"usage":'
            + b'{"nested":' * 150
            + b"0"
            + b"}" * 150
            + b',"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'
        ),
        (b'{"deep":' + b"[" * 1_500 + b"0" + b"]" * 1_500 + b',"choices":[]}'),
    ],
)
def test_deep_outer_json_and_usage_are_bounded_without_recursion_escape(raw_body: bytes) -> None:
    assert len(raw_body) < 2 * 1024 * 1024

    result = _transport(lambda _: httpx.Response(200, content=raw_body)).annotate(SamplingOutcome.ready(_sample()))

    assert (result.status, result.reason) == ("manual_only", "invalid_json")
    canonical_json(result.exchanges).encode("utf-8")
    _validate_http_exchange(result.exchanges[0])


def test_surrogate_message_content_is_rejected_without_uncaught_encoding_or_repair() -> None:
    raw = b'{"choices":[{"message":{"content":"\\ud800"},"finish_reason":"stop"}]}'
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=raw)

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))

    assert (result.status, result.reason, calls) == ("manual_only", "invalid_utf8", 1)
    assert result.initial_content is None
    assert result.exchanges[0]["error"]["class"] == "CosmosEnvelopeError"
    canonical_json(result.exchanges).encode("utf-8")
    _validate_http_exchange(result.exchanges[0])


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
    prefix = (
        "Return only one corrected JSON object matching pnp-trash-cosmos-v2.\n"
        "The segments array must contain exactly seven objects in steps 1 through 7, one for every "
        "required phase. Never omit a phase. When a phase was not visibly attempted, emit it with "
        "status not_observed and null start_s, end_s, confidence, and evidence; caption remains a "
        "required nonempty string. Every segment object must contain all eight keys: step, phase, "
        "status, start_s, end_s, caption, confidence, and evidence. Never omit a key whose value is "
        "null.\n"
        "The exact response schema is:\n"
        + json.dumps(
            COSMOS_RESPONSE_V2_SCHEMA,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    assert repair_text.startswith(prefix)
    repair_payload = json.loads(repair_text[len(prefix) :])
    assert repair_payload["invalid_response"] == "not json"
    assert isinstance(repair_payload["validation_errors"], list)
    assert result.exchanges[-1]["phase"] == "repair"


def test_inner_json_surrogate_validation_error_is_safely_escaped_in_repair_request() -> None:
    requests: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.content)
        if len(requests) == 1:
            return _response('{"\\ud800":1}')
        return _response(json.dumps(_complete_response()))

    result = _transport(handler).annotate(SamplingOutcome.ready(_sample()))

    assert result.status == "succeeded"
    assert len(requests) == 2
    requests[1].decode("utf-8")


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
