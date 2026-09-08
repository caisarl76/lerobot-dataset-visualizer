"""Quality checks report evidence without manufacturing labels or decisions."""

from types import SimpleNamespace

import annotation_quality as quality
from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
import numpy as np
import pytest


def test_missing_grasp_is_a_finding_not_a_fabricated_interval():
    atoms = [{"style": "subtask", "content": "approach", "timestamp": 0.0}]
    issues = quality.check_prompt_sequence(atoms, ["approach", "grasp"])
    assert [issue["code"] for issue in issues] == ["missing_subtask"]
    assert issues[0]["source"] == "deterministic"
    assert len(atoms) == 1


def test_prompt_order_uses_timestamps_and_exact_labels():
    atoms = [
        {"style": "subtask", "content": "grasp", "timestamp": 0.0},
        {"style": "subtask", "content": "approach", "timestamp": 0.5},
        {"style": "subtask", "content": "Grasp", "timestamp": 1.0},
    ]
    assert {issue["code"] for issue in quality.check_prompt_sequence(atoms, ["approach", "grasp"])} == {
        "subtask_order_mismatch",
        "unexpected_subtask",
    }
    assert quality.check_prompt_sequence(list(reversed(atoms[:2])), ["grasp", "approach"]) == []
    assert quality.check_prompt_sequence(atoms, []) == []


class Frames:
    camera_keys = ("observation.images.top",)

    def frames_at(self, record, timestamps, *, camera_key):
        return [np.zeros((24, 32, 3), dtype=np.uint8) for _ in timestamps]


def assess(response):
    record = SimpleNamespace(frame_timestamps=[0.0, 0.5, 1.0], episode_index=0, episode_task="pick cup")
    return quality.assess_episode(record, StubVlmClient(responder=lambda _: response), Frames())


def test_quality_findings_are_advisory_and_keep_evidence():
    issues = assess({"issues": [{"code": "task_failure", "message": "Cup fell", "start": 0.5, "end": 1.0}]})
    assert issues == [
        {
            "code": "task_failure",
            "message": "Cup fell",
            "source": "vlm",
            "severity": "warning",
            "start": 0.5,
            "end": 1.0,
        }
    ]
    assert assess({"issues": []}) == []


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"issues": "fine"},
        {"issues": [], "decision": "delete"},
        {"issues": [{"code": "delete", "message": "bad", "start": 0, "end": 1}]},
        {"issues": [{"code": "task_failure", "message": "bad", "start": -1, "end": 1}]},
        {"issues": [{"code": "task_failure", "message": "bad", "start": 0, "end": 2}]},
        {"issues": [{"code": "task_failure", "message": "bad", "start": float("nan"), "end": 1}]},
        {"issues": [{"code": "task_failure", "message": "bad", "start": True, "end": 1}]},
    ],
)
def test_invalid_quality_reply_is_assessment_failure(response):
    assert [issue["code"] for issue in assess(response)] == ["assessment_failed"]
