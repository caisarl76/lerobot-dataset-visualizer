"""Reviewed boundary accuracy keeps failed attempts and retries distinct."""

import annotation_metrics as metrics
import pytest


def atoms(boundary=2.0, *, last="grasp"):
    return [
        {"style": "subtask", "content": "approach", "timestamp": 0.0},
        {"style": "subtask", "content": last, "timestamp": boundary},
    ]


def reviewed_episode(*predictions, reviewed=None, **overrides):
    return {
        "atoms": reviewed if reviewed is not None else atoms(),
        "review": {"status": "reviewed"},
        "generation_status": "generated",
        "decision": "keep",
        "predictions": [
            {"atoms": prediction, "created_at": f"2026-09-08T00:00:0{i}Z"}
            for i, prediction in enumerate(predictions)
        ],
        **overrides,
    }


def test_changed_topology_is_not_zipped_into_false_accuracy():
    result = metrics.boundary_metrics([3.0], [2.0, 5.0])
    assert result == {
        "status": "topology_mismatch",
        "signed_deltas": [],
        "absolute_deltas": [],
        "mae_seconds": None,
    }


def test_boundary_errors_use_predicted_minus_reviewed():
    result = metrics.boundary_metrics([1.5, 5.0], [2.0, 4.0])
    assert result == {
        "status": "matched",
        "signed_deltas": [-0.5, 1.0],
        "absolute_deltas": [0.5, 1.0],
        "mae_seconds": 0.75,
    }
    assert metrics.boundary_metrics([], []) == {
        "status": "no_boundaries",
        "signed_deltas": [],
        "absolute_deltas": [],
        "mae_seconds": None,
    }


def test_first_pass_and_retry_accuracy_remain_separate():
    summary = metrics.summarize(
        {"0": reviewed_episode(atoms(3.0), atoms(2.25)), "1": reviewed_episode(atoms(1.5))}, []
    )
    first, latest = summary["first_pass"], summary["latest"]
    assert first["signed_deltas"] == [1.0, -0.5]
    assert latest["signed_deltas"] == [0.25, -0.5]
    assert first["mae_seconds"] == 0.75
    assert latest["mae_seconds"] == 0.375
    assert first["eligible_episodes"] == latest["eligible_episodes"] == 2
    assert first["eligible_boundaries"] == latest["eligible_boundaries"] == 2


def test_missing_first_prediction_is_not_replaced_by_successful_retry():
    summary = metrics.summarize({"0": reviewed_episode(None, atoms(2.5)), "1": reviewed_episode()}, [])
    assert summary["first_pass"]["failed_predictions"] == 1
    assert summary["first_pass"]["missing_predictions"] == 1
    assert summary["first_pass"]["eligible_episodes"] == 0
    assert summary["first_pass"]["mae_seconds"] is None
    assert summary["latest"]["failed_predictions"] == 0
    assert summary["latest"]["missing_predictions"] == 1
    assert summary["latest"]["mae_seconds"] == 0.5


def test_failed_retry_does_not_erase_first_pass_score():
    summary = metrics.summarize({"0": reviewed_episode(atoms(3.0), None, generation_status="failed")}, [])
    assert summary["first_pass"]["mae_seconds"] == 1.0
    assert summary["latest"]["failed_predictions"] == 1
    assert summary["latest"]["mae_seconds"] is None


def test_same_number_of_boundaries_with_different_labels_is_mismatch():
    summary = metrics.summarize({"0": reviewed_episode(atoms(last="drop"))}, [])
    assert summary["first_pass"]["topology_mismatch"] == 1
    assert summary["first_pass"]["eligible_boundaries"] == 0
    assert summary["first_pass"]["mae_seconds"] is None


def test_excludes_examples_unreviewed_and_delete_decisions_from_denominators():
    summary = metrics.summarize(
        {
            "0": reviewed_episode(atoms(3.0)),
            "1": reviewed_episode(atoms(4.0), review={"status": "unreviewed"}),
            "2": reviewed_episode(atoms(5.0), decision="delete"),
            "3": reviewed_episode(atoms(2.25)),
        },
        [0],
    )
    assert summary["first_pass"]["eligible_episodes"] == 1
    assert summary["first_pass"]["reviewed_episodes"] == 1
    assert summary["first_pass"]["reviewed_boundaries"] == 1
    assert summary["first_pass"]["mae_seconds"] == 0.25


def test_zero_boundary_episodes_count_without_zero_error_claim():
    one_subtask = atoms()[:1]
    summary = metrics.summarize({"0": reviewed_episode(one_subtask, reviewed=one_subtask)}, [])
    assert summary["first_pass"]["eligible_episodes"] == 1
    assert summary["first_pass"]["eligible_boundaries"] == 0
    assert summary["first_pass"]["mae_seconds"] is None
    assert metrics.summarize({}, [])["latest"]["eligible_episodes"] == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_boundaries_are_not_accuracy_numbers(invalid):
    with pytest.raises(ValueError, match="finite"):
        metrics.boundary_metrics([invalid], [0.0])
