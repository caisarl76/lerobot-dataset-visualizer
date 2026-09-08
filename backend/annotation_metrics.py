"""Boundary accuracy against explicit human reviews, separated by generation attempt."""

import math


def boundary_metrics(predicted: list[float], reviewed: list[float]) -> dict:
    """A signed delta is prediction minus review; mismatched transitions are never paired."""
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in [*predicted, *reviewed]):
        raise ValueError("Boundary timestamps must be finite numbers")
    result = {"status": "topology_mismatch", "signed_deltas": [], "absolute_deltas": [], "mae_seconds": None}
    if len(predicted) != len(reviewed):
        return result
    if not predicted:
        return {**result, "status": "no_boundaries"}
    signed = [prediction - human for prediction, human in zip(predicted, reviewed, strict=True)]
    absolute = [abs(delta) for delta in signed]
    return {
        "status": "matched",
        "signed_deltas": signed,
        "absolute_deltas": absolute,
        "mae_seconds": sum(absolute) / len(absolute),
    }


def _subtasks(atoms):
    rows = sorted((atom for atom in atoms if atom.get("style") == "subtask"), key=lambda atom: atom["timestamp"])
    # The first row begins the episode's initial action; subsequent rows are transitions.
    return [row.get("content") for row in rows], [row["timestamp"] for row in rows[1:]]


def summarize(episodes: dict, example_episode_indices: list[int]) -> dict:
    """Aggregate only retained reviewed episodes, without promoting retries to first-pass results.

    Prediction entries are supplied in attempt order. An entry with atoms=None
    records a failed attempt; an absent list records a missing prediction.
    """
    examples = {str(episode) for episode in example_episode_indices}
    eligible = [
        episode
        for key, episode in episodes.items()
        if str(key) not in examples
        and episode.get("decision") != "delete"
        and (episode.get("review") or {}).get("status") == "reviewed"
    ]
    summary = {}
    for name, attempt_index in (("first_pass", 0), ("latest", -1)):
        result = {
            "eligible_episodes": 0,
            "eligible_boundaries": 0,
            "reviewed_episodes": len(eligible),
            "reviewed_boundaries": 0,
            "missing_predictions": 0,
            "failed_predictions": 0,
            "topology_mismatch": 0,
            "signed_deltas": [],
            "absolute_deltas": [],
            "mae_seconds": None,
        }
        for episode in eligible:
            labels, boundaries = _subtasks(episode.get("atoms") or [])
            result["reviewed_boundaries"] += len(boundaries)
            predictions = episode.get("predictions") or []
            if not predictions:
                result["missing_predictions"] += 1
                continue
            prediction = predictions[attempt_index].get("atoms")
            if prediction is None:
                result["failed_predictions"] += 1
                continue
            predicted_labels, predicted_boundaries = _subtasks(prediction)
            if predicted_labels != labels:
                result["topology_mismatch"] += 1
                continue
            metrics = boundary_metrics(predicted_boundaries, boundaries)
            if metrics["status"] == "topology_mismatch":
                result["topology_mismatch"] += 1
                continue
            result["eligible_episodes"] += 1
            result["eligible_boundaries"] += len(boundaries)
            result["signed_deltas"] += metrics["signed_deltas"]
            result["absolute_deltas"] += metrics["absolute_deltas"]
        if result["eligible_boundaries"]:
            result["mae_seconds"] = sum(result["absolute_deltas"]) / result["eligible_boundaries"]
        summary[name] = result
    return summary
