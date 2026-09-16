# ruff: noqa: F811
import json
from pathlib import Path

import annotation_history as history
from annotation_runs import RunStore
import app
from fastapi import HTTPException
import pytest
from test_annotation_publish import reviewed_run  # noqa: F401


@pytest.mark.parametrize("include_reviewed", [False, True])
def test_detection_preserves_reviewed_and_deleted_and_invalidates_only_changed(
    reviewed_run, tmp_path, monkeypatch, clear_dataset_state, include_reviewed
):
    root = Path(reviewed_run["root"])
    for path in (root / "meta/annotation_predictions").glob("*.json"):
        value = json.loads(path.read_text())
        value["created_at"] = "2026-09-15T00:00:00Z"
        path.write_text(json.dumps(value))
    monkeypatch.setattr(app, "EXPORT_ROOT", tmp_path / "workspace")
    alias = app._register_local_dataset(root).split("/")[1]
    store = RunStore(app.EXPORT_ROOT)
    run = store.create(root, root, None, None, "v3.1", [0, 1, 2])
    run["episodes"]["2"]["decision"] = "delete"
    run["episodes"]["1"]["generation_status"] = "failed"
    run["publication_state"] = "exported"
    run["export"] = {"manifest_sha256": "old"}
    run = store.save(run, run["revision"])
    view = app._workflow_payload(run)
    for ep in [0, 2]:
        atoms = view["episodes"][str(ep)]["atoms"]
        history.save_review(root, ep, atoms, True, history.annotation_hash(atoms))
    reviews = history.read_reviews(root)
    reviews.pop("1", None)
    history.write_reviews(root, reviews)
    seen = []
    def detect(selected, engine):
        seen.append(set(selected["episodes"]))
        for episode in selected["episodes"].values():
            episode["excluded_intervals"] = [{"start_frame": 1, "end_frame": 2}]
            episode["transition_filter_intervals"] = episode["excluded_intervals"]
        return {"enabled": True, "intervals": len(selected["episodes"])}
    monkeypatch.setattr(app, "apply_transition_filter", detect)
    response = app.detect_workflow_transitions(alias, app.WorkflowTransitionDetection(
        expected_revision=run["revision"], include_reviewed=include_reviewed))
    assert seen == [{"0", "1"} if include_reviewed else {"1"}]
    assert response["publication_state"] == "draft" and "export" not in response
    assert response["episodes"]["1"]["generation_status"] == "failed"
    assert response["episodes"]["1"]["excluded_intervals"] == [{"start_frame": 1, "end_frame": 2}]
    assert response["episodes"]["0"]["review"]["status"] == ("unreviewed" if include_reviewed else "reviewed")
    assert history.read_reviews(root)["2"] == reviews["2"]
    with pytest.raises(HTTPException) as error:
        app.detect_workflow_transitions(alias, app.WorkflowTransitionDetection(expected_revision=run["revision"]))
    assert error.value.status_code == 409
    revision = response["revision"]
    again = app.detect_workflow_transitions(alias, app.WorkflowTransitionDetection(
        expected_revision=revision, include_reviewed=include_reviewed))
    assert again["revision"] == revision
    assert again["transition_detection"]["episodes_changed"] == 0
    monkeypatch.setattr(app, "_require_editable", lambda _: (_ for _ in ()).throw(HTTPException(409, "Active job")))
    with pytest.raises(HTTPException) as error:
        app.detect_workflow_transitions(alias, app.WorkflowTransitionDetection(expected_revision=revision))
    assert error.value.status_code == 409
