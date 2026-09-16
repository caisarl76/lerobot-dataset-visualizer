"""A generation job durably checkpoints individual episodes and resumes unfinished work."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from annotation_runs import RunStore, atomic_json, recover_jobs
import app
from fastapi import HTTPException
from official_annotations import REVISION, parse_config
import pandas as pd
import pytest


class SynchronousPool:
    def submit(self, function):
        function()


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    root = tmp_path / "source"
    atomic_json(root / "meta/info.json", {"codebase_version": "v3.1"})
    labels = {
        str(ep): {
            "atoms": [
                {
                    "style": "subtask",
                    "content": f"original {ep}",
                    "timestamp": 0.0,
                    "role": "assistant",
                    "camera": None,
                    "tool_calls": None,
                }
            ]
        }
        for ep in range(3)
    }
    atomic_json(
        root / "meta/lerobot_annotations.json", {"version": 2, "episodes": labels}
    )
    monkeypatch.setattr(app, "EXPORT_ROOT", workspace)
    monkeypatch.setattr(app, "_annotation_jobs", {})
    monkeypatch.setattr(app, "_annotation_pool", SynchronousPool())
    monkeypatch.setattr(app, "_download_full_dataset", lambda _: None)
    store = RunStore(workspace)
    run = store.create(root, root, "team/source", "source-sha", "v3.1", [0, 1, 2])

    def state_for(request):
        state_root = (
            Path(request.local_path)
            if request.local_path
            else Path(app._local_aliases()[request.repo_id])
        )
        saved = json.loads((state_root / "meta/lerobot_annotations.json").read_text())[
            "episodes"
        ]
        return SimpleNamespace(
            root=state_root,
            info={"codebase_version": "v3.1"},
            episodes_df=pd.DataFrame({"episode_index": [0, 1, 2]}),
            annotations={
                int(ep): SimpleNamespace(atoms=value["atoms"])
                for ep, value in saved.items()
            },
        )

    monkeypatch.setattr(app, "_ensure_state", state_for)
    calls = []
    fail = set()
    persisted_before_episode = {}

    def generate(source, output, annotations, config, episodes, **kwargs):
        source, output = Path(source), Path(output)
        assert len(episodes) == 1, (
            "Run orchestration must checkpoint one official target at a time"
        )
        episode = episodes[0]
        persisted_before_episode[episode] = store.read(run["run_id"])
        calls.append({"episode": episode, "source": source, "output": output})
        if episode in fail:
            raise RuntimeError(f"episode {episode} video decode failed")
        shutil.copytree(source, output)
        sidecar = json.loads((output / "meta/lerobot_annotations.json").read_text())
        sidecar["episodes"][str(episode)]["atoms"][0]["content"] = (
            f"generated {episode}"
        )
        atomic_json(output / "meta/lerobot_annotations.json", sidecar)
        prediction = output / f"meta/annotation_predictions/episode-{episode}.json"
        atomic_json(
            prediction, {"episodes": {str(episode): sidecar["episodes"][str(episode)]}}
        )
        return {
            "output_dir": str(output),
            "first_generated_episode_index": episode,
            "validation": {
                "ok": True,
                "errors": [],
                "warnings": [],
                "episodes_checked": 3,
            },
            "prediction_snapshot": str(prediction),
            "phases": [],
            "episode_results": {
                str(episode): {"generation_status": "generated", "issues": []}
            },
        }

    engine = SimpleNamespace(
        parse_config=parse_config, generate_dataset=generate, REVISION=REVISION
    )
    monkeypatch.setattr(app, "_official_engine", lambda: engine)
    return SimpleNamespace(
        root=root,
        store=store,
        run=run,
        calls=calls,
        fail=fail,
        persisted_before_episode=persisted_before_episode,
    )


def test_good_episode_checkpoint_survives_next_failure_and_later_episode_continues(
    workflow,
):
    workflow.fail.add(1)
    queued = app.create_annotation_job(
        app.GenerationRequest(local_path=str(workflow.root))
    )
    job = app.get_annotation_job(queued["job_id"])
    run = workflow.store.read(workflow.run["run_id"])
    assert job["status"] == "completed"
    assert [call["episode"] for call in workflow.calls] == [0, 1, 2]
    first_checkpoint = workflow.calls[0]["output"]
    assert (
        workflow.calls[1]["source"] == workflow.calls[2]["source"] == first_checkpoint
    )
    persisted = workflow.persisted_before_episode[1]
    assert persisted["episodes"]["0"]["generation_status"] == "generated"
    assert Path(persisted["root"]) == first_checkpoint
    assert run["episodes"]["0"]["generation_status"] == "generated"
    assert run["episodes"]["1"]["generation_status"] == "failed"
    assert run["episodes"]["1"]["issues"]
    assert run["episodes"]["2"]["generation_status"] == "generated"
    assert run["current_job_id"] == queued["job_id"] and "status" not in run
    root = Path(run["root"])
    assert job["result"]["output_dir"] == str(root)
    assert app._local_aliases()[job["result"]["repo_id"]] == str(root)
    labels = json.loads((root / "meta/lerobot_annotations.json").read_text())[
        "episodes"
    ]
    assert [labels[str(ep)]["atoms"][0]["content"] for ep in range(3)] == [
        "generated 0",
        "original 1",
        "generated 2",
    ]
    assert (root / "meta/annotation_predictions/episode-0.json").is_file()
    assert (root / "meta/annotation_predictions/episode-2.json").is_file()
    assert not (root / "meta/annotation_predictions/episode-1.json").exists()
    predictions = [
        json.loads(path.read_text())
        for path in (root / "meta/annotation_predictions").glob("*.json")
    ]
    assert all("1" not in prediction["episodes"] for prediction in predictions)
    assert any(
        prediction.get("episode_results", {}).get("1", {}).get("generation_status")
        == "failed"
        for prediction in predictions
    )


def test_restart_marks_job_interrupted_and_resume_selects_only_unfinished_episodes(
    workflow,
):
    store = workflow.store
    run = store.read(workflow.run["run_id"])
    sidecar = workflow.root / "meta/lerobot_annotations.json"
    labels = json.loads(sidecar.read_text())
    labels["episodes"]["0"]["atoms"][0]["content"] = "completed before restart"
    atomic_json(sidecar, labels)
    snapshot = workflow.root / "meta/annotation_predictions/episode-0.json"
    atomic_json(snapshot, {"episodes": {"0": labels["episodes"]["0"]}})
    original_prediction = snapshot.read_bytes()
    run["episodes"]["0"]["generation_status"] = "generated"
    run["episodes"]["1"]["generation_status"] = "failed"
    interrupted_job = "b" * 32
    run["current_job_id"] = interrupted_job
    store.save(run, run["revision"])
    atomic_json(
        app.EXPORT_ROOT / f"jobs/{interrupted_job}.json",
        {
            "job_id": interrupted_job,
            "status": "running",
            "completed_episodes": [0],
        },
    )
    recover_jobs(app.EXPORT_ROOT)
    assert app.get_annotation_job(interrupted_job)["status"] == "interrupted"
    queued = app.create_annotation_job(
        app.GenerationRequest(local_path=str(workflow.root), resume_unfinished=True)
    )
    assert app.get_annotation_job(queued["job_id"])["status"] == "completed"
    assert [call["episode"] for call in workflow.calls] == [1, 2]
    current = store.read(run["run_id"])
    assert current["episodes"]["0"]["generation_status"] == "generated"
    current_root = Path(current["root"])
    restored_labels = json.loads(
        (current_root / "meta/lerobot_annotations.json").read_text()
    )
    assert (
        restored_labels["episodes"]["0"]["atoms"][0]["content"]
        == "completed before restart"
    )
    assert (
        current_root / snapshot.relative_to(workflow.root)
    ).read_bytes() == original_prediction
    assert current["current_job_id"] == queued["job_id"] != interrupted_job
    assert app.get_annotation_job(interrupted_job)["status"] == "interrupted"
    with pytest.raises(HTTPException) as error:
        app.create_annotation_job(
            app.GenerationRequest(local_path=current["root"], resume_unfinished=True)
        )
    assert error.value.status_code in {409, 422}
    assert [call["episode"] for call in workflow.calls] == [1, 2]


def test_revision_change_between_capture_and_admission_rejects_stale_generation(
    workflow, monkeypatch
):
    original = workflow.store.read(workflow.run["run_id"])

    def interleaving_start(operation, on_queued=None):
        newer = workflow.store.read(original["run_id"])
        newer["task_prompt"] = "newer human task"
        workflow.store.save(newer, newer["revision"])
        on_queued("c" * 32)
        pytest.fail("Stale captured generation request was admitted")

    monkeypatch.setattr(app, "_start_annotation_job", interleaving_start)
    with pytest.raises(HTTPException) as error:
        app.create_annotation_job(
            app.GenerationRequest(
                local_path=str(workflow.root), task_prompt="stale task"
            )
        )
    assert error.value.status_code == 409
    current = workflow.store.read(original["run_id"])
    assert current["task_prompt"] == "newer human task"
    assert current["revision"] == original["revision"] + 1
    assert current["current_job_id"] is None
    assert workflow.calls == []


@pytest.mark.parametrize("change_after_review", [False, True])
def test_fewshot_review_includes_current_exclusions(workflow, change_after_review):
    import annotation_history as history

    atoms = json.loads((workflow.root / "meta/lerobot_annotations.json").read_text())[
        "episodes"
    ]["0"]["atoms"]
    intervals = [{"start_frame": 1, "end_frame": 3}]
    run = workflow.store.read(workflow.run["run_id"])
    run["episodes"]["0"]["excluded_intervals"] = intervals
    workflow.store.save(run, run["revision"])
    history.save_review(
        workflow.root,
        0,
        atoms,
        True,
        history.annotation_hash(atoms),
        excluded_intervals=intervals,
    )
    if change_after_review:
        run = workflow.store.read(workflow.run["run_id"])
        run["episodes"]["0"]["excluded_intervals"] = [
            {"start_frame": 1, "end_frame": 4}
        ]
        workflow.store.save(run, run["revision"])
    request = app.GenerationRequest(
        local_path=str(workflow.root), example_episode_indices=[0], episode_indices=[1]
    )
    if change_after_review:
        with pytest.raises(HTTPException, match="explicitly marked reviewed"):
            app.create_annotation_job(request)
        assert workflow.calls == []
    else:
        started = app.create_annotation_job(request)
        assert app.get_annotation_job(started["job_id"])["status"] == "completed"
        assert [call["episode"] for call in workflow.calls] == [1]
