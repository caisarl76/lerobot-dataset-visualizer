from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from curation.exporter import StagingExporter
from curation.validation import (
    run_gr00t_loader_validation,
    run_gr00t_stats_validation,
    validate_gr00t_loader_outer_report,
    validate_gr00t_loader_report,
    validate_gr00t_stats_report,
)
import pytest
from test_exporter import _rich_case


class RecordingRunner:
    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), **kwargs})
        return self.result


_STAT_METRICS = ("mean", "std", "min", "max", "q01", "q99")


def _write_stats_contract(staging: Path) -> None:
    (staging / "meta").mkdir(exist_ok=True)
    info = {
        "features": {
            "observation.state": {"dtype": "float64", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "uint32", "shape": [1]},
            "observation.images.ego_view": {"dtype": "video", "shape": [2, 2, 3]},
        }
    }
    (staging / "meta/info.json").write_text(json.dumps(info))
    stats = {
        feature: {metric: [float(index + 1)] * width for metric in _STAT_METRICS}
        for index, (feature, width) in enumerate((("action", 2), ("observation.state", 2), ("timestamp", 1)))
    }
    (staging / "meta/stats.json").write_text(json.dumps(stats))
    (staging / "meta/relative_stats.json").write_text("{}\n")


def _repo_state() -> dict[str, object]:
    return {
        "name": "Isaac-GR00T",
        "commit": "a" * 40,
        "dirty": False,
        "tracked_diff_sha256": None,
        "untracked_files": [],
    }


def test_gr00t_stats_uses_exact_injected_command_and_records_bounded_report(tmp_path: Path) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = (tmp_path / "staging").resolve()
    staging.mkdir()
    _write_stats_contract(staging)
    runner = RecordingRunner(subprocess.CompletedProcess([], 0, stdout="stats ok\n", stderr=""))

    report = run_gr00t_stats_validation(
        staging_path=staging,
        isaac_root=root,
        runner=runner,
        repository_probe=lambda path, name: _repo_state(),
        environment={"PATH": "/safe/bin", "COSMOS_API_KEY": "must-not-leak", "STAGING_PATH": "wrong"},
    )

    expected = [
        str(root / ".venv/bin/python"),
        "gr00t/data/stats.py",
        "--dataset-path",
        str(staging),
        "--embodiment-tag",
        "UNITREE_G1_SONIC",
        "--modality-config-path",
        "gr00t/configs/data/embodiment_configs.py",
    ]
    assert runner.calls[0]["argv"] == expected
    assert runner.calls[0]["cwd"] == root
    assert runner.calls[0]["env"]["STAGING_PATH"] == str(staging)
    assert "COSMOS_API_KEY" not in report["command"]["environment"]
    assert report["command"]["argv"] == expected
    assert report["repository"]["commit"] == "a" * 40
    assert report["passed"] is True
    assert report["exit_code"] == 0
    assert report["traceback"] is None
    assert [row["path"] for row in report["outputs"]] == [
        "meta/relative_stats.json",
        "meta/stats.json",
    ]
    validate_gr00t_stats_report(report, staging_path=staging, isaac_root=root)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_stats",
        "missing_relative",
        "symlink_stats",
        "symlink_relative",
        "hardlink",
        "empty",
        "malformed",
        "incomplete_feature",
        "wrong_relative",
    ],
)
def test_gr00t_stats_success_requires_exact_authenticated_output_files(tmp_path: Path, corruption: str) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = (tmp_path / "staging").resolve()
    staging.mkdir()
    _write_stats_contract(staging)
    stats = staging / "meta/stats.json"
    relative = staging / "meta/relative_stats.json"
    if corruption == "missing_stats":
        stats.unlink()
    elif corruption == "missing_relative":
        relative.unlink()
    elif corruption in {"symlink_stats", "symlink_relative"}:
        target = staging / "target.json"
        target.write_text("{}\n")
        path = stats if corruption == "symlink_stats" else relative
        path.unlink()
        path.symlink_to(target)
    elif corruption == "hardlink":
        relative.unlink()
        os.link(stats, relative)
    elif corruption == "empty":
        stats.write_bytes(b"")
    elif corruption == "malformed":
        stats.write_text("{not-json")
    elif corruption == "incomplete_feature":
        document = json.loads(stats.read_text())
        del document["action"]["q99"]
        stats.write_text(json.dumps(document))
    else:
        relative.write_text('{"action":{}}\n')
    runner = RecordingRunner(subprocess.CompletedProcess([], 0, stdout="stats ok\n", stderr=""))

    report = run_gr00t_stats_validation(
        staging_path=staging,
        isaac_root=root,
        runner=runner,
        repository_probe=lambda path, name: _repo_state(),
        environment={"PATH": "/safe/bin"},
    )

    assert report["passed"] is False
    assert report["outputs"] == []
    assert report["exception"] is not None


@pytest.mark.parametrize("corruption", ["delete", "corrupt"])
def test_stats_report_orphan_revalidation_authenticates_current_output_bytes(
    tmp_path: Path, corruption: str
) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = (tmp_path / "staging").resolve()
    staging.mkdir()
    _write_stats_contract(staging)
    report = run_gr00t_stats_validation(
        staging_path=staging,
        isaac_root=root,
        runner=RecordingRunner(subprocess.CompletedProcess([], 0, stdout="ok\n", stderr="")),
        repository_probe=lambda path, name: _repo_state(),
        environment={"PATH": "/safe/bin"},
    )
    stats = staging / "meta/stats.json"
    if corruption == "delete":
        stats.unlink()
    else:
        stats.write_text("{}\n")

    with pytest.raises(ValueError):
        validate_gr00t_stats_report(report, staging_path=staging, isaac_root=root)


def test_gr00t_loader_executes_generated_acceptance_script_and_records_episode_results(tmp_path: Path) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = (tmp_path / "staging").resolve()
    staging.mkdir()
    (staging / "meta").mkdir()
    (staging / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 7, "tasks": [f"p{i}" for i in range(7)]}) + "\n"
    )
    loader_result = {
        "schema_version": 1,
        "episodes": [
            {
                "episode_index": 0,
                "row_count": 7,
                "runs": [f"p{i}" for i in range(7)],
                "passed": True,
                "exception": None,
                "traceback": None,
            }
        ],
        "passed": True,
    }
    runner = RecordingRunner(
        subprocess.CompletedProcess([], 0, stdout=json.dumps(loader_result) + "\n", stderr="")
    )

    report = run_gr00t_loader_validation(
        staging_path=staging,
        isaac_root=root,
        runner=runner,
        repository_probe=lambda path, name: _repo_state(),
        environment={"PATH": "/safe/bin"},
    )

    call = runner.calls[0]
    assert call["argv"][:2] == [str(root / ".venv/bin/python"), "-c"]
    script = call["argv"][2]
    assert 'MODALITY_CONFIGS["unitree_g1_sonic"]' in script
    assert "LeRobotEpisodeLoader" in script
    assert call["argv"][3] == str(staging)
    assert report["passed"] is True
    assert report["episodes"][0]["runs"] == [f"p{i}" for i in range(7)]
    validate_gr00t_loader_outer_report(report, staging_path=staging, isaac_root=root)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "argv",
        "cwd",
        "executable",
        "environment",
        "passed",
        "exit",
        "exception",
        "repository",
        "outputs",
    ],
)
def test_gr00t_outer_report_is_closed_and_bound_to_the_exact_invocation(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = (tmp_path / "staging").resolve()
    staging.mkdir()
    _write_stats_contract(staging)
    runner = RecordingRunner(subprocess.CompletedProcess([], 0, stdout="stats ok\n", stderr=""))
    report = run_gr00t_stats_validation(
        staging_path=staging,
        isaac_root=root,
        runner=runner,
        repository_probe=lambda path, name: _repo_state(),
        environment={"PATH": "/safe/bin"},
    )
    if mutation == "extra":
        report["fabricated"] = True
    elif mutation in {"argv", "cwd", "executable", "environment"}:
        report["command"][mutation] = ["wrong"] if mutation == "argv" else "wrong"
    elif mutation == "passed":
        report["passed"] = False
    elif mutation == "exit":
        report["exit_code"] = 9
    elif mutation == "exception":
        report["exception"] = "fabricated"
    elif mutation == "outputs":
        report["outputs"][0]["sha256"] = "0" * 64
    else:
        report["repository"]["commit"] = "invalid"

    with pytest.raises(ValueError):
        validate_gr00t_stats_report(report, staging_path=staging, isaac_root=root)


def test_gr00t_failure_is_a_report_not_an_exception(tmp_path: Path) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    runner = RecordingRunner(subprocess.CompletedProcess([], 3, stdout="", stderr="failed"))

    report = run_gr00t_stats_validation(
        staging_path=staging,
        isaac_root=root,
        runner=runner,
        repository_probe=lambda path, name: _repo_state(),
    )

    assert report["passed"] is False
    assert report["exit_code"] == 3
    assert report["stderr"] == "failed"


def test_validation_child_environment_is_allowlisted_and_all_secret_channels_are_redacted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    secret = "sentinel-super-secret"
    runner = RecordingRunner(subprocess.CompletedProcess([], 2, stdout=f"out:{secret}", stderr=f"err:{secret}"))

    report = run_gr00t_stats_validation(
        staging_path=staging,
        isaac_root=root,
        runner=runner,
        repository_probe=lambda path, name: _repo_state(),
        environment={
            "PATH": "/safe/bin",
            "CURATION_BEARER_TOKEN": secret,
            "COSMOS_API_KEY": secret,
            "UNRELATED": secret,
        },
        secret_values=[secret],
    )

    assert runner.calls[0]["env"] == {"PATH": "/safe/bin", "STAGING_PATH": str(staging.resolve())}
    assert secret not in json.dumps(report)
    assert "[REDACTED]" in report["stdout"]
    assert "[REDACTED]" in report["stderr"]


def test_validation_runner_exception_and_traceback_are_redacted(tmp_path: Path) -> None:
    root = tmp_path / "Isaac-GR00T"
    root.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    secret = "sentinel-exception-secret"

    def failing_runner(*args, **kwargs):
        assert secret not in kwargs["env"].values()
        raise RuntimeError(f"runner failed with {secret}")

    report = run_gr00t_stats_validation(
        staging_path=staging,
        isaac_root=root,
        runner=failing_runner,
        repository_probe=lambda path, name: {**_repo_state(), "tracked_diff_sha256": secret},
        environment={"PATH": "/safe/bin", "CURATION_TOKEN": secret},
        secret_values=[secret],
    )

    encoded = json.dumps(report)
    assert secret not in encoded
    assert "[REDACTED]" in report["exception"]
    assert "[REDACTED]" in report["traceback"]
    assert report["repository"]["tracked_diff_sha256"] == "[REDACTED]"


@pytest.mark.parametrize(
    "mutation",
    ["empty", "omitted", "duplicate", "reordered", "failed", "wrong_rows", "non_seven", "extra"],
)
def test_loader_report_rejects_nonexact_episode_evidence(tmp_path: Path, mutation: str) -> None:
    _, _, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    StagingExporter(database=database, source_registry=registry).run(created["export_id"])
    staging = Path(created["staging_path"])
    episode_rows = [json.loads(line) for line in (staging / "meta/episodes.jsonl").read_text().splitlines()]
    episodes = [
        {
            "episode_index": row["episode_index"],
            "row_count": row["length"],
            "runs": row["tasks"],
            "passed": True,
            "exception": None,
            "traceback": None,
        }
        for row in episode_rows
    ]
    report = {"schema_version": 1, "passed": True, "episodes": episodes}
    if mutation == "empty":
        report["episodes"] = []
    elif mutation == "omitted":
        report["episodes"] = episodes[:-1]
    elif mutation == "duplicate":
        report["episodes"] = [episodes[0], episodes[0]]
    elif mutation == "reordered":
        report["episodes"] = list(reversed(episodes))
    elif mutation == "failed":
        episodes[0] = dict(episodes[0], passed=False, exception="bad", traceback="trace")
    elif mutation == "wrong_rows":
        episodes[0] = dict(episodes[0], row_count=7)
    elif mutation == "non_seven":
        episodes[0] = dict(episodes[0], runs=episodes[0]["runs"][:-1])
    else:
        episodes[0] = dict(episodes[0], fabricated=True)

    with pytest.raises(ValueError):
        validate_gr00t_loader_report(report, staging_path=staging)
