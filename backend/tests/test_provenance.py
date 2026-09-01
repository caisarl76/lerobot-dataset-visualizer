from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from curation.validation import (
    FinalConsistencyError,
    build_curation_provenance,
    seal_staging_tree,
    validate_final_consistency,
    write_checksum_manifest,
    write_provenance,
)
import pytest

APPROVED_DESIGN = Path(
    "/home/jihun/work/GR00T-WholeBodyControl/docs/superpowers/specs/2026-08-18-pnp-trash-cosmos-curation-design.md"
)


def _artifact(root: Path, relative: str, contents: bytes, kind: str) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return {
        "kind": kind,
        "path": relative,
        "bytes": len(contents),
        "media_type": "application/json" if relative.endswith(".json") else "image/png",
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def _document(root: Path) -> dict[str, object]:
    structural = (
        '{"approval_snapshot_sha256":"'
        + "b" * 64
        + '","passed":true,"schema_version":1,"source_manifest_sha256":"'
        + "a" * 64
        + '"}\n'
    ).encode()
    artifacts = [
        _artifact(
            root,
            "meta/curation_artifacts/structural-report.json",
            structural,
            "structural_report",
        ),
        _artifact(root, "meta/curation_artifacts/gr00t-stats-report.json", b"{}\n", "gr00t_stats_report"),
        _artifact(root, "meta/curation_artifacts/gr00t-loader-report.json", b"{}\n", "gr00t_loader_report"),
        _artifact(
            root,
            "meta/curation_artifacts/contact_sheets/final-episode-0.png",
            b"final-sheet",
            "final_contact_sheet",
        ),
        _artifact(
            root,
            "meta/curation_artifacts/contact_sheets/proposal-episode-0.png",
            b"proposal-sheet",
            "proposal_contact_sheet",
        ),
    ]
    return build_curation_provenance(
        source={
            "dataset_alias": "local/pnp_trash",
            "manifest_sha256": "a" * 64,
            "file_count": 1,
            "original_tasks": [{"task_index": 0, "task": "original"}],
        },
        approval={
            "snapshot_sha256": "b" * 64,
            "prompt_template_version": "pnp-trash-prompts-v1",
            "prompt_template_sha256": "c" * 64,
            "templates": [{"step": step, "template": f"template {step}"} for step in range(1, 8)],
        },
        software={
            "exporter_version": "1",
            "repositories": [
                {
                    "name": "lerobot-dataset-visualizer",
                    "commit": "d" * 40,
                    "dirty": False,
                    "tracked_diff_sha256": None,
                    "untracked_files": [],
                },
                {
                    "name": "Isaac-GR00T",
                    "commit": "e" * 40,
                    "dirty": True,
                    "tracked_diff_sha256": "f" * 64,
                    "untracked_files": [{"path": "x", "bytes": 1, "sha256": "0" * 64}],
                },
            ],
        },
        cosmos={
            "model": "cosmos",
            "endpoint_identity": "h100",
            "contract_version": "pnp-trash-cosmos-v2",
            "sampling": {"target_fps": 2, "resize_max_long_edge": 640, "jpeg_quality": 85},
            "limits": {"max_duration_s": 120, "max_frames": 240, "max_payload_bytes": 67108864},
            "job_ids": [],
            "attempt_ids": [],
            "workspace_artifacts": [],
        },
        export={
            "export_id": "11111111-1111-1111-1111-111111111111",
            "created_at_utc": "2026-08-27T00:00:00Z",
            "source_to_output": [{"source_episode_index": 0, "output_episode_index": 0}],
        },
        episodes={
            "kept": [
                {
                    "source_episode_index": 0,
                    "output_episode_index": 0,
                    "object": "can",
                    "hand": "left",
                    "turn": "right",
                    "transition_frames": [1, 2, 3, 4, 5, 6],
                    "reviewer": "human",
                    "revision": 1,
                    "approved_at": "2026-08-27T00:00:00Z",
                }
            ],
            "rejected": [],
        },
        tasks=[{"task_index": index, "ordering_step": index + 1, "prompt": f"p{index}"} for index in range(7)],
        artifacts=artifacts,
    )


def _expectations(document: dict[str, object]) -> dict[str, object]:
    structural = next(artifact for artifact in document["artifacts"] if artifact["kind"] == "structural_report")
    return {
        "approval_snapshot_sha256": document["approval"]["snapshot_sha256"],
        "source_manifest_sha256": document["source"]["manifest_sha256"],
        "prompt_template_version": document["approval"]["prompt_template_version"],
        "prompt_template_sha256": document["approval"]["prompt_template_sha256"],
        "templates": document["approval"]["templates"],
        "source_to_output": document["export"]["source_to_output"],
        "episodes": document["episodes"],
        "tasks": document["tasks"],
        "structural_report_sha256": structural["sha256"],
        "artifacts": document["artifacts"],
    }


def _structural_result(document: dict[str, object]) -> dict[str, object]:
    return {
        "passed": True,
        "approval_snapshot_sha256": document["approval"]["snapshot_sha256"],
        "source_manifest_sha256": document["source"]["manifest_sha256"],
    }


def test_approved_design_provenance_version_matches_the_production_contract(tmp_path: Path) -> None:
    design = APPROVED_DESIGN.read_text()
    section = design.split("The output contains `meta/curation_provenance.json`", 1)[1]
    example = json.loads(section.split("```json", 1)[1].split("```", 1)[0])
    production = _document(tmp_path)

    assert example["schema_version"] == production["schema_version"] == 2


def test_closed_provenance_schema_excludes_secrets_and_raw_reasoning(tmp_path: Path) -> None:
    document = _document(tmp_path)
    written = write_provenance(tmp_path, document)

    assert written["path"] == "meta/curation_provenance.json"
    parsed = json.loads((tmp_path / written["path"]).read_text())
    assert set(parsed) == {
        "schema_version",
        "source",
        "approval",
        "software",
        "cosmos",
        "export",
        "episodes",
        "tasks",
        "artifacts",
    }
    encoded = json.dumps(parsed).lower()
    assert "api_key" not in encoded
    assert "authorization" not in encoded
    assert "reasoning" not in encoded

    contaminated = dict(document)
    contaminated["raw_reasoning"] = "secret"
    with pytest.raises(ValueError, match="closed schema"):
        write_provenance(tmp_path / "other", contaminated)


def test_checksum_seal_and_final_gate_reject_unlisted_symlink_and_changed_bytes(tmp_path: Path) -> None:
    document = _document(tmp_path)
    write_provenance(tmp_path, document)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/episode.parquet").write_bytes(b"parquet")
    checksum = write_checksum_manifest(tmp_path)
    assert checksum.name == "curation_checksums.sha256"
    seal_staging_tree(tmp_path)

    report = validate_final_consistency(
        staging_path=tmp_path,
        provenance=document,
        source_roots=[],
        stats_report_validator=lambda root: None,
        structural_validator=lambda: _structural_result(document),
        expectations=_expectations(document),
    )
    assert report["passed"] is True
    assert report["file_count"] == 7

    (tmp_path / "data/episode.parquet").chmod(0o644)
    (tmp_path / "data/episode.parquet").write_bytes(b"changed")
    with pytest.raises(FinalConsistencyError, match="checksum"):
        validate_final_consistency(
            staging_path=tmp_path,
            provenance=document,
            source_roots=[],
            stats_report_validator=lambda root: None,
            structural_validator=lambda: _structural_result(document),
            expectations=_expectations(document),
        )


@pytest.mark.parametrize(
    "link",
    [
        "approval",
        "source",
        "template",
        "decisions",
        "source_to_output",
        "tasks",
        "structural_report",
        "artifact_media_type",
    ],
)
def test_final_gate_rejects_each_mutated_authenticated_link_even_with_fresh_checksum(
    tmp_path: Path, link: str
) -> None:
    document = _document(tmp_path)
    expectations = _expectations(document)
    mutated = deepcopy(document)
    if link == "approval":
        mutated["approval"]["snapshot_sha256"] = "0" * 64
    elif link == "source":
        mutated["source"]["manifest_sha256"] = "0" * 64
    elif link == "template":
        mutated["approval"]["prompt_template_sha256"] = "0" * 64
    elif link == "decisions":
        mutated["episodes"]["kept"][0]["object"] = "fabricated"
    elif link == "source_to_output":
        mutated["export"]["source_to_output"][0]["output_episode_index"] = 9
    elif link == "tasks":
        mutated["tasks"][0]["prompt"] = "fabricated"
    elif link == "structural_report":
        structural = next(artifact for artifact in mutated["artifacts"] if artifact["kind"] == "structural_report")
        structural["sha256"] = "0" * 64
    else:
        mutated["artifacts"][0]["media_type"] = "text/plain"
    write_provenance(tmp_path, mutated)
    write_checksum_manifest(tmp_path)
    seal_staging_tree(tmp_path)

    with pytest.raises((FinalConsistencyError, ValueError)):
        validate_final_consistency(
            staging_path=tmp_path,
            provenance=mutated,
            source_roots=[],
            stats_report_validator=lambda root: None,
            structural_validator=lambda: _structural_result(document),
            expectations=expectations,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing_stats",
        "missing_loader",
        "replaced_stats",
        "replaced_loader",
        "missing_proposal_sheet",
        "replaced_proposal_sheet",
        "missing_final_sheet",
        "replaced_final_sheet",
    ],
)
def test_final_gate_authenticates_the_exact_artifact_set_with_fresh_checksum(
    tmp_path: Path, mutation: str
) -> None:
    document = _document(tmp_path)
    expectations = _expectations(document)
    mutated = deepcopy(document)
    kind_by_mutation = {
        "missing_stats": "gr00t_stats_report",
        "replaced_stats": "gr00t_stats_report",
        "missing_loader": "gr00t_loader_report",
        "replaced_loader": "gr00t_loader_report",
        "missing_proposal_sheet": "proposal_contact_sheet",
        "replaced_proposal_sheet": "proposal_contact_sheet",
        "missing_final_sheet": "final_contact_sheet",
        "replaced_final_sheet": "final_contact_sheet",
    }
    if mutation == "extra":
        mutated["artifacts"].append(
            _artifact(
                tmp_path,
                "meta/curation_artifacts/contact_sheets/extra.png",
                b"extra",
                "proposal_contact_sheet",
            )
        )
        mutated["artifacts"].sort(key=lambda row: row["path"].encode())
    else:
        kind = kind_by_mutation[mutation]
        artifact = next(row for row in mutated["artifacts"] if row["kind"] == kind)
        if mutation.startswith("missing_"):
            mutated["artifacts"].remove(artifact)
        else:
            contents = f"changed-{mutation}".encode()
            (tmp_path / artifact["path"]).write_bytes(contents)
            artifact["bytes"] = len(contents)
            artifact["sha256"] = hashlib.sha256(contents).hexdigest()
    if mutation in {"missing_stats", "missing_loader"}:
        provenance_path = tmp_path / "meta/curation_provenance.json"
        provenance_path.write_text(
            json.dumps(mutated, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        )
    else:
        write_provenance(tmp_path, mutated)
    write_checksum_manifest(tmp_path)
    seal_staging_tree(tmp_path)

    with pytest.raises(FinalConsistencyError):
        validate_final_consistency(
            staging_path=tmp_path,
            provenance=mutated,
            source_roots=[],
            stats_report_validator=lambda root: None,
            structural_validator=lambda: _structural_result(document),
            expectations=expectations,
        )


def test_seal_makes_all_files_and_directories_read_only(tmp_path: Path) -> None:
    (tmp_path / "a/b").mkdir(parents=True)
    (tmp_path / "a/b/file").write_bytes(b"x")
    seal_staging_tree(tmp_path)
    assert (tmp_path / "a/b/file").stat().st_mode & 0o777 == 0o444
    assert (tmp_path / "a/b").stat().st_mode & 0o777 == 0o555
    assert (tmp_path / "a").stat().st_mode & 0o777 == 0o555
    assert tmp_path.stat().st_mode & 0o777 == 0o555
