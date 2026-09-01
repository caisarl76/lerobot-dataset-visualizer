from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

from curation.db import canonical_json
from curation.exporter import (
    ExportError,
    ValidatedDatasetExporter,
    _copy_independent_regular_file,
    _exclusive_export_execution,
)
from curation.models import ExportState
from curation.publication import (
    PublicationError,
    pin_publication_source,
    preflight_rename_noreplace,
    publish_no_clobber,
    reconcile_publication_paths,
    rename_noreplace,
)
from curation.validation import (
    _LOADER_ACCEPTANCE_SCRIPT,
    ArtifactInstallConflict,
    _hash_file,
    _regular_files,
    install_canonical_file,
    seal_staging_tree,
)
import pytest
from test_exporter import UNKNOWN_XLSX_BYTES, UNKNOWN_XLSX_NAME, _rich_case


def _race_destination(final: str, barrier: multiprocessing.Barrier, sentinel: bytes | None) -> None:
    barrier.wait(10)
    path = Path(final)
    path.mkdir()
    if sentinel is not None:
        (path / "sentinel").write_bytes(sentinel)
    barrier.wait(10)


def test_atomic_publish_renames_then_fsyncs_parent_before_commit(tmp_path: Path) -> None:
    staging = tmp_path / ".staging"
    final = tmp_path / "final"
    staging.mkdir()
    (staging / "sentinel").write_bytes(b"complete")
    events: list[str] = []

    identity = pin_publication_source(staging, final)
    publish_no_clobber(
        staging,
        final,
        expected_identity=identity,
        parent_fsync=lambda path: events.append(f"fsync:{path}"),
        commit_published=lambda: events.append("published"),
    )

    assert not staging.exists()
    assert (final / "sentinel").read_bytes() == b"complete"
    assert events == [f"fsync:{tmp_path}", "published"]


def test_crash_after_rename_is_reconciled_by_final_gate_then_parent_fsync(tmp_path: Path) -> None:
    staging = tmp_path / ".staging"
    final = tmp_path / "final"
    staging.mkdir()
    (staging / "sentinel").write_bytes(b"complete")

    identity = pin_publication_source(staging, final)
    with pytest.raises(SystemExit):
        publish_no_clobber(
            staging,
            final,
            expected_identity=identity,
            after_rename=lambda: (_ for _ in ()).throw(SystemExit(77)),
        )
    assert not staging.exists()
    assert final.is_dir()

    events: list[str] = []
    result = reconcile_publication_paths(
        staging,
        final,
        final_gate=lambda path: events.append(f"gate:{path}"),
        parent_fsync=lambda path: events.append(f"fsync:{path}"),
        commit_published=lambda: events.append("published"),
        return_to_validated=lambda: events.append("must-not-return"),
        fail_operator=lambda code: events.append(code),
    )
    assert result == "published"
    assert events == [f"gate:{final}", f"fsync:{tmp_path}", "published"]


@pytest.mark.parametrize("sentinel", [None, b"competitor"])
def test_no_clobber_process_race_preserves_staging_and_competitor(tmp_path: Path, sentinel: bytes | None) -> None:
    staging = tmp_path / ".staging"
    final = tmp_path / "final"
    staging.mkdir()
    (staging / "source").write_bytes(b"staging")
    barrier = multiprocessing.Barrier(2)
    process = multiprocessing.Process(target=_race_destination, args=(str(final), barrier, sentinel))
    process.start()
    identity = pin_publication_source(staging, final)
    with pytest.raises(PublicationError) as raised:
        publish_no_clobber(
            staging,
            final,
            expected_identity=identity,
            before_rename=lambda: (barrier.wait(10), barrier.wait(10)),
        )
    process.join(10)

    assert raised.value.code == "publish_destination_exists"
    assert (staging / "source").read_bytes() == b"staging"
    assert final.is_dir()
    if sentinel is not None:
        assert (final / "sentinel").read_bytes() == sentinel


def test_rename_noreplace_maps_unsupported_and_existing_errors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    with pytest.raises(PublicationError) as existing:
        rename_noreplace(source, destination)
    assert existing.value.code == "publish_destination_exists"

    def unsupported(*args):
        raise OSError(errno.ENOSYS, "unsupported")

    with pytest.raises(PublicationError) as unavailable:
        rename_noreplace(source, tmp_path / "new", syscall=unsupported)
    assert unavailable.value.code == "publish_noreplace_unsupported"


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_pinned_publication_source_swap_is_never_committed(tmp_path: Path, replacement: str) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    held = tmp_path / "held"
    victim = tmp_path / "victim"
    staging.mkdir()
    victim.mkdir()
    (staging / "expected").write_bytes(b"expected")
    identity = pin_publication_source(staging, final)
    committed: list[bool] = []

    def swap() -> None:
        os.rename(staging, held)
        if replacement == "symlink":
            staging.symlink_to(victim, target_is_directory=True)
        else:
            staging.mkdir()
            (staging / "different").write_bytes(b"different")

    with pytest.raises(PublicationError) as raised:
        publish_no_clobber(
            staging,
            final,
            expected_identity=identity,
            before_rename=swap,
            commit_published=lambda: committed.append(True),
        )
    assert raised.value.code == "publish_source_identity_changed"
    assert committed == []


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_post_rename_identity_swap_is_never_committed(tmp_path: Path, replacement: str) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    held = tmp_path / "held"
    victim = tmp_path / "victim"
    staging.mkdir()
    victim.mkdir()
    identity = pin_publication_source(staging, final)
    committed: list[bool] = []

    def swap_after_rename() -> None:
        os.rename(final, held)
        if replacement == "symlink":
            final.symlink_to(victim, target_is_directory=True)
        else:
            final.mkdir()

    with pytest.raises(PublicationError) as raised:
        publish_no_clobber(
            staging,
            final,
            expected_identity=identity,
            after_rename=swap_after_rename,
            commit_published=lambda: committed.append(True),
        )
    assert raised.value.code == "publish_source_identity_changed"
    assert committed == []


@pytest.mark.parametrize("swap_at", ["final_gate", "parent_fsync"])
def test_reconciliation_rechecks_identity_after_each_external_hook(tmp_path: Path, swap_at: str) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    held = tmp_path / "held"
    staging.mkdir()
    identity = pin_publication_source(staging, final)
    rename_noreplace(staging, final)
    committed: list[bool] = []

    def swap() -> None:
        os.rename(final, held)
        final.mkdir()

    def final_gate(path: Path) -> None:
        if swap_at == "final_gate":
            swap()

    def parent_fsync(path: Path) -> None:
        if swap_at == "parent_fsync":
            swap()

    with pytest.raises(PublicationError) as raised:
        reconcile_publication_paths(
            staging,
            final,
            final_gate=final_gate,
            parent_fsync=parent_fsync,
            commit_published=lambda: committed.append(True),
            return_to_validated=lambda: None,
            fail_operator=lambda code: None,
            expected_identity=identity,
        )
    assert raised.value.code == "publish_source_identity_changed"
    assert committed == []


def test_renameat2_preflight_fails_without_touching_unrelated_parent_entries(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep")

    def unsupported(*args):
        raise OSError(errno.ENOSYS, "unsupported")

    with pytest.raises(PublicationError) as raised:
        preflight_rename_noreplace(tmp_path, syscall=unsupported)
    assert raised.value.code == "publish_noreplace_unsupported"
    assert sentinel.read_bytes() == b"keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sentinel"]


def test_canonical_artifact_install_reconciles_crash_and_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    contents = b'{"passed":true}\n'
    with pytest.raises(SystemExit):
        install_canonical_file(
            path,
            contents,
            after_install=lambda: (_ for _ in ()).throw(SystemExit(91)),
        )
    assert path.read_bytes() == contents
    assert install_canonical_file(path, contents) == "existing"
    with pytest.raises(ArtifactInstallConflict):
        install_canonical_file(path, b"truncated")

    temporary = tmp_path / ".other.json.installing"
    temporary.write_bytes(b"partial")
    with pytest.raises(ArtifactInstallConflict):
        install_canonical_file(tmp_path / "other.json", contents)


def test_partial_contact_sheet_copy_set_resumes_exactly_and_rejects_mismatch(tmp_path: Path) -> None:
    sources = [tmp_path / f"source-{index}.png" for index in range(2)]
    destinations = [tmp_path / "staging" / f"sheet-{index}.png" for index in range(2)]
    for index, source in enumerate(sources):
        source.write_bytes(f"png-{index}".encode())

    _copy_independent_regular_file(sources[0], destinations[0])
    _copy_independent_regular_file(sources[0], destinations[0])
    _copy_independent_regular_file(sources[1], destinations[1])
    assert [path.read_bytes() for path in destinations] == [b"png-0", b"png-1"]
    sources[0].write_bytes(b"changed")
    with pytest.raises(ArtifactInstallConflict):
        _copy_independent_regular_file(sources[0], destinations[0])


def test_export_executor_does_not_relabel_body_oserror(tmp_path: Path) -> None:
    export_id = "11111111-1111-1111-1111-111111111111"
    with pytest.raises(OSError, match="body failure"):
        with _exclusive_export_execution(tmp_path, export_id):
            raise OSError("body failure")


def test_reconcile_all_path_states_and_parent_fsync_retry(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    staging.mkdir()
    events: list[str] = []
    assert (
        reconcile_publication_paths(
            staging,
            final,
            final_gate=lambda path: events.append(f"gate:{path}"),
            parent_fsync=lambda path: events.append(f"fsync:{path}"),
            commit_published=lambda: events.append("published"),
            return_to_validated=lambda: events.append("validated"),
            fail_operator=lambda code: events.append(code),
        )
        == "final_consistency_validated"
    )
    assert events == ["validated"]

    staging.rmdir()
    final.mkdir()
    events.clear()
    assert (
        reconcile_publication_paths(
            staging,
            final,
            final_gate=lambda path: events.append(f"gate:{path}"),
            parent_fsync=lambda path: events.append(f"fsync:{path}"),
            commit_published=lambda: events.append("published"),
            return_to_validated=lambda: events.append("validated"),
            fail_operator=lambda code: events.append(code),
        )
        == "published"
    )
    assert events == [f"gate:{final}", f"fsync:{tmp_path}", "published"]

    def fail_fsync(path: Path) -> None:
        raise OSError("injected")

    with pytest.raises(PublicationError) as raised:
        reconcile_publication_paths(
            staging,
            final,
            final_gate=lambda path: None,
            parent_fsync=fail_fsync,
            commit_published=lambda: events.append("must-not-commit"),
            return_to_validated=lambda: None,
            fail_operator=lambda code: None,
        )
    assert raised.value.code == "publish_parent_fsync_failed"

    staging.mkdir()
    failures: list[str] = []
    assert (
        reconcile_publication_paths(
            staging,
            final,
            final_gate=lambda path: None,
            parent_fsync=lambda path: None,
            commit_published=lambda: None,
            return_to_validated=lambda: None,
            fail_operator=failures.append,
        )
        == "failed"
    )
    assert failures == ["publish_paths_both_present"]

    for child in final.iterdir():
        child.unlink()
    final.rmdir()
    staging.rmdir()
    failures.clear()
    assert (
        reconcile_publication_paths(
            staging,
            final,
            final_gate=lambda path: None,
            parent_fsync=lambda path: None,
            commit_published=lambda: None,
            return_to_validated=lambda: None,
            fail_operator=failures.append,
        )
        == "failed"
    )
    assert failures == ["publish_paths_both_absent"]


def test_export_artifact_and_state_advance_are_one_transaction(tmp_path: Path) -> None:
    _, workspace, database, _, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    database.claim_export_build(export_id=created["export_id"])

    artifact = database.advance_export_with_artifact(
        export_id=created["export_id"],
        expected_state=ExportState.BUILDING,
        state=ExportState.CORE_STRUCTURAL_VALIDATED,
        kind="structural_report",
        relative_path=f"exports/{created['export_id']}/structural-report.json",
        media_type="application/json",
        byte_size=3,
        sha256="a" * 64,
    )

    export = database.get_export(export_id=created["export_id"])
    assert export["state"] == "core_structural_validated"
    assert export["structural_artifact_id"] == artifact["id"]

    with pytest.raises(ValueError):
        database.advance_export_with_artifact(
            export_id=created["export_id"],
            expected_state=ExportState.CORE_STRUCTURAL_VALIDATED,
            state=ExportState.GROOT_STATS_VALIDATED,
            kind="gr00t_stats_report",
            relative_path=f"exports/{created['export_id']}/gr00t-stats-report.json",
            media_type="application/json",
            byte_size=3,
            sha256="not-a-digest",
        )
    assert database.get_export(export_id=created["export_id"])["state"] == "core_structural_validated"


def test_retryable_publication_failure_remains_publishing_until_success(tmp_path: Path) -> None:
    _, _, database, _, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    database.claim_export_build(export_id=created["export_id"])
    with database.open_connection() as connection:
        connection.execute("UPDATE exports SET state='publishing' WHERE id=?", (created["export_id"],))

    failed = database.record_export_failure(
        export_id=created["export_id"],
        expected_state=ExportState.PUBLISHING,
        failure_summary="publish_parent_fsync_failed",
        terminal=False,
    )
    assert failed["state"] == "publishing"
    assert failed["failure_summary"] == "publish_parent_fsync_failed"

    published = database.set_export_state(
        export_id=created["export_id"],
        expected_state=ExportState.PUBLISHING,
        state=ExportState.PUBLISHED,
    )
    assert published["state"] == "published"
    assert published["failure_summary"] is None


def test_validated_exporter_runs_ordered_gates_and_publishes_complete_tree(tmp_path: Path) -> None:
    source, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    events: list[str] = []

    def stats(**kwargs):
        events.append("stats")
        return _fake_stats(**kwargs)

    def loader(**kwargs):
        events.append("loader")
        return _fake_loader(**kwargs)

    exporter = ValidatedDatasetExporter(
        database=database,
        source_registry=registry,
        workspace=workspace,
        isaac_root=tmp_path,
        visualizer_root=tmp_path,
        cosmos_model="cosmos-model",
        cosmos_endpoint_identity="h100",
        video_frame_counter=lambda path: 8,
        stats_validator=stats,
        loader_validator=loader,
        repository_probe=lambda path, name: _clean_repo(
            name, "a" if name == "lerobot-dataset-visualizer" else "b"
        ),
        require_contact_sheets=False,
    )

    result = exporter.run(created["export_id"])

    assert result["state"] == "published"
    assert events == ["stats", "loader"]
    final = Path(created["final_path"])
    assert final.is_dir()
    assert not Path(created["staging_path"]).exists()
    assert (final / "meta/curation_provenance.json").is_file()
    assert (final / "meta/curation_checksums.sha256").is_file()
    assert (final / UNKNOWN_XLSX_NAME).read_bytes() == UNKNOWN_XLSX_BYTES
    provenance = json.loads((final / "meta/curation_provenance.json").read_text())
    assert provenance["source"]["file_count"] == len(registry.records["local/pnp_trash"].file_hashes)
    assert "stale" not in json.loads((final / "meta/stats.json").read_text())
    assert json.loads((final / "meta/relative_stats.json").read_text()) == {}
    assert database.get_export(export_id=created["export_id"])["state"] == "published"
    assert registry.records["local/pnp_trash"].verify_current_inventory()
    assert (source / "meta/tasks.jsonl").read_text() == '{"task_index":0,"task":"original"}\n'
    assert json.loads((source / "meta/stats.json").read_text()) == {"stale": {"mean": [999]}}


def _clean_repo(name: str, digit: str) -> dict[str, object]:
    return {
        "name": name,
        "commit": digit * 40,
        "dirty": False,
        "tracked_diff_sha256": None,
        "untracked_files": [],
    }


def _fake_stats(**kwargs) -> dict[str, object]:
    staging = Path(kwargs["staging_path"]).resolve()
    root = Path(kwargs["isaac_root"]).resolve()
    executable = str(root / ".venv/bin/python")
    info = json.loads((staging / "meta/info.json").read_text())
    stats = {}
    for index, (feature, metadata) in enumerate(sorted(info["features"].items())):
        if not metadata["dtype"].startswith("float"):
            continue
        width = 1
        for dimension in metadata["shape"]:
            width *= dimension
        stats[feature] = {
            metric: [float(index + 1)] * width for metric in ("mean", "std", "min", "max", "q01", "q99")
        }
    outputs = {
        "meta/stats.json": json.dumps(stats, sort_keys=True).encode(),
        "meta/relative_stats.json": b"{}\n",
    }
    for relative, contents in outputs.items():
        (staging / relative).write_bytes(contents)
    return {
        "schema_version": 1,
        "passed": True,
        "command": {
            "executable": executable,
            "cwd": str(root),
            "argv": [
                executable,
                "gr00t/data/stats.py",
                "--dataset-path",
                str(staging),
                "--embodiment-tag",
                "UNITREE_G1_SONIC",
                "--modality-config-path",
                "gr00t/configs/data/embodiment_configs.py",
            ],
            "environment": {"STAGING_PATH": str(staging)},
        },
        "repository": _clean_repo("Isaac-GR00T", "b"),
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "start_time_utc": "2026-08-27T00:00:00Z",
        "end_time_utc": "2026-08-27T00:00:01Z",
        "exception": None,
        "traceback": None,
        "outputs": [
            {
                "path": relative,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "media_type": "application/json",
            }
            for relative, contents in sorted(outputs.items())
        ],
    }


def _fake_loader(**kwargs) -> dict[str, object]:
    staging = Path(kwargs["staging_path"]).resolve()
    root = Path(kwargs["isaac_root"]).resolve()
    executable = str(root / ".venv/bin/python")
    rows = [
        json.loads(line)
        for line in (Path(kwargs["staging_path"]) / "meta/episodes.jsonl").read_text().splitlines()
    ]
    return {
        "schema_version": 1,
        "passed": True,
        "command": {
            "executable": executable,
            "cwd": str(root),
            "argv": [executable, "-c", _LOADER_ACCEPTANCE_SCRIPT, str(staging)],
            "environment": {"STAGING_PATH": str(staging)},
        },
        "repository": _clean_repo("Isaac-GR00T", "b"),
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "start_time_utc": "2026-08-27T00:00:01Z",
        "end_time_utc": "2026-08-27T00:00:02Z",
        "exception": None,
        "traceback": None,
        "episodes": [
            {
                "episode_index": row["episode_index"],
                "row_count": row["length"],
                "runs": row["tasks"],
                "passed": True,
                "exception": None,
                "traceback": None,
            }
            for row in rows
        ],
    }


def _make_exporter(tmp_path: Path, database, registry, workspace: Path, **overrides):
    arguments = {
        "database": database,
        "source_registry": registry,
        "workspace": workspace,
        "isaac_root": tmp_path,
        "visualizer_root": tmp_path,
        "cosmos_model": "cosmos-model",
        "cosmos_endpoint_identity": "h100",
        "video_frame_counter": lambda path: 8,
        "stats_validator": _fake_stats,
        "loader_validator": _fake_loader,
        "repository_probe": lambda path, name: _clean_repo(
            name, "a" if name == "lerobot-dataset-visualizer" else "b"
        ),
        "require_contact_sheets": False,
    }
    arguments.update(overrides)
    return ValidatedDatasetExporter(**arguments)


def _mutate_stats_output(root: Path, mutation: str) -> None:
    stats_path = root / "meta/stats.json"
    relative_path = root / "meta/relative_stats.json"
    if mutation == "delete_stats":
        stats_path.unlink()
    elif mutation == "delete_relative":
        relative_path.unlink()
    elif mutation == "malformed_stats":
        stats_path.write_text("{not-json\n")
    elif mutation == "malformed_relative":
        relative_path.write_text("{not-json\n")
    elif mutation == "numeric":
        stats = json.loads(stats_path.read_text())
        feature = sorted(stats)[0]
        stats[feature]["mean"][0] = 123456.0
        stats_path.write_text(json.dumps(stats, sort_keys=True))
    else:  # pragma: no cover - test helper contract
        raise AssertionError(mutation)


def _rewrite_checksum_manifest(root: Path) -> None:
    checksum_path = root / "meta/curation_checksums.sha256"
    checksum_path.chmod(0o644)
    files = [path for path in _regular_files(root) if path != checksum_path]
    checksum_path.write_text(
        "".join(f"{_hash_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files)
    )
    checksum_path.chmod(0o444)


def _mutate_sealed_modality_and_reseal(root: Path) -> None:
    modality_path = root / "meta/modality.json"
    modality_path.chmod(0o644)
    modality = json.loads(modality_path.read_text())
    modality["review_mutation"] = {"preserved_by_structural_gate": True}
    modality_path.write_text(json.dumps(modality, sort_keys=True))
    _rewrite_checksum_manifest(root)
    seal_staging_tree(root)


@pytest.mark.parametrize(
    ("target", "expected_state", "filename"),
    [
        (ExportState.CORE_STRUCTURAL_VALIDATED, "building", "structural-report.json"),
        (ExportState.GROOT_STATS_VALIDATED, "core_structural_validated", "gr00t-stats-report.json"),
        (ExportState.GROOT_LOADER_VALIDATED, "gr00t_stats_validated", "gr00t-loader-report.json"),
        (ExportState.FINAL_CONSISTENCY_VALIDATED, "provenance_written", "final-consistency-report.json"),
    ],
)
def test_each_gate_report_install_resumes_across_pre_db_crash(
    tmp_path: Path, target: ExportState, expected_state: str, filename: str
) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original = database.advance_export_with_artifact
    crashed = False

    def crash_before_transition(**kwargs):
        nonlocal crashed
        if kwargs["state"] is target and not crashed:
            crashed = True
            raise SystemExit(82)
        return original(**kwargs)

    database.advance_export_with_artifact = crash_before_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    assert database.get_export(export_id=created["export_id"])["state"] == expected_state
    assert (workspace / "exports" / created["export_id"] / filename).is_file()
    database.advance_export_with_artifact = original
    assert exporter.resume(created["export_id"])["state"] == "published"


@pytest.mark.parametrize(
    ("target", "validator_name"),
    [
        (ExportState.GROOT_STATS_VALIDATED, "stats_validator"),
        (ExportState.GROOT_LOADER_VALIDATED, "loader_validator"),
    ],
)
def test_durable_orphan_report_is_adopted_without_nondeterministic_second_validation(
    tmp_path: Path, target: ExportState, validator_name: str
) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    calls: list[int] = []

    def dynamic_validator(**kwargs):
        calls.append(len(calls) + 1)
        report = _fake_stats(**kwargs) if validator_name == "stats_validator" else _fake_loader(**kwargs)
        report["start_time_utc"] = f"2026-08-27T00:00:0{calls[-1]}Z"
        report["stdout"] = f"dynamic-call-{calls[-1]}"
        return report

    exporter = _make_exporter(
        tmp_path,
        database,
        registry,
        workspace,
        **{validator_name: dynamic_validator},
    )
    original = database.advance_export_with_artifact
    crashed = False

    def crash_before_transition(**kwargs):
        nonlocal crashed
        if kwargs["state"] is target and not crashed:
            crashed = True
            raise SystemExit(86)
        return original(**kwargs)

    database.advance_export_with_artifact = crash_before_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    database.advance_export_with_artifact = original
    assert calls == [1]
    assert exporter.resume(created["export_id"])["state"] == "published"
    assert calls == [1]


def test_workspace_only_orphan_report_completes_staged_copy_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import curation.exporter as exporter_module

    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    calls: list[int] = []

    def dynamic_stats(**kwargs):
        calls.append(1)
        return _fake_stats(**kwargs)

    exporter = _make_exporter(
        tmp_path,
        database,
        registry,
        workspace,
        stats_validator=dynamic_stats,
    )
    real_persist = exporter_module.persist_validation_report
    crashed = False

    def crash_between_copies(**kwargs):
        nonlocal crashed
        if kwargs["filename"] == "gr00t-stats-report.json" and not crashed:
            crashed = True
            path = workspace / "exports" / created["export_id"] / kwargs["filename"]
            install_canonical_file(
                path,
                (canonical_json(dict(kwargs["report"])) + "\n").encode(),
                mode=0o600,
            )
            raise SystemExit(87)
        return real_persist(**kwargs)

    monkeypatch.setattr(exporter_module, "persist_validation_report", crash_between_copies)
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    monkeypatch.setattr(exporter_module, "persist_validation_report", real_persist)
    staged = Path(created["staging_path"]) / "meta/curation_artifacts/gr00t-stats-report.json"
    assert not staged.exists()
    assert exporter.resume(created["export_id"])["state"] == "published"
    assert calls == [1]
    assert (Path(created["final_path"]) / "meta/curation_artifacts/gr00t-stats-report.json").is_file()


@pytest.mark.parametrize("corruption", ["invalid", "mismatch", "staging_only"])
def test_invalid_orphan_report_requires_operator_inspection(tmp_path: Path, corruption: str) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original = database.advance_export_with_artifact

    def crash_before_stats_transition(**kwargs):
        if kwargs["state"] is ExportState.GROOT_STATS_VALIDATED:
            raise SystemExit(88)
        return original(**kwargs)

    database.advance_export_with_artifact = crash_before_stats_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    database.advance_export_with_artifact = original
    workspace_report = workspace / "exports" / created["export_id"] / "gr00t-stats-report.json"
    staged_report = Path(created["staging_path"]) / "meta/curation_artifacts/gr00t-stats-report.json"
    if corruption == "invalid":
        workspace_report.write_text("{}\n")
        staged_report.write_text("{}\n")
    elif corruption == "mismatch":
        staged_report.write_text('{"different":true}\n')
    else:
        workspace_report.unlink()

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "export_artifact_reconciliation_conflict"
    assert database.get_export(export_id=created["export_id"])["state"] == "failed"


@pytest.mark.parametrize("corruption", ["delete", "corrupt"])
def test_stats_orphan_output_corruption_is_operator_conflict_without_rerun(
    tmp_path: Path, corruption: str
) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    calls: list[int] = []

    def counted_stats(**kwargs):
        calls.append(1)
        return _fake_stats(**kwargs)

    exporter = _make_exporter(
        tmp_path,
        database,
        registry,
        workspace,
        stats_validator=counted_stats,
    )
    original = database.advance_export_with_artifact

    def crash_before_stats_transition(**kwargs):
        if kwargs["state"] is ExportState.GROOT_STATS_VALIDATED:
            raise SystemExit(90)
        return original(**kwargs)

    database.advance_export_with_artifact = crash_before_stats_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    database.advance_export_with_artifact = original
    stats = Path(created["staging_path"]) / "meta/stats.json"
    if corruption == "delete":
        stats.unlink()
    else:
        stats.write_text("{}\n")

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "export_artifact_reconciliation_conflict"
    assert calls == [1]
    assert database.get_export(export_id=created["export_id"])["state"] == "failed"


@pytest.mark.parametrize(
    "mutation",
    ["delete_stats", "delete_relative", "malformed_stats", "malformed_relative", "numeric"],
)
def test_stats_outputs_are_reauthenticated_immediately_after_the_db_transition(
    tmp_path: Path, mutation: str
) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original = database.advance_export_with_artifact

    def mutate_after_stats_transition(**kwargs):
        result = original(**kwargs)
        if kwargs["state"] is ExportState.GROOT_STATS_VALIDATED:
            _mutate_stats_output(Path(created["staging_path"]), mutation)
        return result

    database.advance_export_with_artifact = mutate_after_stats_transition
    with pytest.raises(ExportError) as raised:
        exporter.run(created["export_id"])

    assert raised.value.payload["error"] == "export_artifact_reconciliation_conflict"
    assert database.get_export(export_id=created["export_id"])["state"] == "failed"
    assert not Path(created["final_path"]).exists()


@pytest.mark.parametrize(
    "mutation",
    ["delete_stats", "delete_relative", "malformed_stats", "malformed_relative", "numeric"],
)
def test_stats_outputs_are_reauthenticated_on_resume_after_the_db_transition(
    tmp_path: Path, mutation: str
) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    calls: list[int] = []

    def counted_stats(**kwargs):
        calls.append(1)
        return _fake_stats(**kwargs)

    exporter = _make_exporter(
        tmp_path,
        database,
        registry,
        workspace,
        stats_validator=counted_stats,
    )
    original = database.advance_export_with_artifact

    def crash_after_stats_transition(**kwargs):
        result = original(**kwargs)
        if kwargs["state"] is ExportState.GROOT_STATS_VALIDATED:
            _mutate_stats_output(Path(created["staging_path"]), mutation)
            raise SystemExit(91)
        return result

    database.advance_export_with_artifact = crash_after_stats_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    database.advance_export_with_artifact = original

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "export_artifact_reconciliation_conflict"
    assert calls == [1]
    assert database.get_export(export_id=created["export_id"])["state"] == "failed"
    assert not Path(created["final_path"]).exists()


def test_stats_output_is_reauthenticated_during_post_rename_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import curation.exporter as exporter_module

    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original_publish = exporter_module.publish_no_clobber

    def crash_after_rename(*args, **kwargs):
        kwargs["after_rename"] = lambda: (_ for _ in ()).throw(SystemExit(92))
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(exporter_module, "publish_no_clobber", crash_after_rename)
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    monkeypatch.setattr(exporter_module, "publish_no_clobber", original_publish)

    final = Path(created["final_path"])
    stats_path = final / "meta/stats.json"
    stats_path.chmod(0o644)
    _mutate_stats_output(final, "numeric")
    stats_path.chmod(0o444)
    _rewrite_checksum_manifest(final)

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "publish_artifact_reconciliation_conflict"
    assert database.get_export(export_id=created["export_id"])["state"] == "failed"


def test_recorded_final_report_must_equal_repeated_gate_before_rename(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original = database.advance_export_with_artifact
    crashed = False

    def mutate_after_final_report_transition(**kwargs):
        nonlocal crashed
        result = original(**kwargs)
        if kwargs["state"] is ExportState.FINAL_CONSISTENCY_VALIDATED and not crashed:
            crashed = True
            _mutate_sealed_modality_and_reseal(Path(created["staging_path"]))
            raise SystemExit(93)
        return result

    database.advance_export_with_artifact = mutate_after_final_report_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    database.advance_export_with_artifact = original

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "export_final_consistency_failed"
    assert database.get_export(export_id=created["export_id"])["state"] == "failed"
    assert Path(created["staging_path"]).is_dir()
    assert not Path(created["final_path"]).exists()


def test_recorded_final_report_must_equal_reconciliation_gate_before_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import curation.exporter as exporter_module

    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    fsync_events: list[Path] = []
    exporter = _make_exporter(
        tmp_path,
        database,
        registry,
        workspace,
        parent_fsync=lambda path: fsync_events.append(path),
    )
    original_publish = exporter_module.publish_no_clobber

    def crash_after_rename(*args, **kwargs):
        kwargs["after_rename"] = lambda: (_ for _ in ()).throw(SystemExit(94))
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(exporter_module, "publish_no_clobber", crash_after_rename)
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    monkeypatch.setattr(exporter_module, "publish_no_clobber", original_publish)
    final = Path(created["final_path"])
    _mutate_sealed_modality_and_reseal(final)

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "publish_artifact_reconciliation_conflict"
    assert database.get_export(export_id=created["export_id"])["state"] == "failed"
    assert fsync_events == []
    assert final.is_dir()


def test_unchanged_final_root_reauthenticates_frozen_stats_command_and_current_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import curation.exporter as exporter_module

    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original_publish = exporter_module.publish_no_clobber

    def crash_after_rename(*args, **kwargs):
        kwargs["after_rename"] = lambda: (_ for _ in ()).throw(SystemExit(95))
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(exporter_module, "publish_no_clobber", crash_after_rename)
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    monkeypatch.setattr(exporter_module, "publish_no_clobber", original_publish)

    assert exporter.resume(created["export_id"])["state"] == "published"
    assert database.get_export(export_id=created["export_id"])["state"] == "published"


def test_provenance_and_checksum_install_resume_across_pre_db_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import curation.exporter as exporter_module

    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original_set_state = database.set_export_state
    crashed = False

    def crash_after_provenance(**kwargs):
        nonlocal crashed
        if kwargs["state"] is ExportState.PROVENANCE_WRITTEN and not crashed:
            crashed = True
            raise SystemExit(83)
        return original_set_state(**kwargs)

    database.set_export_state = crash_after_provenance
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    staging = Path(created["staging_path"])
    assert (staging / "meta/curation_provenance.json").is_file()
    database.set_export_state = original_set_state

    original_final = exporter_module.validate_final_consistency

    def crash_after_checksum(**kwargs):
        assert (staging / "meta/curation_checksums.sha256").is_file()
        raise SystemExit(84)

    monkeypatch.setattr(exporter_module, "validate_final_consistency", crash_after_checksum)
    with pytest.raises(SystemExit):
        exporter.resume(created["export_id"])
    monkeypatch.setattr(exporter_module, "validate_final_consistency", original_final)
    assert exporter.resume(created["export_id"])["state"] == "published"


def test_recorded_report_mismatch_is_terminal_operator_inspection(tmp_path: Path) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original = database.advance_export_with_artifact

    def crash_before_transition(**kwargs):
        if kwargs["state"] is ExportState.GROOT_STATS_VALIDATED:
            raise SystemExit(85)
        return original(**kwargs)

    database.advance_export_with_artifact = crash_before_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    database.advance_export_with_artifact = original
    staged_report = Path(created["staging_path"]) / "meta/curation_artifacts/structural-report.json"
    staged_report.write_bytes(b"truncated")
    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "export_artifact_reconciliation_conflict"
    export = database.get_export(export_id=created["export_id"])
    assert export["state"] == "failed"
    assert export["failure_summary"] == "export_artifact_reconciliation_conflict"


@pytest.mark.parametrize("command", ["run", "resume"])
def test_unsupported_publication_preflight_is_a_sanitized_export_error_before_build(
    tmp_path: Path, command: str
) -> None:
    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(
        tmp_path,
        database,
        registry,
        workspace,
        publication_preflight=lambda parent: (_ for _ in ()).throw(
            PublicationError("publish_noreplace_unsupported", "secret /srv/private/path bearer-token")
        ),
    )
    called: list[bool] = []
    exporter.builder._run_locked = lambda *args, **kwargs: called.append(True)
    with pytest.raises(ExportError) as raised:
        getattr(exporter, command)(created["export_id"])
    assert raised.value.payload == {"error": "publish_noreplace_unsupported"}
    assert "secret" not in str(raised.value)
    assert "/srv/private" not in str(raised.value)
    assert called == []
    assert database.get_export(export_id=created["export_id"])["state"] == "queued"


@pytest.mark.parametrize("swap_at", ["after_pin", "end_of_gate"])
def test_final_read_only_gate_is_bracketed_by_the_same_pinned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap_at: str
) -> None:
    import curation.exporter as exporter_module

    _, workspace, database, registry, service, _ = _rich_case(tmp_path)
    created = service.create("local/pnp_trash")
    exporter = _make_exporter(tmp_path, database, registry, workspace)
    original_advance = database.advance_export_with_artifact
    crashed = False

    def crash_after_final_transition(**kwargs):
        nonlocal crashed
        result = original_advance(**kwargs)
        if kwargs["state"] is ExportState.FINAL_CONSISTENCY_VALIDATED and not crashed:
            crashed = True
            raise SystemExit(89)
        return result

    database.advance_export_with_artifact = crash_after_final_transition
    with pytest.raises(SystemExit):
        exporter.run(created["export_id"])
    database.advance_export_with_artifact = original_advance
    staging = Path(created["staging_path"])
    held = staging.with_name(f"{staging.name}.held")

    def swap_staging() -> None:
        os.rename(staging, held)
        staging.mkdir()

    if swap_at == "after_pin":
        original_pin = exporter_module.pin_publication_source

        def pin_then_swap(source: Path, final: Path):
            identity = original_pin(source, final)
            swap_staging()
            return identity

        monkeypatch.setattr(exporter_module, "pin_publication_source", pin_then_swap)
    else:
        original_gate = exporter_module.validate_final_consistency

        def gate_then_swap(**kwargs):
            report = original_gate(**kwargs)
            swap_staging()
            return report

        monkeypatch.setattr(exporter_module, "validate_final_consistency", gate_then_swap)

    with pytest.raises(ExportError) as raised:
        exporter.resume(created["export_id"])
    assert raised.value.payload["error"] == "publish_source_identity_changed"
    export = database.get_export(export_id=created["export_id"])
    assert export["state"] == "failed"
    assert not Path(created["final_path"]).exists()
