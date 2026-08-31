from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from curation.runbook_validation import build_smoke_authority
import pytest
from test_runbook_validation import ATTEMPT_ID, PROPOSAL_ID, _configuration, _smoke_fixture

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT_REPOSITORY_ROOT = REPOSITORY_ROOT.parents[1]
CURATION_REPO_ROOT = "/home/jihun/work/GR00T-WholeBodyControl/worktrees/lerobot-dataset-visualizer-pnp-trash"
OLD_REPOSITORY_ROOT = "/home/jihun/work/lerobot-dataset-visualizer"

NON_SECRET_EXPORTS = (
    f"export CURATION_REPO_ROOT={CURATION_REPO_ROOT}\n"
    "export CURATION_DATASET_ALIASES_JSON="
    '\'{"local/pnp_trash":'
    '"/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash"}\'\n'
    "export CURATION_WORKSPACE=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation\n"
    "export CURATION_OUTPUT=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned\n"
    "export CURATION_BROWSER_ORIGIN=http://127.0.0.1:3000\n"
    "export CURATION_BACKEND_URL=http://127.0.0.1:8000\n"
    "export NEXT_PUBLIC_DATASET_URL=http://127.0.0.1:8000/api/local-datasets\n"
    "export ISAAC_GROOT_ROOT=/home/jihun/work/Isaac-GR00T"
)

BACKEND_START = """cd "$CURATION_REPO_ROOT"
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000"""

FRONTEND_START = """cd "$CURATION_REPO_ROOT"
bun run dev --hostname 127.0.0.1 --port 3000"""

EXTERNAL_RUNTIME_NAMES = {
    "CURATION_BEARER_TOKEN",
    "COSMOS_BASE_URL",
    "COSMOS_MODEL",
    "COSMOS_API_KEY_ENV",
    "COSMOS_ENDPOINT_IDENTITY",
}

APPROVED_SOURCE_FILE_COUNT = 190
APPROVED_SOURCE_MANIFEST_SHA256 = "5962d8630f06e6260adbae15a3d7ee5f0a1a745c3a12466add8722c2e0da9577"
APPROVED_ANCILLARY_NAME = "pnp_trash.xlsx"
APPROVED_ANCILLARY_SHA256 = "989f6968e5cf8ee0972b850199c948dd75ce140480c82cbe368053cde6ab34c9"
APPROVED_ANCILLARY_SIZE = 13_644


def _source_preflight(runbook: str) -> str:
    return _bash_block_after(runbook, "### 1. Source, final destination, ownership, and free space")


def _bash_block_after(document: str, heading: str) -> str:
    section = document.split(heading, 1)[1]
    return section.split("```bash", 1)[1].split("```", 1)[0].strip()


def _bash_blocks_in_section(document: str, heading: str) -> list[str]:
    section = document.split(heading, 1)[1].split("\n## ", 1)[0]
    return [block.strip() for block in re.findall(r"```bash\n(.*?)\n```", section, re.DOTALL)]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_curation_docs_freeze_example_mapping_and_repository_root_startup() -> None:
    readme = (REPOSITORY_ROOT / "backend" / "README.md").read_text()
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()

    for document in (readme, runbook):
        assert NON_SECRET_EXPORTS in document
        assert BACKEND_START in document
        assert FRONTEND_START in document


def test_curation_docs_use_one_trusted_checkout_root_and_never_the_dirty_primary_path() -> None:
    documents = (
        (REPOSITORY_ROOT / ".env.example").read_text(),
        (REPOSITORY_ROOT / "backend" / "README.md").read_text(),
        (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text(),
        (PARENT_REPOSITORY_ROOT / "docs/superpowers/plans/2026-08-20-pnp-trash-cosmos-curation.md").read_text(),
        (
            PARENT_REPOSITORY_ROOT / "docs/superpowers/specs/2026-08-18-pnp-trash-cosmos-curation-design.md"
        ).read_text(),
    )

    for document in documents:
        assert CURATION_REPO_ROOT in document
        assert OLD_REPOSITORY_ROOT not in document


def _run_checkout_preflight(
    runbook: str,
    tmp_path: Path,
    *,
    approved_commit: str,
    actual_commit: str,
    dirty_output: str,
    heading: str = "#### Checkout authentication gate",
) -> subprocess.CompletedProcess[str]:
    preflight = _bash_block_after(runbook, heading)
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True)
    (checkout / ".git").mkdir()
    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(
        commands / "git",
        """#!/bin/bash
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf '%s\n' true ;;
  *"rev-parse --show-toplevel"*) printf '%s\n' "$CURATION_REPO_ROOT" ;;
  *"rev-parse HEAD"*) printf '%s\n' "$ACTUAL_COMMIT" ;;
  *"merge-base --is-ancestor"*) exit 0 ;;
  *"status --porcelain --untracked-files=all"*) printf '%s' "$DIRTY_OUTPUT" ;;
  *) exit 99 ;;
esac
""",
    )
    return subprocess.run(
        ["bash", "-c", preflight],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "CURATION_REPO_ROOT": str(checkout),
            "APPROVED_CURATION_COMMIT_SHA": approved_commit,
            "ACTUAL_COMMIT": actual_commit,
            "DIRTY_OUTPUT": dirty_output,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_checkout_preflight_rejects_wrong_commit_and_dirty_tree_without_path_disclosure(
    tmp_path: Path,
) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    preflight = _bash_block_after(runbook, "#### Checkout authentication gate")
    assert preflight.startswith("set -euo pipefail\n")
    assert "${APPROVED_CURATION_COMMIT_SHA:?" in preflight
    assert "merge-base --is-ancestor 60ef88c" in preflight
    assert "status --porcelain --untracked-files=all" in preflight
    assert "APPROVED_CURATION_COMMIT_SHA=$(" not in preflight

    wrong = _run_checkout_preflight(
        runbook,
        tmp_path / "wrong",
        approved_commit="a" * 40,
        actual_commit="b" * 40,
        dirty_output="",
    )
    assert wrong.returncode == 1
    assert wrong.stderr == "FAIL: checkout HEAD does not match the independently approved commit\n"
    assert str(tmp_path) not in wrong.stderr

    dirty = _run_checkout_preflight(
        runbook,
        tmp_path / "dirty",
        approved_commit="a" * 40,
        actual_commit="a" * 40,
        dirty_output="?? untracked-sensitive-name",
    )
    assert dirty.returncode == 1
    assert dirty.stderr == "FAIL: curation checkout is not clean\n"
    assert str(tmp_path) not in dirty.stderr


def test_post_static_checkout_gate_reauthenticates_head_and_cleanliness_before_runtime(
    tmp_path: Path,
) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    heading = "#### Post-static checkout reauthentication gate"
    gate = _bash_block_after(runbook, heading)
    assert gate.startswith("set -euo pipefail\n")
    assert "rev-parse HEAD" in gate
    assert "status --porcelain --untracked-files=all" in gate
    assert runbook.index(heading) > runbook.index("bun run format:check")
    assert runbook.index(heading) < runbook.index("## Filesystem and dependency preflight")

    dirty = _run_checkout_preflight(
        runbook,
        tmp_path,
        approved_commit="a" * 40,
        actual_commit="a" * 40,
        dirty_output=" M generated-change",
        heading=heading,
    )
    assert dirty.returncode == 1
    assert dirty.stderr == "FAIL: curation checkout is not clean after static gates\n"


def test_runtime_static_gate_is_nonmutating() -> None:
    readme = (REPOSITORY_ROOT / "backend" / "README.md").read_text()
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    for document in (readme, runbook):
        assert "bun run format &&" not in document
        assert "bun run format:check && bun run validate" in document


def test_task14_docs_truthfully_leave_real_runtime_integration_pending() -> None:
    readme = (REPOSITORY_ROOT / "backend" / "README.md").read_text()
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    for document in (readme, runbook):
        normalized = re.sub(r"\s+", " ", document)
        startup_contract = "creates the canonical source manifest and initializes curation.sqlite3 before serving"
        assert startup_contract in normalized
        assert "Task 14 integration is not complete" in normalized
        assert "approved-source loopback and browser smoke remains pending" in normalized


def test_env_example_never_assigns_external_or_browser_visible_secrets() -> None:
    example = (REPOSITORY_ROOT / ".env.example").read_text()

    for name in EXTERNAL_RUNTIME_NAMES:
        assert name in example
        assert re.search(rf"(?m)^(?:export\s+)?{name}\s*=", example) is None
    assert re.search(r"NEXT_PUBLIC_.*(?:TOKEN|KEY|SECRET|COSMOS)", example) is None


def test_next_launch_docs_never_request_the_complete_python_environment() -> None:
    example = (REPOSITORY_ROOT / ".env.example").read_text()
    readme = (REPOSITORY_ROOT / "backend" / "README.md").read_text()
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()

    assert "both local servers" not in example
    next_launch = (
        "Start Next.js in a second terminal containing only `CURATION_BACKEND_URL`, "
        "`CURATION_BEARER_TOKEN`, and `NEXT_PUBLIC_DATASET_URL`"
    )
    for document in (readme, runbook):
        normalized = re.sub(r"\s+", " ", document)
        assert next_launch in normalized
        assert "second terminal with the same runtime configuration" not in normalized
        assert "second fully configured terminal" not in normalized


def test_capability_probe_matches_production_redirect_and_proxy_policy() -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()

    assert "httpx.Client(trust_env=False, follow_redirects=False)" in runbook
    assert "urllib.request" not in runbook
    assert "urlopen" not in runbook


def test_docs_assign_process_specific_settings_and_only_server_subset_to_next() -> None:
    readme = (REPOSITORY_ROOT / "backend" / "README.md").read_text()
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()

    settings_split = (
        "`backend.app:app` uses `CurationSettings.from_env()`, "
        "`backend/curation_worker.py` uses `WorkerSettings.from_env()`, and "
        "`backend/curation_export.py` uses `ExportSettings.from_env()`"
    )
    next_subset = (
        "Next.js requires only `CURATION_BACKEND_URL`, `CURATION_BEARER_TOKEN`, and `NEXT_PUBLIC_DATASET_URL`"
    )
    capability_reader = (
        "The backend reads the variable named by `COSMOS_API_KEY_ENV` during "
        "batch capability validation before it creates a job"
    )
    least_privilege_key = (
        "Only the backend and worker receive the actual credential variable named by "
        "`COSMOS_API_KEY_ENV`; the exporter must not receive that secret"
    )
    for document in (readme, runbook):
        normalized = re.sub(r"\s+", " ", document)
        assert settings_split in normalized
        assert next_subset in normalized
        assert capability_reader in normalized
        assert least_privilege_key in normalized
        assert "all three Python processes therefore requires" not in normalized
        assert "present in all three Python launch environments" not in normalized
    assert re.search(r"(?m)^\| `CURATION_BEARER_TOKEN` .*\| FastAPI and Next.js only", runbook)
    assert re.search(r"(?m)^\| `COSMOS_BASE_URL` .*\| FastAPI and worker only", runbook)
    assert re.search(r"(?m)^\| `COSMOS_API_KEY_ENV` .*\| FastAPI and worker only", runbook)
    assert re.search(r"(?m)^\| Variable named by `COSMOS_API_KEY_ENV` .*\| Backend and worker only", runbook)


def test_worker_and_exporter_launchers_enforce_process_environment_allowlists(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    launchers = _bash_block_after(runbook, "### Least-privilege Python process launchers")
    visualizer = tmp_path / "visualizer"
    python = visualizer / "backend" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    _write_executable(python, '#!/bin/bash\nenv | sort > "${@: -1}"\n')
    worker_env = tmp_path / "worker.env"
    exporter_env = tmp_path / "exporter.env"
    script = (
        f"{launchers}\n"
        f"run_curation_worker dump {shlex.quote(str(worker_env))}\n"
        f"run_curation_exporter dump {shlex.quote(str(exporter_env))}\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "CURATION_REPO_ROOT": str(visualizer),
            "CURATION_DATASET_ALIASES_JSON": '{"local/pnp_trash":"/source"}',
            "CURATION_WORKSPACE": "/workspace",
            "CURATION_OUTPUT": "/forbidden-output",
            "CURATION_BROWSER_ORIGIN": "http://forbidden.example",
            "CURATION_BEARER_TOKEN": "forbidden-bearer",
            "COSMOS_BASE_URL": "http://127.0.0.1:8001/v1",
            "COSMOS_MODEL": "cosmos3-nano",
            "COSMOS_API_KEY_ENV": "DYNAMIC_COSMOS_SECRET",
            "COSMOS_ENDPOINT_IDENTITY": "h100-cosmos",
            "DYNAMIC_COSMOS_SECRET": "target-secret",
            "ISAAC_GROOT_ROOT": "/isaac",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    worker = worker_env.read_text()
    exporter = exporter_env.read_text()
    assert "DYNAMIC_COSMOS_SECRET=target-secret" in worker
    assert "COSMOS_API_KEY_ENV=DYNAMIC_COSMOS_SECRET" in worker
    for forbidden in (
        "CURATION_BEARER_TOKEN=",
        "CURATION_OUTPUT=",
        "CURATION_BROWSER_ORIGIN=",
        "ISAAC_GROOT_ROOT=",
    ):
        assert forbidden not in worker
    for forbidden in (
        "CURATION_BEARER_TOKEN=",
        "CURATION_OUTPUT=",
        "CURATION_BROWSER_ORIGIN=",
        "COSMOS_BASE_URL=",
        "COSMOS_API_KEY_ENV=",
        "DYNAMIC_COSMOS_SECRET=",
    ):
        assert forbidden not in exporter
    assert "ISAAC_GROOT_ROOT=/isaac" in exporter


def test_worker_launcher_rejects_reserved_dynamic_secret_name_before_child_start(
    tmp_path: Path,
) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    launchers = _bash_block_after(runbook, "### Least-privilege Python process launchers")
    visualizer = tmp_path / "visualizer"
    python = visualizer / "backend" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    marker = tmp_path / "child-started"
    _write_executable(python, f"#!/bin/bash\nprintf started > {shlex.quote(str(marker))}\n")

    result = subprocess.run(
        ["bash", "-c", f"{launchers}\nrun_curation_worker status"],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "CURATION_REPO_ROOT": str(visualizer),
            "CURATION_DATASET_ALIASES_JSON": '{"local/pnp_trash":"/source"}',
            "CURATION_WORKSPACE": "/workspace",
            "CURATION_BEARER_TOKEN": "must-not-leak",
            "COSMOS_BASE_URL": "http://127.0.0.1:8001/v1",
            "COSMOS_MODEL": "cosmos3-nano",
            "COSMOS_API_KEY_ENV": "CURATION_BEARER_TOKEN",
            "COSMOS_ENDPOINT_IDENTITY": "h100-cosmos",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: configured Cosmos credential target is reserved\n"
    assert "must-not-leak" not in result.stdout + result.stderr
    assert not marker.exists()


def test_docs_freeze_the_approved_190_file_source_authority() -> None:
    documents = (
        (REPOSITORY_ROOT / ".env.example").read_text(),
        (REPOSITORY_ROOT / "backend" / "README.md").read_text(),
        (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text(),
    )

    for document in documents:
        normalized = re.sub(r"\s+", " ", document)
        assert "190 immutable regular files" in normalized
        assert APPROVED_SOURCE_MANIFEST_SHA256 in normalized
        assert APPROVED_ANCILLARY_NAME in normalized
        assert APPROVED_ANCILLARY_SHA256 in normalized
        assert "13,644 bytes" in normalized
        assert "189 regular files" not in normalized
        assert "189-file" not in normalized
        assert "No approved canonical hash" not in normalized


def test_source_preflight_fails_closed_on_a_sanitized_191_file_fixture(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    preflight = _source_preflight(runbook)
    assert preflight.startswith("set -euo pipefail\n")
    assert "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}" in preflight
    assert "${CURATION_OUTPUT:?FAIL: CURATION_OUTPUT is required}" in preflight

    source = tmp_path / "source"
    source.mkdir()
    for index in range(191):
        (source / f"file-{index:03d}").write_bytes(b"fixture")
    preflight = preflight.replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash", str(source))
    result = subprocess.run(
        ["bash", "-c", preflight],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "CURATION_WORKSPACE": str(tmp_path / "workspace"),
            "CURATION_OUTPUT": str(tmp_path / "output"),
            "APPROVED_SOURCE_MANIFEST_SHA256": APPROVED_SOURCE_MANIFEST_SHA256,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: source regular-file count must be 190; found 191\n"
    assert str(tmp_path) not in result.stderr


def test_source_preflight_rejects_unapproved_prospective_hash_before_backend_start(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    preflight = _source_preflight(runbook)
    assert "${APPROVED_SOURCE_MANIFEST_SHA256:?FAIL: approved source manifest SHA-256 is required}" in preflight
    assert "PROSPECTIVE_SOURCE_MANIFEST_SHA256" in preflight
    assert f"PINNED_SOURCE_MANIFEST_SHA256={APPROVED_SOURCE_MANIFEST_SHA256}" in preflight
    assert APPROVED_ANCILLARY_NAME in preflight
    assert APPROVED_ANCILLARY_SHA256 in preflight
    assert str(APPROVED_ANCILLARY_SIZE) in preflight
    assert "APPROVED_SOURCE_MANIFEST_SHA256=$(" not in preflight
    assert "from backend.curation.source import _manifest_bytes" in preflight
    assert "_manifest_bytes(root)" in preflight
    assert "os.walk" not in preflight
    assert runbook.index("PROSPECTIVE_SOURCE_MANIFEST_SHA256") < runbook.index(
        "## Start the two loopback services"
    )

    source = tmp_path / "source"
    source.mkdir()
    for index in range(APPROVED_SOURCE_FILE_COUNT):
        (source / f"file-{index:03d}").write_bytes(b"fixture")
    preflight = preflight.replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash", str(source))
    result = subprocess.run(
        ["bash", "-c", preflight],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "CURATION_WORKSPACE": str(tmp_path / "workspace"),
            "CURATION_OUTPUT": str(tmp_path / "output"),
            "CURATION_REPO_ROOT": str(REPOSITORY_ROOT),
            "APPROVED_SOURCE_MANIFEST_SHA256": APPROVED_SOURCE_MANIFEST_SHA256,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: prospective source manifest SHA-256 does not match approved record\n"
    assert str(tmp_path) not in result.stderr


def test_persisted_manifest_gate_reuses_the_pinned_non_live_authority() -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    gate = _bash_block_after(runbook, "### Persisted source manifest")

    assert f"PINNED_SOURCE_MANIFEST_SHA256={APPROVED_SOURCE_MANIFEST_SHA256}" in gate
    assert 'test "$APPROVED_SOURCE_MANIFEST_SHA256" != "$PINNED_SOURCE_MANIFEST_SHA256"' in gate
    assert "APPROVED_SOURCE_MANIFEST_SHA256=$(" not in gate


def test_isaac_preflight_does_not_mask_a_missing_loader_interpreter(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    preflight = _bash_block_after(runbook, "### 3. Isaac-GR00T import and configuration")
    isaac = tmp_path / "isaac"
    stats = isaac / "gr00t" / "data" / "stats.py"
    stats.parent.mkdir(parents=True)
    stats.write_text("# fixture\n")

    result = subprocess.run(
        ["bash", "-c", preflight],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "ISAAC_GROOT_ROOT": str(isaac)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: Isaac-GR00T Python is unavailable\n"
    assert str(tmp_path) not in result.stderr


def test_listener_preflight_requires_both_exact_loopback_ports(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    preflight = _bash_block_after(runbook, "### Listener and exact-origin checks")
    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(
        commands / "ss",
        "#!/bin/bash\nprintf '%s\\n' 'LISTEN 0 128 127.0.0.1:8000 0.0.0.0:* users:'\n",
    )
    _write_executable(
        commands / "curl",
        """#!/bin/bash
case "$*" in
  *localhost:3000*) printf '%s\r\n' 'HTTP/1.1 200 OK' ;;
  *) printf '%s\r\n' 'HTTP/1.1 200 OK' 'access-control-allow-origin: http://127.0.0.1:3000' ;;
esac
""",
    )

    result = subprocess.run(
        ["bash", "-c", preflight],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: Next.js listener 127.0.0.1:3000 is unavailable\n"
    assert str(tmp_path) not in result.stderr


def test_local_asset_smoke_fails_closed_on_http_500_without_real_requests(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    smoke = _bash_block_after(runbook, "## Local asset integration smoke")
    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(commands / "curl", "#!/bin/bash\nprintf '500'\n")

    result = subprocess.run(
        ["bash", "-c", smoke],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: info.json returned HTTP 500\n"
    assert str(tmp_path) not in result.stderr


def test_required_python_preflights_never_use_optimization_sensitive_asserts() -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    frame_gate = _bash_block_after(runbook, "### 4. FFmpeg/PyAV frame-count agreement")
    capability_gate = _bash_blocks_in_section(runbook, "## Cosmos capability and one-episode smoke")[0]

    for gate in (frame_gate, capability_gate):
        assert "assert " not in gate
        assert "raise SystemExit" in gate


def _run_one_episode_smoke(
    runbook: str,
    tmp_path: Path,
    *,
    worker_exit: int,
    final_status: str,
    missing_artifact: str | None = "request",
    status_job_id: str = "job-smoke",
) -> subprocess.CompletedProcess[str]:
    smoke = _bash_blocks_in_section(runbook, "## Cosmos capability and one-episode smoke")[1]
    launchers = _bash_block_after(runbook, "### Least-privilege Python process launchers")
    visualizer = tmp_path / "visualizer"
    worker = visualizer / "backend" / ".venv" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    _write_executable(
        worker,
        (
            "#!/bin/bash\n"
            'case "$*" in\n'
            f"  *backend/curation_worker.py*) exit {worker_exit} ;;\n"
            "  *runbook_validation*representative*)\n"
            "    REPRESENTATIVE_SOURCE=\n"
            '    while test "$#" -gt 0; do\n'
            '      case "$1" in\n'
            "        --source-path) REPRESENTATIVE_SOURCE=$2; shift 2 ;;\n"
            "        *) shift ;;\n"
            "      esac\n"
            "    done\n"
            '    test "$REPRESENTATIVE_SOURCE" = "$EXPECTED_SMOKE_SOURCE" || exit 8\n'
            '    printf \'%s\' \'{"source_episode_index":4,"frame_count":2060,'
            '"duration_s":41.2,"sampled_frame_count":83}\'; exit 0 ;;\n'
            f'  *) exec env PYTHONPATH={shlex.quote(str(REPOSITORY_ROOT))} {sys.executable} "$@" ;;\n'
            "esac\n"
        ),
    )
    status, episode, workspace = _smoke_fixture(tmp_path)
    attempt_state = "succeeded" if final_status == "completed" else "manual_only"
    status["state"] = final_status
    status["job_id"] = status_job_id
    status["counts"] = {attempt_state: 1}
    status["episodes"][0]["state"] = attempt_state
    status["active_proposal_coverage"] = 1 if attempt_state == "succeeded" else 0
    attempt_root = workspace / "artifacts" / "cosmos" / ATTEMPT_ID
    manifest_sha256 = status["configuration"]["source_manifest_sha256"]
    contact_namespace = workspace / "contact_sheets" / "datasets" / f"dataset_1_{manifest_sha256}"
    artifact_paths = {
        "request": attempt_root / "request.json",
        "response": attempt_root / "response.txt",
        "parsed": attempt_root / "parsed.json",
        "contact-sheet": contact_namespace / "proposals" / f"proposal_{PROPOSAL_ID}.png",
        "contact-sheet-receipt": (
            contact_namespace / "receipts" / "proposals" / f"proposal_{PROPOSAL_ID}.png.receipt.json"
        ),
    }
    if missing_artifact is not None:
        artifact_paths[missing_artifact].unlink()

    lexical_parent = tmp_path / "lexical-parent"
    lexical_parent.symlink_to(tmp_path, target_is_directory=True)
    lexical_source = lexical_parent / "source"

    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(
        commands / "curl",
        (
            "#!/bin/bash\n"
            'case "$*" in\n'
            "  */batches/*) printf '%s' \"$SMOKE_STATUS_JSON\" ;;\n"
            "  */episodes/4*) printf '%s' \"$SMOKE_EPISODE_JSON\" ;;\n"
            "  */batches*) printf '%s' '{\"job_id\":\"job-smoke\"}' ;;\n"
            "  *) printf '%s' '{}' ;;\n"
            "esac\n"
        ),
    )
    return subprocess.run(
        ["bash", "-c", f"{launchers}\n{smoke}"],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "CURATION_REPO_ROOT": str(visualizer),
            "CURATION_WORKSPACE": str(workspace),
            "SOURCE_DATASET": str(lexical_source),
            "EXPECTED_SMOKE_SOURCE": str((tmp_path / "source").resolve()),
            "APPROVED_SOURCE_MANIFEST_SHA256": status["configuration"]["source_manifest_sha256"],
            "SMOKE_FINAL_STATUS": final_status,
            "SMOKE_STATUS_JSON": json.dumps(status, separators=(",", ":")),
            "SMOKE_EPISODE_JSON": json.dumps(episode, separators=(",", ":")),
            "CURATION_DATASET_ALIASES_JSON": json.dumps(
                {"local/pnp_trash": str(lexical_source)}, separators=(",", ":")
            ),
            "COSMOS_BASE_URL": "http://127.0.0.1:8001/v1",
            "COSMOS_MODEL": "cosmos3-nano",
            "COSMOS_API_KEY_ENV": "COSMOS_API_KEY",
            "COSMOS_ENDPOINT_IDENTITY": "h100-cosmos",
            "COSMOS_API_KEY": "test-secret",
        },
        capture_output=True,
        text=True,
        input=f"CONFIRM SMOKE EVIDENCE job-smoke {ATTEMPT_ID}\n",
        check=False,
    )


def test_one_episode_smoke_stops_on_worker_failure(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    result = _run_one_episode_smoke(
        runbook,
        tmp_path,
        worker_exit=9,
        final_status="completed",
    )

    assert result.returncode == 9


def test_one_episode_smoke_rejects_a_failed_terminal_state(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    result = _run_one_episode_smoke(
        runbook,
        tmp_path,
        worker_exit=0,
        final_status="failed",
    )

    assert result.returncode == 1
    assert result.stderr == ("FAIL: one-episode Cosmos smoke requires terminal state completed; found failed\n")
    assert str(tmp_path) not in result.stderr


def test_one_episode_smoke_rejects_completed_with_failures_and_manual_only(
    tmp_path: Path,
) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    result = _run_one_episode_smoke(
        runbook,
        tmp_path,
        worker_exit=0,
        final_status="completed_with_failures",
    )

    assert result.returncode == 1
    assert result.stderr == (
        "FAIL: one-episode Cosmos smoke requires terminal state completed; found completed_with_failures\n"
    )
    assert str(tmp_path) not in result.stderr


def test_one_episode_smoke_requires_every_artifact_before_confirmation(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    for missing_artifact in (
        "request",
        "response",
        "parsed",
        "contact-sheet",
        "contact-sheet-receipt",
    ):
        result = _run_one_episode_smoke(
            runbook,
            tmp_path / missing_artifact,
            worker_exit=0,
            final_status="completed",
            missing_artifact=missing_artifact,
        )

        assert result.returncode == 1
        assert result.stderr == f"FAIL: one-episode {missing_artifact} artifact is unavailable\n"
        assert str(tmp_path) not in result.stderr


def test_one_episode_smoke_executes_the_fully_valid_evidence_path(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    result = _run_one_episode_smoke(
        runbook,
        tmp_path,
        worker_exit=0,
        final_status="completed",
        missing_artifact=None,
    )

    assert result.returncode == 0, result.stderr
    assert "Preserve these shell variables before the full batch" in result.stdout


def test_one_episode_smoke_freezes_status_sampling_evidence_and_operator_confirmation() -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    smoke = _bash_blocks_in_section(runbook, "## Cosmos capability and one-episode smoke")[1]

    assert "SMOKE_SOURCE_EPISODE_INDEX=4" in smoke
    assert 'SMOKE_SOURCE_PATH=$(realpath -e -- "$SOURCE_DATASET")' in smoke
    assert "SMOKE_SOURCE_PATH=/home/" not in smoke
    assert "CURATION_DATASET_ALIASES_JSON" in smoke
    assert "runbook_validation representative" in smoke
    assert smoke.index("runbook_validation representative") < smoke.index("SMOKE_RESPONSE=")
    assert "episode_indices:[$index]" in smoke
    assert '"episode_indices":[0]' not in smoke
    assert "episodes/$SMOKE_SOURCE_EPISODE_INDEX" in smoke
    assert 'test "$SMOKE_STATE" = completed' in smoke
    assert 'test "$SMOKE_SUCCEEDED" -eq 1' in smoke
    assert 'test "$SMOKE_MANUAL_ONLY" -eq 0' in smoke
    assert 'test "$SMOKE_RETRYABLE" -eq 0' in smoke
    assert 'test "$SMOKE_PROPOSAL_COVERAGE" -eq 1' in smoke
    assert "completed_with_failures" not in smoke
    for name in ("request.json", "response.txt", "parsed.json", ".png", ".receipt.json"):
        assert name in smoke
    validator = (REPOSITORY_ROOT / "backend" / "curation" / "runbook_validation.py").read_text()
    for key in (
        "original_fps",
        "target_fps",
        "selected_frame_indices",
        "selected_parquet_timestamps_s",
        "media_io_kwargs",
        "do_sample_frames",
        '"redacted"',
        '"sha256"',
    ):
        assert key in validator
    assert "CONFIRM SMOKE EVIDENCE" in smoke
    assert "backend.curation.runbook_validation smoke" in smoke
    assert '--expected-smoke-job-id "$SMOKE_JOB_ID"' in smoke
    assert "CONFIRMED_SMOKE_AUTHORITY_JSON" in smoke
    assert "CONFIRMED_SMOKE_AUTHORITY_SHA256" in smoke


def test_one_episode_preflight_failure_cannot_create_smoke_batch(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    smoke = _bash_blocks_in_section(runbook, "## Cosmos capability and one-episode smoke")[1]
    visualizer = tmp_path / "visualizer"
    python = visualizer / "backend" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    _write_executable(
        python,
        '#!/bin/bash\ncase "$*" in\n  *runbook_validation*representative*) exit 9 ;;\n  *) exit 0 ;;\nesac\n',
    )
    commands = tmp_path / "bin"
    commands.mkdir()
    (tmp_path / "source").mkdir()
    marker = tmp_path / "smoke-posted"
    _write_executable(
        commands / "curl",
        '#!/bin/bash\ncase "$*" in\n'
        "  *workspaces/open*) printf '%s' '{}' ;;\n"
        "  *) printf posted > \"$SMOKE_POST_MARKER\"; printf '%s' '{}' ;;\n"
        "esac\n",
    )

    result = subprocess.run(
        ["bash", "-c", smoke],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "CURATION_REPO_ROOT": str(visualizer),
            "CURATION_WORKSPACE": str(tmp_path / "workspace"),
            "SOURCE_DATASET": str(tmp_path / "source"),
            "CURATION_DATASET_ALIASES_JSON": json.dumps(
                {"local/pnp_trash": str(tmp_path / "source")}, separators=(",", ":")
            ),
            "APPROVED_SOURCE_MANIFEST_SHA256": "a" * 64,
            "SMOKE_POST_MARKER": str(marker),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 9
    assert not marker.exists()


@pytest.mark.parametrize("source_case", ["wrong_alias", "dangling_target", "file_target"])
def test_one_episode_source_authority_gate_rejects_wrong_or_unresolved_canonical_source(
    tmp_path: Path,
    source_case: str,
) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    smoke = _bash_blocks_in_section(runbook, "## Cosmos capability and one-episode smoke")[1]
    visualizer = tmp_path / "visualizer"
    python = visualizer / "backend" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    child_marker = tmp_path / "representative-started"
    _write_executable(
        python,
        f"#!/bin/bash\nprintf started > {shlex.quote(str(child_marker))}\n",
    )
    source = tmp_path / "source"
    if source_case == "dangling_target":
        source.symlink_to(tmp_path / "missing", target_is_directory=True)
    elif source_case == "file_target":
        target = tmp_path / "not-a-directory"
        target.write_text("not source")
        source.symlink_to(target)
    else:
        source.mkdir()
    alias_source = tmp_path / "different-source" if source_case == "wrong_alias" else source
    if source_case == "wrong_alias":
        alias_source.mkdir()

    commands = tmp_path / "bin"
    commands.mkdir()
    post_marker = tmp_path / "smoke-posted"
    _write_executable(
        commands / "curl",
        '#!/bin/bash\ncase "$*" in\n'
        "  *workspaces/open*) printf '%s' '{}' ;;\n"
        "  *) printf posted > \"$SMOKE_POST_MARKER\"; printf '%s' '{}' ;;\n"
        "esac\n",
    )
    result = subprocess.run(
        ["bash", "-c", smoke],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "CURATION_REPO_ROOT": str(visualizer),
            "CURATION_WORKSPACE": str(tmp_path / "workspace"),
            "SOURCE_DATASET": str(source),
            "CURATION_DATASET_ALIASES_JSON": json.dumps(
                {"local/pnp_trash": str(alias_source)}, separators=(",", ":")
            ),
            "APPROVED_SOURCE_MANIFEST_SHA256": "a" * 64,
            "SMOKE_POST_MARKER": str(post_marker),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    expected = (
        "FAIL: smoke source does not match the configured dataset alias\n"
        if source_case == "wrong_alias"
        else "FAIL: smoke source canonical target is unavailable\n"
    )
    assert result.stderr == expected
    assert str(tmp_path) not in result.stderr
    assert not child_marker.exists()
    assert not post_marker.exists()


def test_one_episode_smoke_rejects_a_status_from_another_job_before_confirmation(
    tmp_path: Path,
) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    result = _run_one_episode_smoke(
        runbook,
        tmp_path,
        worker_exit=0,
        final_status="completed",
        missing_artifact=None,
        status_job_id="stale-job",
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: smoke status job identity does not match the posted job\n"
    assert "Required confirmation:" not in result.stdout


def test_full_batch_gate_requires_confirmed_smoke_before_post(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    full_batch = _bash_block_after(runbook, "## Run and monitor the full Cosmos batch")
    commands = tmp_path / "bin"
    commands.mkdir()
    marker = tmp_path / "full-batch-posted"
    _write_executable(
        commands / "curl",
        "#!/bin/bash\nprintf '%s\\n' called >> \"$FULL_BATCH_CALL_MARKER\"\nprintf '%s' '{}'\n",
    )

    result = subprocess.run(
        ["bash", "-c", full_batch],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "FULL_BATCH_CALL_MARKER": str(marker),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: confirmed one-episode smoke job ID is required\n"
    assert not marker.exists()


def _run_full_batch_with_authority(
    runbook: str,
    tmp_path: Path,
    *,
    stale_job_id: bool = False,
    mismatch_configuration: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    launchers = _bash_block_after(runbook, "### Least-privilege Python process launchers")
    full_batch = _bash_block_after(runbook, "## Run and monitor the full Cosmos batch")
    status, episode, workspace = _smoke_fixture(tmp_path)
    authority = build_smoke_authority(
        workspace=workspace,
        status=status,
        episode=episode,
        expected_smoke_job_id="job-smoke",
    )
    authority_json = json.dumps(authority, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    authority_sha256 = hashlib.sha256(authority_json.encode()).hexdigest()
    full_configuration = _configuration(
        tmp_path,
        list(range(92)),
        manifest_sha256=status["configuration"]["source_manifest_sha256"],
    )
    if mismatch_configuration:
        full_configuration["transport"]["timeout_seconds"] = 119
    full_status = {
        "job_id": "job-full",
        "state": "queued",
        "configuration": full_configuration,
        "episodes": [
            {
                "attempt_id": f"attempt-{index}",
                "attempt_number": 0,
                "source_episode_index": index,
                "state": "queued",
            }
            for index in range(92)
        ],
    }

    visualizer = tmp_path / "visualizer"
    python = visualizer / "backend" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    worker_marker = tmp_path / "worker-launched"
    _write_executable(
        python,
        (
            "#!/bin/bash\n"
            'case "$*" in\n'
            f"  *backend/curation_worker.py*) printf launched > {shlex.quote(str(worker_marker))}; exit 0 ;;\n"
            f'  *) exec env PYTHONPATH={shlex.quote(str(REPOSITORY_ROOT))} {sys.executable} "$@" ;;\n'
            "esac\n"
        ),
    )
    commands = tmp_path / "bin"
    commands.mkdir()
    post_marker = tmp_path / "full-batch-posted"
    _write_executable(
        commands / "curl",
        """#!/bin/bash
case "$*" in
  *batches/job-smoke*) printf '%s' "$SMOKE_STATUS_JSON" ;;
  *episodes/4*) printf '%s' "$SMOKE_EPISODE_JSON" ;;
  *batches/job-full*) printf '%s' "$FULL_STATUS_JSON" ;;
  *-d*batches*) printf posted > "$FULL_BATCH_POST_MARKER"; printf '%s' '{"job_id":"job-full"}' ;;
  *) exit 97 ;;
esac
""",
    )
    confirmed_job_id = "stale-job" if stale_job_id else "job-smoke"
    result = subprocess.run(
        ["bash", "-c", f"{launchers}\n{full_batch}"],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "CURATION_REPO_ROOT": str(visualizer),
            "CURATION_WORKSPACE": str(workspace),
            "CURATION_DATASET_ALIASES_JSON": json.dumps(
                {"local/pnp_trash": str(tmp_path / "source")}, separators=(",", ":")
            ),
            "COSMOS_BASE_URL": "http://127.0.0.1:8001/v1",
            "COSMOS_MODEL": "cosmos3-nano",
            "COSMOS_API_KEY_ENV": "COSMOS_API_KEY",
            "COSMOS_ENDPOINT_IDENTITY": "h100-cosmos",
            "COSMOS_API_KEY": "test-secret",
            "CONFIRMED_SMOKE_JOB_ID": confirmed_job_id,
            "CONFIRMED_SMOKE_ATTEMPT_ID": ATTEMPT_ID,
            "CONFIRMED_SMOKE_SOURCE_EPISODE_INDEX": "4",
            "CONFIRMED_SMOKE_AUTHORITY_JSON": authority_json,
            "CONFIRMED_SMOKE_AUTHORITY_SHA256": authority_sha256,
            "SMOKE_EVIDENCE_CONFIRMED": f"CONFIRM SMOKE EVIDENCE {confirmed_job_id} {ATTEMPT_ID}",
            "SMOKE_STATUS_JSON": json.dumps(status, separators=(",", ":")),
            "SMOKE_EPISODE_JSON": json.dumps(episode, separators=(",", ":")),
            "FULL_STATUS_JSON": json.dumps(full_status, separators=(",", ":")),
            "FULL_BATCH_POST_MARKER": str(post_marker),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return result, post_marker, worker_marker


def test_full_batch_refetches_smoke_and_fences_post_and_worker(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()

    stale, stale_post, stale_worker = _run_full_batch_with_authority(
        runbook,
        tmp_path / "stale",
        stale_job_id=True,
    )
    assert stale.returncode != 0
    assert not stale_post.exists()
    assert not stale_worker.exists()

    mismatch, mismatch_post, mismatch_worker = _run_full_batch_with_authority(
        runbook,
        tmp_path / "mismatch",
        mismatch_configuration=True,
    )
    assert mismatch.returncode == 1
    assert mismatch_post.exists()
    assert not mismatch_worker.exists()
    assert "full-batch configuration" in mismatch.stderr

    valid, valid_post, valid_worker = _run_full_batch_with_authority(runbook, tmp_path / "valid")
    assert valid.returncode == 0, valid.stderr
    assert valid_post.exists()
    assert valid_worker.exists()

    full_batch = _bash_block_after(runbook, "## Run and monitor the full Cosmos batch")
    assert "episodes/$SMOKE_SOURCE_EPISODE_INDEX" in full_batch
    assert '--expected-full-job-id "$JOB_ID"' in full_batch
    assert full_batch.count('--expected-smoke-job-id "$CONFIRMED_SMOKE_JOB_ID"') == 2


def test_post_publication_gate_stops_on_final_checksum_failure(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    gate = _bash_block_after(runbook, "## Post-publication verification and handoff")
    final = tmp_path / "final"
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    visualizer = tmp_path / "visualizer"
    (final / "meta").mkdir(parents=True)
    (final / "meta" / "info.json").write_text("{}\n")
    (final / "meta" / "info.json").chmod(0o444)
    source.mkdir()
    workspace.mkdir()
    python = visualizer / "backend" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    _write_executable(python, "#!/bin/bash\nexit 0\n")
    gate = (
        gate.replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned", str(final))
        .replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation", str(workspace))
        .replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash", str(source))
        .replace("/home/jihun/work/lerobot-dataset-visualizer", str(visualizer))
    )

    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(
        commands / "sha256sum",
        """#!/bin/bash
case "$*" in
  *curation_checksums.sha256*) exit 9 ;;
  *) exit 0 ;;
esac
""",
    )
    _write_executable(commands / "bun", "#!/bin/bash\nexit 0\n")
    result = subprocess.run(
        ["bash", "-c", gate],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 9


def test_export_gate_reauthenticates_manifest_and_stops_before_export_on_checksum_failure(
    tmp_path: Path,
) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    gate = _bash_block_after(runbook, "## Build, validate, and publish the separate cleaned dataset")
    assert gate.startswith("set -euo pipefail\n")
    assert f"PINNED_SOURCE_MANIFEST_SHA256={APPROVED_SOURCE_MANIFEST_SHA256}" in gate
    assert 'SOURCE_MANIFEST_FILE_COUNT=$(wc -l < "$SOURCE_MANIFEST")' in gate
    assert 'test "$SOURCE_MANIFEST_FILE_COUNT" -ne 190' in gate
    assert 'SOURCE_MANIFEST_SHA256=$(sha256sum "$SOURCE_MANIFEST"' in gate
    assert gate.index("SOURCE_MANIFEST_SHA256=$(sha256sum") < gate.index("EXPORT_RESPONSE=$(curl")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = workspace / "source-files.sha256"
    manifest.write_text("".join(f"{'0' * 64}  file-{index:03d}\n" for index in range(190)))
    source = tmp_path / "source"
    source.mkdir()
    visualizer = tmp_path / "visualizer"
    exporter = visualizer / "backend" / ".venv" / "bin" / "python"
    exporter.parent.mkdir(parents=True)
    _write_executable(exporter, "#!/bin/bash\nexit 0\n")
    gate = gate.replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash", str(source)).replace(
        "/home/jihun/work/lerobot-dataset-visualizer", str(visualizer)
    )

    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(
        commands / "sha256sum",
        f"""#!/bin/bash
case "$1" in
  --check) exit 9 ;;
  *) printf '%s  %s\n' '{APPROVED_SOURCE_MANIFEST_SHA256}' "$1" ;;
esac
""",
    )
    _write_executable(
        commands / "curl",
        (
            "#!/bin/bash\n"
            "printf '%s' "
            '\'{"export_id":"11111111-1111-1111-1111-111111111111",'
            '"approval_snapshot_sha256":'
            '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\'\n'
        ),
    )
    _write_executable(
        commands / "jq",
        """#!/bin/bash
case "$*" in
  *.export_id*) cat >/dev/null; printf '%s\n' '11111111-1111-1111-1111-111111111111' ;;
  *.approval_snapshot_sha256*) cat >/dev/null; printf '%064d\n' 0 ;;
  *) cat ;;
esac
""",
    )
    result = subprocess.run(
        ["bash", "-c", gate],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "CURATION_WORKSPACE": str(workspace),
            "CURATION_OUTPUT": str(tmp_path / "output"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 9
    assert str(tmp_path) not in result.stderr


def test_export_gate_stops_before_export_when_output_already_exists(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    gate = _bash_block_after(runbook, "## Build, validate, and publish the separate cleaned dataset")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = workspace / "source-files.sha256"
    manifest.write_text("".join(f"{'0' * 64}  file-{index:03d}\n" for index in range(190)))
    source = tmp_path / "source"
    source.mkdir()
    visualizer = tmp_path / "visualizer"
    exporter = visualizer / "backend" / ".venv" / "bin" / "python"
    exporter.parent.mkdir(parents=True)
    _write_executable(exporter, "#!/bin/bash\nexit 0\n")
    gate = gate.replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash", str(source)).replace(
        "/home/jihun/work/lerobot-dataset-visualizer", str(visualizer)
    )

    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(
        commands / "sha256sum",
        f"""#!/bin/bash
case "$1" in
  --check) exit 0 ;;
  *) printf '%s  %s\n' '{APPROVED_SOURCE_MANIFEST_SHA256}' "$1" ;;
esac
""",
    )
    _write_executable(
        commands / "curl",
        (
            "#!/bin/bash\n"
            "printf '%s\\n' called >> \"$EXPORT_CALL_MARKER\"\n"
            "printf '%s' "
            '\'{"export_id":"11111111-1111-1111-1111-111111111111",'
            '"approval_snapshot_sha256":'
            '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\'\n'
        ),
    )
    _write_executable(
        commands / "jq",
        """#!/bin/bash
case "$*" in
  *.export_id*) cat >/dev/null; printf '%s\n' '11111111-1111-1111-1111-111111111111' ;;
  *.approval_snapshot_sha256*) cat >/dev/null; printf '%064d\n' 0 ;;
  *) cat ;;
esac
""",
    )

    for output_kind in ("regular", "symlink"):
        output = tmp_path / f"output-{output_kind}"
        if output_kind == "regular":
            output.write_text("occupied\n")
        else:
            target = tmp_path / "existing-output-target"
            target.mkdir(exist_ok=True)
            output.symlink_to(target, target_is_directory=True)
        marker = tmp_path / f"export-called-{output_kind}"
        result = subprocess.run(
            ["bash", "-c", gate],
            cwd=REPOSITORY_ROOT,
            env={
                **os.environ,
                "PATH": f"{commands}:{os.environ['PATH']}",
                "CURATION_WORKSPACE": str(workspace),
                "CURATION_OUTPUT": str(output),
                "EXPORT_CALL_MARKER": str(marker),
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert result.stderr == "FAIL: curation output already exists\n"
        assert str(tmp_path) not in result.stderr
        assert not marker.exists()
