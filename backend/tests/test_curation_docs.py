from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

NON_SECRET_EXPORTS = (
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

BACKEND_START = """cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000"""

FRONTEND_START = """cd /home/jihun/work/lerobot-dataset-visualizer
bun run dev --hostname 127.0.0.1 --port 3000"""

EXTERNAL_RUNTIME_NAMES = {
    "CURATION_BEARER_TOKEN",
    "COSMOS_BASE_URL",
    "COSMOS_MODEL",
    "COSMOS_API_KEY_ENV",
    "COSMOS_ENDPOINT_IDENTITY",
}

PYTHON_SETTINGS_NAMES = (
    "CURATION_DATASET_ALIASES_JSON",
    "CURATION_WORKSPACE",
    "CURATION_OUTPUT",
    "CURATION_BROWSER_ORIGIN",
    "CURATION_BEARER_TOKEN",
    "COSMOS_BASE_URL",
    "COSMOS_MODEL",
    "COSMOS_API_KEY_ENV",
    "COSMOS_ENDPOINT_IDENTITY",
    "ISAAC_GROOT_ROOT",
    "CURATION_BACKEND_HOST",
)


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


def test_docs_assign_full_settings_to_python_processes_and_only_server_subset_to_next() -> None:
    readme = (REPOSITORY_ROOT / "backend" / "README.md").read_text()
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()

    python_entrypoints = (
        "`backend.app:app`, `backend/curation_worker.py`, and "
        "`backend/curation_export.py` all call `CurationSettings.from_env()`"
    )
    next_subset = (
        "Next.js requires only `CURATION_BACKEND_URL`, `CURATION_BEARER_TOKEN`, and `NEXT_PUBLIC_DATASET_URL`"
    )
    capability_reader = (
        "The backend reads the variable named by `COSMOS_API_KEY_ENV` during "
        "batch capability validation before it creates a job"
    )
    python_mapping_subset = "five non-secret mapping values consumed by `CurationSettings`"
    least_privilege_key = (
        "Only the backend and worker receive the actual credential variable named by "
        "`COSMOS_API_KEY_ENV`; the exporter must not receive that secret"
    )
    for document in (readme, runbook):
        normalized = re.sub(r"\s+", " ", document)
        assert python_entrypoints in normalized
        assert next_subset in normalized
        assert capability_reader in normalized
        assert python_mapping_subset in normalized
        assert least_privilege_key in normalized
        assert "all seven non-secret mapping values" not in normalized
        assert "complete exporter launch environment also carries it" not in normalized
    for name in PYTHON_SETTINGS_NAMES:
        assert re.search(rf"(?m)^\| `{name}` .*\| All three Python processes", runbook)
    assert re.search(r"(?m)^\| Variable named by `COSMOS_API_KEY_ENV` .*\| Backend and worker only", runbook)


def test_source_preflight_fails_closed_on_a_sanitized_190_file_fixture(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    preflight = _source_preflight(runbook)
    assert preflight.startswith("set -euo pipefail\n")
    assert "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}" in preflight
    assert "${CURATION_OUTPUT:?FAIL: CURATION_OUTPUT is required}" in preflight

    source = tmp_path / "source"
    source.mkdir()
    for index in range(190):
        (source / f"file-{index:03d}").write_bytes(b"fixture")
    preflight = preflight.replace("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash", str(source))
    result = subprocess.run(
        ["bash", "-c", preflight],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "CURATION_WORKSPACE": str(tmp_path / "workspace"),
            "CURATION_OUTPUT": str(tmp_path / "output"),
            "APPROVED_SOURCE_MANIFEST_SHA256": "0" * 64,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: source regular-file count must be 189; found 190\n"
    assert str(tmp_path) not in result.stderr


def test_source_preflight_rejects_unapproved_prospective_hash_before_backend_start(tmp_path: Path) -> None:
    runbook = (REPOSITORY_ROOT / "docs" / "pnp-trash-curation-runbook.md").read_text()
    preflight = _source_preflight(runbook)
    assert "${APPROVED_SOURCE_MANIFEST_SHA256:?FAIL: approved source manifest SHA-256 is required}" in preflight
    assert "PROSPECTIVE_SOURCE_MANIFEST_SHA256" in preflight
    assert "APPROVED_SOURCE_MANIFEST_SHA256=$(" not in preflight
    assert "from backend.curation.source import _manifest_bytes" in preflight
    assert "_manifest_bytes(root)" in preflight
    assert "os.walk" not in preflight
    assert runbook.index("PROSPECTIVE_SOURCE_MANIFEST_SHA256") < runbook.index(
        "## Start the two loopback services"
    )

    source = tmp_path / "source"
    source.mkdir()
    for index in range(189):
        (source / f"file-{index:03d}").write_bytes(b"fixture")
    preflight = preflight.replace(
        "/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash", str(source)
    ).replace("/home/jihun/work/lerobot-dataset-visualizer", str(REPOSITORY_ROOT))
    result = subprocess.run(
        ["bash", "-c", preflight],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "CURATION_WORKSPACE": str(tmp_path / "workspace"),
            "CURATION_OUTPUT": str(tmp_path / "output"),
            "APPROVED_SOURCE_MANIFEST_SHA256": "0" * 64,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "FAIL: prospective source manifest SHA-256 does not match approved record\n"
    assert str(tmp_path) not in result.stderr


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
) -> subprocess.CompletedProcess[str]:
    smoke = _bash_blocks_in_section(runbook, "## Cosmos capability and one-episode smoke")[1]
    visualizer = tmp_path / "visualizer"
    worker = visualizer / "backend" / ".venv" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    _write_executable(worker, f"#!/bin/bash\nexit {worker_exit}\n")
    smoke = smoke.replace("/home/jihun/work/lerobot-dataset-visualizer", str(visualizer))

    commands = tmp_path / "bin"
    commands.mkdir()
    _write_executable(
        commands / "curl",
        """#!/bin/bash
case "$*" in
  */batches/*) printf '{"state":"%s"}' "$SMOKE_FINAL_STATUS" ;;
  */batches*) printf '%s' '{"job_id":"job-smoke"}' ;;
  *) printf '%s' '{}' ;;
esac
""",
    )
    _write_executable(
        commands / "jq",
        """#!/bin/bash
case "$*" in
  *.job_id*) cat >/dev/null; printf '%s\n' 'job-smoke' ;;
  *.state*) cat >/dev/null; printf '%s\n' "$SMOKE_FINAL_STATUS" ;;
  *) cat ;;
esac
""",
    )
    return subprocess.run(
        ["bash", "-c", smoke],
        cwd=REPOSITORY_ROOT,
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "SMOKE_FINAL_STATUS": final_status,
        },
        capture_output=True,
        text=True,
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
    assert result.stderr == "FAIL: one-episode Cosmos smoke ended in unexpected state: failed\n"
    assert str(tmp_path) not in result.stderr


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
