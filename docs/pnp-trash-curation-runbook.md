# PnP-trash Cosmos curation runbook

This runbook operates the approved `local/pnp_trash` workflow: Cosmos3-Nano
proposes seven temporal phases, a human approves or rejects every episode, and
the exporter writes a separate v2.1 dataset at
`/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned`.

The source is immutable. Never edit, rename, hardlink from, or write metadata
under `/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash`. Do not
publish to Hugging Face and do not delete workspace/staging evidence during
this procedure.

## Runtime contract

Use this non-secret mapping verbatim:

```bash
export CURATION_REPO_ROOT=/home/jihun/work/GR00T-WholeBodyControl/worktrees/lerobot-dataset-visualizer-pnp-trash
export CURATION_DATASET_ALIASES_JSON='{"local/pnp_trash":"/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash"}'
export CURATION_WORKSPACE=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation
export CURATION_OUTPUT=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned
export CURATION_BROWSER_ORIGIN=http://127.0.0.1:3000
export CURATION_BACKEND_URL=http://127.0.0.1:8000
export NEXT_PUBLIC_DATASET_URL=http://127.0.0.1:8000/api/local-datasets
export ISAAC_GROOT_ROOT=/home/jihun/work/Isaac-GR00T
```

`backend.app:app` uses `CurationSettings.from_env()`,
`backend/curation_worker.py` uses `WorkerSettings.from_env()`, and
`backend/curation_export.py` uses `ExportSettings.from_env()`. FastAPI alone
requires output, browser origin, bearer, and backend-host configuration. The
worker requires aliases, workspace, Cosmos base/model/API-key environment
variable name/endpoint identity, and its frozen limits. The exporter requires
aliases, workspace, Isaac-GR00T root, Cosmos model, and endpoint identity.
Only the backend and worker receive the actual credential variable named by
`COSMOS_API_KEY_ENV`; the exporter must not receive that secret. The backend
reads the variable named by `COSMOS_API_KEY_ENV` during batch capability
validation before it creates a job; the worker reads it only for Cosmos calls.
The exporter receives neither the bearer nor Cosmos base/API-key configuration
and makes no Cosmos call. Neither CLI accepts a secret argument.

Next.js requires only `CURATION_BACKEND_URL`, `CURATION_BEARER_TOKEN`, and
`NEXT_PUBLIC_DATASET_URL`. Its bearer value must exactly match the backend's;
the other Python settings and the Cosmos credential must not be supplied to
browser code.

| Variable                               | Visibility                                     | Owner                               | Startup validation                                                                                                    |
| -------------------------------------- | ---------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `CURATION_REPO_ROOT`                   | Non-secret, operator shell                     | Every documented repository command | Canonical absolute clean Git worktree authenticated by the checkout preflight                                         |
| `CURATION_DATASET_ALIASES_JSON`        | Non-secret, server-only                        | FastAPI, worker, and exporter       | Nonempty JSON object; every key is one `org/dataset` alias and every value is an absolute, existing dataset directory |
| `CURATION_WORKSPACE`                   | Non-secret, server-only                        | FastAPI, worker, and exporter       | Canonical absolute path, separate from every source and output; CLI value must equal the persisted workspace          |
| `CURATION_OUTPUT`                      | Non-secret, server-only                        | FastAPI only                        | Canonical absolute path, separate from source/workspace; must be absent before the no-clobber export                  |
| `CURATION_BROWSER_ORIGIN`              | Non-secret, server-only                        | FastAPI only                        | Exactly one HTTP(S) origin with no credentials, path, query, fragment, or comma-separated alternatives                |
| `CURATION_BACKEND_URL`                 | Non-secret, server-only                        | Next.js only                        | Absolute HTTP(S) URL with no credentials, query, or fragment; this run requires the exact loopback mapping above      |
| `NEXT_PUBLIC_DATASET_URL`              | Public                                         | Next.js and browser URL builder     | Exact loopback local-asset prefix above; it carries no credential                                                     |
| `ISAAC_GROOT_ROOT`                     | Non-secret, server-only                        | FastAPI and exporter                | Canonical absolute path; the preflight must import the exact loader/config and find `gr00t/data/stats.py`             |
| `CURATION_BACKEND_HOST`                | Non-secret, optional                           | FastAPI only                        | Defaults to `127.0.0.1`; must parse as loopback or literal `localhost`                                                |
| `CURATION_BEARER_TOKEN`                | Secret                                         | FastAPI and Next.js only            | Required and nonempty; Next.js and backend values must match; never browser-visible                                   |
| `COSMOS_BASE_URL`                      | External/private configuration                 | FastAPI and worker only             | Absolute HTTP(S), no whitespace, credentials, query, fragment, invalid port, or redirect following                    |
| `COSMOS_MODEL`                         | External/private configuration                 | FastAPI, worker, and exporter       | Required, nonempty, and exactly present in the configured `/v1/models` response                                       |
| `COSMOS_API_KEY_ENV`                   | External/private configuration; names a secret | FastAPI and worker only             | Required distinct variable name; reserved process/configuration collisions are rejected                               |
| Variable named by `COSMOS_API_KEY_ENV` | Secret                                         | Backend and worker only             | Required for backend batch creation and worker calls; prohibited from the exporter environment                        |
| `COSMOS_ENDPOINT_IDENTITY`             | External/private configuration                 | All three Python processes          | Required and nonempty; stable operator identity for the existing H100 service                                         |

The operator's existing server/runtime configuration must provide these names;
their values do not belong in the repository or this runbook:

- `CURATION_BEARER_TOKEN`: present only in the FastAPI and Next.js server
  environments; their values must match.
- `COSMOS_BASE_URL`: OpenAI-compatible H100 endpoint ending at `/v1`.
- `COSMOS_MODEL`: exact model ID returned by `/v1/models`.
- `COSMOS_API_KEY_ENV`: name of the external environment variable that holds
  the Cosmos API key. The key itself is read indirectly by the backend before
  batch creation and by the worker during Cosmos calls; do not supply the
  target secret to the exporter. The target may be a distinct name such as
  `COSMOS_API_KEY`, but it must not collide with an inherited process name,
  any `CURATION_*`/`NEXT_PUBLIC_*` name, or the named Cosmos/Isaac settings in
  the table.
- `COSMOS_ENDPOINT_IDENTITY`: stable non-secret operator identity recorded in
  provenance. For this approved run it is exactly
  `h100-cosmos3-nano-vllm-0.23.0@sha256:f37691f675bb82f734f606de8af90e777d3f80a20b120e699fd43fd10e60b8d7`, combining the H100 service identity, installed vLLM
  `0.23.0`, and immutable container image ID
  `sha256:f37691f675bb82f734f606de8af90e777d3f80a20b120e699fd43fd10e60b8d7`.

`CURATION_BACKEND_HOST` is FastAPI-only, optional, and defaults to
`127.0.0.1`; it may only be a loopback address. The backend owns dataset aliases, workspace/output paths,
the browser-origin allowlist, Cosmos job snapshots, and Isaac-GR00T path. The
worker owns 2 fps sampling and Cosmos calls. The exporter owns staging,
validation, and no-clobber publication. Next.js owns `CURATION_BACKEND_URL`
and server-side bearer injection. Browser code may receive only
`NEXT_PUBLIC_DATASET_URL`.

Never define a `NEXT_PUBLIC_*` token, API key, Cosmos base URL, model, or
endpoint identity. Do not use `set -x` in a shell containing the external
credentials.

### Least-privilege Python process launchers

Define these launchers in every operator shell that will run a worker or
exporter. `env -i` prevents unrelated ambient variables from crossing the
process boundary. The worker receives the credential named by
`COSMOS_API_KEY_ENV`; the exporter receives neither that dynamic credential
nor its name or Cosmos base URL.

```bash
run_curation_worker() {
  : "${CURATION_REPO_ROOT:?FAIL: CURATION_REPO_ROOT is required}"
  : "${CURATION_DATASET_ALIASES_JSON:?FAIL: CURATION_DATASET_ALIASES_JSON is required}"
  : "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}"
  : "${COSMOS_BASE_URL:?FAIL: COSMOS_BASE_URL is required}"
  : "${COSMOS_MODEL:?FAIL: COSMOS_MODEL is required}"
  : "${COSMOS_API_KEY_ENV:?FAIL: COSMOS_API_KEY_ENV is required}"
  : "${COSMOS_ENDPOINT_IDENTITY:?FAIL: COSMOS_ENDPOINT_IDENTITY is required}"
  case "$COSMOS_API_KEY_ENV" in
    PATH|HOME|USER|LOGNAME|SHELL|PWD|OLDPWD|TMPDIR|PYTHONPATH|PYTHONHOME|LD_PRELOAD|LD_LIBRARY_PATH|HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY|SSL_CERT_FILE|SSL_CERT_DIR|REQUESTS_CA_BUNDLE|CURATION_*|NEXT_PUBLIC_*|COSMOS_BASE_URL|COSMOS_MODEL|COSMOS_API_KEY_ENV|COSMOS_ENDPOINT_IDENTITY|ISAAC_GROOT_ROOT)
      printf '%s\n' 'FAIL: configured Cosmos credential target is reserved' >&2
      return 1
      ;;
  esac
  if [[ ! "$COSMOS_API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
    || test -z "${!COSMOS_API_KEY_ENV:-}"; then
    printf '%s\n' 'FAIL: configured Cosmos credential is unavailable' >&2
    return 1
  fi
  env -i \
    PATH="$PATH" \
    CURATION_DATASET_ALIASES_JSON="$CURATION_DATASET_ALIASES_JSON" \
    CURATION_WORKSPACE="$CURATION_WORKSPACE" \
    COSMOS_BASE_URL="$COSMOS_BASE_URL" \
    COSMOS_MODEL="$COSMOS_MODEL" \
    COSMOS_API_KEY_ENV="$COSMOS_API_KEY_ENV" \
    COSMOS_ENDPOINT_IDENTITY="$COSMOS_ENDPOINT_IDENTITY" \
    "$COSMOS_API_KEY_ENV=${!COSMOS_API_KEY_ENV}" \
    "$CURATION_REPO_ROOT/backend/.venv/bin/python" \
    "$CURATION_REPO_ROOT/backend/curation_worker.py" "$@"
}

run_curation_exporter() {
  : "${CURATION_REPO_ROOT:?FAIL: CURATION_REPO_ROOT is required}"
  : "${CURATION_DATASET_ALIASES_JSON:?FAIL: CURATION_DATASET_ALIASES_JSON is required}"
  : "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}"
  : "${COSMOS_MODEL:?FAIL: COSMOS_MODEL is required}"
  : "${COSMOS_ENDPOINT_IDENTITY:?FAIL: COSMOS_ENDPOINT_IDENTITY is required}"
  : "${ISAAC_GROOT_ROOT:?FAIL: ISAAC_GROOT_ROOT is required}"
  env -i \
    PATH="$PATH" \
    CURATION_DATASET_ALIASES_JSON="$CURATION_DATASET_ALIASES_JSON" \
    CURATION_WORKSPACE="$CURATION_WORKSPACE" \
    COSMOS_MODEL="$COSMOS_MODEL" \
    COSMOS_ENDPOINT_IDENTITY="$COSMOS_ENDPOINT_IDENTITY" \
    ISAAC_GROOT_ROOT="$ISAAC_GROOT_ROOT" \
    "$CURATION_REPO_ROOT/backend/.venv/bin/python" \
    "$CURATION_REPO_ROOT/backend/curation_export.py" "$@"
}
```

## Install and static regression gates

### 0. Trusted implementation checkout

Before executing repository code, obtain the full 40-character commit SHA
from an independently reviewed approval record and enter it shell-locally. Do
not derive this trust value from the checkout being authenticated:

```bash
read -r -p 'Paste independently approved curation commit SHA: ' APPROVED_CURATION_COMMIT_SHA
```

#### Checkout authentication gate

Authenticate the one checkout used by every later command:

```bash
set -euo pipefail
: "${CURATION_REPO_ROOT:?FAIL: CURATION_REPO_ROOT is required}"
: "${APPROVED_CURATION_COMMIT_SHA:?FAIL: approved curation commit SHA is required}"
if [[ "$CURATION_REPO_ROOT" != /* ]] || test ! -d "$CURATION_REPO_ROOT"; then
  printf '%s\n' 'FAIL: curation checkout must be an absolute directory' >&2
  exit 1
fi
CANONICAL_CURATION_REPO_ROOT=$(realpath "$CURATION_REPO_ROOT")
if test "$CANONICAL_CURATION_REPO_ROOT" != "$CURATION_REPO_ROOT"; then
  printf '%s\n' 'FAIL: curation checkout path must already be canonical' >&2
  exit 1
fi
if test ! -e "$CURATION_REPO_ROOT/.git" || test -L "$CURATION_REPO_ROOT/.git"; then
  printf '%s\n' 'FAIL: curation checkout is not a Git worktree' >&2
  exit 1
fi
if [[ ! "$APPROVED_CURATION_COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  printf '%s\n' 'FAIL: approved curation commit SHA must be 40 lowercase hex characters' >&2
  exit 1
fi
if test "$(git -C "$CURATION_REPO_ROOT" rev-parse --is-inside-work-tree)" != true \
  || test "$(git -C "$CURATION_REPO_ROOT" rev-parse --show-toplevel)" != "$CURATION_REPO_ROOT"; then
  printf '%s\n' 'FAIL: curation checkout is not the Git worktree root' >&2
  exit 1
fi
if test "$(git -C "$CURATION_REPO_ROOT" rev-parse HEAD)" != "$APPROVED_CURATION_COMMIT_SHA"; then
  printf '%s\n' 'FAIL: checkout HEAD does not match the independently approved commit' >&2
  exit 1
fi
if ! git -C "$CURATION_REPO_ROOT" merge-base --is-ancestor 60ef88c \
  "$APPROVED_CURATION_COMMIT_SHA"; then
  printf '%s\n' 'FAIL: approved checkout predates the minimum curation baseline' >&2
  exit 1
fi
if test -n "$(git -C "$CURATION_REPO_ROOT" status --porcelain --untracked-files=all)"; then
  printf '%s\n' 'FAIL: curation checkout is not clean' >&2
  exit 1
fi
```

From the repository root, create the backend environment and install frontend
dependencies if needed:

```bash
cd "$CURATION_REPO_ROOT"
python -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
bun install
```

Before real data operations, run both complete gates:

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python -m pytest -q backend/tests
```

Expected: PASS, including `test_legacy_v31_regression.py` and the curation
modules.

```bash
cd "$CURATION_REPO_ROOT"
bun run format:check && bun run validate
```

Expected: PASS. Both commands are read-only checks; this gate must not rewrite
the authenticated checkout.

#### Post-static checkout reauthentication gate

Reauthenticate the exact HEAD and clean tree after installation and every
static command, before any runtime preflight or service start:

```bash
set -euo pipefail
: "${CURATION_REPO_ROOT:?FAIL: CURATION_REPO_ROOT is required}"
: "${APPROVED_CURATION_COMMIT_SHA:?FAIL: approved curation commit SHA is required}"
if test "$(git -C "$CURATION_REPO_ROOT" rev-parse HEAD)" != "$APPROVED_CURATION_COMMIT_SHA"; then
  printf '%s\n' 'FAIL: checkout HEAD changed during install or static gates' >&2
  exit 1
fi
if test -n "$(git -C "$CURATION_REPO_ROOT" status --porcelain --untracked-files=all)"; then
  printf '%s\n' 'FAIL: curation checkout is not clean after static gates' >&2
  exit 1
fi
```

Configured FastAPI startup creates the canonical source manifest and
initializes curation.sqlite3 before serving. Task 14 integration is not
complete: the approved-source loopback and browser smoke remains pending until
the operator loads the secure runtime configuration and executes the remaining
runbook gates.

## Filesystem and dependency preflight

Run this section before starting a new curation. Stop on any failed assertion.
The user-approved 2026-08-28 source authority contains 190 immutable regular
files. Its independently reviewed canonical manifest SHA-256 is
`5962d8630f06e6260adbae15a3d7ee5f0a1a745c3a12466add8722c2e0da9577`.
The top-level ancillary `pnp_trash.xlsx` is 13,644 bytes with SHA-256
`989f6968e5cf8ee0972b850199c948dd75ce140480c82cbe368053cde6ab34c9`.
It is immutable source evidence, is copied as an ancillary asset, and is not
annotation authority. In the same shell that runs source preflight, enter the
independently approved manifest value as a shell-local variable:

```bash
read -r -p 'Paste independently approved 190-file manifest SHA-256: ' APPROVED_SOURCE_MANIFEST_SHA256
```

Never populate `APPROVED_SOURCE_MANIFEST_SHA256` with `sha256sum`, command
substitution, or any value derived from the live preflight tree. The live tree
is the candidate being authenticated, not the trust source.

### 1. Source, final destination, ownership, and free space

```bash
set -euo pipefail
: "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}"
: "${CURATION_OUTPUT:?FAIL: CURATION_OUTPUT is required}"
: "${APPROVED_SOURCE_MANIFEST_SHA256:?FAIL: approved source manifest SHA-256 is required}"
PINNED_SOURCE_MANIFEST_SHA256=5962d8630f06e6260adbae15a3d7ee5f0a1a745c3a12466add8722c2e0da9577
PINNED_ANCILLARY_NAME=pnp_trash.xlsx
PINNED_ANCILLARY_SHA256=989f6968e5cf8ee0972b850199c948dd75ce140480c82cbe368053cde6ab34c9
PINNED_ANCILLARY_SIZE=13644
if [[ ! "$APPROVED_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: approved source manifest SHA-256 must be 64 lowercase hex characters' >&2
  exit 1
fi
if test "$APPROVED_SOURCE_MANIFEST_SHA256" != "$PINNED_SOURCE_MANIFEST_SHA256"; then
  printf '%s\n' 'FAIL: entered source manifest SHA-256 does not match the approved record' >&2
  exit 1
fi

SOURCE_DATASET=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash
test -d "$SOURCE_DATASET"
test ! -L "$SOURCE_DATASET"
SOURCE_FILE_COUNT=$(find -P "$SOURCE_DATASET" -type f | wc -l)
if test "$SOURCE_FILE_COUNT" -ne 190; then
  printf 'FAIL: source regular-file count must be 190; found %s\n' "$SOURCE_FILE_COUNT" >&2
  exit 1
fi
test -z "$(find -P "$SOURCE_DATASET" -type l -print -quit)"

test -x "$CURATION_REPO_ROOT/backend/.venv/bin/python"
PROSPECTIVE_SOURCE_AUTHORITY=$(
  PYTHONPATH="$CURATION_REPO_ROOT" SOURCE_DATASET="$SOURCE_DATASET" \
    PINNED_ANCILLARY_NAME="$PINNED_ANCILLARY_NAME" \
    "$CURATION_REPO_ROOT/backend/.venv/bin/python" - <<'PY'
import hashlib
import os
from pathlib import Path

from backend.curation.source import _manifest_bytes

root = Path(os.environ["SOURCE_DATASET"]).resolve(strict=True)
manifest, hashes, identities = _manifest_bytes(root)
ancillary_name = os.environ["PINNED_ANCILLARY_NAME"]
print(hashlib.sha256(manifest).hexdigest())
print(hashes.get(ancillary_name, ""))
identity = identities.get(ancillary_name)
print(-1 if identity is None else identity.size)
PY
)
mapfile -t SOURCE_AUTHORITY_FIELDS <<< "$PROSPECTIVE_SOURCE_AUTHORITY"
if test "${#SOURCE_AUTHORITY_FIELDS[@]}" -ne 3; then
  printf '%s\n' 'FAIL: prospective source authority output is invalid' >&2
  exit 1
fi
PROSPECTIVE_SOURCE_MANIFEST_SHA256=${SOURCE_AUTHORITY_FIELDS[0]}
PROSPECTIVE_ANCILLARY_SHA256=${SOURCE_AUTHORITY_FIELDS[1]}
PROSPECTIVE_ANCILLARY_SIZE=${SOURCE_AUTHORITY_FIELDS[2]}
if [[ ! "$PROSPECTIVE_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: prospective source manifest SHA-256 is invalid' >&2
  exit 1
fi
if test "$PROSPECTIVE_SOURCE_MANIFEST_SHA256" != "$APPROVED_SOURCE_MANIFEST_SHA256"; then
  printf '%s\n' 'FAIL: prospective source manifest SHA-256 does not match approved record' >&2
  exit 1
fi
if test "$PROSPECTIVE_ANCILLARY_SHA256" != "$PINNED_ANCILLARY_SHA256" \
  || test "$PROSPECTIVE_ANCILLARY_SIZE" != "$PINNED_ANCILLARY_SIZE"; then
  printf '%s\n' 'FAIL: pnp_trash.xlsx does not match the approved ancillary asset' >&2
  exit 1
fi

test ! -e "$CURATION_OUTPUT"
test ! -L "$CURATION_OUTPUT"

WORKSPACE_PARENT=$(dirname "$CURATION_WORKSPACE")
OUTPUT_PARENT=$(dirname "$CURATION_OUTPUT")
test -d "$WORKSPACE_PARENT" && test -O "$WORKSPACE_PARENT" && test -w "$WORKSPACE_PARENT"
test -d "$OUTPUT_PARENT" && test -O "$OUTPUT_PARENT" && test -w "$OUTPUT_PARENT"
if test -e "$CURATION_WORKSPACE"; then
  test -d "$CURATION_WORKSPACE" && test ! -L "$CURATION_WORKSPACE"
  test -O "$CURATION_WORKSPACE" && test -w "$CURATION_WORKSPACE"
fi
du -sk "$SOURCE_DATASET"
df -Pk "$WORKSPACE_PARENT" "$OUTPUT_PARENT"
```

Require at least two source-tree sizes of free capacity on both relevant
filesystems, plus local operational headroom for reports and GR00T statistics.
An existing workspace is resumable evidence: inspect it; never clear it to
force a fresh run.

The 190-file count, canonical manifest digest, and ancillary XLSX digest and
size are one indivisible source contract. Do not delete, move, rewrite, ignore,
or parse the XLSX as annotation authority. Any mismatch is a hard stop and
requires a new explicit source approval; never bless a value computed from the
candidate live tree.

### 2. Linux atomic no-clobber support

This executable probe creates and removes two private temporary directories in
the output parent and proves both the `EEXIST` and successful
`renameat2(RENAME_NOREPLACE)` paths:

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python -c 'from pathlib import Path; from backend.curation.publication import preflight_rename_noreplace; preflight_rename_noreplace(Path("/home/jihun/work/GR00T-WholeBodyControl/outputs")); print("renameat2 RENAME_NOREPLACE: PASS")'
```

Any `publish_noreplace_unsupported` result is fatal. There is no fallback to
`os.replace` or a copying publish.

### 3. Isaac-GR00T import and configuration

```bash
set -euo pipefail
if test -z "${ISAAC_GROOT_ROOT:-}"; then
  printf '%s\n' 'FAIL: ISAAC_GROOT_ROOT is required' >&2
  exit 1
fi
if test ! -x "$ISAAC_GROOT_ROOT/.venv/bin/python"; then
  printf '%s\n' 'FAIL: Isaac-GR00T Python is unavailable' >&2
  exit 1
fi
if test ! -f "$ISAAC_GROOT_ROOT/gr00t/data/stats.py"; then
  printf '%s\n' 'FAIL: Isaac-GR00T stats.py is unavailable' >&2
  exit 1
fi
cd "$ISAAC_GROOT_ROOT"
if ! .venv/bin/python -c 'import gr00t.data.dataset.lerobot_episode_loader; import gr00t.configs.data.embodiment_configs; print("Isaac-GR00T loader/config import: PASS")'; then
  printf '%s\n' 'FAIL: Isaac-GR00T loader/config import failed' >&2
  exit 1
fi
```

The export later invokes this checkout with embodiment
`UNITREE_G1_SONIC`; do not substitute another Python environment or config.

### 4. FFmpeg/PyAV frame-count agreement

Run the check against all 92 immutable source videos:

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess
import av

root = Path("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash")
videos = sorted(root.glob("videos/**/*.mp4"))
if len(videos) != 92:
    raise SystemExit(f"FAIL: expected 92 source videos; found {len(videos)}")
for video in videos:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(video),
        ],
        text=True,
    ).strip()
    ffmpeg_count = int(raw)
    with av.open(str(video), mode="r") as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise SystemExit(
                f"FAIL: {video.relative_to(root)} must contain exactly one video stream"
            )
        pyav_count = sum(1 for _ in container.decode(streams[0]))
    if ffmpeg_count != pyav_count:
        raise SystemExit(
            f"FAIL: {video.relative_to(root)} frame-count disagreement: "
            f"ffprobe={ffmpeg_count}, PyAV={pyav_count}"
        )
print(f"FFmpeg/PyAV frame-count agreement: PASS ({len(videos)} videos)")
PY
```

The one-episode Cosmos smoke below additionally proves parquet timestamps,
source FPS, row count, decoded video count, and the deterministic 2 fps sample
indices agree.

## Start the two loopback services

In the fully configured backend terminal, use the repository-root module path
exactly:

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Start Next.js in a second terminal containing only `CURATION_BACKEND_URL`,
`CURATION_BEARER_TOKEN`, and `NEXT_PUBLIC_DATASET_URL`:

```bash
cd "$CURATION_REPO_ROOT"
bun run dev --hostname 127.0.0.1 --port 3000
```

Do not add `--host 0.0.0.0`, `--hostname 0.0.0.0`, or a public tunnel.

### Listener and exact-origin checks

```bash
set -euo pipefail
LISTENERS=$(ss -ltnp)
if ! printf '%s\n' "$LISTENERS" | rg -q '(^|[[:space:]])127\.0\.0\.1:3000([[:space:]]|$)'; then
  printf '%s\n' 'FAIL: Next.js listener 127.0.0.1:3000 is unavailable' >&2
  exit 1
fi
if ! printf '%s\n' "$LISTENERS" | rg -q '(^|[[:space:]])127\.0\.0\.1:8000([[:space:]]|$)'; then
  printf '%s\n' 'FAIL: FastAPI listener 127.0.0.1:8000 is unavailable' >&2
  exit 1
fi
if printf '%s\n' "$LISTENERS" | rg -q '(^|[[:space:]])(\*|0\.0\.0\.0|\[::\]):3000([[:space:]]|$)'; then
  printf '%s\n' 'FAIL: Next.js has a wildcard listener on port 3000' >&2
  exit 1
fi
if printf '%s\n' "$LISTENERS" | rg -q '(^|[[:space:]])(\*|0\.0\.0\.0|\[::\]):8000([[:space:]]|$)'; then
  printf '%s\n' 'FAIL: FastAPI has a wildcard listener on port 8000' >&2
  exit 1
fi

ASSET_INFO=http://127.0.0.1:8000/api/local-datasets/local/pnp_trash/resolve/main/meta/info.json
TRUSTED_CORS_HEADERS=$(curl -fsS -D - -o /dev/null -H 'Origin: http://127.0.0.1:3000' "$ASSET_INFO")
if ! printf '%s\n' "$TRUSTED_CORS_HEADERS" | tr -d '\r' \
  | rg -qi '^access-control-allow-origin: http://127\.0\.0\.1:3000$'; then
  printf '%s\n' 'FAIL: configured browser origin was not allowed exactly' >&2
  exit 1
fi
UNTRUSTED_CORS_HEADERS=$(curl -fsS -D - -o /dev/null -H 'Origin: http://localhost:3000' "$ASSET_INFO")
if printf '%s\n' "$UNTRUSTED_CORS_HEADERS" | tr -d '\r' \
  | rg -qi '^access-control-allow-origin:'; then
  printf '%s\n' 'FAIL: unconfigured browser origin was allowed' >&2
  exit 1
fi
```

The configured browser origin is exactly `http://127.0.0.1:3000`, not the
textually different `http://localhost:3000`.

### Persisted source manifest

Backend startup creates an immutable workspace manifest. In the verification
shell, re-enter `APPROVED_SOURCE_MANIFEST_SHA256` from the same independent
approved record used before startup. Verify the persisted fingerprint equals
that authority; do not promote a newly printed live hash into the approved
value:

```bash
set -euo pipefail
: "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}"
: "${APPROVED_SOURCE_MANIFEST_SHA256:?FAIL: approved source manifest SHA-256 is required}"
PINNED_SOURCE_MANIFEST_SHA256=5962d8630f06e6260adbae15a3d7ee5f0a1a745c3a12466add8722c2e0da9577
if [[ ! "$APPROVED_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: approved source manifest SHA-256 must be 64 lowercase hex characters' >&2
  exit 1
fi
if test "$APPROVED_SOURCE_MANIFEST_SHA256" != "$PINNED_SOURCE_MANIFEST_SHA256"; then
  printf '%s\n' 'FAIL: entered source manifest SHA-256 does not match the approved record' >&2
  exit 1
fi

SOURCE_MANIFEST="$CURATION_WORKSPACE/source-files.sha256"
test -f "$SOURCE_MANIFEST"
SOURCE_MANIFEST_FILE_COUNT=$(wc -l < "$SOURCE_MANIFEST")
if test "$SOURCE_MANIFEST_FILE_COUNT" -ne 190; then
  printf 'FAIL: persisted source manifest must contain 190 files; found %s\n' "$SOURCE_MANIFEST_FILE_COUNT" >&2
  exit 1
fi
SOURCE_MANIFEST_SHA256=$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')
if test "$SOURCE_MANIFEST_SHA256" != "$APPROVED_SOURCE_MANIFEST_SHA256"; then
  printf '%s\n' 'FAIL: persisted source manifest SHA-256 does not match approved record' >&2
  exit 1
fi
printf 'source manifest sha256: %s\n' "$SOURCE_MANIFEST_SHA256"
cd /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash
sha256sum --check "$SOURCE_MANIFEST"
```

Every entry must report `OK`. Later UI/API fingerprints and final provenance
must equal the recorded `SOURCE_MANIFEST_SHA256`.

## Local asset integration smoke

These checks deliberately use the unauthenticated read-only asset server. No
request in this section may carry either the HF token or curation bearer.

```bash
set -euo pipefail
ASSET_ROOT=http://127.0.0.1:8000/api/local-datasets/local/pnp_trash/resolve/main
SMOKE_TMP=$(mktemp -d)
cleanup_local_asset_smoke() {
  rm -r -- "$SMOKE_TMP"
}
trap cleanup_local_asset_smoke EXIT

header_value() {
  local wanted=$1
  local header_file=$2
  awk -v wanted="$wanted" '
    { sub(/\r$/, "") }
    tolower(substr($0, 1, length(wanted) + 1)) == tolower(wanted ":") {
      sub(/^[^:]*:[[:space:]]*/, "")
      value = $0
    }
    END { print value }
  ' "$header_file"
}

INFO_STATUS=$(curl -sS -o "$SMOKE_TMP/info.body" -w '%{http_code}' \
  "$ASSET_ROOT/meta/info.json")
if test "$INFO_STATUS" != 200; then
  printf 'FAIL: info.json returned HTTP %s\n' "$INFO_STATUS" >&2
  exit 1
fi
if ! jq -e \
  '.codebase_version == "v2.1" and .total_episodes == 92 and .fps == 50' \
  "$SMOKE_TMP/info.body" >/dev/null; then
  printf '%s\n' 'FAIL: info.json schema does not match the approved source' >&2
  exit 1
fi

# -I performs a true HEAD request, so curl does not download a response body.
HEAD_STATUS=$(curl -sS -I -D "$SMOKE_TMP/head.headers" -o /dev/null \
  -w '%{http_code}' "$ASSET_ROOT/meta/info.json")
if test "$HEAD_STATUS" != 200; then
  printf 'FAIL: info.json HEAD returned HTTP %s\n' "$HEAD_STATUS" >&2
  exit 1
fi
HEAD_ACCEPT_RANGES=$(header_value Accept-Ranges "$SMOKE_TMP/head.headers")
HEAD_CONTENT_LENGTH=$(header_value Content-Length "$SMOKE_TMP/head.headers")
HEAD_ETAG=$(header_value ETag "$SMOKE_TMP/head.headers")
INFO_BODY_LENGTH=$(wc -c < "$SMOKE_TMP/info.body")
if test "$HEAD_ACCEPT_RANGES" != bytes; then
  printf '%s\n' 'FAIL: info.json HEAD Accept-Ranges must be bytes' >&2
  exit 1
fi
if [[ ! "$HEAD_CONTENT_LENGTH" =~ ^[0-9]+$ ]] \
  || test "$HEAD_CONTENT_LENGTH" -ne "$INFO_BODY_LENGTH"; then
  printf '%s\n' 'FAIL: info.json HEAD Content-Length is not exact' >&2
  exit 1
fi
if [[ ! "$HEAD_ETAG" =~ ^\"[0-9a-f]{64}\"$ ]]; then
  printf '%s\n' 'FAIL: info.json HEAD ETag is invalid' >&2
  exit 1
fi

# Parquet footer suffix range: 206 and exact Content-Range/Content-Length.
PARQUET_STATUS=$(curl -sS -D "$SMOKE_TMP/parquet.headers" \
  -o "$SMOKE_TMP/parquet.body" -w '%{http_code}' -H 'Range: bytes=-8' \
  "$ASSET_ROOT/data/chunk-000/episode_000000.parquet")
if test "$PARQUET_STATUS" != 206; then
  printf 'FAIL: parquet range returned HTTP %s\n' "$PARQUET_STATUS" >&2
  exit 1
fi
PARQUET_CONTENT_RANGE=$(header_value Content-Range "$SMOKE_TMP/parquet.headers")
PARQUET_CONTENT_LENGTH=$(header_value Content-Length "$SMOKE_TMP/parquet.headers")
PARQUET_BODY_LENGTH=$(wc -c < "$SMOKE_TMP/parquet.body")
if test "$PARQUET_CONTENT_LENGTH" != 8 || test "$PARQUET_BODY_LENGTH" -ne 8; then
  printf '%s\n' 'FAIL: parquet suffix range length must be exactly 8 bytes' >&2
  exit 1
fi
if [[ ! "$PARQUET_CONTENT_RANGE" =~ ^bytes\ ([0-9]+)-([0-9]+)/([0-9]+)$ ]]; then
  printf '%s\n' 'FAIL: parquet suffix Content-Range is invalid' >&2
  exit 1
fi
PARQUET_START=$((10#${BASH_REMATCH[1]}))
PARQUET_END=$((10#${BASH_REMATCH[2]}))
PARQUET_TOTAL=$((10#${BASH_REMATCH[3]}))
if (( PARQUET_TOTAL < 8 || PARQUET_START != PARQUET_TOTAL - 8 || PARQUET_END != PARQUET_TOTAL - 1 )); then
  printf '%s\n' 'FAIL: parquet suffix Content-Range is not exact' >&2
  exit 1
fi

# MP4 seek range: 206 and exact Content-Range/Content-Length.
MP4_STATUS=$(curl -sS -D "$SMOKE_TMP/mp4.headers" -o "$SMOKE_TMP/mp4.body" \
  -w '%{http_code}' -H 'Range: bytes=0-1023' \
  "$ASSET_ROOT/videos/chunk-000/observation.images.ego_view/episode_000000.mp4")
if test "$MP4_STATUS" != 206; then
  printf 'FAIL: MP4 range returned HTTP %s\n' "$MP4_STATUS" >&2
  exit 1
fi
MP4_CONTENT_RANGE=$(header_value Content-Range "$SMOKE_TMP/mp4.headers")
MP4_CONTENT_LENGTH=$(header_value Content-Length "$SMOKE_TMP/mp4.headers")
MP4_BODY_LENGTH=$(wc -c < "$SMOKE_TMP/mp4.body")
if test "$MP4_CONTENT_LENGTH" != 1024 || test "$MP4_BODY_LENGTH" -ne 1024; then
  printf '%s\n' 'FAIL: MP4 range length must be exactly 1024 bytes' >&2
  exit 1
fi
if [[ ! "$MP4_CONTENT_RANGE" =~ ^bytes\ 0-1023/([0-9]+)$ ]]; then
  printf '%s\n' 'FAIL: MP4 Content-Range is invalid' >&2
  exit 1
fi
MP4_TOTAL=$((10#${BASH_REMATCH[1]}))
if (( MP4_TOTAL <= 1023 )); then
  printf '%s\n' 'FAIL: MP4 Content-Range total is invalid' >&2
  exit 1
fi

# Percent-decoded traversal must be 404.
TRAVERSAL_STATUS=$(curl --path-as-is -sS -o /dev/null -w '%{http_code}' \
  "$ASSET_ROOT/%2e%2e/meta/info.json")
if test "$TRAVERSAL_STATUS" != 404; then
  printf 'FAIL: traversal probe returned HTTP %s\n' "$TRAVERSAL_STATUS" >&2
  exit 1
fi

# Any Authorization header on a local asset must be 400.
AUTH_STATUS=$(curl -sS -o /dev/null -w '%{http_code}' \
  -H 'Authorization: Bearer deliberate-rejection-probe' "$ASSET_ROOT/meta/info.json")
if test "$AUTH_STATUS" != 400; then
  printf 'FAIL: Authorization rejection probe returned HTTP %s\n' "$AUTH_STATUS" >&2
  exit 1
fi
printf '%s\n' 'local asset integration smoke: PASS'
```

Require the indicated `200`/`206`/`404`/`400` statuses before opening:

```text
http://127.0.0.1:3000/local/pnp_trash/episode_0?tab=annotations
```

In browser developer tools verify `info.json`, `episodes.jsonl`, `tasks.jsonl`,
parquet charts, video seeking, and task-index mode. Inspect page source, loaded
JavaScript, and the Network request headers: no curation bearer, Cosmos
credential/configuration, or HF OAuth token may appear on local asset
requests. Local parquet and MP4 requests must go directly to port 8000, not
through the Next.js JSON proxy.

## Cosmos capability and one-episode smoke

Check `/v1/models` and `/version` without putting the credential in a command
line or output:

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python - <<'PY'
import os
from urllib.parse import urlsplit, urlunsplit

import httpx

from backend.curation.config import CurationSettings

settings = CurationSettings.from_env()
approved_version = "0.23.0"
approved_endpoint_identity = (
    "h100-cosmos3-nano-vllm-0.23.0@sha256:f37691f675bb82f734f606de8af90e777d3f80a20b120e699fd43fd10e60b8d7"
)
if settings.cosmos_endpoint_identity != approved_endpoint_identity:
    raise SystemExit("FAIL: configured Cosmos endpoint identity does not match the approved build")
key_name = settings.cosmos_api_key_env
key = os.environ.get(key_name)
if not key:
    raise SystemExit("FAIL: Cosmos credential variable is unavailable")
model = settings.cosmos_model
url = settings.cosmos_base_url.rstrip("/") + "/models"
base = urlsplit(settings.cosmos_base_url)
version_url = urlunsplit((base.scheme, base.netloc, "/version", "", ""))
with httpx.Client(trust_env=False, follow_redirects=False) as client:
    response = client.get(
        url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=120.0,
    )
    response.raise_for_status()
    document = response.json()
    version_response = client.get(
        version_url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=120.0,
    )
    version_response.raise_for_status()
    version_document = version_response.json()
if not any(item.get("id") == model for item in document.get("data", [])):
    raise SystemExit("FAIL: configured Cosmos model is absent from /v1/models")
if version_document.get("version") != approved_version:
    raise SystemExit("FAIL: deployed vLLM version does not match the approved build")
print(f"Cosmos model identity: PASS ({model}, vLLM {approved_version})")
PY
```

Open the workspace, then create exactly one smoke attempt through the
same-origin Next.js proxy:

```bash
set -euo pipefail
: "${CURATION_REPO_ROOT:?FAIL: CURATION_REPO_ROOT is required}"
: "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}"
CURATION_API=http://127.0.0.1:3000/api/curation
OPEN_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"dataset_alias":"local/pnp_trash","actor":"curator"}' \
  "$CURATION_API/workspaces/open")
printf '%s\n' "$OPEN_RESPONSE" | jq -e .

SMOKE_SOURCE_EPISODE_INDEX=4
: "${SOURCE_DATASET:?FAIL: SOURCE_DATASET is required}"
: "${CURATION_DATASET_ALIASES_JSON:?FAIL: CURATION_DATASET_ALIASES_JSON is required}"
if ! SMOKE_ALIAS_SOURCE=$(printf '%s' "$CURATION_DATASET_ALIASES_JSON" \
  | jq -er '.["local/pnp_trash"] | select(type == "string" and length > 0)' 2>/dev/null); then
  printf '%s\n' 'FAIL: configured smoke dataset alias is invalid' >&2
  exit 1
fi
if test "$SMOKE_ALIAS_SOURCE" != "$SOURCE_DATASET"; then
  printf '%s\n' 'FAIL: smoke source does not match the configured dataset alias' >&2
  exit 1
fi
if ! { SMOKE_SOURCE_PATH=$(realpath -e -- "$SOURCE_DATASET"); } 2>/dev/null; then
  printf '%s\n' 'FAIL: smoke source canonical target is unavailable' >&2
  exit 1
fi
if test ! -d "$SMOKE_SOURCE_PATH" || test -L "$SMOKE_SOURCE_PATH"; then
  printf '%s\n' 'FAIL: smoke source canonical target is unavailable' >&2
  exit 1
fi
: "${APPROVED_SOURCE_MANIFEST_SHA256:?FAIL: approved source manifest SHA-256 is required}"
SMOKE_REPRESENTATIVE_AUTHORITY=$(
  "$CURATION_REPO_ROOT/backend/.venv/bin/python" \
    -m backend.curation.runbook_validation representative \
    --workspace "$CURATION_WORKSPACE" \
    --source-path "$SMOKE_SOURCE_PATH" \
    --source-manifest-sha256 "$APPROVED_SOURCE_MANIFEST_SHA256" \
    --source-episode-index "$SMOKE_SOURCE_EPISODE_INDEX"
)
printf '%s' "$SMOKE_REPRESENTATIVE_AUTHORITY" | jq -e \
  '. == {source_episode_index:4,frame_count:2060,duration_s:41.2,sampled_frame_count:83}' \
  >/dev/null
SMOKE_REQUEST=$(jq -cn --argjson index "$SMOKE_SOURCE_EPISODE_INDEX" \
  '{dataset_alias:"local/pnp_trash",episode_indices:[$index]}')
SMOKE_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d "$SMOKE_REQUEST" \
  "$CURATION_API/batches")
SMOKE_JOB_ID=$(printf '%s' "$SMOKE_RESPONSE" | jq -er '.job_id')
if test -z "$SMOKE_JOB_ID"; then
  printf '%s\n' 'FAIL: one-episode Cosmos smoke did not return a job ID' >&2
  exit 1
fi

run_curation_worker \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --job-id "$SMOKE_JOB_ID"
SMOKE_STATE_RESPONSE=$(curl -fsS "$CURATION_API/batches/$SMOKE_JOB_ID")
printf '%s\n' "$SMOKE_STATE_RESPONSE" | jq
SMOKE_STATE=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.state')
if ! test "$SMOKE_STATE" = completed; then
  printf 'FAIL: one-episode Cosmos smoke requires terminal state completed; found %s\n' \
    "$SMOKE_STATE" >&2
  exit 1
fi
SMOKE_SUCCEEDED=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.counts.succeeded // 0')
SMOKE_MANUAL_ONLY=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.counts.manual_only // 0')
SMOKE_RETRYABLE=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.counts.retryable // 0')
SMOKE_PROPOSAL_COVERAGE=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.active_proposal_coverage')
SMOKE_EPISODE_COUNT=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.episodes | length')
SMOKE_EPISODE_INDEX=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.episodes[0].source_episode_index')
SMOKE_ATTEMPT_STATE=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.episodes[0].state')
SMOKE_ATTEMPT_ID=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.episodes[0].attempt_id')
if ! test "$SMOKE_SUCCEEDED" -eq 1 \
  || ! test "$SMOKE_MANUAL_ONLY" -eq 0 \
  || ! test "$SMOKE_RETRYABLE" -eq 0 \
  || ! test "$SMOKE_PROPOSAL_COVERAGE" -eq 1 \
  || ! test "$SMOKE_EPISODE_COUNT" -eq 1 \
  || ! test "$SMOKE_EPISODE_INDEX" -eq "$SMOKE_SOURCE_EPISODE_INDEX" \
  || ! test "$SMOKE_ATTEMPT_STATE" = succeeded \
  || test -z "$SMOKE_ATTEMPT_ID"; then
  printf '%s\n' 'FAIL: one-episode Cosmos smoke did not produce exactly one successful proposal' >&2
  exit 1
fi

SMOKE_EPISODE_RESPONSE=$(curl -fsS \
  "$CURATION_API/episodes/$SMOKE_SOURCE_EPISODE_INDEX?dataset_alias=local%2Fpnp_trash")
SMOKE_PROPOSAL_ID=$(printf '%s' "$SMOKE_EPISODE_RESPONSE" | jq -er '.active_proposal.id')
SMOKE_PROPOSAL_ATTEMPT_ID=$(printf '%s' "$SMOKE_EPISODE_RESPONSE" \
  | jq -er '.active_proposal.attempt_id')
if test "$SMOKE_PROPOSAL_ATTEMPT_ID" != "$SMOKE_ATTEMPT_ID"; then
  printf '%s\n' 'FAIL: active proposal does not belong to the exact smoke attempt' >&2
  exit 1
fi
SMOKE_DATASET_ID=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.configuration.dataset_id')
SMOKE_SOURCE_MANIFEST_SHA256=$(printf '%s' "$SMOKE_STATE_RESPONSE" \
  | jq -er '.configuration.source_manifest_sha256')
if [[ ! "$SMOKE_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: smoke source manifest identity is invalid' >&2
  exit 1
fi

SMOKE_ATTEMPT_ROOT="$CURATION_WORKSPACE/artifacts/cosmos/$SMOKE_ATTEMPT_ID"
REQUEST_ARTIFACT="$SMOKE_ATTEMPT_ROOT/request.json"
RESPONSE_ARTIFACT="$SMOKE_ATTEMPT_ROOT/response.txt"
PARSED_ARTIFACT="$SMOKE_ATTEMPT_ROOT/parsed.json"
SMOKE_CONTACT_NAMESPACE="$CURATION_WORKSPACE/contact_sheets/datasets/dataset_${SMOKE_DATASET_ID}_${SMOKE_SOURCE_MANIFEST_SHA256}"
PROPOSAL_CONTACT_SHEET="$SMOKE_CONTACT_NAMESPACE/proposals/proposal_${SMOKE_PROPOSAL_ID}.png"
PROPOSAL_CONTACT_RECEIPT="$SMOKE_CONTACT_NAMESPACE/receipts/proposals/proposal_${SMOKE_PROPOSAL_ID}.png.receipt.json"

require_smoke_regular_file() {
  if ! test -f "$1" || test -L "$1"; then
    printf 'FAIL: one-episode %s artifact is unavailable\n' "$2" >&2
    exit 1
  fi
}
require_smoke_regular_file "$REQUEST_ARTIFACT" request
require_smoke_regular_file "$RESPONSE_ARTIFACT" response
require_smoke_regular_file "$PARSED_ARTIFACT" parsed
require_smoke_regular_file "$PROPOSAL_CONTACT_SHEET" contact-sheet
require_smoke_regular_file "$PROPOSAL_CONTACT_RECEIPT" contact-sheet-receipt

cd "$CURATION_REPO_ROOT"
SMOKE_AUTHORITY_JSON=$(
  backend/.venv/bin/python -m backend.curation.runbook_validation smoke \
    --workspace "$CURATION_WORKSPACE" \
    --status-json "$SMOKE_STATE_RESPONSE" \
    --episode-json "$SMOKE_EPISODE_RESPONSE" \
    --expected-smoke-job-id "$SMOKE_JOB_ID"
)
SMOKE_AUTHORITY_SHA256=$(printf '%s' "$SMOKE_AUTHORITY_JSON" | sha256sum | awk '{print $1}')
if [[ ! "$SMOKE_AUTHORITY_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: canonical smoke authority hash is invalid' >&2
  exit 1
fi
AUTHORITATIVE_RESPONSE_RELATIVE_PATH=$(printf '%s' "$SMOKE_AUTHORITY_JSON" \
  | jq -er '.artifacts.authoritative_response.relative_path')
AUTHORITATIVE_RESPONSE_ARTIFACT="$CURATION_WORKSPACE/$AUTHORITATIVE_RESPONSE_RELATIVE_PATH"

printf 'Raw response: %s\nParsed proposal: %s\nSampling request: %s\nContact sheet: %s\n' \
  "$AUTHORITATIVE_RESPONSE_ARTIFACT" "$PARSED_ARTIFACT" "$REQUEST_ARTIFACT" "$PROPOSAL_CONTACT_SHEET"
SMOKE_CONFIRMATION_EXPECTED="CONFIRM SMOKE EVIDENCE $SMOKE_JOB_ID $SMOKE_ATTEMPT_ID"
printf 'Required confirmation: %s\n' "$SMOKE_CONFIRMATION_EXPECTED"
if ! read -r -p 'After inspecting the raw response, parsed proposal, sampling evidence, contact sheet, and UI, type the exact confirmation: ' SMOKE_EVIDENCE_CONFIRMED; then
  printf '%s\n' 'FAIL: one-episode evidence and UI inspection were not explicitly confirmed' >&2
  exit 1
fi
if test "$SMOKE_EVIDENCE_CONFIRMED" != "$SMOKE_CONFIRMATION_EXPECTED"; then
  printf '%s\n' 'FAIL: one-episode evidence and UI inspection were not explicitly confirmed' >&2
  exit 1
fi
export CONFIRMED_SMOKE_JOB_ID="$SMOKE_JOB_ID"
export CONFIRMED_SMOKE_ATTEMPT_ID="$SMOKE_ATTEMPT_ID"
export CONFIRMED_SMOKE_SOURCE_EPISODE_INDEX="$SMOKE_SOURCE_EPISODE_INDEX"
export CONFIRMED_SMOKE_AUTHORITY_JSON="$SMOKE_AUTHORITY_JSON"
export CONFIRMED_SMOKE_AUTHORITY_SHA256="$SMOKE_AUTHORITY_SHA256"
export SMOKE_EVIDENCE_CONFIRMED
printf 'Preserve these shell variables before the full batch: %s\n' \
  "$SMOKE_CONFIRMATION_EXPECTED"
```

Only exact terminal `completed` with one succeeded attempt, no
`manual_only`/`retryable` attempt, and one active proposal is a valid smoke.
The operator confirmation is mandatory after inspecting the raw response,
parsed v2 status/times, strict 50-to-2 fps sampling proof, six-cell proposal
contact sheet and receipt, and UI rendering. Null transitions must render
`not observed` placeholders without fabricated images or deltas.

## Run and monitor the full Cosmos batch

The API only persists a queued job. It never launches a worker. Create the full
92-episode batch only in the same authenticated shell after the exact smoke job
and attempt evidence plus UI inspection were explicitly confirmed:

```bash
set -euo pipefail
if test -z "${CONFIRMED_SMOKE_JOB_ID:-}"; then
  printf '%s\n' 'FAIL: confirmed one-episode smoke job ID is required' >&2
  exit 1
fi
if test -z "${CONFIRMED_SMOKE_ATTEMPT_ID:-}"; then
  printf '%s\n' 'FAIL: confirmed one-episode smoke attempt ID is required' >&2
  exit 1
fi
if test "${CONFIRMED_SMOKE_SOURCE_EPISODE_INDEX:-}" != 4; then
  printf '%s\n' 'FAIL: confirmed smoke must bind pinned representative episode 4' >&2
  exit 1
fi
if test -z "${CONFIRMED_SMOKE_AUTHORITY_JSON:-}" \
  || [[ ! "${CONFIRMED_SMOKE_AUTHORITY_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: canonical confirmed smoke authority is required' >&2
  exit 1
fi
SMOKE_CONFIRMATION_EXPECTED="CONFIRM SMOKE EVIDENCE $CONFIRMED_SMOKE_JOB_ID $CONFIRMED_SMOKE_ATTEMPT_ID"
if test "${SMOKE_EVIDENCE_CONFIRMED:-}" != "$SMOKE_CONFIRMATION_EXPECTED"; then
  printf '%s\n' 'FAIL: smoke evidence and UI confirmation do not match the exact job and attempt' >&2
  exit 1
fi
CURATION_API=http://127.0.0.1:3000/api/curation
CONFIRMED_AUTHORITY_ACTUAL_SHA256=$(printf '%s' "$CONFIRMED_SMOKE_AUTHORITY_JSON" \
  | sha256sum | awk '{print $1}')
if test "$CONFIRMED_AUTHORITY_ACTUAL_SHA256" != "$CONFIRMED_SMOKE_AUTHORITY_SHA256"; then
  printf '%s\n' 'FAIL: confirmed smoke authority hash does not match' >&2
  exit 1
fi
CURRENT_SMOKE_STATUS=$(curl -fsS "$CURATION_API/batches/$CONFIRMED_SMOKE_JOB_ID")
SMOKE_SOURCE_EPISODE_INDEX="$CONFIRMED_SMOKE_SOURCE_EPISODE_INDEX"
CURRENT_SMOKE_EPISODE=$(curl -fsS \
  "$CURATION_API/episodes/$SMOKE_SOURCE_EPISODE_INDEX?dataset_alias=local%2Fpnp_trash")
cd "$CURATION_REPO_ROOT"
CURRENT_SMOKE_AUTHORITY_JSON=$(
  backend/.venv/bin/python -m backend.curation.runbook_validation smoke \
    --workspace "$CURATION_WORKSPACE" \
    --status-json "$CURRENT_SMOKE_STATUS" \
    --episode-json "$CURRENT_SMOKE_EPISODE" \
    --expected-smoke-job-id "$CONFIRMED_SMOKE_JOB_ID"
)
if test "$CURRENT_SMOKE_AUTHORITY_JSON" != "$CONFIRMED_SMOKE_AUTHORITY_JSON"; then
  printf '%s\n' 'FAIL: current smoke evidence does not match the confirmed authority' >&2
  exit 1
fi
BATCH_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"dataset_alias":"local/pnp_trash"}' \
  "$CURATION_API/batches")
JOB_ID=$(printf '%s' "$BATCH_RESPONSE" | jq -er '.job_id')
test -n "$JOB_ID"
printf '%s\n' "$BATCH_RESPONSE" | jq
FULL_BATCH_STATUS=$(curl -fsS "$CURATION_API/batches/$JOB_ID")
backend/.venv/bin/python -m backend.curation.runbook_validation full-batch \
  --workspace "$CURATION_WORKSPACE" \
  --authority-json "$CONFIRMED_SMOKE_AUTHORITY_JSON" \
  --authority-sha256 "$CONFIRMED_SMOKE_AUTHORITY_SHA256" \
  --smoke-status-json "$CURRENT_SMOKE_STATUS" \
  --smoke-episode-json "$CURRENT_SMOKE_EPISODE" \
  --full-status-json "$FULL_BATCH_STATUS" \
  --expected-full-job-id "$JOB_ID" \
  --expected-smoke-job-id "$CONFIRMED_SMOKE_JOB_ID"

run_curation_worker \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --job-id "$JOB_ID"
```

In another terminal, poll persisted status:

```bash
CURATION_API=http://127.0.0.1:3000/api/curation
read -r -p 'Paste JOB_ID: ' JOB_ID
test -n "$JOB_ID"
curl -fsS "$CURATION_API/batches/$JOB_ID" | jq
```

Normal states are `queued`, `running`, and then `completed` or
`completed_with_failures`. The worker is single-concurrency, owns 2 fps local
sampling, renews leases, and reads the API key only from the variable named in
the persisted job. Exit codes are `0` for terminal nonfailed, `1` for failed,
`2` for argument/config/state errors, `3` for a live-lease conflict, and `130`
for a handled interrupt.

### Cancel and retry

Cancellation is idempotent. Queued jobs become `cancelled` immediately;
running jobs become `cancel_requested` and stop claiming new attempts:

```bash
test -n "$JOB_ID"
curl -fsS -X POST -H 'Origin: http://127.0.0.1:3000' \
  "$CURATION_API/batches/$JOB_ID/cancel" | jq
```

Retry creates an immutable child job. Choose explicit episodes or a failure
filter; never edit the parent attempts:

```bash
test -n "$JOB_ID"
RETRY_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"failure_states":["manual_only","retryable"]}' \
  "$CURATION_API/batches/$JOB_ID/retry")
RETRY_JOB_ID=$(printf '%s' "$RETRY_RESPONSE" | jq -er '.job_id')
test -n "$RETRY_JOB_ID"

run_curation_worker \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --job-id "$RETRY_JOB_ID"
```

## Human review

Cosmos supplies proposals only. In the browser, review the entire video for all
92 source episodes:

1. Inspect the proposal, contact sheet, full video, and transition ordering.
2. Reject corrupt, incomplete, unsuccessful, or out-of-order episodes. A kept
   episode must complete all seven steps, including standing straight.
3. For a keep candidate enter normalized free-text object, pickup hand, turn
   direction, and six transition frames.
4. Check grasp/release candidates and every boundary frame; resolve every
   ordering/coverage error and unreadable file.
5. Save draft, then explicitly approve keep or reject. Use approve-and-next.
6. Inspect duration/transition outliers and every grip disagreement over 2.0
   seconds. Cosmos confidence never substitutes for human approval.

The final summary must satisfy:

```text
pending = 0
draft = 0
approved_keep >= 1
approved_keep + approved_reject = 92
invalid approved_keep = 0
```

Check it through the persisted API as well as the UI:

```bash
curl -fsS "$CURATION_API/summary?dataset_alias=local%2Fpnp_trash" | jq
curl -fsS "$CURATION_API/audit?dataset_alias=local%2Fpnp_trash" | jq
```

Reopening an approved episode invalidates that approval and its final contact
sheet. Once an export snapshot is created, later review edits do not alter that
snapshot; they require a new export.

## Build, validate, and publish the separate cleaned dataset

Before export, re-run the source-manifest check, require the final path to be
absent, confirm all 92 decisions and at least one keep, and ensure no export is
active. Create the immutable approval snapshot:

```bash
set -euo pipefail
: "${CURATION_WORKSPACE:?FAIL: CURATION_WORKSPACE is required}"
: "${CURATION_OUTPUT:?FAIL: CURATION_OUTPUT is required}"
PINNED_SOURCE_MANIFEST_SHA256=5962d8630f06e6260adbae15a3d7ee5f0a1a745c3a12466add8722c2e0da9577
SOURCE_MANIFEST="$CURATION_WORKSPACE/source-files.sha256"
if test ! -f "$SOURCE_MANIFEST"; then
  printf '%s\n' 'FAIL: persisted source manifest is unavailable' >&2
  exit 1
fi
SOURCE_MANIFEST_FILE_COUNT=$(wc -l < "$SOURCE_MANIFEST")
if test "$SOURCE_MANIFEST_FILE_COUNT" -ne 190; then
  printf 'FAIL: persisted source manifest must contain 190 files; found %s\n' "$SOURCE_MANIFEST_FILE_COUNT" >&2
  exit 1
fi
SOURCE_MANIFEST_SHA256=$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')
if test "$SOURCE_MANIFEST_SHA256" != "$PINNED_SOURCE_MANIFEST_SHA256"; then
  printf '%s\n' 'FAIL: persisted source manifest SHA-256 does not match approved record' >&2
  exit 1
fi

if test -e "$CURATION_OUTPUT" || test -L "$CURATION_OUTPUT"; then
  printf '%s\n' 'FAIL: curation output already exists' >&2
  exit 1
fi
cd /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash
sha256sum --check "$SOURCE_MANIFEST"

CURATION_API=http://127.0.0.1:3000/api/curation
EXPORT_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"dataset_alias":"local/pnp_trash"}' \
  "$CURATION_API/exports")
EXPORT_ID=$(printf '%s' "$EXPORT_RESPONSE" | jq -er '.export_id')
APPROVAL_SNAPSHOT_SHA256=$(printf '%s' "$EXPORT_RESPONSE" | jq -er '.approval_snapshot_sha256')
test -n "$EXPORT_ID"
test "${#APPROVAL_SNAPSHOT_SHA256}" -eq 64
printf '%s\n' "$EXPORT_RESPONSE" | jq

test -n "$EXPORT_ID"
run_curation_exporter \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --export-id "$EXPORT_ID"
```

The one explicit exporter process performs this order and stops on any failed
gate:

```text
queued
building
core_structural_validated
gr00t_stats_validated
gr00t_loader_validated
provenance_written
final_consistency_validated
publishing
published
```

Monitor the persisted status in a separate terminal:

```bash
CURATION_API=http://127.0.0.1:3000/api/curation
read -r -p 'Paste EXPORT_ID: ' EXPORT_ID
test -n "$EXPORT_ID"
curl -fsS "$CURATION_API/exports/$EXPORT_ID" | jq
```

Publication uses Linux `renameat2(RENAME_NOREPLACE)`, then fsyncs the final
parent, then commits `published`. Any competing destination causes failure;
nothing overwrites it. Do not create the final directory manually.

## Crash recovery

Never start a second live worker/exporter, change SQLite by hand, or delete,
move, chmod, or repair staging/final paths manually.

### Batch recovery

- A queued job uses `run` exactly once.
- After a killed worker, wait for the persisted job/attempt lease to expire,
  inspect status, and use `resume` on the same ID. `resume` accepts only
  `running` or `cancel_requested`; an unexpired lease returns exit 3.
- For `cancel_requested`, `resume` makes no model calls and finalizes remaining
  attempts as cancelled.
- Terminal jobs are immutable. Use the retry API to create a child job for an
  explicit operator retry.

```bash
test -n "$JOB_ID"
run_curation_worker \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  resume --job-id "$JOB_ID"
```

### Export recovery

- A queued export normally uses `run`. After a crash in a nonterminal state,
  inspect the API and use `resume` on the same export ID.
- `resume` continues recorded intermediate validation states. It never resumes
  `failed` or `published`.
- If state is `publishing` with staging absent and final present, recovery
  re-runs the final read-only gate, verifies publication identity, fsyncs the
  parent, and only then commits `published`.
- If staging is present and final absent, recovery returns to the validated
  publication point. If both are present or both absent, it fails for operator
  inspection rather than guessing.
- A parent-fsync failure remains retryable in `publishing`; preserve paths and
  run `resume` after fixing only the external filesystem condition.

```bash
test -n "$EXPORT_ID"
run_curation_exporter \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  resume --export-id "$EXPORT_ID"
```

For any terminal `failed`, retain
`$CURATION_WORKSPACE/exports/$EXPORT_ID`, the unique sibling staging tree, and
the API status for diagnosis. A new attempt requires an explicit new export
snapshot after the failure is understood.

## Post-publication verification and handoff

Only after API state is `published`:

```bash
set -euo pipefail
test -d /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned
test ! -L /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned
test ! -w /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned/meta/info.json

cd /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned
sha256sum --check meta/curation_checksums.sha256

cd /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash
sha256sum --check /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation/source-files.sha256

cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python -m pytest -q backend/tests
bun run validate
```

Inspect these output artifacts:

- `meta/curation_artifacts/structural-report.json`
- `meta/curation_artifacts/gr00t-stats-report.json`
- `meta/curation_artifacts/gr00t-loader-report.json`
- `meta/curation_provenance.json`
- `meta/curation_checksums.sha256`
- workspace-only
  `exports/$EXPORT_ID/final-consistency-report.json`

The loader report must cover every retained episode with exactly seven prompt
runs in order, frame-for-frame equal to parquet `task_index`. Verify the output
contains no symlink, no hardlink to source, no unlisted regular file, and no
raw Cosmos reasoning/credential artifact.

Record the source/final canonical paths; source-manifest and approval-snapshot
SHA-256; kept/rejected/frame counts; `tasks.jsonl` count/hash; Cosmos model and
endpoint identity; contributing job/attempt IDs; visualizer and Isaac-GR00T
commit/dirty states; four report hashes/pass states; and export UUID/published
timestamp. Preserve the workspace as provenance evidence.
