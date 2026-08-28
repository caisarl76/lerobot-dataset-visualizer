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
export CURATION_DATASET_ALIASES_JSON='{"local/pnp_trash":"/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash"}'
export CURATION_WORKSPACE=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation
export CURATION_OUTPUT=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned
export CURATION_BROWSER_ORIGIN=http://127.0.0.1:3000
export CURATION_BACKEND_URL=http://127.0.0.1:8000
export NEXT_PUBLIC_DATASET_URL=http://127.0.0.1:8000/api/local-datasets
export ISAAC_GROOT_ROOT=/home/jihun/work/Isaac-GR00T
```

`backend.app:app`, `backend/curation_worker.py`, and
`backend/curation_export.py` all call `CurationSettings.from_env()`. Launch all
three Python processes with the five non-secret mapping values consumed by
`CurationSettings` (`CURATION_DATASET_ALIASES_JSON`, `CURATION_WORKSPACE`,
`CURATION_OUTPUT`, `CURATION_BROWSER_ORIGIN`, and `ISAAC_GROOT_ROOT`), all five
external runtime names, including the `COSMOS_API_KEY_ENV` name. Only the
backend and worker receive the actual credential variable named by
`COSMOS_API_KEY_ENV`; the exporter must not receive that secret. The backend reads the variable named by
`COSMOS_API_KEY_ENV` during batch capability validation before it creates a
job; the worker reads it for Cosmos calls. The exporter performs no Cosmos call
and launches without the target credential variable. Neither CLI accepts a
secret argument.

Next.js requires only `CURATION_BACKEND_URL`, `CURATION_BEARER_TOKEN`, and
`NEXT_PUBLIC_DATASET_URL`. Its bearer value must exactly match the backend's;
the other Python settings and the Cosmos credential must not be supplied to
browser code.

| Variable                               | Visibility                                     | Owner                                                 | Startup validation                                                                                                    |
| -------------------------------------- | ---------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `CURATION_DATASET_ALIASES_JSON`        | Non-secret, server-only                        | All three Python processes                            | Nonempty JSON object; every key is one `org/dataset` alias and every value is an absolute, existing dataset directory |
| `CURATION_WORKSPACE`                   | Non-secret, server-only                        | All three Python processes                            | Canonical absolute path, separate from every source and output; CLI value must equal the persisted workspace          |
| `CURATION_OUTPUT`                      | Non-secret, server-only                        | All three Python processes                            | Canonical absolute path, separate from source/workspace; must be absent before the no-clobber export                  |
| `CURATION_BROWSER_ORIGIN`              | Non-secret, server-only                        | All three Python processes; backend uses it for CORS  | Exactly one HTTP(S) origin with no credentials, path, query, fragment, or comma-separated alternatives                |
| `CURATION_BACKEND_URL`                 | Non-secret, server-only                        | Next.js only                                          | Absolute HTTP(S) URL with no credentials, query, or fragment; this run requires the exact loopback mapping above      |
| `NEXT_PUBLIC_DATASET_URL`              | Public                                         | Next.js and browser URL builder                       | Exact loopback local-asset prefix above; it carries no credential                                                     |
| `ISAAC_GROOT_ROOT`                     | Non-secret, server-only                        | All three Python processes; exporter uses it          | Canonical absolute path; the preflight must import the exact loader/config and find `gr00t/data/stats.py`             |
| `CURATION_BACKEND_HOST`                | Non-secret, optional                           | All three Python processes; backend guard consumes it | Defaults to `127.0.0.1`; must parse as loopback or literal `localhost`                                                |
| `CURATION_BEARER_TOKEN`                | Secret                                         | All three Python processes; Next.js also consumes it  | Required and nonempty; Next.js and backend values must match; never browser-visible                                   |
| `COSMOS_BASE_URL`                      | External/private configuration                 | All three Python processes                            | Absolute HTTP(S), no whitespace, credentials, query, fragment, invalid port, or redirect following                    |
| `COSMOS_MODEL`                         | External/private configuration                 | All three Python processes                            | Required, nonempty, and exactly present in the configured `/v1/models` response                                       |
| `COSMOS_API_KEY_ENV`                   | External/private configuration; names a secret | All three Python processes                            | Required nonempty variable name                                                                                       |
| Variable named by `COSMOS_API_KEY_ENV` | Secret                                         | Backend and worker only                               | Required for backend batch creation and worker calls; prohibited from the exporter environment                        |
| `COSMOS_ENDPOINT_IDENTITY`             | External/private configuration                 | All three Python processes                            | Required and nonempty; stable operator identity for the existing H100 service                                         |

The operator's existing server/runtime configuration must provide these names;
their values do not belong in the repository or this runbook:

- `CURATION_BEARER_TOKEN`: present in all three Python launch environments and
  the Next.js server process; only FastAPI and Next.js consume it at runtime.
- `COSMOS_BASE_URL`: OpenAI-compatible H100 endpoint ending at `/v1`.
- `COSMOS_MODEL`: exact model ID returned by `/v1/models`.
- `COSMOS_API_KEY_ENV`: name of the external environment variable that holds
  the Cosmos API key. The key itself is read indirectly by the backend before
  batch creation and by the worker during Cosmos calls; do not supply the
  target secret to the exporter.
- `COSMOS_ENDPOINT_IDENTITY`: stable non-secret operator identity recorded in
  provenance, supplied from the existing Cosmos runtime configuration.

`CURATION_BACKEND_HOST` is optional and defaults to `127.0.0.1`; it may only be
a loopback address. The backend owns dataset aliases, workspace/output paths,
the browser-origin allowlist, Cosmos job snapshots, and Isaac-GR00T path. The
worker owns 2 fps sampling and Cosmos calls. The exporter owns staging,
validation, and no-clobber publication. Next.js owns `CURATION_BACKEND_URL`
and server-side bearer injection. Browser code may receive only
`NEXT_PUBLIC_DATASET_URL`.

Never define a `NEXT_PUBLIC_*` token, API key, Cosmos base URL, model, or
endpoint identity. Do not use `set -x` in a shell containing the external
credentials.

## Install and static regression gates

From the repository root, create the backend environment and install frontend
dependencies if needed:

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
python -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
bun install
```

Before real data operations, run both complete gates:

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/python -m pytest -q backend/tests
```

Expected: PASS, including `test_legacy_v31_regression.py` and the curation
modules.

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
bun run format && bun run validate
```

Expected: PASS. `format` may only produce formatting changes that are reviewed
before continuing.

## Filesystem and dependency preflight

Run this section before starting a new curation. Stop on any failed assertion.
Before touching the live tree, obtain the canonical 189-file manifest SHA-256
from an independently reviewed and approved inventory record. In the same
shell that runs source preflight, enter that recorded value as a shell-local
variable:

```bash
read -r -p 'Paste independently approved 189-file manifest SHA-256: ' APPROVED_SOURCE_MANIFEST_SHA256
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
if [[ ! "$APPROVED_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: approved source manifest SHA-256 must be 64 lowercase hex characters' >&2
  exit 1
fi

SOURCE_DATASET=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash
test -d "$SOURCE_DATASET"
test ! -L "$SOURCE_DATASET"
SOURCE_FILE_COUNT=$(find -P "$SOURCE_DATASET" -type f | wc -l)
if test "$SOURCE_FILE_COUNT" -ne 189; then
  printf 'FAIL: source regular-file count must be 189; found %s\n' "$SOURCE_FILE_COUNT" >&2
  exit 1
fi
test -z "$(find -P "$SOURCE_DATASET" -type l -print -quit)"

VISUALIZER_ROOT=/home/jihun/work/lerobot-dataset-visualizer
test -x "$VISUALIZER_ROOT/backend/.venv/bin/python"
PROSPECTIVE_SOURCE_MANIFEST_SHA256=$(
  PYTHONPATH="$VISUALIZER_ROOT" SOURCE_DATASET="$SOURCE_DATASET" \
    "$VISUALIZER_ROOT/backend/.venv/bin/python" - <<'PY'
import hashlib
import os
from pathlib import Path

from backend.curation.source import _manifest_bytes

root = Path(os.environ["SOURCE_DATASET"]).resolve(strict=True)
manifest, _, _ = _manifest_bytes(root)
print(hashlib.sha256(manifest).hexdigest())
PY
)
if [[ ! "$PROSPECTIVE_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: prospective source manifest SHA-256 is invalid' >&2
  exit 1
fi
if test "$PROSPECTIVE_SOURCE_MANIFEST_SHA256" != "$APPROVED_SOURCE_MANIFEST_SHA256"; then
  printf '%s\n' 'FAIL: prospective source manifest SHA-256 does not match approved record' >&2
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

The approved source snapshot contains exactly 189 regular files. A count of
190 is a hard stop, including when the additional file is a top-level
`pnp_trash.xlsx`. Do not delete, move, ignore, or otherwise mutate the source
implicitly. Task 15 remains blocked until the operator explicitly resolves the
post-snapshot source drift and a new immutable manifest expectation is agreed.
No approved canonical hash for the intended 189-file tree is currently
recorded in this repository or runbook. Do not invent or bless one. Task 15
remains blocked until the XLSX drift is explicitly resolved by the operator
without implicit source mutation and the exact canonical 189-file manifest
hash is independently recorded and approved.

### 2. Linux atomic no-clobber support

This executable probe creates and removes two private temporary directories in
the output parent and proves both the `EEXIST` and successful
`renameat2(RENAME_NOREPLACE)` paths:

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
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
cd /home/jihun/work/lerobot-dataset-visualizer
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
cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Start Next.js in a second terminal containing only `CURATION_BACKEND_URL`,
`CURATION_BEARER_TOKEN`, and `NEXT_PUBLIC_DATASET_URL`:

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
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
if [[ ! "$APPROVED_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'FAIL: approved source manifest SHA-256 must be 64 lowercase hex characters' >&2
  exit 1
fi

SOURCE_MANIFEST="$CURATION_WORKSPACE/source-files.sha256"
test -f "$SOURCE_MANIFEST"
SOURCE_MANIFEST_FILE_COUNT=$(wc -l < "$SOURCE_MANIFEST")
if test "$SOURCE_MANIFEST_FILE_COUNT" -ne 189; then
  printf 'FAIL: persisted source manifest must contain 189 files; found %s\n' "$SOURCE_MANIFEST_FILE_COUNT" >&2
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

Check `/v1/models` without putting the credential in a command line or output:

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/python - <<'PY'
import os

import httpx

key_name = os.environ["COSMOS_API_KEY_ENV"]
key = os.environ.get(key_name)
if not key:
    raise SystemExit("FAIL: Cosmos credential variable is unavailable")
model = os.environ["COSMOS_MODEL"]
url = os.environ["COSMOS_BASE_URL"].rstrip("/") + "/models"
with httpx.Client(trust_env=False, follow_redirects=False) as client:
    response = client.get(
        url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=120.0,
    )
    response.raise_for_status()
    document = response.json()
if not any(item.get("id") == model for item in document.get("data", [])):
    raise SystemExit("FAIL: configured Cosmos model is absent from /v1/models")
print(f"Cosmos model identity: PASS ({model})")
PY
```

Open the workspace, then create exactly one smoke attempt through the
same-origin Next.js proxy:

```bash
set -euo pipefail
CURATION_API=http://127.0.0.1:3000/api/curation
OPEN_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"dataset_alias":"local/pnp_trash","actor":"curator"}' \
  "$CURATION_API/workspaces/open")
printf '%s\n' "$OPEN_RESPONSE" | jq -e .

SMOKE_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"dataset_alias":"local/pnp_trash","episode_indices":[0]}' \
  "$CURATION_API/batches")
SMOKE_JOB_ID=$(printf '%s' "$SMOKE_RESPONSE" | jq -er '.job_id')
if test -z "$SMOKE_JOB_ID"; then
  printf '%s\n' 'FAIL: one-episode Cosmos smoke did not return a job ID' >&2
  exit 1
fi

cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/python backend/curation_worker.py \
  --workspace /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  run --job-id "$SMOKE_JOB_ID"
SMOKE_STATE_RESPONSE=$(curl -fsS "$CURATION_API/batches/$SMOKE_JOB_ID")
printf '%s\n' "$SMOKE_STATE_RESPONSE" | jq
SMOKE_STATE=$(printf '%s' "$SMOKE_STATE_RESPONSE" | jq -er '.state')
case "$SMOKE_STATE" in
  completed | completed_with_failures) ;;
  *)
    printf 'FAIL: one-episode Cosmos smoke ended in unexpected state: %s\n' "$SMOKE_STATE" >&2
    exit 1
    ;;
esac
```

Require terminal `completed` or `completed_with_failures`; a `manual_only`
episode remains eligible for human review and is never automatically rejected.
For a successful proposal, inspect its immutable artifacts:

```bash
find "$CURATION_WORKSPACE/artifacts/cosmos" -type f -path '*/request.json' -print
find "$CURATION_WORKSPACE/artifacts/cosmos" -type f -path '*/response.txt' -print
find "$CURATION_WORKSPACE/artifacts/cosmos" -type f -path '*/parsed.json' -print
find "$CURATION_WORKSPACE/contact_sheets" -type f -path '*/proposals/*.png' -print
```

In `request.json`, require `sampling.original_fps == 50`,
`sampling.target_fps == 2`, increasing source-frame indices and recorded
parquet timestamps, plus `request_body.media_io_kwargs.video.fps == 50` and
`do_sample_frames == false`. Confirm the base64 payload is replaced by a hash.
Inspect the raw response, parsed v2 status/times, six-cell proposal contact
sheet, and UI rendering. Null transitions must render `not observed`
placeholders without fabricated images or deltas.

## Run and monitor the full Cosmos batch

The API only persists a queued job. It never launches a worker. After the smoke
job is terminal, create the full 92-episode batch:

```bash
BATCH_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"dataset_alias":"local/pnp_trash"}' \
  "$CURATION_API/batches")
JOB_ID=$(printf '%s' "$BATCH_RESPONSE" | jq -er '.job_id')
test -n "$JOB_ID"
printf '%s\n' "$BATCH_RESPONSE" | jq

cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/python backend/curation_worker.py \
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

cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/python backend/curation_worker.py \
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
test ! -e "$CURATION_OUTPUT" && test ! -L "$CURATION_OUTPUT"
cd /home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash
sha256sum --check "$CURATION_WORKSPACE/source-files.sha256"

EXPORT_RESPONSE=$(curl -fsS -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:3000' \
  -d '{"dataset_alias":"local/pnp_trash"}' \
  "$CURATION_API/exports")
EXPORT_ID=$(printf '%s' "$EXPORT_RESPONSE" | jq -er '.export_id')
APPROVAL_SNAPSHOT_SHA256=$(printf '%s' "$EXPORT_RESPONSE" | jq -er '.approval_snapshot_sha256')
test -n "$EXPORT_ID"
test "${#APPROVAL_SNAPSHOT_SHA256}" -eq 64
printf '%s\n' "$EXPORT_RESPONSE" | jq

cd /home/jihun/work/lerobot-dataset-visualizer
test -n "$EXPORT_ID"
backend/.venv/bin/python backend/curation_export.py \
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
cd /home/jihun/work/lerobot-dataset-visualizer
test -n "$JOB_ID"
backend/.venv/bin/python backend/curation_worker.py \
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
cd /home/jihun/work/lerobot-dataset-visualizer
test -n "$EXPORT_ID"
backend/.venv/bin/python backend/curation_export.py \
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

cd /home/jihun/work/lerobot-dataset-visualizer
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
