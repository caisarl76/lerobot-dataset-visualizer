# Annotations and PnP-trash curation backend

This FastAPI service supports two isolated workflows:

- the existing LeRobot v3.1 annotation and export API
  (`language_persistent`, `language_events`, and `tools`); and
- the local v2.1 `task_index` workflow for Cosmos proposals, human review,
  deterministic export, GR00T validation, and atomic publication.

The v3.1 API remains available when curation is not configured. Curation is
all-or-nothing: if any required curation variable is present, every required
variable must validate before the backend starts.

## Install

From the repository root:

```bash
cd "$CURATION_REPO_ROOT"
python -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
bun install
```

`backend/requirements.txt` includes `pytest`, so the documented backend gate
uses the same virtual environment as the service.

## PnP-trash runtime configuration

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

The user-approved 2026-08-28 source authority contains 190 immutable regular
files. Its canonical manifest SHA-256 is
`5962d8630f06e6260adbae15a3d7ee5f0a1a745c3a12466add8722c2e0da9577`.
The top-level ancillary `pnp_trash.xlsx` is 13,644 bytes with SHA-256
`989f6968e5cf8ee0972b850199c948dd75ce140480c82cbe368053cde6ab34c9`;
it remains an immutable source asset and is not annotation authority.

The operator's existing server/runtime configuration must separately provide
the variables `CURATION_BEARER_TOKEN`, `COSMOS_BASE_URL`, `COSMOS_MODEL`,
`COSMOS_API_KEY_ENV`, and `COSMOS_ENDPOINT_IDENTITY`. Do not put their values
in this repository.

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
validation before it creates a job; the worker reads it only when building a
Cosmos attempt processor. A distinct name such as `COSMOS_API_KEY` is valid;
FastAPI and the worker reject collisions with inherited process names,
`CURATION_*`/`NEXT_PUBLIC_*`, and named Cosmos/Isaac settings before reading
or forwarding the target. The exporter receives neither the bearer nor Cosmos
base/API-key configuration and makes no Cosmos call. Neither CLI accepts a
secret argument. `CURATION_BACKEND_HOST` is FastAPI-only, optional, defaults
to `127.0.0.1`, and rejects a non-loopback value.

Do not launch either CLI directly from the ambient operator environment. Use
the runbook's explicit `env -i` `run_curation_worker` and
`run_curation_exporter` functions. They enforce the process-specific
allowlists and remove the dynamic Cosmos credential from the exporter even
when it is present in the parent shell.

Next.js requires only `CURATION_BACKEND_URL`, `CURATION_BEARER_TOKEN`, and
`NEXT_PUBLIC_DATASET_URL`. Its bearer value must exactly match the backend's.

Only `NEXT_PUBLIC_DATASET_URL` is browser-visible. Never create a
`NEXT_PUBLIC_*` token, key, Cosmos endpoint, model, or endpoint-identity
variable. The Next.js server injects `CURATION_BEARER_TOKEN` into same-origin
JSON requests on the server side.

See [.env.example](../.env.example) for the non-secret dotenv mapping and
[the curation runbook](../docs/pnp-trash-curation-runbook.md) for validation,
batch, review, export, and recovery procedures.

## Start the curation services

Start the backend from the repository root. This package-relative import is
intentional and is the supported curation entrypoint:

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

Open
`http://127.0.0.1:3000/local/pnp_trash/episode_0?tab=annotations` only after
the runbook preflight passes. FastAPI never starts a Cosmos worker or exporter
in the background; their persisted IDs must be run through the separate CLIs.
Configured FastAPI startup creates the canonical source manifest and
initializes curation.sqlite3 before serving. Task 14 integration is not
complete: the approved-source loopback and browser smoke remains pending until
the operator loads the secure runtime configuration and executes the runbook.

## Curation routes

Read-only dataset bytes are available at
`/api/local-datasets/{org}/{dataset}/resolve/main/{asset_path}`. The route
supports full, range, and `HEAD` responses and deliberately rejects every
request carrying `Authorization`.

Mutable JSON routes require the curation bearer token:

| Method  | Path                                              | Purpose                                     |
| ------- | ------------------------------------------------- | ------------------------------------------- |
| `POST`  | `/api/curation/workspaces/open`                   | Register/open the immutable source snapshot |
| `GET`   | `/api/curation/summary`                           | Dataset review counts and suggestions       |
| `GET`   | `/api/curation/episodes/{episode}`                | Review state and active proposal            |
| `PATCH` | `/api/curation/episodes/{episode}/draft`          | Save an optimistic-concurrency draft        |
| `POST`  | `/api/curation/episodes/{episode}/approve-keep`   | Approve a complete seven-step episode       |
| `POST`  | `/api/curation/episodes/{episode}/approve-reject` | Approve rejection                           |
| `POST`  | `/api/curation/episodes/{episode}/reopen`         | Invalidate approval and reopen review       |
| `POST`  | `/api/curation/batches`                           | Freeze and queue a Cosmos batch             |
| `GET`   | `/api/curation/batches/{job-id}`                  | Read persisted batch status                 |
| `POST`  | `/api/curation/batches/{job-id}/retry`            | Create an immutable child retry batch       |
| `POST`  | `/api/curation/batches/{job-id}/cancel`           | Idempotently request cancellation           |
| `POST`  | `/api/curation/exports`                           | Freeze and queue an approval snapshot       |
| `GET`   | `/api/curation/exports/{export-id}`               | Read persisted export status                |

## Existing v3.1 annotation API

The existing service writes per-episode persistent identity and exact-frame
events to `data/chunk-*/file-*.parquet`. Persistent styles are `subtask`,
`plan`, and `memory`; event styles are `interjection` and `vqa`; speech tool
calls use event storage. The dataset-level `tools` column carries the `say`
tool schema, and the legacy `subtask_index` column is dropped.

| Method | Path                                       | Purpose                                |
| ------ | ------------------------------------------ | -------------------------------------- |
| `GET`  | `/api/health`                              | Liveness and style catalog             |
| `POST` | `/api/dataset/load`                        | Cache and read dataset metadata        |
| `GET`  | `/api/episodes/{episode}/atoms`            | Read saved atoms                       |
| `POST` | `/api/episodes/{episode}/atoms`            | Write exact-frame atoms                |
| `GET`  | `/api/episodes/{episode}/frame_timestamps` | Read frame timestamps                  |
| `POST` | `/api/export`                              | Rewrite shards into a new directory    |
| `POST` | `/api/push_to_hub`                         | Export and push to a target repository |

Legacy annotations are stored in `<dataset_root>/meta/lerobot_annotations.json`
and v1 `subtasks`/`high_levels` data is migrated on load. For a legacy-only
run, the original `NEXT_PUBLIC_ANNOTATE_BACKEND_URL` configuration remains
supported; it is not used by the PnP-trash curation workspace.

## Regression gates

```bash
cd "$CURATION_REPO_ROOT"
backend/.venv/bin/python -m pytest -q backend/tests
```

```bash
cd "$CURATION_REPO_ROOT"
bun run format:check && bun run validate
```

The backend suite includes the v3.1 compatibility gate. Both commands are
mandatory before running real curation or publication.
