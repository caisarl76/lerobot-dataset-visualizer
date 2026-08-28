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
python -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
bun install
```

`backend/requirements.txt` includes `pytest`, so the documented backend gate
uses the same virtual environment as the service.

## PnP-trash runtime configuration

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

`backend.app:app`, `backend/curation_worker.py`, and
`backend/curation_export.py` all call `CurationSettings.from_env()`. Every one
of those three Python processes therefore requires the complete settings
environment: the five non-secret mapping values consumed by
`CurationSettings` (`CURATION_DATASET_ALIASES_JSON`, `CURATION_WORKSPACE`,
`CURATION_OUTPUT`, `CURATION_BROWSER_ORIGIN`, and `ISAAC_GROOT_ROOT`), all five
external runtime names, including the `COSMOS_API_KEY_ENV` name. Only the
backend and worker receive the actual credential variable named by
`COSMOS_API_KEY_ENV`; the exporter must not receive that secret. The backend reads the variable named by
`COSMOS_API_KEY_ENV` during batch capability validation before it creates a
job; the worker reads it for Cosmos calls. The exporter makes no Cosmos call
and launches without the target credential variable. Neither CLI accepts a
secret argument. `CURATION_BACKEND_HOST` is optional and defaults to
`127.0.0.1`; all three Python entrypoints reject a non-loopback value while
loading settings.

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
cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Start Next.js in a second terminal containing only `CURATION_BACKEND_URL`,
`CURATION_BEARER_TOKEN`, and `NEXT_PUBLIC_DATASET_URL`:

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
bun run dev --hostname 127.0.0.1 --port 3000
```

Open
`http://127.0.0.1:3000/local/pnp_trash/episode_0?tab=annotations` only after
the runbook preflight passes. FastAPI never starts a Cosmos worker or exporter
in the background; their persisted IDs must be run through the separate CLIs.

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
cd /home/jihun/work/lerobot-dataset-visualizer
backend/.venv/bin/python -m pytest -q backend/tests
```

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
bun run format && bun run validate
```

The backend suite includes the v3.1 compatibility gate. Both commands are
mandatory before running real curation or publication.
