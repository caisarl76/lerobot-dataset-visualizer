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

## Official annotation pipeline

Open `/annotate` to prepare a local or Hub dataset, then use its **Annotations**
tab to generate and review a draft. This workflow calls LeRobot's official
plan/subtask/memory/task augmentation, interjection/speech, and general VQA
modules, its `StagingValidator`, and its `LanguageColumnsWriter`. The source is
pinned to [`3f2c29e`](https://github.com/huggingface/lerobot/tree/3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e).
There are no PnP phase labels in this path. Official prompts and output schemas
remain unchanged; optional reviewed examples are added as VLM context.

### Few-shot annotation

Manually annotate and save 3–5 representative episodes in the **same draft
dataset**, then enter their numeric IDs in **Example episodes** (for example,
`0, 4, 12`). Enable **All episodes** and click **Generate draft**. The selected
examples are validated, preserved, and excluded from generation. Other episodes
are generated using their saved labels and timestamped camera contact sheets.
Generated labels are never automatically treated as reviewed examples.

This is in-context guidance, not model training. Use examples showing the
object names and subtask granularity you want, including representative task
variations. Quality still needs human review. Official modules, staging,
validation, and writing remain in use.

The API accepts `example_episode_indices` alongside `episode_indices` and
`config`. At most five distinct examples are allowed. Empty/invalid examples,
missing video, and jobs with no remaining target episodes fail explicitly.
The deployed 32k context is bounded to 12 example camera sheets and 20,000
characters of example labels per module; select fewer examples if exceeded.
Each sheet contains six sampled frames. Selected IDs and annotation hashes are
recorded in `meta/annotation_pipeline.json` for reproducibility.

### Runtime setup

Use **Annotations → Delete episodes** to enter unwanted episode IDs and
**Create cleaned draft**. The official `delete_episodes` operation creates a
new dataset, rewrites affected video segments, renumbers the remaining episodes,
and updates metadata/statistics. Saved annotations and robot modality metadata
are retained. The source is kept for recovery. `meta/annotation_pipeline.json`
records the deleted IDs and old-to-new episode mapping. Example IDs must refer
to the new numbering when using few-shot generation on the cleaned draft.

The matching API is `POST /api/annotation/delete-episodes` with a dataset
reference and `episode_indices`. At least one episode must remain.

Use a separate Python 3.12+ environment, outside collection/training runtimes:

```bash
uv venv --python 3.13 backend/.venv
uv pip install --python backend/.venv/bin/python -r backend/requirements-annotations.txt --torch-backend cpu
LEROBOT_ANNOTATE_BROWSER_ORIGIN=http://127.0.0.1:3000 \
  backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7861
```

In a second terminal, from the same checkout:

```bash
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun run dev --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000/annotate`. No `CURATION_*` configuration is needed.
A CPU annotation backend can decode video and call an existing VLM server;
it does not need to load the VLM locally. Configure `vlm.api_base` and
`vlm.model_id` under **Advanced options**. The default endpoint is
`http://localhost:8000/v1`. If authentication is needed, set
`LEROBOT_VLM_API_KEY` in the backend's environment, never a `NEXT_PUBLIC_*`
variable. Server launch commands and keys cannot be submitted by the UI.

Module switches preserve their nested settings. Advanced options accept the
upstream `plan`, `interjections`, `vqa`, `vlm`, `executor`, `seed`, and
`video_backend` configurations, including task derivation, rephrasing axes,
seeded relabeling, sampling density, memory/plan emission, interjection limits,
VQA question types, cameras, and concurrency. All three modules are enabled
by default. Use upstream's CLI for auto-serving and HF Jobs infrastructure;
the visualizer runs the official executor against an existing endpoint.
See the [official guide](https://huggingface.co/docs/lerobot/main/en/annotation_pipeline).

Preparation uses the official v2.1-to-v3 converter functions on a **new**
directory. Existing v3 datasets are copied. Conversion preserves episode/frame
identities and writes actual shard/video offsets; it does not merely change
the version string. The pinned converter calls its output `v3.0`, while the
annotation documentation calls the language schema v3.1. Both display in this
UI. Video-stored RGB cameras are required for automatic generation.

Generation produces another independent draft. All episodes are staged before
writing, so selecting one episode preserves others sharing its parquet shard.
Saved edits seed staging; disabled modules retain their annotations. Enabled
modules replace their own annotations **in the generated draft**. Source
parquet and source edits remain intact. Videos use hardlinks with a copy
fallback and must be treated as immutable. Failed drafts are retained for
inspection but are not registered as successful outputs.

The existing atom editor can save incomplete drafts to
`meta/lerobot_annotations.json`. **Validate** reports upstream errors/warnings;
**Save dataset** saves current edits and exports only after official validation
passes. Export destinations must be new directories outside the source.
The complete rich columns remain available for other models/recipes.
`tools` is dataset metadata in `meta/info.json`, not a per-frame column.

`LEROBOT_ANNOTATE_EXPORT` selects the workspace (default
`/tmp/lerobot_visualizer_annotate_exports`); `LEROBOT_ANNOTATE_CACHE` selects the
Hub cache. Use persistent storage for durable work. Local aliases and job
results persist under the workspace. Run one backend process: a restart marks
unfinished jobs failed; open the prepared dataset and start a new draft.

| Route                           | Purpose                                                 |
| ------------------------------- | ------------------------------------------------------- |
| `GET /api/annotation/config`    | Pinned engine revision and upstream defaults            |
| `POST /api/annotation/prepare`  | Copy/convert a local or Hub source                      |
| `POST /api/annotation/jobs`     | Generate all or selected episode indices                |
| `GET /api/annotation/jobs/{id}` | Job status, validation and draft location               |
| `POST /api/annotation/validate` | Validate edited atoms against source timestamps/cameras |

### GR00T N1.7 training view

Export reviewed annotations through **Save dataset**, then map that rich output
onto the matching original GR00T-compatible v2.1 dataset:

```bash
backend/.venv/bin/python backend/groot_export.py \
  --annotated-root /path/to/validated-rich-export \
  --source-root /path/to/original-v21 \
  --output-root /path/to/new-groot-training-view \
  --mode subtask
```

`subtask` uses the current subtask with the original task as fallback. `task`
uses the original task; `--task-variant N` selects an official task rephrasing.
`context` composes the task plus active subtask, plan, memory, and latest
interjection. Only atoms at or before the current frame are eligible. The
adapter rejects mismatched episode/frame/timestamp identities and keeps action,
state, video and continuous statistics unchanged. Its output supplies
`annotation.human.task_description` through per-frame `task_index` and
`meta/tasks.jsonl`, using the existing N1.7 loader without modifying the model.
The v2 source must be materialized; symlinked files are rejected.

This conditions **action prediction**. It does not add VQA, language or speech
output losses to GR00T N1.7. Context must also be available at inference;
additional context's policy benefit has not been measured. Use the rich source
with a separate language-generation training recipe when those outputs are
needed.

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
events to `data/chunk-*/file-*.parquet`. Persistent styles include `task_aug`, `subtask`,
`plan`, and `memory`; generated event styles are `interjection` and `vqa`; speech tool
calls use event storage. The dataset-level `tools` metadata carries the `say`
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

# Prediction snapshots and human review

Successful generation captures the postprocessed official annotations in a new
`meta/annotation_predictions/<timestamp>_<id>.json` before exposing the draft.
It records generated episode IDs, few-shot labels and IDs, model configuration
(without API keys or launch commands), source path, and LeRobot revision.
Edits never rewrite these files; retries copy previous snapshots and add a new one.
Snapshots contain only generation targets, so examples are not evaluation targets.

In the annotation panel, save edits, then explicitly select **Mark reviewed**.
Saving alone never marks an episode reviewed. Review records are persisted in
`meta/annotation_reviews.json`, bound to an annotation hash. Changed saved labels
or regeneration invalidate the affected reviews. **Reopen review** clears the mark.
The review API rejects stale hashes with HTTP 409.

Existing datasets start unreviewed; old predictions are not inferred from edited
labels. Earlier manually frozen evaluation artifacts remain separate. Episode
deletion creates a new identity mapping; its source retains the original history.

## Hosted review workflow

Prepare a Hub revision through `/api/annotation/prepare`; the backend resolves
it to a commit before downloading. Prepared drafts carry a run pointer and the
persistent run store tracks the current job, episode decisions and publication
state. Review decisions never delete files. Export applies the complete deletion
set once to an independent copy. A saved edit invalidates review and export.

`GET /api/workflow/<local-alias>` returns review/quality history and first/latest
boundary MAE. Its `/decision`, `/export`, and `/publish` POST endpoints require
the current run revision. Export and publication use persisted annotation jobs;
UI polling can reconnect after navigation. Atom saves on these drafts require
the hash returned by GET atoms to reject stale browser tabs.

H100 deployment: `deployment/annotation-hosted.compose.yaml` runs one CPU
annotation worker with durable storage and an HTTPS gateway. It connects to the
existing Qwen GPU0 service on loopback 34002; it does not create a second VLM.
Provide ANNOTATION*HOSTNAME, ANNOTATION_BROWSER_ORIGIN, ANNOTATION_WORKSPACE,
ANNOTATION_BACKEND_TOKEN, ANNOTATION_HUB_REPOS and HF_TOKEN in the server
runtime environment. The backend credential must match the Space secret.
Do not copy local legacy CURATION*\* settings into this hosted deployment.

The Space must be private. Set runtime variables ANNOTATION_BACKEND_URL,
ANNOTATION_BROWSER_ORIGIN, ANNOTATION_HOSTED_PRIVATE_SPACE=1 and the backend
credential as a secret. The Docker build sets
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=/api/annotation. Private Space access is the
team authentication boundary; exact-origin checks alone do not authorize users.
Browser video ranges and API requests pass through the same-origin proxy.

Do not configure the production target for smoke tests. Use a disposable Hub
repository in ANNOTATION_HUB_REPOS and review the frozen export before publishing.
The full official language artifact is published to annotations/<run_id>;
source-compatible main is updated with an explicit parent-commit check. An
oversized atomic publication is rejected before main changes. Supply actual
Space and HTTPS names before deployment; these files do not allocate a hostname,
Space, storage or GPU.

Corrupt raw v2.1 episodes remain in a source-review workspace with their original
IDs and failure evidence. Healthy targets and reviewed examples run through an
ephemeral official-converted subset; only their generated sidecars return to the
complete review workspace. At frozen export, retained episodes are materialized
once with the final identity mapping and then converted and validated officially.
This recovery exception is necessary because the pinned official deletion API
requires v3 and whole-source conversion cannot read a corrupt v2 video/shard.
Healthy prepared v3 datasets continue to use official deletion once at export.
A damaged shared v3 shard that cannot be read still requires source repair;
publication must not reinterpret its contents or fabricate missing frames.

Run creation captures original file hashes; export rejects later source changes.
Pre-binding experimental runs must be prepared again to obtain that provenance.
Generation checkpoints each episode, and **Resume unfinished episodes** reuses
saved prompts/configuration/examples while skipping successful targets and delete
flags. Each failed attempt is retained separately from successful predictions.
