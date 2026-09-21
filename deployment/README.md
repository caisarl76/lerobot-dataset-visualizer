# Local checkout

The active checkout is `/home/jihun/work/lerobot-dataset-visualizer`.
Run the launch commands below from that directory. Future feature worktrees
belong under its `.worktrees/` directory; see [AGENTS.md](../AGENTS.md).
The annotation backend requires Python 3.12 or newer and the pinned packages
in `backend/requirements-annotations.txt`.

# Current default: Qwen3.6-27B

The local annotation backend defaults to `Qwen/Qwen3.6-27B` on H100 through the
loopback SSH tunnel at port 34002. Start the default stack after reboot with
`bash deployment/start-annotation-local.sh`. Refresh
http://127.0.0.1:3000/annotate after switching providers. Existing saved jobs
retain their recorded generation configuration for reproducibility.

# Optional local annotation with Genon

This optional setup for http://127.0.0.1:3000/annotate uses the CPU annotation
backend at port 7861 and `qwen/qwen3.5-397b-a17b-fp8` at
`https://api.genon.ai/v1`. No HF Space or local GPU is required. Existing review
workspace and caches stay under
`/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations`.

The backend-only dotenv file must define `GENON_API_KEY` and
`LEROBOT_VLM_API_KEY=${GENON_API_KEY}`. Start after reboot from this repository:

```bash
bash deployment/start-annotation-genon.sh /home/jihun/work/GR00T-WholeBodyControl/.env
```

The script creates the `lerobot-annotation` tmux session and refuses to replace
an existing session. It loads the credential file only into the backend.
Generation sends selected dataset frames to Genon. Dataset publication remains
an explicit action in the review workflow. Refresh the annotation page after
switching providers so previously loaded configuration is refreshed.

# Annotation service on h100 GPU 0

The local visualizer and CPU annotation backend use Qwen3.6-27B through an
SSH tunnel. All three upstream modules remain enabled. Deployment defaults
limit concurrency to four VLM requests and one episode; the UI can override
individual settings without losing the endpoint configuration.

## GPU service

Qwen weights use `/mnt/data01/huggingface` on h100. Download them before the
switch (the existing Cosmos container provides the Hugging Face client):

```bash
ssh h100 'docker exec jihun-cosmos3-nano python3 -c '\''from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.6-27B", ignore_patterns=["*.md", "*.png", "*.jpg"]))'\'''
ssh h100 bash -s < deployment/start-qwen-h100.sh
ssh h100 docker logs --tail 40 jihun-lerobot-qwen36-gpu0
```

The script stops Cosmos and retains its container for rollback. Qwen uses
only GPU 0, 90% GPU memory, a 32,768-token context, eight CPUs, and a 96 GiB
host-memory limit. The existing vLLM 0.23.0 image is pinned by local image ID.
Its API binds only to h100 loopback port 34002; no public endpoint is opened.

## Local services

After installing the backend and building the UI as described in
[the backend guide](../backend/README.md), run from the repository root:

```bash
bash deployment/start-annotation-local.sh
curl --fail http://127.0.0.1:34002/v1/models
curl --fail http://127.0.0.1:7861/api/annotation/config
```

Open <http://127.0.0.1:3000/annotate>. The dedicated `lerobot-annotation`
tmux session contains `tunnel`, `backend`, and `ui` windows. Logs, caches,
and independent dataset drafts live under
`/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations`.
The local services survive closing the terminal; rerun the script after a
workstation reboot. Qwen's container restarts with Docker unless stopped.

`LEROBOT_ANNOTATE_CONFIG` selects the partial JSON defaults in
`annotation-h100.json`. API credentials, if needed, belong only in the
backend's `LEROBOT_VLM_API_KEY` environment variable.

To stop this local stack, use `tmux kill-session -t lerobot-annotation`.
To restore the former Cosmos service on h100:

```bash
ssh h100 docker stop jihun-lerobot-qwen36-gpu0
ssh h100 docker start jihun-cosmos3-nano
```

This rollback restores the former port 34001 service; change the annotation
model/endpoint defaults if the annotation backend should use Cosmos.

## Verified deployment (2026-09-07)

- Model revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- GPU 0: NVIDIA H100 80 GB; Qwen serving uses approximately 73,060 MiB.
- Real camera image inference succeeded through the SSH tunnel.
- Original `pnp_trash` converted to an independent 92-episode draft;
  official validation passed with no errors or warnings.
- All three enabled generation modules ran for episode 34 (21.88 seconds).
  Output: 13 persistent rows and 45 event rows; validation passed across
  the resulting 92-episode dataset. Only episode 34 was selected for generation.
- Emitted styles: task augmentation (11), plan (1), subtask (1), speech (1),
  VQA (44). No interjection or memory atoms were emitted for this episode;
  enabled modules do not guarantee every style for every episode.
- Prepared draft: `local/annotation-17ad579896f60259`.
- Generated draft: `local/annotation-74de5f7d90831524`;
  [review episode 34](http://127.0.0.1:3000/local/annotation-74de5f7d90831524/34).
- Generation job: `8c8604a1624048818a93d1a2c5632117`.

Validation checks structure and timing. Generated semantic labels still need
human review in the visualizer before training export.

## Combined private Space with hosted Qwen

The root Dockerfile now runs the existing Next UI on port 7860 and the official
annotation backend on container loopback port 7861. It calls the external
`qwen/qwen3.5-397b-a17b-fp8` model through an OpenAI-compatible Genon endpoint.
It does not load model weights or require an H100 endpoint. The local H100 launch
scripts above remain available for the existing local service.

Create a **private Docker Space** in an account/organization that supports Docker
hosting. The default dataset is `mncai/G1_Dex3_PickAndPlaceTrash`; the Space and
dataset are different repository types even if they share that ID.

Configure these Space secrets:

- `GENON_API_KEY`: the existing provider key; never upload the `.env` file.
- `HF_TOKEN`: a Hub token authorized to read the input and write the publication
  allowlist. Only the backend process receives either credential.

Configure these runtime variables:

- `ANNOTATION_VLM_API_BASE=https://api.genon.ai/v1`: the confirmed Genon endpoint.
- `ANNOTATION_WORKSPACE`: a writable **persistent mounted filesystem**. Configure
  storage before use; temporary container disk is not durable annotation storage.
- `ANNOTATION_HOSTED_PRIVATE_SPACE=1`: set only after verifying Space visibility.
- `ANNOTATION_HUB_REPOS=mncai/G1_Dex3_PickAndPlaceTrash`: allowed publication targets.

`SPACE_HOST` normally supplies the exact browser origin; an explicit
`ANNOTATION_BROWSER_ORIGIN` can override it. The launcher generates a shared
internal service credential for its two processes at each boot. Do not configure
an external backend URL or send model credentials to browser code. If either
process exits, the launcher stops the other so Space health cannot mask failure.

The deployment is not complete until the provider URL and image runtime are
verified, durable storage is configured, and an external-browser review/export
smoke test passes. The initial mncai creation attempt returned HTTP 402 requiring
an organization hosting plan; no subscription was purchased or Space created.

Provider verification (2026-09-09): the root `.env` key authenticated successfully
with HTTP 200 for the configured Qwen model and correctly identified a synthetic
red image. The supplied `thinking_token_budget=2048` was accepted by the API.
The pinned official LeRobot client does not expose that provider-specific field;
its `max_new_tokens` is a total completion limit, not a separate reasoning cap.
The key was not printed or committed. HF hosting and persistent storage remain
required before deployment.

The official LeRobot OpenAI client also returned valid image-based JSON
(`{"color":"#FF0000"}`) without provider-specific request extensions. The smoke
assertion initially expected the word `red`; the returned hex code is the correct
color. This checks client compatibility, not episode boundary accuracy.


### Default transition-pause import filter

On **Prepare annotation dataset**, **Import filters → Exclude transition pauses**
is checked by default for both local and Hub sources. Uncheck it to import without
this filter. The API equivalent is `exclude_transition_pauses: false` on
`POST /api/annotation/prepare`.

The filter handles Pose (stream mode 1) ↔ normal Planner (mode 2) and
frozen-upper-body Planner (mode 3). It excludes from the switch to the first of
three consecutive forward 0.2-second windows whose largest monitored joint
angle range exceeds 0.1 radians. Planner entry monitors the 12 leg joints;
Pose entry monitors all 29 body joints, so either arm manipulation or leg
movement ends the pause. Hands are not used as body-motion evidence.
All windows must lie within the same destination-mode segment. Joints are
identified by name, not column position. A transition is skipped if its required
joint fields are missing or no movement onset is detected. Pose pause/off/VR
3-point modes are not covered by this rule.

The preparation result reports the number of intervals and frames excluded.
Intervals are saved in the new workflow and appear on its **Exclude** timeline;
they can be adjusted or removed normally. They are materialized only during
export. Opening an existing workspace does not rerun the filter or overwrite
manual edits. The original source is unchanged. Workflow metadata retains the
filter settings and initially detected intervals for traceability.

This detects transition-related low motion, not proven stale network packets.
Review the excluded intervals before exporting the dataset.

# Local dataset monitor

Open [Monitor](http://127.0.0.1:3000/monitor) from the homepage to inspect local
collection folders, annotation runs, prompt coverage, frozen training exports,
and recorded Hugging Face publications. Monitoring is read-only: it does not
import datasets, launch generation, change reviews or exclusions, export data,
or upload to HF. **Prepare** only opens the existing preparation page with the
source path filled in; importing remains an explicit action there.

## Configuration and scope

`deployment/start-annotation-local.sh` passes `LEROBOT_MONITOR_ROOT` to the backend,
with `/home/jihun/work/GR00T-WholeBodyControl/outputs` as its default. That path
currently resolves to `/mnt/data/jihun/datasets/G1_WBT_GR00T`. Override it before
starting the local stack when a different collection root is needed:

```bash
LEROBOT_MONITOR_ROOT=/absolute/collection/root bash deployment/start-annotation-local.sh
```

The launcher refuses to replace an existing session. An already running backend
must receive the setting on its next start. For an existing stack, first confirm
that annotation generation/export/publication jobs are idle, build the UI, and
restart only the annotation backend and UI with their existing environment plus
this setting. Preserve the VLM tunnel, credentials, workspace, caches, provider
settings, and unrelated services.

The monitor is local-deployment functionality. With no backend root setting it
shows an unconfigured state; it does not guess a root or accept paths from the
browser. Existing annotation authentication and origin controls still apply.
Local filesystem paths are shown inside that boundary, not made into a general
filesystem browser.

Discovery reads immediate child dataset directories using LeRobot v2.1, v3.0,
and v3.1 metadata. Symlinks are resolved and deduplicated; candidates escaping
the configured root are excluded. Hidden, staging, cache, workspace, and export
infrastructure directories are excluded from collection discovery. Partial or
malformed metadata stays visible with an Updating/diagnostic state. Exports
outside the collection scan appear only through recorded workflow references.

Relationships require recorded provenance. Confirmed single-parent derivatives
are grouped; multi-source merges stay independent with their sources listed.
Folder names alone do not establish relationships. Group cards count displayed
collection groups and can overlap; source and derivative episode counts must
not be added as a count of unique demonstrations.

**Open review** uses the existing full `local/annotation-…` alias registered for
the selected run's checkpoint and its first retained episode. The monitor reads
the alias registry without adding entries. If the checkpoint has no registered
alias or retained episode, it cannot offer that review link. It remembers the
selected run per folder in browser storage; otherwise it chooses the most
recent workflow activity, not the most recently published run.

## Reading the counts and prompts

Collected episodes come from the current source metadata. Every run counter is
scoped to the selected run's checkpoint, without adding together separate runs.

| Field | Meaning |
| --- | --- |
| Imported | Episode identities captured by the selected run. |
| Accepted / Rejected / Pending | Explicit Keep / Delete / Pending decisions. |
| Retained | All imported episodes except Delete, including Pending. |
| Decision completion | `(Accepted + Rejected) / Imported`. |
| Current reviewed / review rate | Retained episodes with matching saved annotation and exclusion review hashes; rate is `Current reviewed / Retained`. |
| Accepted duration | Retained frames in Keep episodes divided by their FPS. |
| Not imported | Current source episode IDs absent from the run's original source identities. |
| Export frames/duration | Counts from the particular frozen export, independent of later edits. |

A current review does not imply Keep: a reviewed Pending episode is still
retained. Generation success does not imply human review, and accepted advisory
findings remain distinct from unresolved findings. No run means Not imported;
a known zero is different from unknown/unreadable evidence. Empty denominators
show an em dash with No episodes/No retained episodes, not 100%. Incomplete
counts and coverage remain visibly unknown or partial.

Expanded prompt ratios use **current-reviewed, non-deleted episodes**, including
reviewed Pending episodes. Counts follow the literal saved `subtask` prompt
active at each actual source-frame timestamp. Half-open excluded frame intervals
`[start_frame, end_frame)` are removed; a prompt starting inside a cut can remain
active when retained frames resume. Ratios divide by all calculable eligible
retained frames, including Unlabeled and Ambiguous buckets. Frame shares sum to
100% apart from rounding; episode presence counts can overlap. Durations use
the relevant episode FPS.

Literal whitespace differences remain distinct and are marked in the prompt
view. Missing labels do not fall back to `task_aug` or generic task prompts.
Conflicting active labels are Ambiguous. Unreadable annotations or timestamps
exclude the affected episode from the calculable denominator and make coverage
incomplete; they do not establish a complete zero-frame distribution.

## Freshness and publication history

The page polls cached lightweight summaries every 30 seconds while visible and
pauses while hidden. **Refresh** invalidates the lightweight summary cache.
Expanded details are cached and invalidated when their relevant run, review,
exclusion, or metadata signatures change. Updating snapshots preserve the last
good display and timestamp rather than mixing partially written records.

Source checks compare metadata identities and lengths; they do not hash or
decode videos. New IDs show Not imported; missing IDs or changed lengths show
Source changed. Unchanged metadata is not proof of identical source bytes; the
existing export process remains responsible for full source validation.

Publication history, current local changes, and source changes are separate.
An old receipt stays visible after local edits. Frozen export review digests can
establish Unpublished changes when linkage is available; missing history or a
missing manifest makes freshness unverified. Saved receipts and completed
publication jobs are the available history, not a reconstruction of every
past upload. Recorded unpublished exports also remain visible. A missing local
export is unavailable and has no working copy-path action.

**Check HF** is an explicit read-only remote check of a recorded repository and
revision using server-side credentials. It compares the current head with the
recorded commit and reports match, changed/advanced, missing, unavailable, or
access denied, with a check timestamp. It performs no upload or full content
audit and does not run on periodic refresh. A failed check does not erase the
historical receipt. Remote-check results are held in memory and reset on backend
restart; publication receipts remain durable.

## Verified reference dataset

The 2026-09-16 reference is `pnp_table_260909`, run
`b937e3f6925a4bafa5eb1bb70ea74ec8`: 87 collected/imported, 71 Keep, 16 Delete,
0 Pending, and 71/71 retained episodes currently reviewed. The recorded export
is GR00T v2.1 (`groot_v21`), subtask mode, 34,594 frames / 691.88 seconds.
These values describe that observation, not a promise that later edits cannot
change it.

- [Current review](http://127.0.0.1:3000/local/annotation-99d858fff3f3e9ff/episode_1?tab=annotations).
- [Recorded HF revision](https://huggingface.co/datasets/mncai/G1_Dex3_PickTable/tree/260915), commit `f3d31d6b481a61fc1779890df46be58fd38cd811`.
- Available training export: `/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations/workspace/drafts/8b78dd2573054344b16b77fd514ef6c7/pnp_table_260915`.

Use the export path shown alongside its format, mode, and revision when training.
The original 87-episode collection and the current editable checkpoint are
separate artifacts. See the [acceptance record](../docs/artifacts/dataset-monitor/acceptance.md)
for verification evidence and limitations.

## Fork deployment workflow

The upstream GitHub Actions Space deployment is restricted to
`huggingface/lerobot-dataset-visualizer`. Pushing to `caisarl76` or `genonai`
does not deploy to upstream's `lerobot/visualize_dataset` Space. Local service
startup is unchanged. To deploy a fork to a Space, explicitly configure its
repository identity, Space destination, credentials, and job condition first.

## Upstream episode-loader compatibility

The September 21, 2026 merge includes Hugging Face upstream through `80b0f987`.
Remote v3.0 datasets use `@huggingface/lerobot` for indexed episode lookup.
Local datasets, v3.1, and custom dataset roots retain the compatible metadata
reader and destination-scoped authentication. Video URLs remain appropriate
for browser access; internal backend credentials are not serialized into them.

The pinned JavaScript package currently limits index traversal to 64 files.
Missing v3.0 episodes or incomplete index listings fall back to the existing
reader. V2 video URLs continue to use metadata path templates directly, avoiding
the package's capped JSONL reader. Real-Parquet regression fixtures exercise
these cases, including camera-specific segment offsets and global frame indices.
