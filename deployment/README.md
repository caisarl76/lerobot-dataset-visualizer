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
