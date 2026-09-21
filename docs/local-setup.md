# Local visualizer and annotation guide

For teammates using [caisarl76/lerobot-dataset-visualizer](https://github.com/caisarl76/lerobot-dataset-visualizer) on their own computer. These instructions cover the annotation and Dataset Monitor implementation on `main` as of September 21, 2026.

## What runs locally

| Component | Address | Purpose |
| --- | --- | --- |
| Next.js UI | `http://127.0.0.1:3000` | Browse, annotate, review, export, and monitor datasets. |
| Python backend | `http://127.0.0.1:7861` | Read local files, run official LeRobot annotation modules, save reviews, and materialize exports. |
| Existing VLM endpoint | Configured separately | Generate labels from episode images. Required for automatic generation only. |

The UI and backend can run on CPU when using a remote VLM. Browsing and manual review do not require a VLM. Automatic generation requires an OpenAI-compatible **vision** endpoint that accepts images and the official annotation requests.

Start with the portable commands below. `deployment/start-annotation-local.sh` is configured for Jihun's workstation paths and `h100` SSH alias; it is not a generic installer.

## 1. Install the application

Use Linux with Git, Bun, uv, and a separate Python 3.12+ environment. The commands below use Python 3.13, matching the documented backend setup. Internet access is required to install dependencies and access Hub datasets or a remote VLM. Keep this environment separate from robot collection and training environments.

```bash
git clone https://github.com/caisarl76/lerobot-dataset-visualizer.git
cd lerobot-dataset-visualizer
bun install --frozen-lockfile
uv venv --python 3.13 backend/.venv
uv pip install --python backend/.venv/bin/python \
  -r backend/requirements-annotations.txt --torch-backend cpu
```

Install `requirements-annotations.txt`, not just `requirements.txt`: it includes the pinned official LeRobot annotation pipeline. Do not substitute an arbitrary LeRobot release. Video decoding uses PyAV (`video_backend: pyav`).

Allow enough disk space for source datasets, independent annotation drafts, Hub downloads, and frozen exports. Conversion and video export can take substantial time and disk space.

## 2. Configure persistent storage and the VLM

Run from the repository root. The example stores application state outside the checkout and monitors datasets under `~/datasets/collections`:

```bash
mkdir -p "$HOME/.local/share/lerobot-visualizer/workspace" \
  "$HOME/.cache/lerobot-visualizer" \
  "$HOME/.config/lerobot-visualizer" \
  "$HOME/datasets/collections"

cp deployment/annotation-h100.json \
  "$HOME/.config/lerobot-visualizer/annotation.json"
```

Edit `~/.config/lerobot-visualizer/annotation.json` before generation:

- `vlm.api_base`: your existing VLM's OpenAI-compatible URL, including `/v1`.
- `vlm.model_id`: the exact served model ID.
- Keep `vlm.auto_serve` set to `false`; this application does not launch model weights.
- The copied configuration selects `Qwen/Qwen3.6-27B` at `http://127.0.0.1:34002/v1`, with thinking disabled. Change model-specific settings when using another provider.

If your team has access to the existing H100 deployment, establish an SSH tunnel in a separate terminal (replace the SSH destination with your account):

```bash
ssh -N -L 34002:127.0.0.1:34002 user@your-vlm-server
```

This assumes a VLM is already serving on the remote host's loopback port 34002. It does not start one. For a hosted provider, use that provider's URL directly and its exact model ID. `deployment/annotation-genon.json` is an optional example for Genon's Qwen3.5 endpoint, not the default Qwen3.6 deployment.

## 3. Start the backend — terminal A

From the repository root:

```bash
export LEROBOT_ANNOTATE_CONFIG="$HOME/.config/lerobot-visualizer/annotation.json"
export LEROBOT_ANNOTATE_EXPORT="$HOME/.local/share/lerobot-visualizer/workspace"
export LEROBOT_ANNOTATE_CACHE="$HOME/.cache/lerobot-visualizer"
export LEROBOT_MONITOR_ROOT="$HOME/datasets/collections"
export LEROBOT_ANNOTATE_BROWSER_ORIGIN=http://127.0.0.1:3000
```

Change `LEROBOT_MONITOR_ROOT` to your actual collection parent directory. Each immediate child should be a dataset directory, for example:

```text
~/datasets/collections/
  pnp_table_260909/
    meta/info.json
    data/...
    videos/...
  another_collection/
    meta/info.json
    data/...
    videos/...
```

If the VLM requires a key, enter it only in this backend terminal:

```bash
read -rsp 'VLM API key: ' LEROBOT_VLM_API_KEY
export LEROBOT_VLM_API_KEY
printf '\n'
```

For private Hub datasets or publication, likewise provide a token with the required read/write permissions:

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
printf '\n'
```

These prompts avoid putting literal credentials into shell history. The backend does not automatically load a repository `.env` file. Never put credentials in `NEXT_PUBLIC_*` variables or commit them. Optionally restrict publication targets with a comma-separated backend `ANNOTATION_HUB_REPOS` allowlist.

Start one backend process:

```bash
backend/.venv/bin/python -m uvicorn backend.app:app \
  --host 127.0.0.1 --port 7861
```

Keep this terminal running. Preserve the workspace path across restarts: it holds workflow state, aliases, review history, drafts, and exports. The default workspace under `/tmp` is unsuitable for durable work, which is why these instructions override it.

## 4. Start the UI — terminal B

Open a separate terminal, return to the repository root, and run:

```bash
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 \
  bun run dev --hostname 127.0.0.1 --port 3000
```

Open **http://127.0.0.1:3000/**. Use `127.0.0.1` consistently; the backend checks the browser origin, so `localhost` is not interchangeable with this configuration.

For a production build, stop the development UI and use:

```bash
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun run build
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 \
  bun run start --hostname 127.0.0.1 --port 3000
```

The public backend URL is baked into the frontend build. Rebuild after changing it. These instructions bind both services to loopback for use on the same computer; shared/hosted deployment has a separate authentication configuration.

## 5. Verify the setup

In another terminal:

```bash
curl --fail http://127.0.0.1:7861/api/annotation/config
```

Expect JSON describing the pinned engine and annotation configuration. When using the example tunnel, also check:

```bash
curl --fail http://127.0.0.1:34002/v1/models
```

Use the appropriate provider authentication if required. A model listing only verifies connectivity; generate one episode before starting a whole-dataset batch to verify image inference and annotation compatibility.

## 6. Prepare, generate, and review a dataset

1. On the homepage, choose **Local directory**, enter an absolute dataset directory, and click **Prepare**. You can also open `/annotate` directly and choose a local or Hugging Face source such as `mncai/G1_Dex3_PickTable`. Local paths refer to files accessible by the backend process.
2. Prepare the annotation draft. The source is preserved; review and generation operate on independent drafts/checkpoints. Use the alias generated on your machine, not another teammate's `local/annotation-…` URL.
3. Review **Import filters → Exclude transition pauses**, enabled by default. For compatible G1 recordings it detects low-motion intervals after both Pose → Planner and Planner → Pose switches. Inspect the resulting red **Exclude** intervals; low motion is a candidate, not proof of stale data. Unsupported/missing joint or mode information may yield no intervals. For an already opened workflow, use **Detect transition pauses**.
4. In **Annotations**, enter the intended subtask prompts and generation settings. For a single full-episode instruction, use a single subtask spanning the whole episode, for example `approach the table and pick the bottle`. Inspect object names rather than leaving `OBJECT` placeholders.
5. Optionally annotate 3–5 representative episodes, **save**, and explicitly **Mark reviewed** on each. Enter their current draft episode IDs in **Example episodes**. Saving alone does not make an example eligible. Examples must belong to the same draft; generation preserves them.
6. Start with one target episode using **Generate draft**. Inspect the result, then select the remaining episodes or **All episodes**. Follow the job status and open the current checkpoint. Review failures and use **Resume unfinished episodes** when available.
7. Inspect video, available G1 motion replay, prompts, and boundaries. Use **+ Add subtask** to add an interval. Adjust timing on the timeline. On **Exclude**, drag/select a range or enter start/end seconds, then use **Exclude interval**. **Preview clipping** previews retained playback; actual clipping happens at export.
8. Use **Review & export** to mark unwanted episodes **Delete**, resolve quality findings, and **Keep remaining** when appropriate. Delete is a reversible workflow decision until export. Save edits and mark retained episodes reviewed only after inspecting their prompts and exclusions. Changed labels or exclusions invalidate their previous review.

Generation calls official LeRobot modules and validation, but output still requires human review. Quality findings can block export even when generation has completed. Do not clear findings or mark episodes reviewed solely to bypass those checks.

## 7. Export locally or publish to Hugging Face

In **Review & export → Export & publish**:

1. Select **GR00T v2.1** for the standalone GR00T training dataset, or **Rich annotations** to retain official language columns.
2. Choose instruction mode. **Subtask** uses the reviewed subtask text over each interval, including a single full-episode subtask. Task mode uses task augmentation prompts and rejects ambiguous variants; it is not a way to silently select among generated rephrasings.
3. Enter a dataset folder name. Leave the HF repository blank for local export. For upload, enter the exact repository and destination revision/branch; confirm you have write access.
4. Click **Preview export**. This materializes a frozen dataset with deletion and exclusion decisions applied and displays its path, counts, format, and destination. Use this **export path** for training, not the original collection or editable draft.
5. Inspect the preview. To upload, confirm its displayed destination and publish. Changing options or reviewed content requires a new preview. Keep the resulting Hub revision and commit receipt.

Original collections remain intact. Exported episodes may be renumbered; the export records source-to-output identity and retained-frame mappings. Back up the persistent workspace and keep source files available and unchanged while working: source changes can invalidate export provenance.

## 8. Monitor collections

Open **http://127.0.0.1:3000/monitor**. It reads immediate child dataset directories under `LEROBOT_MONITOR_ROOT` and links recorded annotation runs and exports.

- **Collected / Imported** distinguish current source episodes from those captured by a selected workflow.
- **Accepted / Rejected / Pending** reflect Keep / Delete / undecided decisions. Retained includes Pending episodes as well as Keep.
- **Review rate** counts retained episodes whose saved review still matches their current annotations and exclusions.
- **Prompt ratios** measure literal reviewed subtask text over retained frames; unlabeled and ambiguous coverage are shown separately.
- **Export and HF history** show recorded artifacts and publication receipts. **Check HF** explicitly verifies the recorded remote revision; it does not upload anything.

Monitoring is read-only. It does not import new episodes or discover arbitrary past uploads from folder names. A fresh teammate installation has no other workstation's review history unless that workflow state and its required datasets are migrated; cloning Git only transfers application code.

## Troubleshooting and restart

| Symptom | Check / action |
| --- | --- |
| Local directory option is missing | Start/build the UI with the absolute `NEXT_PUBLIC_ANNOTATE_BACKEND_URL` above. |
| Backend unavailable or origin rejected | Check terminal A and port 7861; use `http://127.0.0.1:3000` exactly. |
| `plan: Connection error` | Check the VLM URL, model ID, tunnel, credentials, and provider availability. Backend health alone does not verify VLM access. |
| Few-shot examples must be explicitly marked reviewed | Save and Mark reviewed on those exact draft episode IDs after the latest edits. |
| Generation completed but labels appear absent | Open the job's current checkpoint and its Annotations view, rather than an older source/draft alias. |
| Subtask prompt mismatch or unresolved findings at export | Inspect the affected episode and workflow prompt rules; correct/review labels and resolve findings before creating a new preview. |
| Ambiguous `task_aug` variants | Review the task annotations or use Subtask mode when reviewed subtask intervals are the intended training instructions. |
| Monitor is unconfigured/empty | Set the backend's `LEROBOT_MONITOR_ROOT`, check child dataset metadata and permissions, and restart the backend. |
| New collection episodes are not in a workflow | Monitoring may show Not imported or Source changed; prepare a new workflow for the changed source. |
| Port already in use | Inspect the existing process before starting another instance; run only one backend against a workspace. |

Stop the UI/backend with Ctrl+C in their respective terminals when jobs are idle. Restart using the same configuration and storage paths. A backend restart marks unfinished jobs failed; inspect the checkpoint and resume/retry through the workflow. Do not delete the workspace to fix a connection error.

For code updates, stop services when idle, run `git pull --ff-only` in a clean checkout, reinstall dependencies with the commands in step 1, and restart. Rebuild first if using production mode. Preserve local configuration and workspace data.

## Further reference

- [Backend behavior and API details](../backend/README.md)
- [Existing workstation/hosted deployment and monitor semantics](../deployment/README.md)
- [Dataset Monitor acceptance record](artifacts/dataset-monitor/acceptance.md)
