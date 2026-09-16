# Dataset collection, review, and publication monitor

Date: 2026-09-16
Repository: `/home/jihun/work/lerobot-dataset-visualizer`
Status: Design sections approved; written specification awaiting user review.

## Purpose

Add a `/monitor` page that answers, for each collection folder: how much data exists, how much has been accepted or rejected, how much has a current review, what training prompts it contains, and which reviewed version was published to Hugging Face.

The configured local collection root is `/home/jihun/work/GR00T-WholeBodyControl/outputs`, currently a symlink to `/mnt/data/jihun/datasets/G1_WBT_GR00T`. Dataset folders, annotation checkpoints, and published exports are different objects. The monitor must show their relationships without conflating their counts or freshness.

The user approved a folder overview with confirmed derivatives grouped beneath their source, expandable annotation runs and publications, reviewed-frame prompt distributions, and read-only monitoring. Existing collection, annotation, and publishing workflows remain responsible for their mutations.

## First-version scope

- Discover local LeRobot datasets immediately beneath the configured collection root.
- Associate existing annotation runs by canonical source path.
- Group confirmed derivative datasets and expose all runs and known exports.
- Show collection counts, retention decisions, current reviews, exclusions, retained duration, generation problems, and publication state.
- Show prompt distribution for a selected run, using reviewed retained frames.
- Open existing review pages, prepare an unimported source, open HF versions, and copy an available training path.
- Manual refresh, lightweight periodic refresh while visible, and explicit remote HF verification.

Non-goals: launching collection, automatic imports/generation/filtering, review or deletion controls on this page, automatic publication, training controls, new VLM accuracy analysis, arbitrary filesystem browsing, historical reconstruction from naming conventions, and a separate database or always-running filesystem watcher.

## Identity and discovery

### Collection folders

Use a server-side `LEROBOT_MONITOR_ROOT` setting; configure the local launcher with the root above. Resolve the root and candidate paths before comparing identities. Root access is local-deployment functionality; leave the monitor unavailable when no root is configured. Existing authentication and origin protections apply to monitor endpoints. Browser requests cannot supply arbitrary scan roots or filesystem paths.

Scan immediate child directories for `meta/info.json`. A complete metadata file identifies a dataset even if it has zero completed episodes. Read completed episode records using the existing v2.1 and v3 metadata conventions; distinguish metadata totals from successfully readable committed records if they disagree. Exclude hidden directories, configured workspace/cache/export infrastructure, and temporary staging directories. Do not recursively scan the annotation workspace for collection folders. Discover exports through workflow metadata instead.

Resolve symlinks for deduplication. Only discover collection candidates whose resolved paths remain inside the configured root. Retain a readable collection path and canonical identity. References to registered workflow/export paths outside a direct collection child can appear as linked artifacts, but are not scan roots.

A candidate with partial metadata is shown as Updating rather than silently disappearing. A directory that contains neither dataset metadata nor evidence of a dataset is omitted. A stable parse error or missing referenced asset is a per-folder diagnostic, not a page-wide failure.

### Relationships

Use explicit provenance: run `source_root`, run/export identities, `source_episode_mapping.json`, supported merge manifests, and recorded export reports. A mapping of episode numbers alone does not prove a parent folder; it must resolve to a recorded run or source identity. Never infer parentage from `_full_task`, `_subtask`, dates, or other name patterns.

A derivative with exactly one confirmed parent is nested beneath it. A merged dataset with multiple parents stays an independent row and lists its sources, avoiding duplicate placement. Unresolved or conflicting provenance remains ungrouped with a clear indicator. Prevent cycles when following provenance.

Do not sum source and derived folders into a purported count of unique physical demonstrations. Top-level summary cards count displayed collection groups and groups needing attention, not globally deduplicated episodes.

### Runs and exports

Resolve run `source_root` against the canonical collection path. Temporary test runs outside the configured root do not appear as collection entries. Each run remains individually identifiable; never add counts across runs for the same source.

The browser remembers the selected run per folder in local storage. Without a valid remembered choice, select the most recently updated run, using the existing run record's file modification time with run ID as a deterministic tie-breaker. Label the timestamp as workflow activity, not collection time. Add no new persisted execution-state model. A published run is not silently preferred over a newer active run.

Expose known publications independently of the selected run. Read the current run publication receipt and completed publication-job receipts where available; deduplicate by repository, revision, and commit. Show associated format, frame counts, and export path only when a receipt can be reliably linked to its frozen export. Do not invent publication history when old receipts are absent. Missing training files are shown as unavailable rather than producing a working-looking copy-path action.

## Metrics and their denominators

All run metrics are scoped to the visibly selected run and its checkpoint frame counts/timestamps. Source totals are scoped to the live collection folder. Do not substitute live source frame counts into checkpoint metrics after source changes.

| Metric | Definition |
| --- | --- |
| Collected episodes | Completed episode identities in the current source metadata; zero is a valid value. |
| Imported episodes | Episode identities captured by the selected run. |
| Accepted | Run episodes explicitly marked Keep. |
| Rejected | Run episodes explicitly marked Delete. |
| Pending decisions | Imported episodes whose decision is Pending. |
| Retained | Imported episodes not marked Delete; includes Pending. |
| Decision completion | `(Accepted + Rejected) / Imported`. |
| Current reviewed | Retained episodes whose saved annotation and exclusion hashes match their explicit review record. |
| Review rate | `Current reviewed / Retained`. |
| Not imported | Source episode identities absent from the run's immutable original identity set. |
| Usable duration | Accepted episodes' retained frame counts divided by their FPS, summed across episodes. This measures accepted data, not a separate judgment of task success. |
| Exported frames/duration | Values belonging to the specific frozen export, never recomputed from a subsequently edited run. |

For zero denominators show an em dash and `No episodes` or `No retained episodes`, not 100%. For no run, review and decision fields are Not imported, not zero percent reviewed. Display missing imported source IDs separately; a reduced folder total is not evidence of intentional rejection.

Keep review and retention decisions separate. A reviewed episode can still have a Pending decision. Generation success does not constitute review. A VLM warning does not constitute rejection. The first version uses Accepted instead of the ambiguous label Valid.

Current reviews reuse `annotation_history.annotation_hash` and `exclusions_hash` semantics. Batch-load sidecars; do not call the full workflow payload for every row because it loads predictions and computes accuracy metrics unrelated to this page. Where a sidecar episode is absent, read the authoritative dataset annotations through existing readers before determining review state; never substitute an empty list for unreadable data.

Needs attention distinguishes unreviewed/stale reviews, pending decisions, generation failure, unreadable data, and unresolved findings. Accepted findings remain visible as accepted advisory findings rather than unresolved blockers. Derive generation execution status from the current persisted job, as the existing workflow does.

## Prompt distribution

Calculate distributions on demand for the selected run's non-deleted episodes with current explicit reviews. Display this eligibility scope and counts, including reviewed episodes still awaiting a Keep/Delete decision. Export preview counts remain a separate concept.

Use the saved `subtask` atoms and actual source-frame timestamps. At each retained source frame, use the subtask active at that timestamp, with the same boundary semantics as the training exporter. Exclusions are half-open source-frame intervals; all excluded frames are removed before counting. A prompt that begins inside an excluded interval can still be active when retained frames resume.

For each literal prompt display retained-frame count, share of all eligible retained frames, retained duration, and distinct episode count containing at least one frame assigned to that prompt. Frame percentages, including Unlabeled/Ambiguous buckets, sum to 100% apart from rounding. Episode percentages can overlap and must not be presented as mutually exclusive slices. Aggregate durations per episode FPS.

Do not substitute generated `task_aug` variants or generic OBJECT prompts when no subtask covers a frame. Report that frame as Unlabeled. Conflicting active atoms are Ambiguous; do not choose arbitrarily. Preserve literal prompt differences; mark otherwise invisible whitespace differences in the detail view instead of silently merging labels. Missing source timestamps or unreadable annotation data make the affected coverage unknown and visibly excluded from the calculable denominator.

Return useful coverage counts even when some episodes fail to load. Do not claim a complete distribution unless all eligible episodes were evaluated.

## Source growth and publication freshness

Source growth, review changes, and remote publication state are independent indicators.

- New source IDs: show the Not imported count even if the selected run is fully reviewed and published.
- Missing source IDs or changed episode lengths: show Source changed and require investigation; never silently reassign review identities.
- Current annotation/exclusion hashes: determine whether reviews are stale.
- Existing run publication state and frozen review digest: distinguish Published from Unpublished changes. Reuse the export digest rules when a linked frozen manifest is available; otherwise state that freshness cannot be verified.
- An old publication receipt remains visible after local edits; it is not proof that the current draft was uploaded.

Lightweight refresh uses metadata and sidecar file signatures, not whole-video hashing. Unchanged IDs and frame counts do not prove byte-for-byte source identity. Label the result as a metadata check; retain the existing full source hash validation at export. Do not imply continuous content verification.

`Check HF` checks only a known publication record. Use server-side HF credentials and the recorded repository/revision; return the current head and compare it with the recorded publication commit. States: matches recorded commit, branch advanced/changed, missing revision, unavailable, and access denied. A timeout preserves the historical publication receipt and marks the remote check unavailable, not unpublished. Show the last check time. This verifies revision identity, not a new full download/content audit. Do not contact HF on every periodic refresh.

## Page design

Add Monitor to homepage navigation and make `/monitor` directly accessible. Reuse the visualizer's existing typography, controls, spacing, status colors, and responsive patterns.

Top area:

- Root path and last successful refresh time.
- Refresh button and refresh/error indicator.
- Cards for collection groups, groups awaiting review, groups with new episodes, and groups with recorded publications. Cards can overlap and are labeled as group counts.
- Search by folder name or publication repository; filters for attention, unimported, reviewing, and published.

Main table columns: folder; collected/imported counts; accepted/rejected/pending; review progress; accepted duration; HF publication summary; attention; actions. Make selected-run scope visible, not hidden only in a tooltip. Keep the primary Open review action visible without expanding. Unimported folders link to the existing prepare page with their source path prefilled and no automatic import.

An expanded row contains the run selector, frame-based prompt distribution, known derived datasets and provenance, recorded exports/publications, generation diagnostics, and full paths. Show a compact summary of exclusions with a link to the existing editor. Display training path, format, instruction mode, frame count, and publication revision together to avoid copying the wrong draft for training.

On narrow screens stack row fields or use accessible disclosure cards; important actions must not rely on hover or horizontal scrolling. Status colors always have text labels. Loading, partial data, unavailable paths, empty root, no configured root, and backend-unavailable states each have explicit copy.

## Backend and frontend boundaries

Add a focused `backend/annotation_monitor.py` module for discovery, relationships, summary aggregation, and detail calculations. Keep HTTP handlers thin in `backend/app.py`. Reuse existing readers, review hashes, exclusion utilities, job persistence, and publication metadata; do not move unrelated workflow code as part of this feature.

Proposed endpoint contract:

- `GET /api/monitor`: root availability, collection rows, run summaries, publication summaries, scan time, and per-folder diagnostics.
- `GET /api/monitor/datasets/{dataset_id}?run_id=...`: selected-run detail and prompt distribution, with eligibility/coverage fields.
- `POST /api/monitor/publications/{publication_id}/check`: read-only remote verification, with no repository mutation.

Dataset/publication IDs are opaque identifiers derived from discovered or registered records. Resolve and validate them server-side; never accept an arbitrary path, repository, or URL through these identifiers. Detail endpoints reject unrelated run IDs. Filesystem paths are exposed only within the configured local deployment and existing authentication boundary.

Add frontend types/client methods separate from editor state. The monitor must not hydrate the annotation editor merely to show a summary. Use component state and browser-local selected-run preferences; no shared persisted counters.

Poll lightweight summaries every 30 seconds while the page is visible; pause when hidden. Allow only one refresh request at a time and ignore responses from obsolete root/run selections. Manual Refresh invalidates lightweight summary cache; expanded details invalidate when relevant run, review, exclusion, or episode metadata signatures change. Cache detail computations in memory with bounded entries and replace them on changed signatures. No queue, database, watcher, or background service is needed for the first version.

HF check results may use a bounded in-memory cache; server restart returns them to Not checked. Existing publication receipts remain durable independently.

## Consistency and error handling

Read the file signatures before and after a dataset/run snapshot. If relevant metadata changes during the read, retry once. If still changing, report Updating and retain the previous successful display with its timestamp; do not combine totals from different snapshots. A root that cannot be read produces a clear page-level error. A broken dataset or run affects only its own row/detail.

Do not decode videos, load model weights, or hash large video files for routine monitoring. Fetch frame timestamps and labels only for requested details. Mark source byte integrity as unverified by monitoring; export remains the authoritative full validation path.

## Verification and acceptance

Backend tests must cover:

1. Root symlink deduplication and rejection of browser-supplied/unregistered paths.
2. v2.1/v3 discovery, empty datasets, partial writes, malformed metadata, and internal-directory exclusion.
3. Confirmed derivatives, unknown provenance, merges with several parents, and cycles.
4. Multiple runs without double counting, deterministic default selection, and unavailable exports.
5. Appended and missing source episodes without changing saved review identities.
6. Review hashes invalidated by annotation or exclusion edits; zero denominators and Pending versus Keep semantics.
7. Prompt counts across excluded spans and exact boundaries, mixed FPS, unlabeled/ambiguous frames, whitespace differences, and partial coverage.
8. Published receipts with newer local edits; unavailable history; remote match, changed head, missing/access denied, and timeouts without mutating HF.
9. Summary reads do not invoke VLM, video decoding, full inventory hashing, annotation preparation, or workflow mutations.

Frontend tests cover run selection persistence, stale-response handling, visible polling pause/resume, filters, partial failures, explicit status labels, and correct review/HF/training links. Verify desktop and narrow viewport layout in the browser.

Read-only acceptance against `pnp_table_260909`, selecting run `b937e3f6925a4bafa5eb1bb70ea74ec8`:

- Source: 87 collected episodes; imported: 87; accepted: 71; rejected: 16; pending: 0.
- Review rate: 71/71 retained episodes, 100%, provided no later edits occurred.
- Known export: GR00T v2.1, subtask instruction mode, 34,594 frames.
- Known publication: `mncai/G1_Dex3_PickTable`, revision `260915`, commit `f3d31d6b481a61fc1779890df46be58fd38cd811`.
- Correct current review link uses the full registered alias `local/annotation-99d858fff3f3e9ff`.
- Training path resolves to the recorded `pnp_table_260915` export, not the original 87-episode folder.
- A source-growth fixture adds Not imported episodes while preserving the completed run's review and publication record.

## Delivery boundaries

Implementation follows only after the user reviews this written specification and the implementation plan is prepared. Continue in the canonical visualizer repository; future isolated worktrees belong under its `.worktrees/` directory. Preserve existing unrelated feature changes. No service restart, dataset mutation, HF upload, or feature implementation is part of this design-document step.
