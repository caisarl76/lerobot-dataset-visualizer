# Dataset Monitor whole-branch review

Range reviewed: `ad36ff9f..c8680d94` in `/home/jihun/work/lerobot-dataset-visualizer/.worktrees/dataset-monitor`.

## Assessment

**Ready: Yes, with one non-blocking wording correction recommended.**

- **Spec: PASS.** Discovery, explicit provenance, selected-run denominators, current review hashes, retained-frame prompt coverage, separate frozen exports/publication history, explicit HF checking, cache consistency, local-only boundaries, and frontend selection/polling match the agreed feature. The minor job label below overstates the kind of job represented.
- **Quality: PASS.** No critical or important cross-component defect found. The implementation isolates corrupt evidence, exposes unknown coverage, uses bounded in-memory caches, rejects unrelated identifiers, and avoids monitor-side workflow mutations.

## Strengths

- Backend source-growth comparisons use immutable original identities and checkpoint lengths; frontend review denominators remain selected-run scoped.
- Publication records remain dataset-wide regardless of run selection. Export linkage requires recorded destination/digest evidence, unavailable training files disable copying, and remote checks preserve historical receipts.
- Detail signatures include run/checkpoint and artifact state. The frontend rejects obsolete detail responses and visibly retains prior details when a consistent replacement cannot be read.
- Summary and detail code reuse review hashing without invoking editor payloads, prediction metrics, VLM work, video decoding, or full asset hashing. The only remote action is the bounded recorded-revision lookup after publication-ID validation.
- Existing dirty-baseline changes are outside the reviewed feature delta. The supplied acceptance record explicitly separates observed browser behavior, clipboard-stub verification, and unchanged monitored metadata.

## Findings

### Minor — P3: Current workflow job is labeled as generation

- **Location:** `src/components/dataset-monitor-row.tsx:720`.
- **Evidence:** The component displays `run.job.status` as `Generation job`. `backend/annotation_monitor.py:758-770` reads the run's current persisted job and projects only job ID/status/error. Existing `_queue_workflow_operation` at `backend/app.py:1640` assigns this same pointer for export and publication operations. The underlying job record does not reliably identify its operation kind.
- **Impact:** A completed upload can appear as completed generation, and an export/upload error can appear as a generation failure in Run evidence. Counts and recorded publications remain correct.
- **Recommendation:** Rename the text to `Current workflow job` (or `Current job`). This avoids inventing a generation-specific state or changing persisted schemas. A focused component assertion is sufficient if the correction is made.

No critical or important findings.

## Review and verification scope

Read the production backend module, route/config integration diff, client contracts, page lifecycle and row rendering, responsive styles, existing workflow/job writers and review digest, selected monitor tests, design/plan/contracts, and acceptance record. The review focused on identity, stale snapshots, current-review coverage, artifact association, and mutation boundaries.

Used one independent temporary-directory probe for a concrete concern: a collection containing a self-looping symlink plus a healthy dataset. Both discovery and `MonitorService.summary()` returned the healthy row successfully; the concern did not reproduce and is not a finding. The probe wrote only temporary fixture files and did not modify the checkout.

Did not rerun already-passed suites or deploy/restart anything. Regression/build/browser results are the provided integration evidence: 329 backend/access checks, 115 existing workflow regressions, 46 UI tests / 204 assertions, type checks, canonical production build, 1440/390 browser checks, and monitored-byte comparisons. This report does not represent a new execution of those checks.

## Scoped correction and deployed verification

The only finding was corrected in `2e87e3a3`: the label now says **Current workflow job**. Scoped re-review passed specification and quality with no new findings. The existing 46 frontend tests passed (204 assertions), the canonical production build passed, and a final browser smoke check confirmed the deployed label, 71/71 reviewed episodes, and 34,594 retained frames.

The feature branch and its worktree remain available. Only guarded feature diffs were applied to the canonical checkout; existing unfinished changes were preserved. No remote Git push was performed.
