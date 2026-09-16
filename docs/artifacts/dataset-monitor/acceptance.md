# Dataset monitor acceptance — 2026-09-16

## Delivery status and scope

Implementation Tasks 1–7 are integrated on `feat/dataset-monitor` in
`.worktrees/dataset-monitor`; Task 8 adds operator documentation and the verified
v3.1 compatibility correction. The parent integrator applied the guarded feature
patch `ad36ff9f..81dc2372` to the canonical dirty checkout while preserving its
preexisting changes. Feature commits and the worktree are retained; no push or
cleanup is part of this delivery.

Integrated backend/access checks passed after the v3.1 correction in commit
`1f88a88b`; the parent applied that patch to the canonical checkout. The canonical
production build passed and the idle annotation backend/UI were restarted with
the monitor root configured, preserving the VLM tunnel. Browser checks passed
at 1440 px and 390 px; the explicit HF check matched the recorded commit.
All 415 preexisting monitored files were unchanged. Seven new metadata files
in two other folders predated the browser checks, as recorded below.

Browser evidence verifies review/HF link targets, actual review-page navigation
with video rendered, and the training-path copy handler using a clipboard stub.
The recorded HF tree URL returned HTTP 200. Operating-system clipboard
integration was not tested.

The feature is read-only. Preparation links navigate to an existing workflow;
they do not import. Monitor reads must not change datasets, review records,
run decisions, exclusions, the alias registry, or HF repositories. The explicit
remote-check POST reads a recorded HF revision and changes only an in-memory
check cache.

## Local reference evidence

Source observations: `/tmp/monitor-live-summary.json`,
`/tmp/monitor-live-detail.json`, and `/tmp/monitor-acceptance-evidence.json`.
The source/detail snapshot was read at `2026-09-16T04:47:57.827030+00:00`.

| Field | Observed value |
| --- | --- |
| Configured collection root | `/home/jihun/work/GR00T-WholeBodyControl/outputs` |
| Resolved collection root | `/mnt/data/jihun/datasets/G1_WBT_GR00T` |
| Source | `pnp_table_260909` |
| Source path | `/mnt/data/jihun/datasets/G1_WBT_GR00T/pnp_table_260909` |
| Selected run | `b937e3f6925a4bafa5eb1bb70ea74ec8` |
| Collected / imported | 87 / 87 |
| Accepted / rejected / pending | 71 / 16 / 0 |
| Retained / current reviewed | 71 / 71; review rate 100% |
| Decision completion | 100% |
| Accepted retained frames / duration | 34,594 / 691.88 seconds |
| Exclusions | 10,375 frames across 70 episodes |
| Findings | 0 unresolved, 66 accepted advisory, 0 generation failures, 0 unreadable |
| Source metadata changes | No new, missing, or changed-length IDs |
| Local publication freshness | Published; no local changes; linked frozen digest verifiable |
| Export format / instruction mode | `groot_v21` / `subtask` |
| Export frames / duration | 34,594 / 691.88 seconds |
| Export availability | Available at observation time |
| Publication repository / revision | `mncai/G1_Dex3_PickTable` / `260915` |
| Recorded commit | `f3d31d6b481a61fc1779890df46be58fd38cd811` |
| Export manifest SHA-256 | `0c8d6e7ebd6e7b541675cbc28081150eb70245b752a8f5c048f7216d86755c21` |
| HF remote check in initial snapshot | Not checked |

Current review:
[episode 1 using the registered alias](http://127.0.0.1:3000/local/annotation-99d858fff3f3e9ff/episode_1?tab=annotations).
The first retained episode is 1. The frozen output has a different alias,
`local/annotation-dd5160c5c35f2943`.

Available training path:

```text
/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations/workspace/drafts/8b78dd2573054344b16b77fd514ef6c7/pnp_table_260915
```

[Recorded HF version](https://huggingface.co/datasets/mncai/G1_Dex3_PickTable/tree/260915).
The historical commit is a receipt, not a claim that the remote revision still
points to that commit. The live check below records that separately.

## Prompt acceptance

The selected run has 71 eligible and 71 evaluated current-reviewed retained
episodes, 34,594 calculable retained frames, no unknown episodes, and zero
Unlabeled/Ambiguous frames. Coverage is complete. Independent comparison against
the frozen export's task-index rows matched the 7 literal labels and all frame
counts (Task 3 acceptance in the progress ledger).

The literal labels below use JSON quoting so that the trailing newline remains
visible. No normalization merges the two apple prompts.

| Literal subtask | Frames | Frame share | Seconds | Episodes containing prompt |
| --- | ---: | ---: | ---: | ---: |
| "pick up the apple" | 5,945 | 17.1851% | 118.90 | 12 |
| "pick up the apple\n" | 701 | 2.0264% | 14.02 | 1 |
| "pick up the bottle" | 23,362 | 67.5319% | 467.24 | 46 |
| "pick up the brown bottle" | 3,051 | 8.8194% | 61.02 | 6 |
| "put the apple on the table" | 141 | 0.4076% | 2.82 | 1 |
| "put the bottle on the table" | 1,137 | 3.2867% | 22.74 | 5 |
| "put the brown bottle on the table" | 257 | 0.7429% | 5.14 | 1 |

Ratios count the active saved subtask at each actual source-frame timestamp,
after removing half-open excluded source-frame intervals. The denominator is
calculable frames in current-reviewed non-deleted episodes, including reviewed
Pending episodes. Unreadable episodes make coverage incomplete and are excluded
from that denominator; they are not counted as known zero. Frame shares include
Unlabeled/Ambiguous buckets when present. Episode counts can overlap and are not
a partition. Accepted duration separately counts only Keep episodes.

## Verification evidence

These are observed results, with original log paths retained for audit. The
final acceptance wave below supersedes preliminary counts where applicable.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Monitor/backend access suite after Task 5 fixes | 275 passed; 2 existing FastAPI lifecycle warnings | `/tmp/monitor-task5-round1-tests.log` |
| Existing annotation/history/review/delivery/publication regressions | 115 passed; 4 existing warnings | `/tmp/monitor-existing-regressions.log` |
| Monitor client/row/page lifecycle after Task 7 fixes | 46 passed, 204 assertions | `/tmp/monitor-task7-round1-tests.log` |
| Frontend type checks | Passed | `/tmp/monitor-task7-round1-types.log` |
| Earlier worktree production build | Passed before the final Task 7 UI fix | `/tmp/monitor-build.log` |
| Canonical production build after guarded integration | Passed; backend/UI then restarted by parent | `/tmp/monitor-canonical-build.log` |
| Final monitor/backend access suite after v3.1 correction | 329 passed; 2 existing lifecycle warnings | `/tmp/monitor-task8-backend-tests.log` |
| v3.1 red regression | v2.1/v3.0 passed; v3.1 failed because discovery returned Updating instead of Ready | `/tmp/monitor-task8-red.log` |
| v3.1 green prompt suite | 157 passed; 2 existing lifecycle warnings | `/tmp/monitor-task8-green.log` |
| Task 8 Python lint | Passed on staged monitor module and prompt tests | `ruff check --no-cache` |
| Browser acceptance | 1440/390 px passed; correct selected run, prompts, search/Refresh, link targets, copy-handler path, explicit HF check | `/tmp/monitor-browser-check.log`, `/tmp/monitor-browser-results.json` |
| Read-only comparison | All 415 original files unchanged; 7 independently added metadata files; none modified/removed | `/tmp/monitor-readonly-before.json`, `/tmp/monitor-readonly-comparison.json` |

The compatibility test uses on-disk v3 parquet episode metadata and frame shards,
saved annotation and exclusion review hashes, and source timestamps. Fixture
parameterization covers v2.1, v3.0, and v3.1. It verifies Ready discovery, imported
and retained counts, current review eligibility, 7 retained frames, and literal
prompt counts A=2/B=5 across an exclusion. The implementation change only adds
v3.1 to discovery's supported-version allowlist, matching the existing
`official_annotations.prepare_dataset` support; it introduces no new layout or
dependency.

Commands for the integrated acceptance wave, from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 backend/.venv/bin/python -m pytest backend/tests/test_annotation_monitor.py backend/tests/test_annotation_monitor_prompts.py backend/tests/test_annotation_monitor_publications.py backend/tests/test_annotation_monitor_api.py -q
bun test src/utils/__tests__/monitorClient.test.ts src/components/__tests__/dataset-monitor-row.test.tsx src/app/monitor/__tests__/page.test.tsx
bun run type-check
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun run build
ruff check --no-cache backend/annotation_monitor.py backend/tests/test_annotation_monitor.py backend/tests/test_annotation_monitor_prompts.py backend/tests/test_annotation_monitor_publications.py backend/tests/test_annotation_monitor_api.py
git diff --check
```

HTTP tests run in the parent's integration environment. A child sandbox HTTP
transport hang is not a product failure; this worker's v3.1 regression uses
non-HTTP tests. No actual HF check or production-data write was performed by the
documentation worker.

## Final rollout and browser evidence

The parent verified idle jobs, restarted only the annotation backend/UI with
the monitor root configured, and preserved the VLM tunnel. Final backend/access
checks: 329 passed, with Ruff clean; the canonical production build passed.
The deployed summary at `2026-09-16T07:07:34.634094+00:00` contains 35 folders,
compared with 33 in the earlier baseline. Evidence:
`/tmp/monitor-deployed-summary.json`.

Browser checks in `/tmp/monitor-browser-check.log` and
`/tmp/monitor-browser-results.json` passed at 1440 px and 390 px, including no
horizontal overflow hiding controls. The selected published run shows 87
collected/imported, 71/71 current reviews, 71/16/0 decisions, the seven literal
prompts with 34,594 retained frames, 70 exclusion-bearing episodes, and the
correct frozen export path. Search and manual Refresh were exercised. The
recorded review and HF hrefs match the links above. The parent clicked **Open
review**, navigated to the exact registered episode-1 URL, and verified that video
rendered (`/tmp/monitor-review-link-result.json`). The recorded HF tree URL
returned HTTP 200 through curl. The copy handler passed the exact training path
to a stubbed clipboard API and displayed success; an actual OS clipboard read
was not tested.

The parent visually inspected all four responsive screenshots and confirmed
readable layouts (copied into this artifact directory by the parent):

- [Desktop overview, 1440 px](monitor-desktop-1440.png).
- [Mobile overview, 390 px](monitor-mobile-390.png).
- [Desktop expanded details, 1440 px](monitor-desktop-details-1440.png).
- [Mobile expanded details, 390 px](monitor-mobile-details-390.png).

The explicit **Check HF** action ran once at
`2026-09-16T07:07:33.086624+00:00`. It returned current head
`f3d31d6b481a61fc1779890df46be58fd38cd811`, with **Matches recorded commit**.
This result is scoped to that check time; the historical receipt remains a
separate durable record.

The browser request trace contains monitor summary/detail GETs, one explicit
read-only publication-check POST, and unrelated page/auth GETs. It contains no
annotation prepare, review-save, export, generation, or upload request.

The after-monitoring comparison in `/tmp/monitor-readonly-comparison.json` found
all **415 original files byte-unchanged**, with **no modified or removed files**.
Seven new metadata files appeared in two independently created folders:

- `pnp_bottle_260916`: `episodes.jsonl`, `episodes_stats.jsonl`, `info.json`,
  `modality.json`, and `tasks.jsonl` under `meta/`.
- `pnp_table_260915_subtask_eval_20260916T050458970984929Z`: `meta/info.json`
  and `meta/modality.json`.

Their recorded modification times range from `05:05:48` to `06:32:41` UTC,
before the browser check at `07:07` UTC. These additions explain the larger
collection inventory; they are separate from the monitor browser actions.
A subsequent comparison of `/tmp/monitor-readonly-after.json` with
`/tmp/monitor-readonly-post-links.json` found all 422 files identical: zero
additions, changes, or deletions during the final monitoring and actual review
navigation window. The file comparison covers its recorded metadata/run/review/alias
scope, not a full byte audit of every video. Source growth is also verified with
appended-episode fixtures rather than changes to a real collection.

## Operational limits

The monitor needs an explicitly configured local root and existing annotation
authentication/origin controls. It discovers only immediate collection children;
registered workflow paths can expose linked exports. Existing aliases are read,
never registered by monitoring. Relationships use explicit provenance, and
collection group cards do not establish globally unique demonstration totals.

Summary polling is cached, visible-only, and every 30 seconds, with manual
refresh. Source checks compare IDs, lengths, and metadata signatures, not video
bytes; export remains the full source-validation boundary. Current reviews use
annotation and exclusion hashes, separately from Keep/Delete decisions.

Publication history is limited to saved receipts and completed publication jobs.
Missing manifests make local freshness unverified; missing exports have no copy
path. Explicit HF checks verify remote revision identity only, use server-side
credentials, and never upload. Their in-memory cache is lost on restart; saved
receipts remain. A receipt survives local edits and remote timeouts.

## Decision audit

The following are all `Ruling:` entries from the implementation progress ledger
at documentation capture. They retain the rationale and stated cost if wrong so
that review decisions survive outside transient worker reports.

Ruling: Add structured findings and exclusions to run/detail responses: findings {unresolved, accepted_advisory, generation_failed, unreadable}, exclusions {episodes, frames}. Diagnostics use {code,message,severity}; provenance uses {status,sources}. — Required display fields missing from plan — Cost if wrong: adjust API/frontend types.

Ruling: Add separate exports array to dataset/detail responses with {id,run_id,path,available,format,instruction_mode,frames,seconds,manifest_sha256,output_repo_id}. — Unpublished frozen exports must remain visible — Cost if wrong: small response/UI adjustment.

Ruling: Corrupt discovered folders remain diagnostic rows; only corrupt run records may be omitted with diagnostics. — Spec requires visible failures — Cost if wrong: extra rows in monitor.

Ruling: Escalate Task 1 fixes to gpt-6-astra/high after incomplete implementation and unaddressed direct corrections. — User routing requires escalation on repeated worker failures — Cost if wrong: additional model cost.

Ruling: Child implementers stage files under /tmp; main applies, verifies, and commits in worktree. — Child approval requests blocked for several minutes; main escalation works — Cost if wrong: extra integration step, guarded by tests and review.

Ruling: Escalate Task3 and remaining integration-heavy implementation to gpt-6-astra/high after repeated lower-tier omissions. — Explicit acceptance instructions were not followed across tasks — Cost if wrong: higher model cost, lower rework risk.

Ruling: Integrate only the feature diff onto the canonical dirty checkout; retain feature branch/worktree instead of merging the baseline snapshot. — Baseline snapshot contains existing unfinished edits that must remain preserved — Cost if wrong: branch still needs later Git-history integration; live feature remains reversible.

Additional Task 8 decisions:

- Accept v3.1 using the existing v3 parquet layout after a failing discovery test;
  cost if wrong: incorrect support claims. Real-file discovery, current reviews,
  and prompt coverage now exercise v3.1 alongside v3.0 and v2.1.
- Integrate a guarded feature patch onto the canonical dirty checkout while
  preserving preexisting changes, and retain feature-branch commits/worktree;
  cost if wrong: integration overlap or lost unrelated edits. The parent checked
  clean patch application and owns final integrated verification.
- Attribute browser and filesystem evidence only to the checks actually run;
  cost if wrong: overstated delivery claims. Actual review navigation and HF
  HTTP availability are recorded separately; clipboard verification remains
  limited to the stubbed browser handler.
