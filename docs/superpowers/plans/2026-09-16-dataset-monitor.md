# Dataset Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local dataset monitor showing collection, review, usable data, prompt distribution, and HF publication status without modifying datasets or workflows.

**Architecture:** A focused Python monitoring module reads collection metadata and existing run/review/publication records; thin FastAPI endpoints expose summaries and on-demand detail. A Next.js `/monitor` page groups confirmed derivatives, selects one annotation run per source, and displays independent publication history. Reuse existing identity, review-hash, clipping, and export semantics; introduce no database, worker, or filesystem watcher.

**Tech Stack:** Existing Python 3.13 annotation virtualenv, FastAPI, PyArrow, huggingface_hub, Next.js 15, React 19, TypeScript, Bun, pytest, and Testing Library. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-16-dataset-monitor-design.md` (commit `a2fab9d9`). Read this document and the spec before execution.

## Global Constraints

- "Use a server-side `LEROBOT_MONITOR_ROOT` setting; configure the local launcher with the root above."
- "Browser requests cannot supply arbitrary scan roots or filesystem paths."
- "Never infer parentage from `_full_task`, `_subtask`, dates, or other name patterns."
- "Each run remains individually identifiable; never add counts across runs for the same source."
- "The first version uses Accepted instead of the ambiguous label Valid."
- "Poll lightweight summaries every 30 seconds while the page is visible; pause when hidden."
- "Do not decode videos, load model weights, or hash large video files for routine monitoring."
- "Do not contact HF on every periodic refresh."
- "No queue, database, watcher, or background service is needed for the first version."
- "Preserve existing unrelated feature changes."

Configured collection root: `/home/jihun/work/GR00T-WholeBodyControl/outputs`. Canonical repository: `/home/jihun/work/lerobot-dataset-visualizer`. Existing workflow workspace is supplied by `LEROBOT_ANNOTATE_EXPORT`; never hardcode it inside the monitoring module.

---

## Execution preparation and file ownership

Before changing implementation, inspect `git status --short`, `git diff`, and applicable `AGENTS.md`. The canonical checkout contains uncommitted annotation/export/UI work that the running service uses. Do not create a worktree from HEAD and assume it contains that work. If isolation is selected, use the worktree skill and a path below this repository's `.worktrees/`; first carry the required baseline into that worktree without discarding canonical edits. Otherwise work in the canonical checkout and stage only task-owned changes. Never use `git add .` for these tasks.

Paths below are repository-relative. Backend commands use `backend/.venv/bin/python`. File edits and tests in this repository may require filesystem escalation from the WBC workspace. Do not deploy until verification passes and annotation jobs are idle.

| File | Responsibility |
| --- | --- |
| `backend/annotation_monitor.py` | Discovery, provenance, run summaries, prompt distributions, publication records, consistency/cache handling. |
| `backend/tests/monitor_fixtures.py` | Small real metadata/Parquet fixtures; no models or videos. |
| `backend/tests/test_annotation_monitor.py` | Discovery, relationships, metrics, read-only behavior. |
| `backend/tests/test_annotation_monitor_prompts.py` | Actual timestamps, exclusions, prompt denominators. |
| `backend/tests/test_annotation_monitor_publications.py` | Publication history, freshness, injected HF client. |
| `backend/app.py` | Thin monitor endpoints only. Preserve other unfinished changes. |
| `backend/tests/test_annotation_monitor_api.py` | Endpoint configuration, identity, auth, and error behavior. |
| `src/utils/monitorClient.ts` | HTTP client and response types; reuse configured annotation backend URL. |
| `src/utils/__tests__/monitorClient.test.ts` | URL encoding, HTTP errors, cancellation. |
| `src/app/monitor/page.tsx` | Client-side state, polling, filtering, navigation. |
| `src/app/monitor/page.module.css` | Responsive monitor layout using existing visual conventions. |
| `src/components/dataset-monitor-row.tsx` | Folder row/card, selected-run metrics, expanded details. |
| `src/components/__tests__/dataset-monitor-row.test.tsx` | Counters, progress, prompt detail, links, accessibility. |
| `src/app/monitor/__tests__/page.test.tsx` | Selection persistence, refresh lifecycle, errors, races. |
| `src/app/page.tsx` | Local Monitor navigation link. |
| `deployment/start-annotation-local.sh` | Configure monitor root for local service. |
| `deployment/README.md` | Operator instructions, denominator definitions, limitations. |

Keep monitoring logic out of editor context and avoid unrelated refactors. If delegated, one worker owns the backend module at a time; frontend work can proceed after the API contract is fixed. Follow the user's model routing: bounded implementation to gpt-5.6-luna/low, difficult diagnosis/reviews to gpt-6-astra/high, main agent owns integration.

## Shared contracts

Use JSON-serializable dictionaries internally and typed TypeScript responses. No new ORM or generic repository layer. Internal dictionaries may carry private `_root: Path` and `_run: dict` fields for computation; strip private fields before serializing HTTP responses. Define all public functions below in `annotation_monitor.py`; helper functions stay private.

```python
# root and workspace are trusted server configuration, never request values.
def discover_datasets(root: Path, workspace: Path) -> list[dict]: ...
def read_runs(workspace: Path) -> list[dict]: ...
def attach_relationships(datasets: list[dict], runs: list[dict]) -> list[dict]: ...
def summarize_run(dataset: dict, run: dict, workspace: Path) -> dict: ...
def prompt_distribution(run: dict) -> dict: ...
def publication_records(run: dict, workspace: Path) -> list[dict]: ...
def check_publication(publication: dict, api) -> dict: ...

class MonitorService:
    def __init__(self, root: Path, workspace: Path): ...
    def summary(self, *, refresh: bool = False) -> dict: ...
    def detail(self, dataset_id: str, run_id: str | None = None) -> dict: ...
    def check(self, publication_id: str) -> dict: ...
```

These signatures are the cross-task interface, not implementation stubs to check in. New dataset IDs use `sha256(str(path.resolve()).encode()).hexdigest()[:24]`. Publication IDs use the same digest length over canonical JSON `[repo_id, revision, commit]`. All lookups validate IDs against the current server-discovered snapshot.

Summary response: `{configured, root, scanned_at, updating, diagnostics, datasets}`. Each dataset has `{id, name, path, canonical_path, state, collected, reported_collected, episode_lengths, parent_ids, child_ids, provenance, default_run_id, runs, publications, diagnostics}`. `episode_lengths` maps source IDs to metadata frame counts and supports summaries without reading frame data. Each run summary has `{run_id, updated_at, current_repo_id, first_retained_episode, detail_signature, metrics, job, freshness, diagnostics}`. The default run is newest run-file mtime, descending, then run ID ascending; preserve a valid browser-selected run across refreshes.

`metrics`: `{imported, accepted, rejected, pending, retained, reviewed, decision_rate, review_rate, new_episode_ids, missing_episode_ids, changed_length_ids, accepted_frames, accepted_seconds, counts_complete}`. Rates are fractions in `[0,1]` or null. Unknown counts are null rather than zero. A diagnostic identifies the missing evidence.

Detail response: `{dataset_id, run_id, signature, metrics, prompts, publications, diagnostics}`. `prompts` contains `{eligible_episodes, evaluated_episodes, unknown_episode_ids, retained_frames, unlabeled_frames, ambiguous_frames, complete, rows}`. Each prompt row has `{text, frames, seconds, episodes, ratio}`; bucket ratios use the same `retained_frames` denominator.

Publication record: `{id, repo_id, revision, commit, url, export_path, export_available, format, instruction_mode, exported_frames, manifest_sha256, linked_run_id, remote_check}`. Unknown optional fields are null. `remote_check` is null until explicitly checked; then `{status, checked_at, current_commit, message}`. Status values: `match`, `changed`, `missing`, `access_denied`, `unavailable`.

Freshness: `{publication_state, metadata_only: true, source_changed, local_changes, verifiable}`. `local_changes` is null when old publication evidence cannot be linked. Do not treat a completed generation job as a publication or infer upload success from a destination string.

## Task 1: Discover datasets and confirm relationships

**Files:** Create `backend/annotation_monitor.py`, `backend/tests/monitor_fixtures.py`, `backend/tests/test_annotation_monitor.py`.

**Interfaces:** Produces `discover_datasets`, `read_runs`, `attach_relationships` and stable IDs. Consumes configured root/workspace and existing metadata formats.

- [ ] **Step 1: Add realistic small fixtures and failing discovery tests.** Create `write_v21(root, lengths)` that writes `meta/info.json`, `meta/episodes.jsonl`, and matching Parquet files. Provide `write_run(workspace, source, checkpoint, episodes, run_id)` that directly writes a valid run fixture without hashing a video tree. Fixture code starts with:

```python
def write_v21(root, lengths):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    (root / "meta").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "fps": 10,
            "total_episodes": len(lengths), "total_frames": sum(lengths),
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {}}
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = [{"episode_index": ep, "length": n, "tasks": ["pick"]}
            for ep, n in enumerate(lengths)]
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    for ep, n in enumerate(lengths):
        path = root / info["data_path"].format(episode_chunk=0, episode_index=ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"episode_index": [ep]*n,
                                "frame_index": list(range(n)),
                                "timestamp": [i/10 for i in range(n)]}), path)
    return root

def test_discovery_deduplicates_alias_and_keeps_empty(tmp_path):
    root, workspace = tmp_path / "collection", tmp_path / "workspace"
    source = write_v21(root / "source", [4, 6])
    write_v21(root / "empty", [])
    (root / "alias").symlink_to(source, target_is_directory=True)
    rows = discover_datasets(root, workspace)
    assert sorted(r["collected"] for r in rows) == [0, 2]
    assert len({r["canonical_path"] for r in rows}) == 2
```

Also test v3 episode metadata with a small `meta/episodes/chunk-000/file-000.parquet`, duplicates/negative IDs, mismatched info count, partial JSON, hidden/infrastructure directories, and links escaping the root. Count completed records; do not assume IDs are `range(total_episodes)`.

- [ ] **Step 2: Run the tests red.** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 backend/.venv/bin/python -m pytest backend/tests/test_annotation_monitor.py -q`. Expected: missing monitor implementation.
- [ ] **Step 3: Implement discovery and provenance.** Build metadata readers with `json` and projected PyArrow metadata reads. Existing `official_annotations.source_episode_rows` rejects empty datasets, so handle valid zero-episode metadata before invoking compatible readers. Stable file signatures are `(relative_path, size, mtime_ns)` over relevant metadata, including directory listings. Sample before/after, retry once, then return Updating if changing.

```python
def _inside(path, root):
    resolved = path.resolve()
    return resolved == root or root in resolved.parents

# Alias IDs must be based on resolved paths, not the display path.
def _dataset_id(path):
    return sha256(str(path.resolve()).encode()).hexdigest()[:24]
```

Support actual provenance keys: `source_episode_mapping.original_root`; `source_episode_mapping.run_id` resolved via run source; `merge_manifest.sources[*].root`; `groot_instruction_export.source_root` only when it resolves to a discovered source or a confirmed provenance chain. Do not turn an unknown intermediate staging path into a parent. Multiple parents stay top-level. Detect cycles and retain the rows with diagnostics. `read_runs` isolates corrupt records and never calls `recover_jobs` or `RunStore.create`.

- [ ] **Step 4: Verify tests and lint.** Run the task test file and `ruff check --no-cache backend/annotation_monitor.py backend/tests/monitor_fixtures.py backend/tests/test_annotation_monitor.py`.
- [ ] **Step 5: Commit only the new task files.** Commit message: `feat: discover local datasets and recorded lineage`.

## Task 2: Aggregate review, decision, and source-growth metrics

**Files:** Modify `backend/annotation_monitor.py`, `backend/tests/monitor_fixtures.py`, `backend/tests/test_annotation_monitor.py`.

**Interfaces:** Consumes dataset `episode_lengths`, `read_runs` records. Produces `summarize_run(dataset, run, workspace)` with the metrics contract.

- [ ] **Step 1: Write failing review and growth tests.** Add a three-episode fixture: Keep/current review, Pending/current review, Delete. A fourth episode exists only in live source metadata. Fixture annotations use `annotation_history.annotation_hash` and `exclusions_hash`; an exclusion edit without updating the review must make the review stale.

```python
def test_source_growth_does_not_dilute_imported_review_rate(review_fixture):
    dataset, run, workspace = review_fixture
    result = summarize_run(dataset, run, workspace)["metrics"]
    assert (result["imported"], result["accepted"], result["rejected"], result["pending"]) == (3, 1, 1, 1)
    assert result["review_rate"] == 1.0
    assert result["decision_rate"] == 2 / 3
    assert result["new_episode_ids"] == [3]
```

Define `review_fixture` in `monitor_fixtures.py` using `write_v21` and `write_run`, and import it explicitly in tests. Include missing original IDs, changed lengths, all-deleted, no-run, missing checkpoint, absent sidecar with real parquet fallback, and invalid exclusions. Patch full source inventory, video decode, and generation entrypoints to raise if a summary calls them.

- [ ] **Step 2: Run targeted tests red.** Run `test_annotation_monitor.py` with `-k 'review or growth or summary or exclusions'`.
- [ ] **Step 3: Implement summary aggregation.** Batch-load review/annotation sidecars once per checkpoint; map live source IDs through `original_episode_index`. Use checkpoint metadata for run frame counts. Reuse annotation/exclusion hashing and `normalize_exclusions`; inspect persisted jobs without rewriting their statuses.

```python
retained = [ep for ep, state in run["episodes"].items() if state["decision"] != "delete"]
accepted = [ep for ep, state in run["episodes"].items() if state["decision"] == "keep"]
review_rate = reviewed_count / len(retained) if retained else None
# reviewed_count requires matching annotation and exclusion hashes and reviewed_at.
```

For malformed episode evidence set `counts_complete=False`, retain known decision counts, and mark review/duration values unknown where a denominator cannot be established. Use batch fallback readers only for sidecar-missing episodes. No `_workflow_payload`, prediction enumeration, full video hashing, or model import on summary paths. Attach current job status from its persisted JSON. Accepted advisory findings remain distinct from unresolved findings.

- [ ] **Step 4: Verify all discovery/summary tests pass and read-only assertions hold.** Snapshot dataset/workspace file contents before/after the service calls, excluding no files: monitoring must write none.
- [ ] **Step 5: Commit task-owned hunks.** Commit message: `feat: summarize current reviews and newly collected episodes`.

## Task 3: Calculate retained-frame prompt distributions

**Files:** Modify `backend/annotation_monitor.py`, `backend/tests/monitor_fixtures.py`; create `backend/tests/test_annotation_monitor_prompts.py`.

**Interfaces:** Produces `prompt_distribution(run)` with prompt and coverage contracts. Consumes current reviewed, non-deleted checkpoint episodes, actual timestamp arrays, and normalized exclusions.

- [ ] **Step 1: Add failing interval tests.** Fixture has 10 frames at 10 FPS, prompt A at 0.0, B at 0.3, exclusion `[2,5)`, a current review, and decision Pending. Expected counts: A=2, B=5, denominator=7. Add an exclusion edit with a stale review and assert eligibility becomes zero.

```python
def test_prompt_inside_exclusion_remains_active_after_cut(prompt_fixture):
    result = prompt_distribution(prompt_fixture)
    counts = {row["text"]: row["frames"] for row in result["rows"]}
    assert counts == {"A": 2, "B": 5}
    assert result["retained_frames"] == 7
    assert result["eligible_episodes"] == result["evaluated_episodes"] == 1
    assert result["complete"] is True
```

Define `prompt_fixture` in the task test file using shared writers and real review hashes. Add exact-boundary, before-first-label, same-time conflicting labels, identical duplicates, whitespace-distinct text, deleted/unreviewed episodes, mixed FPS, corrupt timestamps, and partial-coverage tests. Cross-check active text against the existing exporter on a valid fixture.

- [ ] **Step 2: Run `test_annotation_monitor_prompts.py` red.** Expected missing function or incorrect boundary/count behavior, not an unrelated import error.
- [ ] **Step 3: Implement frame counting.** Read only timestamps and required language columns for eligible episodes, respecting v3 row offsets and shared shards. For each retained source frame, advance the sorted subtask state to that timestamp. A conflicting latest same-time label set goes to Ambiguous; no active label goes to Unlabeled. Identical duplicate text at the same time counts once. Later unambiguous changes end an ambiguous span.

```python
# For one successfully read episode:
kept = retained_indices(frame_count, exclusions)
# Sum per episode rather than applying a global FPS to mixed datasets.
seconds_for_prompt = frames_for_prompt / fps
ratio = frames_for_prompt / total_evaluated_retained_frames if total_evaluated_retained_frames else None
```

Preserve exact text in keys; episode count is the number of eligible episodes with at least one counted frame. Unknown episodes are excluded from the evaluated denominator and listed explicitly. Never fall back to task_aug. Ensure numeric ratios plus Unlabeled/Ambiguous sum to one for nonempty evaluated coverage.

- [ ] **Step 4: Verify distribution tests and summary regressions.** Run both monitor test files; check no video API calls.
- [ ] **Step 5: Commit.** Message: `feat: measure reviewed prompt distribution after exclusions`.

## Task 4: Read publication history and verify known HF revisions

**Files:** Modify `backend/annotation_monitor.py`; create `backend/tests/test_annotation_monitor_publications.py`.

**Interfaces:** Produces `publication_records(run, workspace)` and `check_publication(publication, api)`. Uses only saved receipts, job JSON, linked frozen manifests, and an injected `HfApi` client for explicit checks.

- [ ] **Step 1: Write failing receipt and fake-Hub tests.** Add a publication with later local draft edits; preserve its revision but report unpublished changes. Completed publication jobs use the current persisted job result shape (`repo_id`, `revision`, `main_commit`, `urls`). Unrelated runs' receipts must not be attached. Deduplicate identical commits.

```python
class FakeHub:
    def __init__(self, commit):
        self.commit = commit
        self.calls = []
    def repo_info(self, repo_id, *, repo_type, revision, timeout):
        from types import SimpleNamespace
        self.calls.append((repo_id, repo_type, revision, timeout))
        return SimpleNamespace(sha=self.commit)

def test_remote_check_uses_only_recorded_destination():
    pub = {"repo_id": "team/data", "revision": "260915", "commit": "old"}
    api = FakeHub("new")
    result = check_publication(pub, api)
    assert result["status"] == "changed"
    assert result["current_commit"] == "new"
    assert api.calls == [("team/data", "dataset", "260915", 10)]
```

Add missing revision, 401/403, network timeout, missing local export, no frozen receipt, and unchanged commit tests. The installed `HfApi.repo_info` accepts `timeout`; pass `timeout=10` explicitly.

- [ ] **Step 2: Run publication tests red.** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 backend/.venv/bin/python -m pytest backend/tests/test_annotation_monitor_publications.py -q`.
- [ ] **Step 3: Implement history and freshness.** Follow run `publication` first, then completed publish-job results only when a reliable job/run link exists. Do not invent historical export paths. Reuse `annotation_publish._review_digest` lazily when a linked frozen manifest is available; digest comparison reads review data, not video bytes. Unknown historical links yield `verifiable=False`.

```python
# A failed remote check never clears the saved publication.
result = {"status": status, "checked_at": checked_at,
          "current_commit": current_commit, "message": message}
```

Use the installed Hub exception classes. A missing branch is Missing; explicit auth rejection is Access denied; a repository-not-found response that cannot distinguish privacy from absence gets an explanatory unavailable/missing-or-inaccessible message rather than claiming deletion. No create_repo/create_branch/create_commit/download calls.

- [ ] **Step 4: Run tests and confirm summary makes zero HF calls.** Bind a client that raises on all methods during summary evaluation.
- [ ] **Step 5: Commit.** Message: `feat: expose publication history and explicit Hub verification`.

## Task 5: Expose consistent cached monitor endpoints

**Files:** Modify `backend/annotation_monitor.py`, `backend/app.py`, `deployment/start-annotation-local.sh`; create `backend/tests/test_annotation_monitor_api.py`.

**Interfaces:** Produces `MonitorService` and the three spec endpoints. Consumes Tasks 1–4 functions and existing FastAPI middleware. No new public API accepts paths or arbitrary repositories.

- [ ] **Step 1: Add failing endpoint/service tests.** Use tmp roots and the current app test patterns. Assert unconfigured summary returns `configured:false`, unknown IDs return 404, invalid run ownership returns 404, POST remote check has no body-supplied destination, and responses cannot escape the configured root. Hosted authentication behavior must remain unchanged; do not broadly add local-monitor routes to the hosted allowlist.

```python
def test_detail_rejects_a_run_from_another_source(service, unrelated_run_id):
    dataset = service.summary()["datasets"][0]
    with pytest.raises(KeyError):
        service.detail(dataset["id"], unrelated_run_id)
```

Define these fixtures using the earlier fixture writers. Add before/after signature changes on consecutive reads, stale snapshot preservation, detail cache invalidation after exclusions, and bounded caches. Compare all workflow/dataset bytes before and after summary/detail/check operations. Review-link aliases must be derived/read without calling the mutating `_register_local_dataset`.

- [ ] **Step 2: Run API tests red.** Run `test_annotation_monitor_api.py` with plugin autoload disabled.
- [ ] **Step 3: Implement the service.** Keep a single latest summary snapshot, a lock protecting its replacement, and an `OrderedDict` cache capped at 32 detail responses plus 64 explicit remote-check results. Do not hold the lock through an HF request. Invalidate detail by run/review/annotation/episode metadata and needed data-shard signatures. Retry inconsistent reads once; preserve the prior successful result with its old timestamp and `updating:true` after a second change. Skip corrupt rows independently.

```python
# Existing aliases are read-only inputs. Compute the existing alias convention
# only for display if already registered; otherwise offer prepare/open guidance.
# Never allocate annotation drafts or rewrite the alias registry on GET.
```

Add thin handlers with this shape:

```python
@app.get("/api/monitor")
def monitor_summary(refresh: bool = False):
    service = _monitor_service()
    if service is None:
        return {"configured": False, "root": None, "scanned_at": None,
                "updating": False, "diagnostics": [], "datasets": []}
    return service.summary(refresh=refresh)
```

Define `_monitor_service()` in `app.py`: lazily read configured root and `EXPORT_ROOT`, reuse one service for that pair, and rebuild only when configuration changes. Other handlers translate service KeyError to 404 and root-unavailable errors to an actionable 503. Register both relative/legacy imports as app.py already does. Remote checking uses `POST /api/monitor/publications/{publication_id}/check`; validate the ID using an existing publication record before creating an HF client.

Configure local launcher with `export LEROBOT_MONITOR_ROOT="${LEROBOT_MONITOR_ROOT:-/home/jihun/work/GR00T-WholeBodyControl/outputs}"` and pass it into the backend tmux command. Keep the monitor disabled in hosted deployment. Do not change other startup settings.

- [ ] **Step 4: Run all four monitor test files and relevant access tests.** Also run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 backend/.venv/bin/python -m pytest backend/tests/test_annotation_access.py -q`. Confirm monitor handlers remain protected by existing middleware.
- [ ] **Step 5: Commit only monitor changes.** `git add -p backend/app.py deployment/start-annotation-local.sh` for these already-dirty files; stage new monitor files explicitly. Message: `feat: serve read-only dataset monitor endpoints`.

## Task 6: Build typed monitor client and folder details

**Files:** Create `src/utils/monitorClient.ts`, `src/utils/__tests__/monitorClient.test.ts`, `src/components/dataset-monitor-row.tsx`, `src/components/__tests__/dataset-monitor-row.test.tsx`, `src/app/monitor/page.module.css`.

**Interfaces:** Export `MonitorSummary`, `MonitorDataset`, `MonitorRun`, `MonitorDetail`, `MonitorPublication`, `RemoteCheck` matching Shared contracts. Export `fetchMonitorSummary(refresh?: boolean, signal?: AbortSignal)`, `fetchMonitorDetail(datasetId: string, runId: string, signal?: AbortSignal)`, `checkMonitorPublication(publicationId: string, signal?: AbortSignal)`. Row props: `{dataset, selectedRunId, detail, loading, error, onSelectRun, onExpand, onCheckPublication}`; annotate callback types explicitly.

- [ ] **Step 1: Add failing client and row tests.** Test absolute and relative backend bases, encoded identifiers, HTTP failures, and forwarded AbortSignal. Row fixture has 87/71/16/0 counts and review rate 1. Assert Accepted, Rejected, Review progress and Not imported labels are visible; test null denominator, loading details, missing export path, preserved publication during local edits, and a prompt containing a trailing newline.

```tsx
expect(screen.getByText("71 / 71 reviewed")).toBeTruthy();
expect(screen.getByRole("link", { name: "Open review" }).getAttribute("href"))
  .toBe("/local/annotation-99d858fff3f3e9ff/episode_1?tab=annotations");
```

Review links choose the first retained episode from the run, not hardcoded episode zero. Include that ID as `first_retained_episode` on the run summary and in the TypeScript type. This is a navigation field, not a new metric.

- [ ] **Step 2: Run both test files red with `bun test`.** Use existing `src/test-setup.ts`; do not introduce a test runner.
- [ ] **Step 3: Implement the client, row, and styles.** Reuse `getAnnotateBackendUrl` from annotationsClient. Distinguish backend-unconfigured from network failures. Render simple frame-share bars with text and an accessible table; use no new chart library. Show Unlabeled/Ambiguous rows and incomplete coverage explicitly. Use `<details>` for folder expansion, labeled selects/buttons, links for navigation, and clipboard only after user action. Preserve exact literal prompt text; add a visible whitespace marker when necessary.

```ts
function requestUrl(path: string): string {
  const base = getAnnotateBackendUrl();
  if (!base) throw new Error("Annotation backend is not configured.");
  return base.startsWith("/")
    ? `${base.replace(/\/$/, "")}${path}`
    : new URL(path, base).toString();
}
```

Style desktop rows and mobile stacked cards so counts/actions remain visible at 390 px; no hover-only controls. Expose errors through `role="alert"` and refresh/loading copy through `role="status"`.

- [ ] **Step 4: Run row/client tests and `bun run type-check`.** Ensure types cover nullable unknown values rather than coercing them to zero.
- [ ] **Step 5: Commit new files.** Message: `feat: display dataset review and publication details`.

## Task 7: Assemble monitor navigation, selection, and polling

**Files:** Create `src/app/monitor/page.tsx`, `src/app/monitor/__tests__/page.test.tsx`; modify `src/app/page.tsx` and the monitor CSS/row only as needed.

**Interfaces:** Consumes Task 6 client/components. `localStorage` preference key is `lerobot-monitor:selected-runs:v1`, containing `{[datasetId]: runId}`. No persistence of counters, annotations, or job status.

- [ ] **Step 1: Write failing page lifecycle tests.** Mock monitorClient promises and visibility events. Verify valid selection survives refresh; removed selection falls back to default; expanded run fetches details; an old detail response cannot replace the newly selected run; hidden page stops polling; visible page refreshes; refresh errors preserve the last good snapshot with its timestamp.

```tsx
fireEvent.change(screen.getByLabelText("Annotation run for source"),
  { target: { value: "second-run" } });
await waitFor(() => expect(client.fetchMonitorDetail)
  .toHaveBeenCalledWith("source-id", "second-run", expect.any(AbortSignal)));
expect(JSON.parse(localStorage.getItem("lerobot-monitor:selected-runs:v1")!))
  .toEqual({ "source-id": "second-run" });
```

Use actual mock functions/spies in each test and restore them after each test. Add a counter proving periodic summary polling never calls HF checking or detail loading for collapsed rows.

- [ ] **Step 2: Run page tests red.** `bun test src/app/monitor/__tests__/page.test.tsx`.
- [ ] **Step 3: Implement the page.** Use one summary fetch at a time, AbortController on unmount, and request sequence checks for obsolete detail results. Invalidate expanded details when the summary reports a changed signature; include `detail_signature` on each run summary and its TypeScript type. Schedule the next visible refresh 30 seconds after the prior request finishes rather than piling requests with setInterval. A visibilitychange cancels the timer; a visible page requests fresh summary. Manual Refresh requests `refresh=true`, preserving old data during the request.

```tsx
const loadSequence = useRef(0);
// Capture ++loadSequence.current with each detail request; apply its result
// only if the sequence, dataset ID, and selected run still match.
```

Display summary cards as overlapping group counts, not total unique demonstrations. Filter/search folder names and publication repository strings. Preserve source merges as independent rows and show their source links. Add an enabled-local-backend Monitor link beside the existing homepage annotation navigation. Missing configuration explains the root setting; unimported rows link to `/annotate?local_path=` with encoded paths and never POST preparation automatically.

- [ ] **Step 4: Run page/client/component tests and type checks.** Verify rendered links target exact registered aliases and export paths; show diagnostics when no valid alias is recorded rather than creating one on read.
- [ ] **Step 5: Commit task-owned changes.** Use selective staging for already-dirty homepage. Message: `feat: add dataset monitor page with visible-only refresh`.

## Task 8: Acceptance, documentation, and local rollout

**Files:** Modify `deployment/README.md`; create `docs/artifacts/dataset-monitor/acceptance.md`. Update implementation tests only for actual failures found during acceptance.

**Interfaces:** Consumes the completed read-only page/endpoints. Produces verified local deployment and an evidence record. No dataset or HF publication mutation is part of this task.

- [ ] **Step 1: Run focused backend and frontend checks.** Commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 backend/.venv/bin/python -m pytest backend/tests/test_annotation_monitor.py backend/tests/test_annotation_monitor_prompts.py backend/tests/test_annotation_monitor_publications.py backend/tests/test_annotation_monitor_api.py -q
bun test src/utils/__tests__/monitorClient.test.ts src/components/__tests__/dataset-monitor-row.test.tsx src/app/monitor/__tests__/page.test.tsx
bun run type-check
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun run build
```

Run relevant existing access, annotation review, and publication regression tests if shared helpers or routes changed. Run `ruff check --no-cache` on task Python files and `git diff --check`. Do not invoke global formatting over unrelated changes.

- [ ] **Step 2: Reconcile live acceptance with current source state.** Check the selected run `b937e3f6925a4bafa5eb1bb70ea74ec8`: expected historical baseline 87 collected/imported, 71 accepted/currently reviewed, 16 rejected, 0 pending, 34,594 exported frames, publication `mncai/G1_Dex3_PickTable` / `260915` / `f3d31d6b481a61fc1779890df46be58fd38cd811`. If the user has changed data since design, record the actual changed state instead of forcing old counts. Compare output with source metadata, review hashes, and export manifest.
- [ ] **Step 3: Deploy only when annotation jobs are idle.** Inspect persisted/current job statuses and live tmux services. Rebuild before restarting UI; restart only the annotation backend/UI with configured monitor root, preserving all other env settings and the VLM tunnel. Do not interrupt generation or upload to deploy this page.
- [ ] **Step 4: Browser verification at 1440 px and 390 px.** Open `/monitor`; inspect baseline row, select the published run, expand prompt ratios, copy the verified training path, follow review and HF links, test manual refresh and search, then trigger the explicit read-only Check HF action. Record returned head as checked now; do not assert it remains the historical commit if the branch legitimately advanced. Confirm no horizontal overflow hides controls and status labels remain readable.
- [ ] **Step 5: Confirm no mutations.** Compare dataset/run/review/alias-registry content signatures from before and after browser monitoring. Changes due to an independent active collector must be distinguished from the monitor; monitor traces must show only read calls and the explicit remote-check POST. Use the appended-episode fixture for growth verification rather than modifying a real collection.
- [ ] **Step 6: Document and commit.** Add root configuration, local-only availability, metric denominators, metadata-only source checks, periodic refresh, manual HF checks, and known-history limitations to deployment README. Record commands/results, responsive screenshots, acceptance values, and available review/training URLs in `docs/artifacts/dataset-monitor/acceptance.md`. Commit only task hunks with message `docs: document and verify dataset monitoring workflow`.

## Plan self-review and handoff

Coverage map: discovery/provenance (Task 1); metric denominators and source growth (Task 2); interval-aware prompt ratios (Task 3); receipts and remote checks (Task 4); consistency, caching, identifiers, configuration/auth (Task 5); responsive detail presentation (Task 6); selection/polling/navigation (Task 7); regression checks and live acceptance (Task 8).

Implementation can begin after choosing subagent-driven or inline execution. This plan does not authorize changing reviewed prompts, retention decisions, exclusions, source datasets, or Hub content. The monitoring feature reads those records and links to existing workflows for edits.
