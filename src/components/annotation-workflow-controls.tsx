"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  fetchWorkflow,
  detectWorkflowTransitions,
  postWorkflowDecision,
  keepWorkflowRemaining,
  reviewWorkflowRetained,
  exportWorkflow,
  publishWorkflow,
  getAnnotationJob,
  createAnnotationJob,
  type WorkflowRun,
  type WorkflowMetric,
  type WorkflowExportOptions,
} from "../utils/annotationsClient";
import { useAnnotations } from "../context/annotations-context";
import type { LanguageAtom } from "../types/language.types";

const subtasks = (atoms: LanguageAtom[]) =>
  atoms
    .filter((atom) => atom.style === "subtask")
    .sort((a, b) => a.timestamp - b.timestamp);
const seconds = (value: number) => `${value.toFixed(3)} s`;
function Metric({ label, metric }: { label: string; metric?: WorkflowMetric }) {
  return (
    <div>
      <strong>{label}</strong>
      {metric ? (
        <>
          <p>
            MAE:{" "}
            {metric.mae_seconds == null
              ? "Not eligible"
              : seconds(metric.mae_seconds)}
          </p>
          <p>
            Eligible episodes: {metric.eligible_episodes}; boundaries:{" "}
            {metric.eligible_boundaries}; missing predictions:{" "}
            {metric.missing_predictions}; topology mismatch:{" "}
            {metric.topology_mismatch}
          </p>
        </>
      ) : (
        <p>No accuracy report yet.</p>
      )}
    </div>
  );
}
const defaultOptions = (): WorkflowExportOptions => ({
  export_format: "groot_v21",
  instruction_mode: "task",
  dataset_name: "retained-dataset",
  destination_repo_id: null,
  destination_revision: "main",
  destination_private: true,
});
function frozenExportOptions(
  value: WorkflowRun["export"],
): WorkflowExportOptions | null {
  if (!value?.format || !value.instruction_mode || !value.dataset_name)
    return null;
  return {
    export_format: value.format,
    instruction_mode: value.instruction_mode,
    dataset_name: value.dataset_name,
    destination_repo_id: value.destination?.repo_id ?? null,
    destination_revision: value.destination?.revision ?? "main",
    destination_private: value.destination?.private ?? true,
  };
}
function normalizedOptions(
  value: WorkflowExportOptions,
): WorkflowExportOptions {
  return {
    ...value,
    dataset_name: value.dataset_name.trim(),
    destination_repo_id: value.destination_repo_id?.trim() || null,
    destination_revision: value.destination_revision.trim() || "main",
  };
}
export function AnnotationWorkflowControls() {
  const { ident, episodeId, dirty, saving, atoms, frameTimestamps } =
    useAnnotations();
  const alias = ident.repoId?.startsWith("local/")
    ? ident.repoId.slice(6)
    : null;
  const datasetKey = JSON.stringify([alias, ident.revision, ident.localPath]);
  const contextKey = JSON.stringify([datasetKey, episodeId]);
  const contextRef = useRef(contextKey);
  contextRef.current = contextKey;
  const [loaded, setLoaded] = useState<{
    key: string;
    run: WorkflowRun;
  } | null>(null);
  const run = loaded?.key === contextKey ? loaded.run : null;
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);
  const [reason, setReason] = useState("");
  const [filter, setFilter] = useState("all");
  const [includeReviewedTransitions, setIncludeReviewedTransitions] =
    useState(false);
  const [queueEpisodeId, setQueueEpisodeId] = useState<number | null>(null);
  const inspectedEpisode = queueEpisodeId ?? episodeId;
  const [refresh, setRefresh] = useState(0);
  const [reviewConsent, setReviewConsent] = useState<string | null>(null);
  const [destinationConsent, setDestinationConsent] = useState<string | null>(
    null,
  );
  const [options, setOptions] = useState<WorkflowExportOptions>(defaultOptions);
  const hydratedDataset = useRef<string | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const exportDetails = useRef<HTMLDetailsElement>(null);
  const scrollRequested = useRef(false);
  const generation = useRef(0);
  const activeMutation = useRef(false);
  const previousContext = useRef<string | null>(null);
  const mounted = useRef(true);
  const setRun = (next: WorkflowRun) =>
    setLoaded({ key: contextKey, run: next });

  useEffect(() => {
    const requests = generation;
    mounted.current = true;
    return () => {
      mounted.current = false;
      requests.current++;
    };
  }, []);
  useEffect(() => {
    hydratedDataset.current = null;
    setOptions(defaultOptions());
    setReviewConsent(null);
    setDestinationConsent(null);
    setExportOpen(false);
    setFilter("all");
    setIncludeReviewedTransitions(false);
  }, [datasetKey]);
  useEffect(() => {
    if (run && hydratedDataset.current !== datasetKey) {
      setOptions(frozenExportOptions(run.export) ?? defaultOptions());
      hydratedDataset.current = datasetKey;
    }
  }, [run, datasetKey]);
  useEffect(() => {
    const refreshRun = () => setRefresh((value) => value + 1);
    const openExport = () => {
      scrollRequested.current = true;
      setExportOpen(true);
    };
    window.addEventListener("annotation-workflow-changed", refreshRun);
    window.addEventListener("annotation-open-export", openExport);
    return () => {
      window.removeEventListener("annotation-workflow-changed", refreshRun);
      window.removeEventListener("annotation-open-export", openExport);
    };
  }, []);
  useEffect(() => {
    if (exportOpen && run && scrollRequested.current && exportDetails.current) {
      scrollRequested.current = false;
      exportDetails.current.scrollIntoView?.({
        block: "start",
        behavior: "smooth",
      });
      exportDetails.current
        .querySelector<HTMLElement>("summary")
        ?.focus({ preventScroll: true });
    }
  }, [exportOpen, run]);
  useEffect(() => {
    const changedContext = previousContext.current !== contextKey;
    previousContext.current = contextKey;
    if (changedContext) {
      activeMutation.current = false;
      setLoaded(null);
      setQueueEpisodeId(null);
      setReason("");
      setStatus("");
      setReviewConsent(null);
      setDestinationConsent(null);
    } else if (activeMutation.current) return;
    const requests = generation;
    const token = ++requests.current;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const current = () =>
      !cancelled &&
      mounted.current &&
      contextRef.current === contextKey &&
      generation.current === token;
    if (!alias) {
      setBusy(false);
      return;
    }
    setBusy(true);
    setLoadFailed(false);
    async function load() {
      try {
        let next = await fetchWorkflow(alias!);
        if (!current()) return;
        setLoaded({ key: contextKey, run: next });
        if (changedContext)
          setReason(next.episodes[String(episodeId)]?.decision_reason ?? "");
        if (next.current_job_id) {
          let job = await getAnnotationJob(next.current_job_id);
          if (!current()) return;
          const wasActive = ["queued", "running"].includes(job.status);
          while (current() && ["queued", "running"].includes(job.status)) {
            setStatus(`Job ${job.status}…`);
            await new Promise<void>((resolve) => {
              timer = setTimeout(resolve, 1000);
            });
            if (!current()) return;
            job = await getAnnotationJob(next.current_job_id);
          }
          if (!current()) return;
          if (job.status === "completed") {
            next = await fetchWorkflow(alias!);
            if (!current()) return;
            setLoaded({ key: contextKey, run: next });
            if (wasActive || changedContext) setStatus("Job completed.");
            // Old bookmarks still point at the imported draft. Reload the whole
            // viewer so media, parquet rows, and editor atoms share the checkpoint.
            if (
              !dirty &&
              !saving &&
              episodeId !== null &&
              next.current_repo_id?.startsWith("local/") &&
              next.current_repo_id !== ident.repoId
            ) {
              window.location.replace(
                `/${next.current_repo_id}/episode_${episodeId}${window.location.search || "?tab=annotations"}${window.location.hash}`,
              );
            }
          }
          if (job.status === "failed" || job.status === "interrupted")
            setStatus(
              job.error ||
                `Job ${job.status}. Retry generation for unfinished episodes.`,
            );
        }
      } catch (error) {
        if (current()) {
          setStatus(error instanceof Error ? error.message : String(error));
          setLoadFailed(true);
        }
      } finally {
        if (current()) setBusy(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
      if (requests.current === token) requests.current++;
      if (timer) clearTimeout(timer);
    };
  }, [alias, contextKey, episodeId, dirty, saving, refresh]);

  if (!alias) return null;
  if (!run)
    return (
      <section aria-label="Annotation workflow">
        <p>{status || "Loading workflow…"}</p>
        <Link href="/annotate">Prepare review draft</Link>
      </section>
    );
  const current = run.episodes[String(inspectedEpisode)];
  const disabled =
    dirty ||
    saving ||
    busy ||
    loadFailed ||
    episodeId === null ||
    hydratedDataset.current !== datasetKey;
  const exportOptions = normalizedOptions(options);
  const frozenOptions = frozenExportOptions(run.export);
  // Existing repository visibility is observed metadata, not an export option.
  const comparableOptions =
    run.export?.destination?.exists && frozenOptions
      ? {
          ...exportOptions,
          destination_private: frozenOptions.destination_private,
        }
      : exportOptions;
  const optionsChanged =
    !!run.export &&
    (!frozenOptions ||
      JSON.stringify(frozenOptions) !== JSON.stringify(comparableOptions));
  const snapshotKey = JSON.stringify([
    contextKey,
    run.run_id,
    run.revision,
    run.review_snapshot_sha256,
  ]);
  const publishKey = JSON.stringify([
    snapshotKey,
    run.export?.manifest_sha256,
    run.export?.destination,
  ]);
  const destination = run.export?.destination;
  const publishAllowed =
    !disabled &&
    run.publication_state === "exported" &&
    !!run.export &&
    !!destination &&
    !!frozenOptions &&
    !optionsChanged &&
    destinationConsent === publishKey;
  const updateOptions = (update: Partial<WorkflowExportOptions>) => {
    setOptions((value) => ({ ...value, ...update }));
    setDestinationConsent(null);
  };
  const notify = (review = false) => {
    window.dispatchEvent(new window.Event("annotation-workflow-changed"));
    if (review)
      window.dispatchEvent(new window.Event("annotation-review-changed"));
  };
  const mutate = async (
    work: () => Promise<WorkflowRun>,
    pending: string,
    completed: string | ((next: WorkflowRun) => string),
    review = false,
  ) => {
    if (disabled || activeMutation.current) return;
    const token = ++generation.current;
    const valid = () =>
      mounted.current &&
      contextRef.current === contextKey &&
      generation.current === token;
    activeMutation.current = true;
    setBusy(true);
    setStatus(pending);
    setReviewConsent(null);
    setDestinationConsent(null);
    try {
      const next = await work();
      if (!valid()) return;
      setRun(next);
      setStatus(typeof completed === "function" ? completed(next) : completed);
      notify(review);
    } catch (error) {
      if (valid())
        setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (valid()) {
        activeMutation.current = false;
        setBusy(false);
        setRefresh((value) => value + 1);
      }
    }
  };
  const decision = (value: "keep" | "delete" | "pending") =>
    mutate(
      () =>
        postWorkflowDecision(alias, {
          episode_index: inspectedEpisode!,
          decision: value,
          reason: reason.trim(),
          expected_revision: run.revision,
        }),
      "Saving decision…",
      "Decision saved. Delete remains reversible until export.",
    );
  const submit = async (publish: boolean, resume = false) => {
    if (disabled || activeMutation.current || (publish && !publishAllowed))
      return;
    const requestedOptions = { ...exportOptions };
    if (!publish && !resume && !requestedOptions.dataset_name) return;
    const token = ++generation.current;
    const valid = () =>
      mounted.current &&
      contextRef.current === contextKey &&
      generation.current === token;
    activeMutation.current = true;
    setBusy(true);
    setDestinationConsent(null);
    setReviewConsent(null);
    setStatus(
      resume
        ? "Resuming unfinished episodes…"
        : publish
          ? "Publishing frozen export…"
          : "Preparing export preview…",
    );
    try {
      const request = resume
        ? await createAnnotationJob({
            repoId: run.current_repo_id || ident.repoId,
            resume_unfinished: true,
            episode_indices: null,
            task_prompt: run.task_prompt,
            subtask_prompts: run.subtask_prompts,
            example_episode_indices: run.example_episode_indices || [],
            config: {},
            assess_quality: true,
          })
        : publish
          ? await publishWorkflow(
              alias,
              run.export!.manifest_sha256,
              run.revision,
            )
          : await exportWorkflow(alias, run.revision, requestedOptions);
      if (!valid()) return;
      let job = await getAnnotationJob(request.job_id);
      if (!valid()) return;
      while (["queued", "running"].includes(job.status)) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (!valid()) return;
        job = await getAnnotationJob(request.job_id);
        if (!valid()) return;
      }
      if (job.status !== "completed")
        throw new Error(job.error || `Job ${job.status}`);
      const next = await fetchWorkflow(alias);
      if (!valid()) return;
      setRun(next);
      if (!resume && !publish) {
        const nextOptions =
          frozenExportOptions(next.export) ?? requestedOptions;
        setOptions(
          next.export?.destination?.exists
            ? {
                ...nextOptions,
                destination_private: requestedOptions.destination_private,
              }
            : nextOptions,
        );
      }
      setStatus(
        resume
          ? "Generation finished. Open the current checkpoint to review the results."
          : publish
            ? "Publication completed."
            : "Export validated. Inspect the frozen destination and files before publishing.",
      );
      notify();
    } catch (error) {
      if (valid())
        setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (valid()) {
        activeMutation.current = false;
        setBusy(false);
        setRefresh((value) => value + 1);
      }
    }
  };
  const queue = Object.entries(run.episodes)
    .sort(([a], [b]) => Number(a) - Number(b))
    .filter(
      ([, ep]) =>
        filter === "all" ||
        (filter === "flagged" && ep.issues.length > 0) ||
        (filter === "failed" && ep.generation_status === "failed") ||
        (filter === "reviewed" &&
          ep.decision !== "delete" &&
          ep.review?.status === "reviewed") ||
        (filter === "unreviewed" &&
          ep.decision !== "delete" &&
          ep.review?.status !== "reviewed") ||
        (filter === "deleted" && ep.decision === "delete"),
    );
  const original = subtasks(current?.predictions?.[0]?.atoms ?? []);
  const edited = subtasks(atoms);
  const comparable =
    original.length > 0 &&
    original.length === edited.length &&
    original.every((a, i) => a.content === edited[i].content);
  const endTime = frameTimestamps.at(-1);
  const kept = Object.values(run.episodes).filter(
    (ep) => ep.decision !== "delete",
  ).length;
  const deleted = Object.values(run.episodes).filter(
    (ep) => ep.decision === "delete",
  ).length;
  const unreviewed = Object.values(run.episodes).filter(
    (ep) => ep.decision !== "delete" && ep.review?.status !== "reviewed",
  ).length;
  return (
    <section aria-label="Annotation workflow" className="annotation-workflow">
      <h3>Annotation workflow</h3>
      {run.current_repo_id && run.current_repo_id !== ident.repoId && (
        <Link
          href={`/${run.current_repo_id}/episode_${episodeId ?? 0}?tab=annotations`}
        >
          Open current checkpoint
        </Link>
      )}
      <div className="workflow-episode-actions">
        <h4>Episode {inspectedEpisode}</h4>
        <p>Generation: {current?.generation_status ?? "pending"}</p>
        <p>Review decision: {current?.decision ?? "pending"}</p>
        {(current?.issues ?? []).map((issue, i) => (
          <div key={`${issue.code}-${i}`}>
            <strong>
              {issue.code} · {issue.source} · {issue.severity}
            </strong>
            <p>{issue.message}</p>
            {typeof issue.start === "number" && (
              <p>
                Evidence: {seconds(issue.start)}
                {typeof issue.end === "number"
                  ? ` – ${seconds(issue.end)}`
                  : ""}
              </p>
            )}
          </div>
        ))}
        <label>
          Decision reason{" "}
          <input
            aria-label="Decision reason"
            placeholder="Required to delete an episode"
            value={reason}
            onInput={(e) => setReason(e.currentTarget.value)}
            disabled={disabled}
          />
        </label>
        <p>
          Provide a reason before deleting an episode. Deletion is reversible
          until export.
        </p>
        <div className="workflow-action-buttons">
          <button disabled={disabled} onClick={() => decision("keep")}>
            Keep
          </button>
          <button
            disabled={disabled || !reason.trim()}
            onClick={() => decision("delete")}
          >
            Delete episode
          </button>
          <button disabled={disabled} onClick={() => decision("pending")}>
            Undo decision
          </button>
        </div>
        {dirty && (
          <p>
            Save the current episode before changing decisions or exporting.
          </p>
        )}
      </div>
      <div className="workflow-transition-actions">
        <h4>Transition pauses</h4>
        <p>
          Detect Pose ↔ Planner pauses and merge them into Exclude for
          unreviewed, retained episodes.
        </p>
        <label>
          <input
            type="checkbox"
            checked={includeReviewedTransitions}
            disabled={disabled}
            onChange={(event) =>
              setIncludeReviewedTransitions(event.target.checked)
            }
          />
          Include reviewed episodes
        </label>
        {includeReviewedTransitions && (
          <p>
            Reviewed episodes with changed exclusions will need review again.
          </p>
        )}
        <div className="workflow-action-buttons">
          <button
            disabled={disabled}
            onClick={() =>
              void mutate(
                () =>
                  detectWorkflowTransitions(alias, {
                    expected_revision: run.revision,
                    include_reviewed: includeReviewedTransitions,
                  }),
                "Detecting transition pauses…",
                (next) =>
                  next.transition_detection?.skipped_reason
                    ? `Transition filter skipped: ${next.transition_detection.skipped_reason}`
                    : `Updated exclusions in ${next.transition_detection?.episodes_changed ?? 0} episodes. Review them on the Exclude timeline.`,
                true,
              )
            }
          >
            Detect transition pauses
          </button>
        </div>
      </div>
      <details className="workflow-details">
        <summary>Review queue ({queue.length} episodes)</summary>
        {Object.entries(run.episodes).some(
          ([id, ep]) =>
            ep.generation_status !== "generated" &&
            ep.decision !== "delete" &&
            !(run.example_episode_indices || []).includes(Number(id)),
        ) && (
          <button disabled={disabled} onClick={() => void submit(false, true)}>
            Resume unfinished episodes
          </button>
        )}
        <p>
          Run {run.run_id} · {run.publication_state}
        </p>
        <button onClick={() => setRefresh((x) => x + 1)} disabled={busy}>
          Refresh workflow
        </button>
        <label>
          Queue filter{" "}
          <select
            aria-label="Queue filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">All episodes</option>
            <option value="flagged">Flagged</option>
            <option value="failed">Generation failed</option>
            <option value="unreviewed">Needs review</option>
            <option value="reviewed">Reviewed</option>
            <option value="deleted">Marked for deletion</option>
          </select>
        </label>
        <div className="workflow-queue">
          <table>
            <thead>
              <tr>
                <th>Episode</th>
                <th>Generation</th>
                <th>Issues</th>
                <th>Decision</th>
                <th>Review</th>
                <th>Inspect</th>
              </tr>
            </thead>
            <tbody>
              {queue.map(([id, ep]) => (
                <tr
                  key={id}
                  aria-current={Number(id) === episodeId ? "true" : undefined}
                >
                  <td>
                    <a
                      href={`/${run.current_repo_id || `local/${encodeURIComponent(alias)}`}/episode_${id}?tab=annotations`}
                    >
                      Episode {id}
                    </a>
                  </td>
                  <td>{ep.generation_status}</td>
                  <td>{ep.issues.length}</td>
                  <td>{ep.decision}</td>
                  <td>{ep.review?.status ?? "unreviewed"}</td>
                  <td>
                    <button
                      disabled={busy}
                      aria-label={`Inspect episode ${id} issues`}
                      onClick={() => {
                        setQueueEpisodeId(Number(id));
                        setReason(ep.decision_reason ?? "");
                        setStatus("");
                      }}
                    >
                      Inspect issues
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!queue.length && <p>No episodes match this filter.</p>}
        </div>
      </details>
      <details className="workflow-details">
        <summary>Prediction comparison</summary>
        <h4>Original prediction and edited subtasks</h4>
        {inspectedEpisode !== episodeId ? (
          <p>
            <a
              href={`/${run.current_repo_id || `local/${encodeURIComponent(alias)}`}/episode_${inspectedEpisode}?tab=annotations`}
            >
              Open this episode’s video and timeline
            </a>{" "}
            to inspect or edit its annotations. You can decide from the recorded
            issues above if the episode cannot be loaded.
          </p>
        ) : !original.length ? (
          <p>
            No original prediction. This episode is excluded from boundary
            accuracy.
          </p>
        ) : (
          <>
            {!comparable && (
              <p>
                Subtask topology changed. Boundary differences are not
                comparable.
              </p>
            )}
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Original subtask</th>
                    <th>Original interval</th>
                    <th>Edited subtask</th>
                    <th>Edited interval</th>
                    <th>Signed start change</th>
                    <th>Absolute start change</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from(
                    { length: Math.max(original.length, edited.length) },
                    (_, i) => (
                      <tr key={i}>
                        <td>{original[i]?.content ?? "—"}</td>
                        <td>
                          {original[i]
                            ? `${seconds(original[i].timestamp)} – ${original[i + 1] ? seconds(original[i + 1].timestamp) : endTime == null ? "episode end" : seconds(endTime)}`
                            : "—"}
                        </td>
                        <td>{edited[i]?.content ?? "—"}</td>
                        <td>
                          {edited[i]
                            ? `${seconds(edited[i].timestamp)} – ${edited[i + 1] ? seconds(edited[i + 1].timestamp) : endTime == null ? "episode end" : seconds(endTime)}`
                            : "—"}
                        </td>
                        <td>
                          {comparable
                            ? `${edited[i].timestamp - original[i].timestamp >= 0 ? "+" : ""}${seconds(edited[i].timestamp - original[i].timestamp)}`
                            : "—"}
                        </td>
                        <td>
                          {comparable
                            ? seconds(
                                Math.abs(
                                  edited[i].timestamp - original[i].timestamp,
                                ),
                              )
                            : "—"}
                        </td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </details>
      <details className="workflow-details">
        <summary>Reviewed boundary accuracy</summary>
        <p>
          Only reviewed, generated, non-example episodes qualify. These reports
          use saved annotations.
        </p>
        <Metric label="First pass" metric={run.metrics.first_pass} />
        <Metric label="Latest prediction" metric={run.metrics.latest} />
      </details>
      <details
        className="workflow-details workflow-export"
        ref={exportDetails}
        open={exportOpen}
        onToggle={(event) => setExportOpen(event.currentTarget.open)}
      >
        <summary>Export &amp; publish</summary>
        <p>
          Retained: {kept} · Delete: {deleted} · Unreviewed retained:{" "}
          {unreviewed}
        </p>
        <div className="workflow-bulk-actions">
          <button
            disabled={
              disabled ||
              !Object.values(run.episodes).some(
                (ep) => ep.decision !== "delete" && ep.decision !== "keep",
              )
            }
            onClick={() =>
              void mutate(
                () => keepWorkflowRemaining(alias, run.revision),
                "Keeping remaining episodes…",
                "Remaining episodes kept; deletion decisions preserved.",
              )
            }
          >
            Keep remaining
          </button>
          <label>
            <input
              type="checkbox"
              checked={reviewConsent === snapshotKey}
              onChange={(event) =>
                setReviewConsent(
                  event.currentTarget.checked ? snapshotKey : null,
                )
              }
              disabled={disabled || !kept || !run.review_snapshot_sha256}
            />{" "}
            I verified the retained prompts and clipping intervals
          </label>
          <button
            disabled={
              disabled ||
              !kept ||
              reviewConsent !== snapshotKey ||
              !run.review_snapshot_sha256
            }
            onClick={() => {
              if (reviewConsent === snapshotKey && run.review_snapshot_sha256)
                void mutate(
                  () =>
                    reviewWorkflowRetained(alias, {
                      expected_revision: run.revision,
                      expected_review_sha256: run.review_snapshot_sha256!,
                      confirmed: true,
                    }),
                  "Reviewing retained episodes…",
                  "Retained episodes marked reviewed.",
                  true,
                );
            }}
          >
            Mark retained reviewed
          </button>
        </div>
        <fieldset className="workflow-export-options" disabled={disabled}>
          <legend>Export options</legend>
          <label>
            Export format{" "}
            <select
              value={options.export_format}
              onChange={(event) =>
                updateOptions({
                  export_format: event.target
                    .value as WorkflowExportOptions["export_format"],
                })
              }
            >
              <option value="groot_v21">GR00T v2.1</option>
              <option value="rich">Rich annotations</option>
            </select>
          </label>
          <label>
            Instruction mode{" "}
            <select
              value={options.instruction_mode}
              onChange={(event) =>
                updateOptions({
                  instruction_mode: event.target
                    .value as WorkflowExportOptions["instruction_mode"],
                })
              }
            >
              <option value="task">Task</option>
              <option value="subtask">Subtask</option>
            </select>
          </label>
          <label>
            Dataset folder name{" "}
            <input
              value={options.dataset_name}
              onInput={(event) =>
                updateOptions({ dataset_name: event.currentTarget.value })
              }
            />
          </label>
          <label>
            HF repository (optional){" "}
            <input
              value={options.destination_repo_id ?? ""}
              onInput={(event) =>
                updateOptions({
                  destination_repo_id: event.currentTarget.value,
                })
              }
              placeholder="org/dataset"
            />
          </label>
          <label>
            Revision{" "}
            <input
              value={options.destination_revision}
              disabled={!exportOptions.destination_repo_id}
              onInput={(event) =>
                updateOptions({
                  destination_revision: event.currentTarget.value,
                })
              }
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={options.destination_private}
              disabled={!exportOptions.destination_repo_id}
              onChange={(event) =>
                updateOptions({
                  destination_private: event.currentTarget.checked,
                })
              }
            />{" "}
            Create new repositories privately
          </label>
        </fieldset>
        <p>
          Leave the HF repository empty for a local export. Preview freezes the
          selected format and destination; publishing uses that exact preview.
        </p>
        {unreviewed > 0 && (
          <p>
            Mark the retained episodes reviewed before previewing an export.
          </p>
        )}
        <button
          disabled={
            disabled || !kept || unreviewed > 0 || !exportOptions.dataset_name
          }
          onClick={() => void submit(false)}
        >
          Preview export
        </button>
        {run.export && (
          <div className="workflow-frozen-export">
            <h4>Frozen export preview</h4>
            <p>
              Retained: {run.export.retained_episodes}; deleted:{" "}
              {Array.isArray(run.export.deleted_episodes)
                ? run.export.deleted_episodes.length
                : run.export.deleted_episodes}
              {run.export.retained_frames !== undefined
                ? `; frames: ${run.export.retained_frames}`
                : ""}
            </p>
            <p>
              Frozen format:{" "}
              {run.export.format === "groot_v21"
                ? "GR00T v2.1"
                : run.export.format === "rich"
                  ? "Rich annotations"
                  : "Not recorded — preview again"}{" "}
              · instructions: {run.export.instruction_mode ?? "not recorded"} ·
              folder: {run.export.dataset_name ?? "not recorded"}
            </p>
            {destination ? (
              <>
                <p>
                  Frozen destination:{" "}
                  <strong>
                    {destination.repo_id} @ {destination.revision}
                  </strong>
                </p>
                <p>
                  {destination.exists
                    ? "Existing repository"
                    : "New repository"}{" "}
                  ·{" "}
                  {destination.revision_exists
                    ? "Existing branch"
                    : "New branch"}
                  {typeof destination.private === "boolean"
                    ? ` · ${destination.private ? "Private" : "Public"}`
                    : ""}
                </p>
                {destination.expected_commit && (
                  <p>
                    Expected destination commit:{" "}
                    <code>{destination.expected_commit}</code>
                  </p>
                )}
                {destination.exists && (
                  <a
                    href={`https://huggingface.co/datasets/${destination.repo_id.split("/").map(encodeURIComponent).join("/")}${destination.revision_exists ? `/tree/${encodeURIComponent(destination.revision)}` : ""}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Inspect destination on Hugging Face
                  </a>
                )}
              </>
            ) : (
              <p>
                Frozen destination: local export only. Enter an HF repository
                and preview again to publish.
              </p>
            )}
            {run.export.local_path && (
              <p>
                Local dataset path: <code>{run.export.local_path}</code>
              </p>
            )}
            {run.export.output_repo_id && (
              <p>
                <Link
                  href={`/${run.export.output_repo_id}/episode_0?tab=annotations`}
                >
                  Open exported dataset
                </Link>
              </p>
            )}
            {optionsChanged && (
              <p role="alert">
                Export options changed. Preview again before publishing.
              </p>
            )}
            <p>
              Manifest: <code>{run.export.manifest_sha256}</code>
            </p>
            {run.export.managed_changes && (
              <details>
                <summary>File changes from the pinned source</summary>
                <p>
                  Add/update:{" "}
                  {run.export.managed_changes.added_or_updated.length}; remove:{" "}
                  {run.export.managed_changes.deleted.length}
                </p>
                <pre>
                  {run.export.managed_changes.added_or_updated
                    .map((path) => `+ ${path}`)
                    .concat(
                      run.export.managed_changes.deleted.map(
                        (path) => `- ${path}`,
                      ),
                    )
                    .join("\n")}
                </pre>
              </details>
            )}
            {Array.isArray(run.export.deleted_episodes) &&
              run.export.deleted_episodes.length > 0 && (
                <p>
                  Deleted episode IDs: {run.export.deleted_episodes.join(", ")}
                </p>
              )}
            {run.export.validation && (
              <div>
                <p>
                  Validation: {run.export.validation.ok ? "passed" : "failed"};
                  episodes checked: {run.export.validation.episodes_checked}
                </p>
                {run.export.validation.errors.map((message, index) => (
                  <p key={`error-${index}`}>Error: {message}</p>
                ))}
                {run.export.validation.warnings.length > 0 && (
                  <details>
                    <summary>
                      Validation warnings (
                      {run.export.validation.warnings.length})
                    </summary>
                    {run.export.validation.warnings.map((message, index) => (
                      <p key={`warning-${index}`}>Warning: {message}</p>
                    ))}
                  </details>
                )}
              </div>
            )}
            {destination && (
              <label className="workflow-destination-consent">
                <input
                  type="checkbox"
                  checked={destinationConsent === publishKey}
                  disabled={
                    disabled ||
                    optionsChanged ||
                    !frozenOptions ||
                    run.publication_state !== "exported"
                  }
                  onChange={(event) =>
                    setDestinationConsent(
                      event.currentTarget.checked ? publishKey : null,
                    )
                  }
                />{" "}
                I confirm publishing this frozen export to {destination.repo_id}{" "}
                @ {destination.revision}
              </label>
            )}
            <button
              disabled={!publishAllowed}
              onClick={() => void submit(true)}
            >
              Update Hugging Face
            </button>
          </div>
        )}
        {run.publication && (
          <div>
            <p>Published commit: {run.publication.main_commit}</p>
            {(run.publication.main_url || run.publication.urls?.main) && (
              <a href={run.publication.main_url || run.publication.urls?.main}>
                Published dataset
              </a>
            )}{" "}
            {(run.publication.rich_url || run.publication.urls?.rich) && (
              <a href={run.publication.rich_url || run.publication.urls?.rich}>
                Rich annotation revision
              </a>
            )}{" "}
            {(run.publication.raw_url || run.publication.urls?.raw) && (
              <a href={run.publication.raw_url || run.publication.urls?.raw}>
                Original source revision
              </a>
            )}
          </div>
        )}
      </details>
      {status && <p role="status">{status}</p>}
    </section>
  );
}
