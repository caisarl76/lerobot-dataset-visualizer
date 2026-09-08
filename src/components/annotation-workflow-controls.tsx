"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  fetchWorkflow,
  postWorkflowDecision,
  exportWorkflow,
  publishWorkflow,
  getAnnotationJob,
  createAnnotationJob,
  type WorkflowRun,
  type WorkflowMetric,
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
export function AnnotationWorkflowControls() {
  const { ident, episodeId, dirty, saving, atoms, frameTimestamps } =
    useAnnotations();
  const alias = ident.repoId?.startsWith("local/")
    ? ident.repoId.slice(6)
    : null;
  const [run, setRun] = useState<WorkflowRun | null>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [reason, setReason] = useState("");
  const [filter, setFilter] = useState("all");
  const [queueEpisodeId, setQueueEpisodeId] = useState<number | null>(null);
  const inspectedEpisode = queueEpisodeId ?? episodeId;
  const [refresh, setRefresh] = useState(0);
  const generation = useRef(0);
  useEffect(() => {
    const refreshRun = () => setRefresh((value) => value + 1);
    window.addEventListener("annotation-workflow-changed", refreshRun);
    return () =>
      window.removeEventListener("annotation-workflow-changed", refreshRun);
  }, []);
  useEffect(() => {
    const scope = generation;
    const token = ++scope.current;
    setQueueEpisodeId(null);
    setRun(null);
    setReason("");
    setStatus("");
    setBusy(false);
    if (!alias) return;
    let cancelled = false;
    async function load() {
      try {
        let next = await fetchWorkflow(alias!);
        if (cancelled || token !== generation.current) return;
        setRun(next);
        setReason(next.episodes[String(episodeId)]?.decision_reason ?? "");
        if (next.current_job_id) {
          let job = await getAnnotationJob(next.current_job_id);
          while (
            !cancelled &&
            token === generation.current &&
            ["queued", "running"].includes(job.status)
          ) {
            setBusy(true);
            setStatus(`Job ${job.status}…`);
            await new Promise((resolve) => setTimeout(resolve, 1000));
            if (cancelled || token !== generation.current) return;
            job = await getAnnotationJob(next.current_job_id);
          }
          if (cancelled || token !== generation.current) return;
          if (job.status === "completed") {
            next = await fetchWorkflow(alias!);
            if (cancelled || token !== generation.current) return;
            setRun(next);
          }
          setStatus(
            job.status === "failed" || job.status === "interrupted"
              ? job.error ||
                  `Job ${job.status}. Retry generation for unfinished episodes.`
              : "",
          );
          setBusy(false);
        }
      } catch (error) {
        if (!cancelled && token === generation.current) {
          setStatus(error instanceof Error ? error.message : String(error));
          setBusy(false);
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
      scope.current++;
    };
  }, [alias, episodeId, dirty, saving, refresh]);
  if (!alias) return null;
  if (!run)
    return (
      <section aria-label="Annotation workflow">
        <p>{status || "Loading workflow…"}</p>
        <Link href="/annotate">Prepare hosted workflow</Link>
      </section>
    );
  const current = run.episodes[String(inspectedEpisode)];
  const disabled = dirty || saving || busy || episodeId === null;
  const decision = async (value: "keep" | "delete" | "pending") => {
    if (disabled) return;
    const token = ++generation.current;
    setBusy(true);
    setStatus("Saving decision…");
    try {
      const next = await postWorkflowDecision(alias, {
        episode_index: inspectedEpisode!,
        decision: value,
        reason: reason.trim(),
        expected_revision: run.revision,
      });
      if (token !== generation.current) return;
      setRun(next);
      setStatus("Decision saved. Delete remains reversible until export.");
    } catch (error) {
      if (token === generation.current)
        setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (token === generation.current) setBusy(false);
    }
  };
  const submit = async (publish: boolean, resume = false) => {
    if (disabled || (publish && !run.export)) return;
    const token = ++generation.current;
    setBusy(true);
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
          : await exportWorkflow(alias, run.revision);
      if (token !== generation.current) return;
      let job = await getAnnotationJob(request.job_id);
      while (
        ["queued", "running"].includes(job.status) &&
        token === generation.current
      ) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (token !== generation.current) return;
        job = await getAnnotationJob(request.job_id);
      }
      if (token !== generation.current) return;
      if (job.status !== "completed")
        throw new Error(job.error || `Job ${job.status}`);
      const next = await fetchWorkflow(alias);
      if (token !== generation.current) return;
      setRun(next);
      setStatus(
        resume
          ? "Generation finished. Open the current checkpoint to review the results."
          : publish
            ? "Publication completed."
            : "Export validated. Inspect the preview before updating Hugging Face.",
      );
    } catch (error) {
      if (token === generation.current)
        setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (token === generation.current) setBusy(false);
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
  return (
    <section
      aria-label="Annotation workflow"
      className="annotation-composer"
      style={{ display: "block" }}
    >
      <h3>Annotation workflow</h3>
      {run.current_repo_id && run.current_repo_id !== ident.repoId && (
        <Link
          href={`/${run.current_repo_id}/episode_${episodeId ?? 0}?tab=annotations`}
        >
          Open current checkpoint
        </Link>
      )}
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
      <div style={{ maxHeight: 240, overflow: "auto" }}>
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
      <h4>
        Episode {inspectedEpisode}: {current?.decision ?? "pending"}
      </h4>
      {(current?.issues ?? []).map((issue, i) => (
        <div key={`${issue.code}-${i}`}>
          <strong>
            {issue.code} · {issue.source} · {issue.severity}
          </strong>
          <p>{issue.message}</p>
          {typeof issue.start === "number" && (
            <p>
              Evidence: {seconds(issue.start)}
              {typeof issue.end === "number" ? ` – ${seconds(issue.end)}` : ""}
            </p>
          )}
        </div>
      ))}
      <label>
        Decision reason{" "}
        <input
          aria-label="Decision reason"
          value={reason}
          onInput={(e) => setReason(e.currentTarget.value)}
          disabled={disabled}
        />
      </label>
      <button disabled={disabled} onClick={() => decision("keep")}>
        Keep
      </button>
      <button
        disabled={disabled || !reason.trim()}
        onClick={() => decision("delete")}
      >
        Delete
      </button>
      <button disabled={disabled} onClick={() => decision("pending")}>
        Undo decision
      </button>
      <p>
        Delete flags this episode for the export. Keep or undo restores it
        without changing files.
      </p>
      {dirty && (
        <p>Save the current episode before changing decisions or exporting.</p>
      )}
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
              Subtask topology changed. Boundary differences are not comparable.
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
      <details>
        <summary>Reviewed boundary accuracy</summary>
        <p>
          Only reviewed, generated, non-example episodes qualify. These reports
          use saved annotations.
        </p>
        <Metric label="First pass" metric={run.metrics.first_pass} />
        <Metric label="Latest prediction" metric={run.metrics.latest} />
      </details>
      <h4>Export and publication</h4>
      <button disabled={disabled} onClick={() => submit(false)}>
        Preview export
      </button>
      {run.export && (
        <div>
          <p>
            Retained: {run.export.retained_episodes}; deleted:{" "}
            {Array.isArray(run.export.deleted_episodes)
              ? run.export.deleted_episodes.length
              : run.export.deleted_episodes}
          </p>
          <p>
            Destination: {run.repo_id || "Local export only"} · main format:{" "}
            {run.source_format || "source format"} · rich revision: annotations/
            {run.run_id}
          </p>
          <p>
            Manifest: <code>{run.export.manifest_sha256}</code>
          </p>
          {run.export.managed_changes && (
            <details>
              <summary>File changes from the pinned source</summary>
              <p>
                Add/update: {run.export.managed_changes.added_or_updated.length}
                ; remove: {run.export.managed_changes.deleted.length}
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
              {run.export.validation.warnings.map((message, index) => (
                <p key={`warning-${index}`}>Warning: {message}</p>
              ))}
            </div>
          )}
          <button
            disabled={disabled || run.publication_state !== "exported"}
            onClick={() => submit(true)}
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
      {status && <p role="status">{status}</p>}
    </section>
  );
}
