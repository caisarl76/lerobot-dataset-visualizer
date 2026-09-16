"use client";

import React, { useId, useState } from "react";
import type {
  MonitorDataset,
  MonitorDetail,
  MonitorDiagnostic,
  MonitorExport,
  MonitorPrompts,
  MonitorPublication,
  MonitorRun,
  RemoteCheck,
} from "../utils/monitorClient";
import styles from "../app/monitor/page.module.css";

export interface DatasetMonitorRowProps {
  dataset: MonitorDataset;
  selectedRunId: string | null;
  detail: MonitorDetail | null;
  detailStale?: boolean;
  loading: boolean;
  error: string | null;
  onSelectRun: (runId: string) => void;
  onExpand: (expanded: boolean) => void;
  onCheckPublication: (publicationId: string) => Promise<RemoteCheck>;
}
const count = (value: number | null | undefined): string =>
  value == null ? "Unknown" : value.toLocaleString("en-US");
const percent = (value: number | null): string =>
  value == null
    ? "—"
    : `${(value * 100).toLocaleString("en-US", { maximumFractionDigits: 1 })}%`;
const duration = (seconds: number | null | undefined): string => {
  if (seconds == null) return "Unknown";
  if (seconds < 60)
    return `${seconds.toLocaleString("en-US", { maximumFractionDigits: 2 })} s`;
  return `${Math.floor(seconds / 60)} min ${(seconds % 60).toLocaleString("en-US", { maximumFractionDigits: 1 })} s`;
};
const ids = (values: (number | string)[] | null): string =>
  values == null ? "Unknown" : values.length ? values.join(", ") : "None";
const format = (value: string | null): string =>
  value === "groot_v21" ? "GR00T v2.1" : (value ?? "Unknown");
function reviewUrl(run: MonitorRun | undefined): string | null {
  if (
    !run?.current_repo_id ||
    run.first_retained_episode == null ||
    !Number.isInteger(run.first_retained_episode) ||
    run.first_retained_episode < 0
  )
    return null;
  const parts = run.current_repo_id.split("/");
  if (
    parts.length !== 2 ||
    parts.some((part) => !part || part === "." || part === "..")
  )
    return null;
  return `/${parts.map(encodeURIComponent).join("/")}/episode_${run.first_retained_episode}?tab=annotations`;
}
function hfUrl(raw: string | null): string | null {
  if (!raw) return null;
  try {
    const url = new URL(raw);
    return url.protocol === "https:" &&
      url.hostname === "huggingface.co" &&
      !url.username &&
      !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
}
function mergeRecords<T extends { id: string }>(
  history: T[],
  details: T[],
): T[] {
  return Array.from(
    new Map(
      [...details, ...history].map((record) => [record.id, record]),
    ).values(),
  );
}
function Diagnostics({ items }: { items: MonitorDiagnostic[] }) {
  const unique = Array.from(
    new Map(
      items.map((item) => [`${item.code}:${item.message}`, item]),
    ).values(),
  );
  if (!unique.length) return null;
  return (
    <ul className={styles.diagnostics}>
      {unique.map((item) => (
        <li
          key={`${item.code}:${item.message}`}
          className={
            item.severity === "error"
              ? styles.error
              : item.severity === "warning"
                ? styles.warning
                : styles.muted
          }
        >
          <strong>
            {item.severity === "error"
              ? "Error"
              : item.severity === "warning"
                ? "Warning"
                : "Info"}
            :
          </strong>{" "}
          {item.message}
        </li>
      ))}
    </ul>
  );
}
function TrainingPath({
  path,
  available,
  label,
}: {
  path: string | null;
  available: boolean;
  label: string;
}) {
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  async function copy() {
    if (!path || !available) return;
    setMessage(null);
    setError(null);
    try {
      await navigator.clipboard.writeText(path);
      setMessage("Training path copied.");
    } catch {
      setError(
        "Could not copy the training path. Select and copy the displayed path.",
      );
    }
  }
  return (
    <div className={styles.trainingPath}>
      <span className={styles.label}>Training path</span>
      <code className={styles.path}>{path ?? "Unknown"}</code>
      {available && path ? (
        <button
          type="button"
          className={styles.button}
          aria-label={`Copy training path for ${label}`}
          onClick={() => void copy()}
        >
          Copy training path
        </button>
      ) : (
        <span className={styles.warning}>Path unavailable · cannot copy</span>
      )}
      {message && <span role="status">{message}</span>}
      {error && (
        <span className={styles.error} role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
function ExportCard({ record }: { record: MonitorExport }) {
  return (
    <article
      className={styles.artifact}
      aria-label={`Recorded export ${record.id}`}
    >
      <div className={styles.artifactHeader}>
        <strong>{format(record.format)}</strong>
        <span className={styles.badge}>Recorded export</span>
      </div>
      <p>Instruction mode: {record.instruction_mode ?? "Unknown"}</p>
      <p>
        {count(record.frames)} frames · {duration(record.seconds)}
      </p>
      <p className={styles.muted}>Run: {record.run_id ?? "Unknown"}</p>
      <TrainingPath
        path={record.path}
        available={record.available}
        label={`export ${record.id}`}
      />
    </article>
  );
}
const remoteLabels: Record<RemoteCheck["status"], string> = {
  match: "Matches recorded commit",
  changed: "Branch advanced/changed",
  missing: "Missing revision",
  access_denied: "Access denied",
  unavailable: "Unavailable",
};
function PublicationCard({
  publication,
  exports,
  onCheck,
}: {
  publication: MonitorPublication;
  exports: MonitorExport[];
  onCheck: DatasetMonitorRowProps["onCheckPublication"];
}) {
  const [checking, setChecking] = useState(false);
  const [checked, setChecked] = useState<RemoteCheck | null>(null);
  const [error, setError] = useState<string | null>(null);
  const remote =
    checked &&
    (!publication.remote_check ||
      checked.checked_at >= publication.remote_check.checked_at)
      ? checked
      : publication.remote_check;
  const link = hfUrl(publication.url);
  const linkedExport = exports.find(
    (record) =>
      record.manifest_sha256 &&
      record.manifest_sha256 === publication.manifest_sha256 &&
      record.path === publication.export_path,
  );
  async function check() {
    setChecking(true);
    setError(null);
    try {
      setChecked(await onCheck(publication.id));
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "HF verification failed. Try again.",
      );
    } finally {
      setChecking(false);
    }
  }
  return (
    <article
      className={styles.artifact}
      aria-label={`Recorded publication ${publication.repo_id} ${publication.revision ?? "unknown revision"}`}
    >
      <div className={styles.artifactHeader}>
        <strong className={styles.path}>{publication.repo_id}</strong>
        <span className={styles.badge}>Recorded publication</span>
      </div>
      <p>
        Revision: <code>{publication.revision ?? "Unknown"}</code>
      </p>
      <p className={styles.path}>
        Recorded commit: <code>{publication.commit ?? "Unknown"}</code>
      </p>
      <p>
        {format(publication.format)} · Instruction mode:{" "}
        {publication.instruction_mode ?? "Unknown"}
      </p>
      <p>
        {count(publication.exported_frames)} frames ·{" "}
        {duration(linkedExport?.seconds)}
      </p>
      <p className={styles.muted}>
        Recorded run: {publication.linked_run_id ?? "Unknown"}
      </p>
      <TrainingPath
        path={publication.export_path}
        available={publication.export_available}
        label={`publication ${publication.id}`}
      />
      <div className={styles.actions}>
        {link ? (
          <a
            className={styles.button}
            href={link}
            target="_blank"
            rel="noreferrer"
          >
            Open HF · {publication.revision ?? "recorded version"}
          </a>
        ) : (
          <span className={styles.muted}>HF link unavailable</span>
        )}
        <button
          className={styles.button}
          type="button"
          disabled={checking}
          onClick={() => void check()}
        >
          {checking ? "Checking HF…" : "Check HF"}
        </button>
      </div>
      <div className={styles.remoteCheck} role="status">
        {checking ? (
          "Checking recorded HF revision…"
        ) : remote ? (
          <>
            <strong>{remoteLabels[remote.status]}</strong>
            <span>Last checked: {remote.checked_at}</span>
            {remote.current_commit && (
              <span className={styles.path}>
                Current head: <code>{remote.current_commit}</code>
              </span>
            )}
            {remote.message && <span>{remote.message}</span>}
          </>
        ) : (
          "HF: Not checked"
        )}
      </div>
      {error && (
        <p role="alert" className={styles.error}>
          {error}
        </p>
      )}
    </article>
  );
}
function PromptDistribution({ prompts }: { prompts: MonitorPrompts }) {
  const buckets = [
    { label: "Unlabeled (bucket)", frames: prompts.unlabeled_frames },
    { label: "Ambiguous (bucket)", frames: prompts.ambiguous_frames },
  ];
  return (
    <section className={styles.section}>
      <h3>Reviewed-frame prompts</h3>
      <p className={styles.muted}>
        Current reviewed, non-deleted episodes, including Pending decisions.
        Excluded frames are removed.
      </p>
      <p>
        {prompts.evaluated_episodes} of {prompts.eligible_episodes} eligible
        episodes evaluated ·{" "}
        <strong className={prompts.complete ? styles.success : styles.warning}>
          {prompts.complete ? "Complete coverage" : "Incomplete coverage"}
        </strong>
      </p>
      <p>Unknown episode IDs: {ids(prompts.unknown_episode_ids)}</p>
      <p className={styles.muted}>
        Denominator: {count(prompts.retained_frames)} retained frames from
        evaluated episodes. Episode counts can overlap.
      </p>
      {prompts.retained_frames === 0 && (
        <p>No retained frames available for a distribution.</p>
      )}
      <table className={styles.promptTable} aria-label="Prompt distribution">
        <thead>
          <tr>
            <th scope="col">Literal prompt / bucket</th>
            <th scope="col">Frames</th>
            <th scope="col">Frame share</th>
            <th scope="col">Duration</th>
            <th scope="col">Episodes</th>
          </tr>
        </thead>
        <tbody>
          {prompts.rows.map((row, index) => (
            <tr key={`${row.text}:${index}`}>
              <th scope="row" data-label="Literal prompt">
                <code className={styles.literal}>{row.text}</code>
                {/(^\s|\s$|\t|\r|\n| {2})/.test(row.text) && (
                  <span className={styles.whitespace}>
                    <span className={styles.muted}>Whitespace: </span>
                    <code>
                      {row.text
                        .replace(/ /g, "·")
                        .replace(/\t/g, "⇥")
                        .replace(/\r/g, "␍")
                        .replace(/\n/g, "↵")}
                    </code>
                  </span>
                )}
              </th>
              <td data-label="Frames">{count(row.frames)}</td>
              <td data-label="Frame share">
                {percent(row.ratio)}
                <span className={styles.frameTrack} aria-hidden="true">
                  <span
                    style={{
                      width: `${Math.min(100, Math.max(0, (row.ratio ?? 0) * 100))}%`,
                    }}
                  />
                </span>
              </td>
              <td data-label="Duration">{duration(row.seconds)}</td>
              <td data-label="Episodes">{count(row.episodes)}</td>
            </tr>
          ))}
          {buckets.map((bucket) => {
            const ratio = prompts.retained_frames
              ? bucket.frames / prompts.retained_frames
              : null;
            return (
              <tr key={bucket.label} className={styles.bucket}>
                <th scope="row" data-label="Coverage bucket">
                  {bucket.label}
                </th>
                <td data-label="Frames">{count(bucket.frames)}</td>
                <td data-label="Frame share">
                  {percent(ratio)}
                  <span className={styles.frameTrack} aria-hidden="true">
                    <span style={{ width: `${(ratio ?? 0) * 100}%` }} />
                  </span>
                </td>
                <td data-label="Duration">—</td>
                <td data-label="Episodes">—</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <Diagnostics items={prompts.diagnostics ?? []} />
    </section>
  );
}

export function DatasetMonitorRow({
  dataset,
  selectedRunId,
  detail,
  detailStale = false,
  loading,
  error,
  onSelectRun,
  onExpand,
  onCheckPublication,
}: DatasetMonitorRowProps) {
  const selectId = useId();
  const run = dataset.runs.find(
    (candidate) => candidate.run_id === selectedRunId,
  );
  const currentDetail =
    detail?.dataset_id === dataset.id &&
    detail.run_id === (run?.run_id ?? null) &&
    (detail.updating ||
      detailStale ||
      detail.signature === (run?.detail_signature ?? null))
      ? detail
      : null;
  const metrics = run?.metrics;
  const unpublishedChanges =
    run?.freshness.local_changes === true ||
    run?.freshness.publication_state === "unpublished_changes";
  const review = reviewUrl(run);
  const publications = mergeRecords(
    dataset.publications,
    currentDetail?.publications ?? [],
  );
  const exports = mergeRecords(dataset.exports, currentDetail?.exports ?? []);
  const diagnostics = [
    ...dataset.diagnostics,
    ...(run?.diagnostics ?? []),
    ...(currentDetail?.diagnostics ?? []),
  ];
  return (
    <article
      className={styles.datasetRow}
      id={`dataset-${dataset.id}`}
      aria-label={dataset.name}
    >
      <div className={styles.rowHeader}>
        <div className={styles.folderIdentity}>
          <div className={styles.folderTitle}>
            <h2>{dataset.name}</h2>
            <span
              className={
                dataset.state === "Ready" ? styles.badge : styles.warningBadge
              }
            >
              {dataset.state}
            </span>
          </div>
          <p className={styles.scope}>
            {run
              ? `Selected run: ${run.run_id}`
              : dataset.runs.length
                ? "Select an annotation run"
                : "Not imported"}
          </p>
          {run && (
            <p className={styles.activity}>
              Workflow activity: {run.updated_at ?? "Unknown"}
            </p>
          )}
        </div>
        <div className={styles.actions}>
          {review ? (
            <a href={review} className={styles.primaryButton}>
              Open review
            </a>
          ) : !dataset.runs.length ? (
            <a
              className={styles.primaryButton}
              href={`/annotate?local_path=${encodeURIComponent(dataset.path)}`}
            >
              Prepare dataset
            </a>
          ) : (
            <span className={styles.muted}>
              {!run?.current_repo_id
                ? "Review alias unavailable. Open the existing annotation workflow to restore its registered alias."
                : "No retained episode available to open."}
            </span>
          )}
        </div>
      </div>
      <dl className={styles.metrics}>
        <div>
          <dt>Collected</dt>
          <dd>{count(dataset.collected)}</dd>
        </div>
        <div>
          <dt>Imported</dt>
          <dd>{metrics ? count(metrics.imported) : "Not imported"}</dd>
        </div>
        <div>
          <dt>Accepted</dt>
          <dd className={styles.success}>
            {metrics ? count(metrics.accepted) : "—"}
          </dd>
        </div>
        <div>
          <dt>Rejected</dt>
          <dd>{metrics ? count(metrics.rejected) : "—"}</dd>
        </div>
        <div>
          <dt>Pending</dt>
          <dd className={metrics?.pending ? styles.warning : undefined}>
            {metrics ? count(metrics.pending) : "—"}
          </dd>
        </div>
        <div>
          <dt>Usable duration</dt>
          <dd>{metrics ? duration(metrics.accepted_seconds) : "—"}</dd>
        </div>
      </dl>
      <div className={styles.progressGrid}>
        <div>
          <span className={styles.label}>Review progress</span>
          <strong>
            {!metrics
              ? "Not imported"
              : metrics.retained === 0
                ? "— · No retained episodes"
                : metrics.retained == null || metrics.reviewed == null
                  ? "— · Unknown review coverage"
                  : `${count(metrics.reviewed)} / ${count(metrics.retained)} reviewed`}
          </strong>
          {metrics && metrics.retained !== 0 && (
            <span className={styles.muted}>
              {percent(metrics.review_rate)} of retained episodes
            </span>
          )}
          {metrics?.review_rate != null && metrics.retained !== 0 && (
            <progress
              className={styles.progress}
              value={metrics.review_rate}
              max={1}
              aria-label="Review progress"
            />
          )}
        </div>
        <div>
          <span className={styles.label}>Decision completion</span>
          <strong>
            {!metrics
              ? "Not imported"
              : metrics.imported === 0
                ? "— · No episodes"
                : metrics.imported == null
                  ? "— · Unknown imported count"
                  : percent(metrics.decision_rate)}
          </strong>
          <span className={styles.muted}>Accepted + Rejected / Imported</span>
        </div>
        <div>
          <span className={styles.label}>Not imported</span>
          <strong>
            {metrics
              ? count(metrics.new_episode_ids?.length)
              : count(dataset.collected)}
          </strong>
          <span className={styles.muted}>New source episodes</span>
        </div>
        <div>
          <span className={styles.label}>HF publications</span>
          <strong>
            {publications.length} recorded{" "}
            {publications.length === 1 ? "publication" : "publications"}
          </strong>
          <span className={styles.muted}>
            {unpublishedChanges ? (
              <span className={styles.warning}>Unpublished changes</span>
            ) : run?.freshness.local_changes === false &&
              run.freshness.verifiable ? (
              "No local changes since linked export"
            ) : (
              "Publication freshness unverified"
            )}
          </span>
          {unpublishedChanges && !run?.freshness.verifiable && (
            <span className={styles.muted}>Digest comparison unverified</span>
          )}
        </div>
      </div>
      <div className={styles.attention}>
        {run && (
          <>
            <span
              className={
                run.findings.unresolved ? styles.warning : styles.muted
              }
            >
              Unresolved findings: {count(run.findings.unresolved)}
            </span>
            <span className={styles.muted}>
              Accepted advisory findings:{" "}
              {count(run.findings.accepted_advisory)}
            </span>
            {run.findings.generation_failed > 0 && (
              <span className={styles.warning}>
                Generation failed: {count(run.findings.generation_failed)}
              </span>
            )}
            {run.findings.unreadable > 0 && (
              <span className={styles.warning}>
                Unreadable episodes: {count(run.findings.unreadable)}
              </span>
            )}
            {run.freshness.source_changed && (
              <span className={styles.warning}>Source changed</span>
            )}
            {!metrics?.counts_complete && (
              <span className={styles.warning}>
                Incomplete counts · inspect diagnostics
              </span>
            )}
          </>
        )}
        {dataset.diagnostics.length > 0 && (
          <span className={styles.warning}>
            {dataset.diagnostics.length} folder{" "}
            {dataset.diagnostics.length === 1 ? "diagnostic" : "diagnostics"}
          </span>
        )}
      </div>
      <details
        className={styles.details}
        onToggle={(event) => onExpand(event.currentTarget.open)}
      >
        <summary>Runs, prompts &amp; recorded artifacts</summary>
        <div className={styles.detailContent}>
          {dataset.runs.length > 0 && (
            <div className={styles.runSelect}>
              <label htmlFor={selectId}>
                Annotation run for {dataset.name}
              </label>
              <select
                id={selectId}
                value={selectedRunId ?? ""}
                onChange={(event) => onSelectRun(event.target.value)}
              >
                {!run && (
                  <option value="" disabled>
                    Select a run
                  </option>
                )}
                {dataset.runs.map((candidate) => (
                  <option key={candidate.run_id} value={candidate.run_id}>
                    {candidate.run_id} ·{" "}
                    {candidate.updated_at ?? "Unknown activity"}
                  </option>
                ))}
              </select>
            </div>
          )}
          {loading && (
            <p role="status" className={styles.muted}>
              Loading details…
            </p>
          )}
          {error && (
            <p role="alert" className={styles.error}>
              {error}
            </p>
          )}
          {currentDetail?.updating && (
            <p role="status" className={styles.warning}>
              {currentDetail.signature === null ? (
                "Updating · no consistent detail snapshot is available yet."
              ) : (
                <>
                  Updating · showing the last consistent detail snapshot from{" "}
                  {currentDetail.scanned_at ?? "an unknown time"}.
                </>
              )}
            </p>
          )}
          {currentDetail && detailStale && !currentDetail.updating && (
            <p role="status" className={styles.warning}>
              Stale details · showing the last consistent detail snapshot from{" "}
              {currentDetail.scanned_at ?? "an unknown time"}.
            </p>
          )}
          {currentDetail?.prompts ? (
            <PromptDistribution prompts={currentDetail.prompts} />
          ) : run && !loading && !error ? (
            <p className={styles.muted}>
              Prompt details are not loaded for this run.
            </p>
          ) : null}
          {run && (
            <section className={styles.section}>
              <h3>Run evidence</h3>
              <p>
                Exclusions: {count(run.exclusions.episodes)} episodes ·{" "}
                {count(run.exclusions.frames)} frames removed from retained
                episodes {review && <a href={review}>Open exclusion editor</a>}
              </p>
              <p>Missing source IDs: {ids(run.metrics.missing_episode_ids)}</p>
              <p>Changed length IDs: {ids(run.metrics.changed_length_ids)}</p>
              <p>Not imported source IDs: {ids(run.metrics.new_episode_ids)}</p>
              <p>
                Current workflow job: {run.job?.status ?? "No recorded job"}
                {run.job?.error ? ` · ${run.job.error}` : ""}
              </p>
              <p className={styles.muted}>
                Metadata check only. Source byte integrity is not verified;
                export performs full validation.
              </p>
            </section>
          )}
          <section className={styles.section}>
            <h3>Recorded exports</h3>
            <p className={styles.muted}>
              Frozen artifacts across all runs. An export does not imply
              publication.
            </p>
            {exports.length ? (
              <div className={styles.artifactGrid}>
                {exports.map((record) => (
                  <ExportCard key={record.id} record={record} />
                ))}
              </div>
            ) : (
              <p>No recorded exports.</p>
            )}
          </section>
          <section className={styles.section}>
            <h3>Recorded publications</h3>
            <p className={styles.muted}>
              Receipts across all runs remain visible after local edits. Check
              HF verifies the recorded revision&apos;s current commit.
            </p>
            {publications.length ? (
              <div className={styles.artifactGrid}>
                {publications.map((publication) => (
                  <PublicationCard
                    key={publication.id}
                    publication={publication}
                    exports={exports}
                    onCheck={onCheckPublication}
                  />
                ))}
              </div>
            ) : (
              <p>No recorded publications.</p>
            )}
          </section>
          <section className={styles.section}>
            <h3>Folder &amp; provenance</h3>
            <p className={styles.path}>
              Collection path: <code>{dataset.path}</code>
            </p>
            <p className={styles.path}>
              Canonical path: <code>{dataset.canonical_path}</code>
            </p>
            <p>Provenance: {dataset.provenance.status}</p>
            {dataset.provenance.sources.map((source) => (
              <p className={styles.path} key={source}>
                Recorded source: <code>{source}</code>
              </p>
            ))}
            {dataset.child_ids.length > 0 && (
              <p>
                Known derivatives:{" "}
                {dataset.child_ids.map((id, index) => (
                  <React.Fragment key={id}>
                    {index > 0 && ", "}
                    <a href={`#dataset-${id}`}>{id}</a>
                  </React.Fragment>
                ))}
              </p>
            )}
            {dataset.parent_ids.length > 0 && (
              <p>
                Confirmed parents:{" "}
                {dataset.parent_ids.map((id, index) => (
                  <React.Fragment key={id}>
                    {index > 0 && ", "}
                    <a href={`#dataset-${id}`}>{id}</a>
                  </React.Fragment>
                ))}
              </p>
            )}
            {dataset.reported_collected !== dataset.collected && (
              <p className={styles.warning}>
                Metadata reports {count(dataset.reported_collected)} episodes;{" "}
                {count(dataset.collected)} completed episode records are
                readable.
              </p>
            )}
          </section>
          <Diagnostics items={diagnostics} />
        </div>
      </details>
    </article>
  );
}
