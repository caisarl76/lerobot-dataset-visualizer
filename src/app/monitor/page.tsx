"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { DatasetMonitorRow } from "@/components/dataset-monitor-row";
import {
  fetchMonitorSummary,
  fetchMonitorDetail,
  checkMonitorPublication,
  type MonitorDataset,
  type MonitorSummary,
  type MonitorDetail,
  type MonitorRun,
  type RemoteCheck,
} from "@/utils/monitorClient";
import styles from "./page.module.css";

const isVisible = () => document.visibilityState !== "hidden";
const STORAGE_KEY = "lerobot-monitor:selected-runs:v1";
type Selections = Record<string, string>;
function selectedRun(dataset: MonitorDataset, selections: Selections) {
  return (
    dataset.runs.find((run) => run.run_id === selections[dataset.id]) ??
    dataset.runs.find((run) => run.run_id === dataset.default_run_id) ??
    dataset.runs[0]
  );
}
function awaitingReview(run?: MonitorRun) {
  return (
    !!run &&
    ((run.metrics.pending ?? 0) > 0 ||
      (run.metrics.retained !== null &&
        run.metrics.reviewed !== null &&
        run.metrics.reviewed < run.metrics.retained))
  );
}
function needsAttention(dataset: MonitorDataset, run?: MonitorRun) {
  return (
    !run ||
    awaitingReview(run) ||
    dataset.diagnostics.some((d) => d.severity !== "info") ||
    (!!run &&
      (run.findings.unresolved > 0 ||
        run.findings.generation_failed > 0 ||
        run.findings.unreadable > 0 ||
        run.metrics.counts_complete === false ||
        (run.metrics.new_episode_ids?.length ?? 0) > 0 ||
        (run.metrics.missing_episode_ids?.length ?? 0) > 0 ||
        (run.metrics.changed_length_ids?.length ?? 0) > 0 ||
        run.freshness.local_changes === true ||
        run.freshness.publication_state === "unpublished_changes" ||
        run.diagnostics.some((d) => d.severity !== "info")))
  );
}
// Only confirmed, acyclic, single-parent edges determine placement.
function parentsOf(datasets: MonitorDataset[]) {
  const ids = new Set(datasets.map((d) => d.id));
  const candidates = new Map(
    datasets
      .filter(
        (d) =>
          d.provenance.status === "confirmed" &&
          d.parent_ids.length === 1 &&
          ids.has(d.parent_ids[0]),
      )
      .map((d) => [d.id, d.parent_ids[0]]),
  );
  const parents = new Map<string, string>();
  for (const [id, parent] of candidates) {
    const visited = new Set([id]);
    let cursor: string | undefined = parent;
    while (cursor && !visited.has(cursor)) {
      visited.add(cursor);
      cursor = candidates.get(cursor);
    }
    if (!cursor) parents.set(id, parent);
  }
  return parents;
}

function DatasetEntry({
  dataset,
  run,
  onSelectRun,
  onCheckPublication,
  refreshVersion,
  summaryVersion,
}: {
  dataset: MonitorDataset;
  run?: MonitorRun;
  onSelectRun: (id: string) => void;
  onCheckPublication: (id: string) => Promise<RemoteCheck>;
  refreshVersion: number;
  summaryVersion: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState<MonitorDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryVersion, setRetryVersion] = useState(0);
  const needsRetry = useRef(false);
  const runId = run?.run_id;
  const signature = run?.detail_signature;
  useEffect(() => {
    setDetail((previous) =>
      expanded &&
      previous?.dataset_id === dataset.id &&
      previous.run_id === runId
        ? previous
        : null,
    );
    setError(null);
    setLoading(false);
    needsRetry.current = false;
    if (!expanded || !runId) return;
    const controller = new AbortController();
    setLoading(true);
    void fetchMonitorDetail(dataset.id, runId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        if (
          result.dataset_id !== dataset.id ||
          result.run_id !== runId ||
          (!result.updating && result.signature !== signature)
        ) {
          needsRetry.current = true;
          setError("Detail snapshot changed; waiting for the next refresh.");
          return;
        }
        needsRetry.current = result.updating;
        setDetail((previous) =>
          result.updating &&
          result.signature === null &&
          previous?.dataset_id === dataset.id &&
          previous.run_id === runId &&
          previous.signature !== null
            ? {
                ...previous,
                updating: true,
                diagnostics: [
                  ...previous.diagnostics.filter(
                    (old) =>
                      !result.diagnostics.some(
                        (item) => item.code === old.code,
                      ),
                  ),
                  ...result.diagnostics,
                ],
              }
            : result,
        );
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          needsRetry.current = true;
          setError(
            reason instanceof Error
              ? reason.message
              : "Could not load details.",
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [dataset.id, runId, signature, expanded, refreshVersion, retryVersion]);
  useEffect(() => {
    // A stale/failed detail gets one retry after a later successful summary.
    // Detail completion never triggers this effect, so Updating cannot spin.
    if (needsRetry.current && isVisible()) {
      needsRetry.current = false;
      setRetryVersion((value) => value + 1);
    }
  }, [summaryVersion]);
  return (
    <DatasetMonitorRow
      dataset={dataset}
      selectedRunId={runId ?? null}
      detail={detail}
      detailStale={
        !!detail && (loading || !!error || detail.signature !== signature)
      }
      loading={loading}
      error={error}
      onSelectRun={onSelectRun}
      onExpand={setExpanded}
      onCheckPublication={onCheckPublication}
    />
  );
}

export default function MonitorPage() {
  const [snapshot, setSnapshot] = useState<MonitorSummary | null>(null);
  const [selections, setSelections] = useState<Selections>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [summaryVersion, setSummaryVersion] = useState(0);
  const refresh = useRef<(manual?: boolean) => void>(() => {});
  const checks = useRef(new Map<string, RemoteCheck>());
  const checkControllers = useRef(new Set<AbortController>());

  useEffect(() => {
    try {
      const saved: unknown = JSON.parse(
        window.localStorage.getItem(STORAGE_KEY) ?? "{}",
      );
      if (saved && typeof saved === "object" && !Array.isArray(saved))
        setSelections(
          Object.fromEntries(
            Object.entries(saved).filter(
              ([, value]) => typeof value === "string",
            ),
          ),
        );
    } catch {
      /* Storage can be disabled; in-memory selection still works. */
    }
    const activeChecks = checkControllers.current;
    let disposed = false;
    let active: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let queued = false;
    const clearTimer = () => {
      clearTimeout(timer);
      timer = undefined;
    };
    async function load(manual = false) {
      clearTimer();
      if (disposed || !isVisible()) return;
      if (active) {
        queued = true;
        return;
      }
      const controller = new AbortController();
      active = controller;
      setLoading(true);
      setError(null);
      try {
        const result = await fetchMonitorSummary(manual, controller.signal);
        if (disposed || controller.signal.aborted) return;
        // Checks completed since this request began must outlive older snapshots.
        const publicationIds = new Set(
          result.datasets.flatMap((d) => d.publications.map((p) => p.id)),
        );
        for (const id of checks.current.keys())
          if (!publicationIds.has(id)) checks.current.delete(id);
        const datasets = result.datasets.map((dataset) => ({
          ...dataset,
          publications: dataset.publications.map((publication) => {
            const checked = checks.current.get(publication.id);
            return checked &&
              (!publication.remote_check ||
                checked.checked_at >= publication.remote_check.checked_at)
              ? { ...publication, remote_check: checked }
              : publication;
          }),
        }));
        setSnapshot({ ...result, datasets });
        setSummaryVersion((value) => value + 1);
        setSelections((previous) => {
          const next = Object.fromEntries(
            datasets.flatMap((dataset) => {
              const run = selectedRun(dataset, previous);
              return run ? [[dataset.id, run.run_id]] : [];
            }),
          );
          try {
            window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
          } catch {
            /* Optional preference. */
          }
          return next;
        });
        if (manual) setRefreshVersion((value) => value + 1);
      } catch (reason) {
        if (!disposed && !controller.signal.aborted)
          setError(
            reason instanceof Error
              ? reason.message
              : "Monitor backend is unavailable.",
          );
      } finally {
        active = null;
        if (!disposed) {
          setLoading(false);
          if (isVisible()) {
            if (queued) {
              queued = false;
              void load();
            } else timer = setTimeout(() => void load(), 30000);
          }
        }
      }
    }
    refresh.current = (manual) => {
      void load(manual);
    };
    const visibility = () => {
      clearTimer();
      if (isVisible()) void load();
      else queued = false;
    };
    document.addEventListener("visibilitychange", visibility);
    void load();
    return () => {
      disposed = true;
      active?.abort();
      clearTimer();
      for (const controller of activeChecks) controller.abort();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);

  async function checkPublication(id: string) {
    const controller = new AbortController();
    checkControllers.current.add(controller);
    try {
      const checked = await checkMonitorPublication(id, controller.signal);
      if (!controller.signal.aborted) {
        checks.current.set(id, checked);
        setSnapshot(
          (previous) =>
            previous && {
              ...previous,
              datasets: previous.datasets.map((dataset) => ({
                ...dataset,
                publications: dataset.publications.map((publication) =>
                  publication.id === id
                    ? { ...publication, remote_check: checked }
                    : publication,
                ),
              })),
            },
        );
      }
      return checked;
    } finally {
      checkControllers.current.delete(controller);
    }
  }
  function selectRun(datasetId: string, runId: string) {
    setSelections((previous) => {
      const next = { ...previous, [datasetId]: runId };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* Optional preference. */
      }
      return next;
    });
  }
  // Source links remain usable even when their target is outside the current filter.
  useEffect(() => {
    function showAnchor() {
      if (window.location.hash.startsWith("#dataset-")) {
        setSearch("");
        setFilter("all");
      }
    }
    window.addEventListener("hashchange", showAnchor);
    showAnchor();
    return () => window.removeEventListener("hashchange", showAnchor);
  }, []);
  useEffect(() => {
    if (
      !search &&
      filter === "all" &&
      window.location.hash.startsWith("#dataset-")
    ) {
      document
        .getElementById(window.location.hash.slice(1))
        ?.scrollIntoView?.({ block: "start" });
    }
  }, [search, filter, snapshot]);

  const datasets = snapshot?.datasets ?? [];
  const parents = parentsOf(datasets);
  const children = (id: string) =>
    datasets.filter((dataset) => parents.get(dataset.id) === id);
  const roots = datasets.filter((dataset) => !parents.has(dataset.id));
  const members = (dataset: MonitorDataset): MonitorDataset[] => [
    dataset,
    ...children(dataset.id).flatMap(members),
  ];
  const groups = roots.map(members);
  const groupCounts = [
    groups.length,
    groups.filter((group) =>
      group.some((dataset) => awaitingReview(selectedRun(dataset, selections))),
    ).length,
    groups.filter((group) =>
      group.some(
        (dataset) =>
          (selectedRun(dataset, selections)?.metrics.new_episode_ids?.length ??
            0) > 0,
      ),
    ).length,
    groups.filter((group) =>
      group.some((dataset) => dataset.publications.length > 0),
    ).length,
  ];
  function matches(dataset: MonitorDataset) {
    const query = search.trim().toLocaleLowerCase();
    if (
      query &&
      ![dataset.name, ...dataset.publications.map((p) => p.repo_id)].some(
        (text) => text.toLocaleLowerCase().includes(query),
      )
    )
      return false;
    const run = selectedRun(dataset, selections);
    return (
      filter === "all" ||
      (filter === "attention" && needsAttention(dataset, run)) ||
      (filter === "unimported" && !run) ||
      (filter === "reviewing" && awaitingReview(run)) ||
      (filter === "published" && dataset.publications.length > 0)
    );
  }
  const shown = new Set(datasets.filter(matches).map((dataset) => dataset.id));
  const shownRoots = datasets.filter(
    (dataset) =>
      shown.has(dataset.id) && !shown.has(parents.get(dataset.id) ?? ""),
  );
  function renderDataset(dataset: MonitorDataset): React.ReactNode {
    const descendants = children(dataset.id).filter((child) =>
      shown.has(child.id),
    );
    return (
      <React.Fragment key={dataset.id}>
        <DatasetEntry
          dataset={dataset}
          run={selectedRun(dataset, selections)}
          onSelectRun={(id) => selectRun(dataset.id, id)}
          onCheckPublication={checkPublication}
          refreshVersion={refreshVersion}
          summaryVersion={summaryVersion}
        />
        {descendants.length > 0 && (
          <div className={styles.derivatives}>
            {descendants.map(renderDataset)}
          </div>
        )}
      </React.Fragment>
    );
  }
  return (
    <div className={styles.page}>
      <nav className={styles.navigation} aria-label="Monitor navigation">
        <Link href="/">LeRobot Dataset Visualizer</Link>
        <Link href="/annotate">Prepare annotations</Link>
      </nav>
      <main className={styles.content}>
        <header className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Local datasets</p>
            <h1>Dataset monitor</h1>
            <p className={styles.description}>
              Collection, review, and recorded publications in one place. Choose
              a run to inspect its review progress.
            </p>
            {snapshot?.root && (
              <p className={styles.rootPath}>Root: {snapshot.root}</p>
            )}
            <p className={styles.muted}>
              Last successful refresh:{" "}
              {snapshot?.scanned_at ?? "Not yet available"}
            </p>
          </div>
          <button
            className={styles.primaryButton}
            disabled={loading}
            onClick={() => refresh.current(true)}
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </header>
        {loading && (
          <p className={styles.status} role="status">
            Refreshing dataset summary…
          </p>
        )}
        {error && (
          <p className={styles.error} role="alert">
            {error}
            {snapshot
              ? " Showing the last successful snapshot."
              : " Check the annotation backend connection (NEXT_PUBLIC_ANNOTATE_BACKEND_URL)."}
          </p>
        )}
        {snapshot?.updating && (
          <p className={styles.warning} role="status">
            Updating: showing the last consistent snapshot and its timestamp.
          </p>
        )}
        {snapshot?.diagnostics.map((diagnostic, index) => (
          <p
            key={`${diagnostic.code}-${index}`}
            className={
              diagnostic.severity === "error" ? styles.error : styles.warning
            }
          >
            {diagnostic.severity}: {diagnostic.message}
          </p>
        ))}
        {snapshot && !snapshot.configured ? (
          <div className={styles.emptyState}>
            No monitor root configured. Set LEROBOT_MONITOR_ROOT on the
            annotation backend to the local collection directory.
          </div>
        ) : (
          snapshot && (
            <>
              <section
                className={styles.summaryCards}
                aria-label="Collection group counts"
              >
                {[
                  "Collection groups",
                  "Groups awaiting review",
                  "Groups with new episodes",
                  "Groups with recorded publications",
                ].map((label, index) => (
                  <div
                    className={styles.summaryCard}
                    key={label}
                    aria-label={label}
                  >
                    <strong>{groupCounts[index]}</strong>
                    <span>{label}</span>
                  </div>
                ))}
              </section>
              <p className={styles.muted}>
                Group counts overlap. Source and derived folders are grouped
                where provenance is confirmed; these are not counts of unique
                demonstrations.
              </p>
              <div className={styles.toolbar}>
                <label className={styles.searchControl}>
                  Search datasets
                  <input
                    aria-label="Search datasets"
                    type="search"
                    placeholder="Folder name or publication repository"
                    value={search}
                    onInput={(event) => setSearch(event.currentTarget.value)}
                  />
                </label>
                <label className={styles.filter}>
                  Status filter
                  <select
                    aria-label="Status filter"
                    value={filter}
                    onChange={(event) => setFilter(event.target.value)}
                  >
                    {[
                      ["all", "All datasets"],
                      ["attention", "Needs attention"],
                      ["unimported", "Unimported"],
                      ["reviewing", "Reviewing"],
                      ["published", "Published"],
                    ].map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {datasets.length === 0 ? (
                <p className={styles.emptyState}>
                  No datasets found in the configured root.
                </p>
              ) : shownRoots.length === 0 ? (
                <p className={styles.emptyState}>
                  No datasets match these filters.
                </p>
              ) : (
                <div
                  className={styles.datasetList}
                  onClick={(event) => {
                    const anchor = (
                      event.target as HTMLElement
                    ).closest<HTMLAnchorElement>('a[href^="#dataset-"]');
                    if (anchor) {
                      setSearch("");
                      setFilter("all");
                    }
                  }}
                >
                  {shownRoots.map((dataset) => (
                    <div
                      className={styles.group}
                      role="group"
                      aria-label={`Collection group ${dataset.name}`}
                      key={dataset.id}
                    >
                      {renderDataset(dataset)}
                    </div>
                  ))}
                </div>
              )}
            </>
          )
        )}
      </main>
    </div>
  );
}
