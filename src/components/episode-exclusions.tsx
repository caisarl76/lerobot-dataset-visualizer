"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAnnotations } from "../context/annotations-context";
import { useTime } from "../context/time-context";
import {
  fetchWorkflow,
  getAnnotationJob,
  postWorkflowExclusions,
  type WorkflowRun,
} from "../utils/annotationsClient";

type Interval = { start_frame: number; end_frame: number };

function mergeIntervals(intervals: Interval[]): Interval[] {
  const merged: Interval[] = [];
  for (const interval of [...intervals].sort(
    (a, b) => a.start_frame - b.start_frame,
  )) {
    const previous = merged[merged.length - 1];
    if (previous && interval.start_frame <= previous.end_frame)
      previous.end_frame = Math.max(previous.end_frame, interval.end_frame);
    else merged.push({ ...interval });
  }
  return merged;
}

function nearestBoundary(boundaries: number[], time: number): number {
  let low = 0;
  let high = boundaries.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (boundaries[middle] < time) low = middle + 1;
    else high = middle;
  }
  return low > 0 && time - boundaries[low - 1] <= boundaries[low] - time
    ? low - 1
    : low;
}

export const EpisodeExclusions: React.FC<{ duration: number }> = ({
  duration,
}) => {
  const { ident, episodeId, frameTimestamps, dirty, saving } = useAnnotations();
  const { currentTime, seek, subscribe, setIsPlaying, isPlaying } = useTime();
  const alias = ident.repoId?.startsWith("local/")
    ? ident.repoId.slice(6)
    : null;
  const scope = JSON.stringify([alias, episodeId, ident.revision]);
  const scopeRef = useRef(scope);
  scopeRef.current = scope;
  const mounted = useRef(true);
  const operation = useRef(0);
  const [loaded, setLoaded] = useState<{
    scope: string;
    run: WorkflowRun;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [jobStatus, setJobStatus] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [error, setError] = useState("");
  const [start, setStart] = useState("0");
  const [end, setEnd] = useState("0");
  const [selected, setSelected] = useState<number | null>(null);
  const [preview, setPreview] = useState(false);
  const drag = useRef<{
    pointerId: number;
    anchor: number;
    start: string;
    end: string;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const run = loaded?.scope === scope ? loaded.run : null;
  const intervals = useMemo(
    () =>
      mergeIntervals(
        run?.episodes[String(episodeId)]?.excluded_intervals ?? [],
      ),
    [run, episodeId],
  );
  const frameCount = frameTimestamps.length;
  const step =
    frameCount > 1
      ? (frameTimestamps[frameCount - 1] - frameTimestamps[0]) /
        (frameCount - 1)
      : duration;
  // The last timestamp labels the last frame, not the exclusive episode end.
  const boundaries = useMemo(
    () =>
      frameCount
        ? [...frameTimestamps, frameTimestamps[frameCount - 1] + step]
        : [],
    [frameTimestamps, frameCount, step],
  );
  const endTime = boundaries[frameCount] ?? duration;
  const activeJob = jobStatus === "queued" || jobStatus === "running";
  const canEdit =
    !!alias &&
    episodeId !== null &&
    !!run &&
    !!run.episodes[String(episodeId)] &&
    frameCount > 0 &&
    !loading &&
    !loadFailed &&
    !busy &&
    !dirty &&
    !saving &&
    !activeJob &&
    run.episodes[String(episodeId)].decision !== "delete";

  useEffect(() => {
    const pendingOperation = operation;
    mounted.current = true;
    return () => {
      mounted.current = false;
      pendingOperation.current++;
    };
  }, []);

  useEffect(() => {
    operation.current++;
    setBusy(false);
    setLoaded(null);
    setLoading(true);
    setLoadFailed(false);
    setJobStatus("");
    setError("");
    setStart("0");
    setEnd("0");
    setSelected(null);
    setPreview(false);
    drag.current = null;
    setDragging(false);
  }, [scope]);

  useEffect(() => {
    const reload = () => setRefresh((value) => value + 1);
    window.addEventListener("annotation-workflow-changed", reload);
    return () =>
      window.removeEventListener("annotation-workflow-changed", reload);
  }, []);

  useEffect(() => {
    if (!alias || episodeId === null || busy) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const current = () =>
      !cancelled && mounted.current && scopeRef.current === scope;
    setLoading(true);
    setLoadFailed(false);
    async function load() {
      try {
        let next = await fetchWorkflow(alias!);
        if (!current()) return;
        setLoaded({ scope, run: next });
        if (next.current_job_id) {
          let job = await getAnnotationJob(next.current_job_id);
          if (!current()) return;
          setJobStatus(job.status);
          setLoading(false);
          while (current() && ["queued", "running"].includes(job.status)) {
            await new Promise<void>((resolve) => {
              timer = setTimeout(resolve, 1000);
            });
            if (!current()) return;
            job = await getAnnotationJob(next.current_job_id);
            if (!current()) return;
            setJobStatus(job.status);
          }
          if (!current()) return;
          setLoading(true);
          next = await fetchWorkflow(alias!);
          if (!current()) return;
          setLoaded({ scope, run: next });
        } else setJobStatus("");
      } catch (cause) {
        if (current()) {
          setError(
            cause instanceof Error
              ? cause.message
              : "Unable to load exclusions",
          );
          setLoadFailed(true);
        }
      } finally {
        if (current()) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [alias, episodeId, scope, refresh, busy]);

  const first = Number(start);
  const last = Number(end);
  const validTimes =
    start !== "" &&
    end !== "" &&
    Number.isFinite(first) &&
    Number.isFinite(last) &&
    first >= (boundaries[0] ?? 0) &&
    last <= endTime &&
    first < last;
  const firstFrame = validTimes ? nearestBoundary(boundaries, first) : 0;
  const endFrame = validTimes ? nearestBoundary(boundaries, last) : 0;
  const nextIntervals = mergeIntervals([
    ...intervals,
    { start_frame: firstFrame, end_frame: endFrame },
  ]);
  const allRemoved =
    nextIntervals.reduce(
      (count, interval) => count + interval.end_frame - interval.start_frame,
      0,
    ) >= frameCount;
  const validSelection = validTimes && endFrame > firstFrame && !allRemoved;
  const retainedCount =
    frameCount -
    intervals.reduce(
      (count, interval) => count + interval.end_frame - interval.start_frame,
      0,
    );

  const save = async (next: Interval[]) => {
    if (!canEdit || !alias || episodeId === null || !run) return;
    const token = ++operation.current;
    const current = () =>
      mounted.current &&
      scopeRef.current === scope &&
      operation.current === token;
    setBusy(true);
    setError("");
    try {
      const updated = await postWorkflowExclusions(alias, {
        episode_index: episodeId,
        expected_revision: run.revision,
        excluded_intervals: next,
      });
      if (!current()) return;
      setLoaded({ scope, run: updated });
      setSelected(null);
      window.dispatchEvent(new window.Event("annotation-workflow-changed"));
      window.dispatchEvent(new window.Event("annotation-review-changed"));
      window.dispatchEvent(new window.Event("annotation-clipping-changed"));
    } catch (cause) {
      if (current())
        setError(
          cause instanceof Error ? cause.message : "Unable to save exclusions",
        );
    } finally {
      if (current()) setBusy(false);
    }
  };

  const pointerFrame = (event: React.PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction =
      rect.width > 0
        ? Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
        : 0;
    return nearestBoundary(
      boundaries,
      boundaries[0] + fraction * (endTime - boundaries[0]),
    );
  };
  const updateDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const state = drag.current;
    if (!state || state.pointerId !== event.pointerId || !canEdit) return;
    const frame = pointerFrame(event);
    setStart(String(boundaries[Math.min(state.anchor, frame)]));
    setEnd(String(boundaries[Math.max(state.anchor, frame)]));
  };

  useEffect(() => {
    if (!preview || !isPlaying || !frameCount) return;
    let seeking = false;
    const skipExcluded = (time: number) => {
      if (seeking) return;
      const hit = intervals.find(
        (interval) =>
          time >= boundaries[interval.start_frame] &&
          (time < boundaries[interval.end_frame] ||
            interval.end_frame === frameCount),
      );
      if (!hit) return;
      seeking = true;
      if (hit.end_frame === frameCount) {
        setIsPlaying(false);
        seek(boundaries[hit.start_frame - 1], "external");
      } else seek(boundaries[hit.end_frame], "external");
      seeking = false;
    };
    skipExcluded(currentTime);
    return subscribe(skipExcluded);
  }, [
    preview,
    isPlaying,
    intervals,
    boundaries,
    frameCount,
    currentTime,
    seek,
    subscribe,
    setIsPlaying,
  ]);

  if (!alias || episodeId === null || !frameCount) return null;
  const displayTime = (frame: number) => boundaries[frame].toFixed(3);
  const playheadFraction = Math.max(
    0,
    Math.min(1, (currentTime - boundaries[0]) / (endTime - boundaries[0] || 1)),
  );
  const playheadLabelPosition =
    playheadFraction <= 0.05
      ? "start"
      : playheadFraction >= 0.95
        ? "end"
        : "middle";
  const rangeStyle = (a: number, b: number) => ({
    left: `${((boundaries[a] - boundaries[0]) / (endTime - boundaries[0])) * 100}%`,
    width: `${((boundaries[b] - boundaries[a]) / (endTime - boundaries[0])) * 100}%`,
  });

  return (
    <section className="episode-exclusions" aria-label="Episode clipping">
      <div className="episode-exclusions-lane">
        <span>Exclude</span>
        <div
          className="episode-exclusions-track"
          data-testid="exclusion-track"
          aria-label="Drag to select an excluded source interval"
          aria-disabled={!canEdit}
          onPointerDown={(event) => {
            if (
              !canEdit ||
              event.button !== 0 ||
              (event.target as HTMLElement).closest("button")
            )
              return;
            event.preventDefault();
            drag.current = {
              pointerId: event.pointerId,
              anchor: pointerFrame(event),
              start,
              end,
            };
            setDragging(true);
            setSelected(null);
            event.currentTarget.setPointerCapture?.(event.pointerId);
            updateDrag(event);
          }}
          onPointerMove={updateDrag}
          onPointerUp={(event) => {
            if (drag.current?.pointerId !== event.pointerId) return;
            updateDrag(event);
            drag.current = null;
            setDragging(false);
            if (event.currentTarget.hasPointerCapture?.(event.pointerId))
              event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onPointerCancel={() => {
            if (drag.current) {
              setStart(drag.current.start);
              setEnd(drag.current.end);
            }
            drag.current = null;
            setDragging(false);
          }}
        >
          {intervals.map((interval, index) => (
            <button
              key={`${interval.start_frame}-${interval.end_frame}`}
              type="button"
              className="episode-exclusion-bar"
              style={rangeStyle(interval.start_frame, interval.end_frame)}
              aria-label={`Select exclusion ${index + 1}`}
              aria-pressed={selected === index}
              onClick={() => {
                setSelected(index);
                seek(boundaries[interval.start_frame], "external");
              }}
            />
          ))}
          {validTimes && endFrame > firstFrame && (
            <div
              className={`episode-exclusion-selection${dragging ? " dragging" : ""}`}
              style={rangeStyle(firstFrame, endFrame)}
            />
          )}
          <div
            className={`episode-exclusion-playhead ${playheadLabelPosition}`}
            data-testid="exclusion-playhead"
            style={{ left: `${playheadFraction * 100}%` }}
            aria-hidden="true"
          >
            <span>{currentTime.toFixed(3)} s</span>
          </div>
        </div>
      </div>
      <p className="episode-exclusions-summary">
        Source timeline · {frameCount} frames · {endTime.toFixed(3)} s.
        Retained: {(retainedCount * step).toFixed(3)} s ({retainedCount}{" "}
        frames).
      </p>
      <div className="episode-exclusions-controls">
        <label>
          Start (s)
          <input
            aria-label="Start seconds"
            type="number"
            min={boundaries[0]}
            max={endTime}
            step={step}
            value={start}
            disabled={!canEdit}
            onInput={(event) => setStart(event.currentTarget.value)}
          />
        </label>
        <label>
          End (s)
          <input
            aria-label="End seconds"
            type="number"
            min={boundaries[0]}
            max={endTime}
            step={step}
            value={end}
            disabled={!canEdit}
            onInput={(event) => setEnd(event.currentTarget.value)}
          />
        </label>
        <button
          type="button"
          disabled={!canEdit || !validSelection}
          onClick={() => {
            if (validSelection) void save(nextIntervals);
          }}
        >
          Exclude interval
        </button>
        <button
          type="button"
          disabled={!canEdit || !intervals.length}
          onClick={() => void save([])}
        >
          Clear
        </button>
        <label>
          <input
            type="checkbox"
            checked={preview}
            onChange={(event) => {
              setPreview(event.target.checked);
              setIsPlaying(false);
            }}
          />
          Preview clipping
        </label>
      </div>
      {validTimes && endFrame > firstFrame && (
        <p className="episode-exclusions-summary">
          Selection: source frames [{firstFrame}, {endFrame}) ·{" "}
          {displayTime(firstFrame)}–{displayTime(endFrame)} s.
        </p>
      )}
      {validTimes && allRemoved && (
        <p role="alert">
          At least one source frame must remain. Use the episode deletion
          decision to remove the whole episode.
        </p>
      )}
      <p className="episode-exclusions-summary">
        Drag the Exclude lane or enter times, then select Exclude interval.
        Review the episode, then use Export &amp; publish → Preview export to
        create the clipped copy. The source timeline stays unchanged.
      </p>
      {intervals.length > 0 && (
        <ul className="episode-exclusions-list">
          {intervals.map((interval, index) => (
            <li
              key={`${interval.start_frame}-${interval.end_frame}`}
              className={selected === index ? "selected" : ""}
            >
              <span>
                {displayTime(interval.start_frame)}–
                {displayTime(interval.end_frame)} s · source frames [
                {interval.start_frame}, {interval.end_frame})
              </span>
              <button
                type="button"
                disabled={!canEdit}
                aria-label={`Remove exclusion ${index + 1}`}
                onClick={() =>
                  void save(
                    intervals.filter((_, candidate) => candidate !== index),
                  )
                }
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
      {(loading ||
        busy ||
        activeJob ||
        dirty ||
        saving ||
        run?.episodes[String(episodeId)]?.decision === "delete") && (
        <p role="status">
          {busy
            ? "Saving exclusions…"
            : loading
              ? "Loading exclusions…"
              : activeJob
                ? `Workflow job ${jobStatus}…`
                : dirty || saving
                  ? "Save annotation edits before changing exclusions."
                  : "This episode is marked for deletion."}
        </p>
      )}
      {error && (
        <p role="alert">
          {error}{" "}
          <button
            type="button"
            disabled={busy || loading}
            onClick={() => {
              setError("");
              setRefresh((value) => value + 1);
            }}
          >
            Reload exclusions
          </button>
        </p>
      )}
    </section>
  );
};
