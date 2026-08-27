"use client";

import { useEffect, useMemo, useState } from "react";

import { useTime } from "../context/time-context";
import type { GripDiagnostic, TransitionFrames } from "../types/curation.types";
import {
  buildTimelineSpans,
  snapToNearestFrame,
} from "../utils/curationTimeline";

const TRANSITION_STEPS = [2, 3, 4, 5, 6, 7] as const;

type ProposalStatus = "completed" | "partial" | "not_observed";

export interface TaskIndexCurationTimelineProps {
  timestamps: readonly number[];
  transitions: TransitionFrames;
  proposalTransitions?: TransitionFrames | null;
  proposalStatuses?: readonly ProposalStatus[];
  grip?: GripDiagnostic | null;
  disabled?: boolean;
  onChange: (transitions: TransitionFrames) => void;
}

function finiteFrame(frame: number | null): frame is number {
  return frame !== null && Number.isInteger(frame);
}

function editableBounds(
  transitions: TransitionFrames,
  transitionIndex: number,
  frameCount: number,
): { lower: number; upper: number } {
  let lower = 1;
  for (let index = transitionIndex - 1; index >= 0; index -= 1) {
    const previous = transitions[index];
    if (finiteFrame(previous)) {
      lower = previous + 1;
      break;
    }
  }

  let upper = frameCount - 1;
  for (
    let index = transitionIndex + 1;
    index < transitions.length;
    index += 1
  ) {
    const next = transitions[index];
    if (finiteFrame(next)) {
      upper = next - 1;
      break;
    }
  }
  return { lower, upper };
}

function transitionCopy(
  transitions: TransitionFrames,
  transitionIndex: number,
  frame: number,
): TransitionFrames {
  const next = [...transitions] as TransitionFrames;
  next[transitionIndex] = frame;
  return next;
}

function frameLabel(frame: number, timestamps: readonly number[]): string {
  return `frame ${frame} · ${timestamps[frame].toFixed(3)} s`;
}

export function TaskIndexCurationTimeline({
  timestamps,
  transitions,
  proposalTransitions = null,
  proposalStatuses = [],
  grip = null,
  disabled = false,
  onChange,
}: TaskIndexCurationTimelineProps) {
  const { currentTime, seek } = useTime();
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    if (selectedIndex >= transitions.length) setSelectedIndex(0);
  }, [selectedIndex, transitions.length]);

  const spans = useMemo(
    () => buildTimelineSpans(transitions, timestamps),
    [timestamps, transitions],
  );

  const setFrame = (transitionIndex: number, frame: number) => {
    if (disabled || !Number.isInteger(frame)) return;
    const { lower, upper } = editableBounds(
      transitions,
      transitionIndex,
      timestamps.length,
    );
    if (frame < lower || frame > upper) return;
    onChange(transitionCopy(transitions, transitionIndex, frame));
    seek(timestamps[frame]);
  };

  const selectedFrame = transitions[selectedIndex];
  const selectedBounds = editableBounds(
    transitions,
    selectedIndex,
    timestamps.length,
  );
  const canMoveBack =
    finiteFrame(selectedFrame) && selectedFrame > selectedBounds.lower;
  const canMoveForward =
    finiteFrame(selectedFrame) && selectedFrame < selectedBounds.upper;

  const setFromPlayer = () => {
    const frame = snapToNearestFrame(timestamps, currentTime);
    if (selectedBounds.lower > selectedBounds.upper) return;
    setFrame(
      selectedIndex,
      Math.min(selectedBounds.upper, Math.max(selectedBounds.lower, frame)),
    );
  };

  const nudge = (direction: -1 | 1) => {
    const frame = transitions[selectedIndex];
    if (!finiteFrame(frame)) return;
    setFrame(selectedIndex, frame + direction);
  };

  return (
    <section
      className="panel p-4 space-y-4"
      aria-labelledby="curation-timeline-heading"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3
            id="curation-timeline-heading"
            className="text-sm font-medium text-slate-100"
          >
            Seven-phase task timeline
          </h3>
          <p className="mt-1 text-xs text-slate-400">
            Select the first frame of steps 2–7. Every saved frame is a
            source-frame index.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className="rounded border border-slate-600 px-2 py-1 text-xs text-slate-200 disabled:opacity-40"
            aria-label="Move selected boundary back one frame"
            disabled={disabled || !canMoveBack}
            onClick={() => nudge(-1)}
          >
            −1 frame
          </button>
          <button
            type="button"
            className="rounded border border-slate-600 px-2 py-1 text-xs text-slate-200 disabled:opacity-40"
            aria-label="Move selected boundary forward one frame"
            disabled={disabled || !canMoveForward}
            onClick={() => nudge(1)}
          >
            +1 frame
          </button>
          <button
            type="button"
            className="rounded border border-cyan-500/60 px-2 py-1 text-xs text-cyan-200 disabled:opacity-40"
            aria-label="Set selected boundary from current frame"
            disabled={disabled}
            onClick={setFromPlayer}
          >
            Set from player
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {TRANSITION_STEPS.map((step, transitionIndex) => {
          const frame = transitions[transitionIndex];
          const proposalFrame = proposalTransitions?.[transitionIndex] ?? null;
          const bounds = editableBounds(
            transitions,
            transitionIndex,
            timestamps.length,
          );
          const rangeValue = finiteFrame(frame)
            ? frame
            : Math.min(bounds.upper, bounds.lower);
          const selected = selectedIndex === transitionIndex;
          const gripMarker =
            transitionIndex === 0 && grip?.status === "available"
              ? { kind: "grasp", point: grip.grasp }
              : transitionIndex === 4 && grip?.status === "available"
                ? { kind: "release", point: grip.release }
                : null;

          return (
            <div
              key={step}
              className={`rounded border p-3 ${selected ? "border-cyan-500/70 bg-cyan-950/20" : "border-slate-700/80"}`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  className="text-left text-xs font-medium text-slate-200"
                  aria-label={`Select step ${step} boundary`}
                  aria-pressed={selected}
                  onClick={() => setSelectedIndex(transitionIndex)}
                >
                  Step {step} starts ·{" "}
                  {finiteFrame(frame)
                    ? frameLabel(frame, timestamps)
                    : "Not set"}
                </button>
                <span className="text-[10px] uppercase tracking-wide text-cyan-300">
                  Human
                </span>
              </div>
              <input
                className="mt-2 w-full accent-cyan-400"
                type="range"
                aria-label={`Step ${step} transition frame`}
                min={bounds.lower}
                max={Math.max(bounds.lower, bounds.upper)}
                step={1}
                value={rangeValue}
                aria-valuetext={
                  finiteFrame(frame) ? frameLabel(frame, timestamps) : "Not set"
                }
                disabled={disabled || bounds.lower > bounds.upper}
                onFocus={() => setSelectedIndex(transitionIndex)}
                onInput={(event) =>
                  setFrame(transitionIndex, Number(event.currentTarget.value))
                }
                onKeyDown={(event) => {
                  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight")
                    return;
                  event.preventDefault();
                  setSelectedIndex(transitionIndex);
                  setFrame(
                    transitionIndex,
                    finiteFrame(frame)
                      ? frame + (event.key === "ArrowLeft" ? -1 : 1)
                      : rangeValue,
                  );
                }}
              />
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                {finiteFrame(proposalFrame) ? (
                  <button
                    type="button"
                    className="rounded bg-violet-950/50 px-2 py-1 text-violet-200"
                    aria-label={`Seek to proposal frame ${proposalFrame}`}
                    onClick={() => seek(timestamps[proposalFrame])}
                  >
                    Proposal · {frameLabel(proposalFrame, timestamps)} ·{" "}
                    {proposalStatuses[transitionIndex] ?? "completed"}
                  </button>
                ) : proposalTransitions ? (
                  <span className="rounded bg-slate-800 px-2 py-1 text-slate-300">
                    Proposal: NOT OBSERVED
                  </span>
                ) : (
                  <span>No proposal</span>
                )}
                {gripMarker && (
                  <button
                    type="button"
                    className="rounded bg-amber-950/50 px-2 py-1 text-amber-200"
                    aria-label={`Seek to grip ${gripMarker.kind} frame ${gripMarker.point.frame}`}
                    onClick={() => seek(gripMarker.point.timestampSeconds)}
                  >
                    Grip {gripMarker.kind} · frame {gripMarker.point.frame}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {spans === null ? (
        <p className="text-xs text-amber-300">
          Complete all six boundaries to preview seven phase spans.
        </p>
      ) : (
        <ol className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {spans.map((span) => (
            <li
              key={span.step}
              data-testid="curation-phase-span"
              className="rounded border border-slate-700 bg-slate-900/40 p-2"
            >
              <span className="text-xs font-medium text-slate-200">
                Step {span.step}
              </span>
              <p className="mt-1 text-[11px] text-slate-400">
                Frames {span.startFrame}–{span.endFrame} · {span.durationLabel}
              </p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
