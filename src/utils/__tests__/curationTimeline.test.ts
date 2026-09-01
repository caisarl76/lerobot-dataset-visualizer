import { describe, expect, test } from "bun:test";

import {
  buildTimelineSpans,
  formatDurationLabel,
  isCompleteTransitionSet,
  nudgeTransitionFrame,
  snapToNearestFrame,
  validateTransitionFrames,
} from "../curationTimeline";

describe("curation timeline", () => {
  test("snaps seconds to the nearest source frame and breaks exact ties lower", () => {
    const timestamps = [0, 0.1, 0.21, 0.31];

    expect(snapToNearestFrame(timestamps, -4)).toBe(0);
    expect(snapToNearestFrame(timestamps, 0.05)).toBe(0);
    expect(snapToNearestFrame(timestamps, 0.08)).toBe(1);
    expect(snapToNearestFrame(timestamps, 0.155)).toBe(1);
    expect(snapToNearestFrame(timestamps, 20)).toBe(3);
  });

  test("requires exactly six strictly increasing in-range integer transitions", () => {
    expect(validateTransitionFrames([1, 2, 3, 4, 5, 6], 8)).toEqual({
      valid: true,
      transitions: [1, 2, 3, 4, 5, 6],
    });
    expect(validateTransitionFrames([0, 2, 3, 4, 5, 6], 8)).toEqual({
      valid: false,
      reason: "transition_out_of_range",
    });
    expect(validateTransitionFrames([1, 2, 2, 4, 5, 6], 8)).toEqual({
      valid: false,
      reason: "transitions_not_strictly_increasing",
    });
    expect(validateTransitionFrames([1, 2, 3, 4, 5, 8], 8)).toEqual({
      valid: false,
      reason: "transition_out_of_range",
    });
    expect(validateTransitionFrames([1, 2, 3], 8)).toEqual({
      valid: false,
      reason: "six_transitions_required",
    });
  });

  test("nudges exactly one frame without crossing neighbors or emptying edge spans", () => {
    const transitions: [number, number, number, number, number, number] = [
      1, 3, 5, 7, 9, 11,
    ];

    expect(nudgeTransitionFrame(transitions, 1, -1, 13)).toEqual([
      1, 2, 5, 7, 9, 11,
    ]);
    expect(nudgeTransitionFrame([1, 2, 5, 7, 9, 11], 1, -1, 13)).toEqual([
      1, 2, 5, 7, 9, 11,
    ]);
    expect(nudgeTransitionFrame(transitions, 1, 1, 13)).toEqual([
      1, 4, 5, 7, 9, 11,
    ]);
    expect(nudgeTransitionFrame([1, 4, 5, 7, 9, 11], 1, 1, 13)).toEqual([
      1, 4, 5, 7, 9, 11,
    ]);
    expect(nudgeTransitionFrame(transitions, 0, -1, 13)).toEqual(transitions);
    expect(nudgeTransitionFrame(transitions, 5, 1, 12)).toEqual(transitions);
  });

  test("builds seven gap-free nonoverlapping spans that cover every source frame", () => {
    const timestamps = Array.from({ length: 14 }, (_, frame) => frame * 0.1);
    const spans = buildTimelineSpans([2, 4, 6, 8, 10, 12], timestamps);

    expect(
      spans?.map(({ step, startFrame, endFrame }) => ({
        step,
        startFrame,
        endFrame,
      })),
    ).toEqual([
      { step: 1, startFrame: 0, endFrame: 1 },
      { step: 2, startFrame: 2, endFrame: 3 },
      { step: 3, startFrame: 4, endFrame: 5 },
      { step: 4, startFrame: 6, endFrame: 7 },
      { step: 5, startFrame: 8, endFrame: 9 },
      { step: 6, startFrame: 10, endFrame: 11 },
      { step: 7, startFrame: 12, endFrame: 13 },
    ]);
    expect(
      spans?.flatMap((span) =>
        Array.from(
          { length: span.endFrame - span.startFrame + 1 },
          (_, offset) => span.startFrame + offset,
        ),
      ),
    ).toEqual(Array.from({ length: 14 }, (_, frame) => frame));
    expect(spans?.every((span) => span.durationFrames === 2)).toBe(true);
    expect(spans?.every((span) => span.durationSeconds === 0.2)).toBe(true);
    expect(spans?.every((span) => span.durationLabel === "0.2 s")).toBe(true);
  });

  test("formats finite nonnegative durations consistently", () => {
    expect(formatDurationLabel(0)).toBe("0.0 s");
    expect(formatDurationLabel(1.24)).toBe("1.2 s");
    expect(formatDurationLabel(12.96)).toBe("13.0 s");
  });

  test("preserves nullable incomplete proposals without inventing spans", () => {
    const incomplete = [1, 2, null, 4, null, 6] as const;

    expect(isCompleteTransitionSet(incomplete)).toBe(false);
    expect(
      buildTimelineSpans(
        incomplete,
        Array.from({ length: 8 }, (_, frame) => frame / 10),
      ),
    ).toBeNull();
    expect(incomplete).toEqual([1, 2, null, 4, null, 6]);
  });
});
