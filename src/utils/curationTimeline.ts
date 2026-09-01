export type TransitionFrame = number | null;
export type CompleteTransitionFrames = [
  number,
  number,
  number,
  number,
  number,
  number,
];

export type TransitionValidation =
  | { valid: true; transitions: CompleteTransitionFrames }
  | {
      valid: false;
      reason:
        | "six_transitions_required"
        | "transition_not_integer"
        | "transition_out_of_range"
        | "transitions_not_strictly_increasing";
    };

export interface TimelineSpan {
  step: 1 | 2 | 3 | 4 | 5 | 6 | 7;
  startFrame: number;
  endFrame: number;
  durationFrames: number;
  startTimeSeconds: number;
  endTimeSeconds: number;
  durationSeconds: number;
  durationLabel: string;
}

export function snapToNearestFrame(
  timestamps: readonly number[],
  seconds: number,
): number {
  if (
    timestamps.length === 0 ||
    !Number.isFinite(seconds) ||
    timestamps.some(
      (timestamp, index) =>
        !Number.isFinite(timestamp) ||
        (index > 0 && timestamp <= timestamps[index - 1]),
    )
  ) {
    throw new RangeError(
      "timestamps must be a nonempty, finite, strictly increasing array",
    );
  }

  let nearest = 0;
  let distance = Math.abs(timestamps[0] - seconds);
  for (let index = 1; index < timestamps.length; index += 1) {
    const candidateDistance = Math.abs(timestamps[index] - seconds);
    if (candidateDistance < distance) {
      nearest = index;
      distance = candidateDistance;
    }
  }
  return nearest;
}

export function isCompleteTransitionSet(
  transitions: readonly TransitionFrame[],
): transitions is CompleteTransitionFrames {
  return (
    transitions.length === 6 &&
    transitions.every((frame) => Number.isInteger(frame))
  );
}

export function validateTransitionFrames(
  transitions: readonly TransitionFrame[],
  frameCount: number,
): TransitionValidation {
  if (transitions.length !== 6)
    return { valid: false, reason: "six_transitions_required" };
  if (!isCompleteTransitionSet(transitions)) {
    return { valid: false, reason: "transition_not_integer" };
  }
  if (
    !Number.isInteger(frameCount) ||
    frameCount < 7 ||
    transitions.some((frame) => frame < 1 || frame >= frameCount)
  ) {
    return { valid: false, reason: "transition_out_of_range" };
  }
  if (
    transitions.some(
      (frame, index) => index > 0 && frame <= transitions[index - 1],
    )
  ) {
    return { valid: false, reason: "transitions_not_strictly_increasing" };
  }
  return {
    valid: true,
    transitions: [...transitions] as CompleteTransitionFrames,
  };
}

export function nudgeTransitionFrame(
  transitions: readonly number[],
  transitionIndex: number,
  direction: -1 | 1,
  frameCount: number,
): CompleteTransitionFrames {
  const validated = validateTransitionFrames(transitions, frameCount);
  if (!validated.valid) throw new RangeError(validated.reason);
  if (
    !Number.isInteger(transitionIndex) ||
    transitionIndex < 0 ||
    transitionIndex >= 6
  ) {
    throw new RangeError("transition index must be between zero and five");
  }

  const next = [...validated.transitions] as CompleteTransitionFrames;
  const lowerBound = transitionIndex === 0 ? 1 : next[transitionIndex - 1] + 1;
  const upperBound =
    transitionIndex === 5 ? frameCount - 1 : next[transitionIndex + 1] - 1;
  next[transitionIndex] = Math.max(
    lowerBound,
    Math.min(upperBound, next[transitionIndex] + direction),
  );
  return next;
}

export function formatDurationLabel(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new RangeError("duration must be finite and nonnegative");
  }
  return `${seconds.toFixed(1)} s`;
}

function roundedSeconds(seconds: number): number {
  return Number(seconds.toFixed(12));
}

export function buildTimelineSpans(
  transitions: readonly TransitionFrame[],
  timestamps: readonly number[],
): TimelineSpan[] | null {
  const validated = validateTransitionFrames(transitions, timestamps.length);
  if (!validated.valid) return null;
  if (
    timestamps.some(
      (timestamp, index) =>
        !Number.isFinite(timestamp) ||
        (index > 0 && timestamp <= timestamps[index - 1]),
    )
  ) {
    throw new RangeError("timestamps must be finite and strictly increasing");
  }

  const starts = [0, ...validated.transitions];
  const finalFrameInterval = timestamps.at(-1)! - timestamps.at(-2)!;
  const episodeEndSeconds = timestamps.at(-1)! + finalFrameInterval;

  return starts.map((startFrame, index) => {
    const nextStart = starts[index + 1] ?? timestamps.length;
    const endFrame = nextStart - 1;
    const endTimeSeconds =
      nextStart < timestamps.length ? timestamps[nextStart] : episodeEndSeconds;
    const durationSeconds = roundedSeconds(
      endTimeSeconds - timestamps[startFrame],
    );
    return {
      step: (index + 1) as TimelineSpan["step"],
      startFrame,
      endFrame,
      durationFrames: endFrame - startFrame + 1,
      startTimeSeconds: timestamps[startFrame],
      endTimeSeconds: roundedSeconds(endTimeSeconds),
      durationSeconds,
      durationLabel: formatDurationLabel(durationSeconds),
    };
  });
}
