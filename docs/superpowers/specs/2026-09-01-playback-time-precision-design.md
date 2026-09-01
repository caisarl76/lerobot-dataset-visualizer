# Playback Time Precision Design

**Date:** 2026-09-01

**Status:** Approved for implementation planning

## Context

The shared playback bar currently renders both the current playback time and
the video duration with `Math.floor`. Although its range input already seeks in
`0.01`-second steps, the visible label discards the fractional part. That makes
the player label unsuitable for checking precise curation interval boundaries.

The `pnp_trash` source is sampled at 50 Hz, so a two-decimal seconds label can
represent its 0.02-second frame spacing. The curation timeline remains the
authority for exact source-frame selection and storage.

## Formatting Contract

The playback label displays the current slider value and duration as seconds
with exactly two fractional digits:

```text
8.24 / 128.70
```

- The integer portion expands naturally; it is not padded or truncated.
- The fractional portion is rounded to two decimal places and zero-padded.
- The existing spaces around `/` are retained.
- Non-finite or negative transient values render as `0.00` rather than leaking
  `NaN`, `Infinity`, or a negative time into the interface.
- The current-time side continues to use the local slider value so it updates
  immediately during a drag.

## Component Design

`PlaybackBar` owns a small pure seconds-formatting function and uses it for both
halves of the label. Keeping the formatter beside the component makes the
contract independently testable without introducing a broader time-formatting
API.

The label keeps tabular numerals and right alignment. Its fixed width increases
from the current narrow allocation so values such as
`128.70 / 128.70` remain readable without clipping or wrapping.

## Behavior That Does Not Change

- The range input retains its existing `0.01`-second step.
- Playback, pause, drag, keyboard, and video synchronization behavior are
  unchanged.
- Curation boundaries continue to snap to authoritative source-frame
  timestamps.
- Stored transition frame indices, interval semantics, API payloads, and
  exported annotations are unchanged.
- The change applies wherever the shared playback bar is rendered, including
  the Episodes and Annotations tabs.

## Test Strategy

Implementation follows test-driven development:

1. Add failing formatter tests for zero padding, rounding, values over 99
   seconds, and invalid/negative inputs.
2. Add a failing `PlaybackBar` wiring test that renders it inside `TimeProvider`
   and proves the initial duration and a subsequent seek appear with two
   decimals.
3. Make the smallest component change that passes those tests.
4. Run the focused test, the complete frontend validation suite, and a browser
   smoke check against the local `pnp_trash` annotations view.

## Acceptance Criteria

- The review UI displays playback time in the form `8.24 / 128.70`.
- Both values always have exactly two fractional digits.
- Long durations remain visible and stable as the current time changes.
- Seeking and annotation interval data behave exactly as before.
- Focused tests and the existing frontend validation suite pass.
