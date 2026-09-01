# Precise Playback Time Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the shared video player's current time and duration with fixed hundredth-second precision so curators can read precise interval times.

**Architecture:** Add one pure formatter beside `PlaybackBar`, then route both displayed values through it while leaving the existing `0.01`-second slider and frame-authoritative curation state untouched. Cover the formatter and its component wiring in one focused Testing Library test file, widen the label, and verify the result in the loopback `pnp_trash` review UI.

**Tech Stack:** React 19, TypeScript, Bun test runner, Testing Library, happy-dom, Tailwind CSS, Next.js 15

---

## File Map

- Create `src/components/__tests__/playback-bar.test.tsx`: own the formatting contract and prove `PlaybackBar` renders slider state through it.
- Modify `src/components/playback-bar.tsx:13-100`: define the local pure formatter, use it for current time and duration, and allocate enough non-wrapping width for the new label.
- Reference `docs/superpowers/specs/2026-09-01-playback-time-precision-design.md`: authoritative behavior and acceptance criteria; do not modify it during implementation.

No backend, API, curation-state, timestamp, or export file changes are required.

### Task 1: Implement the Precise Playback Label with TDD

**Files:**

- Create: `src/components/__tests__/playback-bar.test.tsx`
- Modify: `src/components/playback-bar.tsx:13-100`
- Test: `src/components/__tests__/playback-bar.test.tsx`

- [ ] **Step 1: Write the failing formatter and component-wiring tests**

Create `src/components/__tests__/playback-bar.test.tsx` with this complete content:

```tsx
import React from "react";
import { describe, expect, test } from "bun:test";
import {
  fireEvent,
  getQueriesForElement,
  render,
} from "@testing-library/react";

import PlaybackBar, { formatPlaybackSeconds } from "../playback-bar";
import { TimeProvider } from "../../context/time-context";

const screen = getQueriesForElement(document.body);

describe("formatPlaybackSeconds", () => {
  test("renders non-negative finite seconds with fixed hundredth precision", () => {
    expect(formatPlaybackSeconds(0)).toBe("0.00");
    expect(formatPlaybackSeconds(8.2)).toBe("8.20");
    expect(formatPlaybackSeconds(8.239)).toBe("8.24");
    expect(formatPlaybackSeconds(128.7)).toBe("128.70");
    expect(formatPlaybackSeconds(-0.01)).toBe("0.00");
    expect(formatPlaybackSeconds(Number.NaN)).toBe("0.00");
    expect(formatPlaybackSeconds(Number.POSITIVE_INFINITY)).toBe("0.00");
  });
});

describe("PlaybackBar", () => {
  test("shows precise current and total seconds and updates while seeking", () => {
    render(
      <TimeProvider duration={128.7}>
        <PlaybackBar />
      </TimeProvider>,
    );

    expect(screen.getByText("0.00 / 128.70")).toBeTruthy();

    fireEvent.change(screen.getByRole("slider", { name: "Seek video" }), {
      target: { value: "8.239" },
    });

    const label = screen.getByText("8.24 / 128.70");
    expect(label).toBeTruthy();
    expect(label.className).toContain("w-28");
    expect(label.className).toContain("whitespace-nowrap");
  });
});
```

- [ ] **Step 2: Run the focused test and confirm the red state**

Run:

```bash
bun test src/components/__tests__/playback-bar.test.tsx
```

Expected: FAIL because `playback-bar.tsx` does not export
`formatPlaybackSeconds`; the old component also renders floored whole seconds.
Do not proceed unless the new test fails for that missing behavior.

- [ ] **Step 3: Add the minimal formatter and route the label through it**

In `src/components/playback-bar.tsx`, add this named export immediately after
the imports:

```tsx
export function formatPlaybackSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0.00";
  }
  return seconds.toFixed(2);
}
```

Replace the current time-label span with:

```tsx
<span className="w-28 shrink-0 whitespace-nowrap text-right tabular text-[11px] text-slate-400">
  {formatPlaybackSeconds(sliderValue)} / {formatPlaybackSeconds(duration)}
</span>
```

Do not change the range input's `step={0.01}`, seek handlers, `TimeProvider`, or
curation timeline code.

- [ ] **Step 4: Format only the two touched source files**

Run:

```bash
./node_modules/.bin/prettier --write src/components/playback-bar.tsx src/components/__tests__/playback-bar.test.tsx
```

Expected: both paths are reported as formatted or unchanged; no other file is
rewritten.

- [ ] **Step 5: Run the focused test and confirm the green state**

Run:

```bash
bun test src/components/__tests__/playback-bar.test.tsx
```

Expected: 2 tests pass with no failures. The rendered label changes from
`0.00 / 128.70` to `8.24 / 128.70` after the range change.

- [ ] **Step 6: Commit the focused implementation**

Run:

```bash
git add src/components/playback-bar.tsx src/components/__tests__/playback-bar.test.tsx
git commit -m "feat: show precise playback time"
```

Expected: one commit containing only the component and focused test.

### Task 2: Validate the Frontend and Browser Result

**Files:**

- Verify: `src/components/playback-bar.tsx`
- Verify: `src/components/__tests__/playback-bar.test.tsx`
- Verify: `src/components/task-index-curation-timeline.tsx`

- [ ] **Step 1: Run the repository formatting check**

Run:

```bash
bun run format:check
```

Expected: PASS with all matched files using Prettier formatting.

- [ ] **Step 2: Run the complete frontend validation suite**

Run:

```bash
bun run validate
```

Expected: type checking, lint, formatting, and all tests pass. Based on the
current 241-test baseline plus the two new tests, Bun reports 243 passing tests
and zero failures. The three previously recorded React-hooks lint warnings may
remain; no new warning or error is acceptable.

- [ ] **Step 3: Confirm the implementation commit is clean**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` prints nothing and exits zero; `git status
--short` prints nothing.

- [ ] **Step 4: Start fresh loopback review services**

First confirm neither review port is owned by an unknown process:

```bash
lsof -nP -iTCP:3000 -sTCP:LISTEN
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Expected before startup: neither command lists a listener. If a listener is
present, identify it and stop only a review process started by this task; do
not terminate an unrelated process.

In a dedicated terminal, run this exact loopback-only review harness from the
feature worktree:

```bash
set -euo pipefail
task14_repo=/home/jihun/work/GR00T-WholeBodyControl/worktrees/lerobot-dataset-visualizer-pnp-trash
task14_bearer_token=$(openssl rand -hex 32)
task14_cosmos_token=$(openssl rand -hex 32)
task14_backend_pid=
task14_next_pid=

task14_cleanup() {
  if test -n "$task14_next_pid"; then kill "$task14_next_pid" 2>/dev/null || true; fi
  if test -n "$task14_backend_pid"; then kill "$task14_backend_pid" 2>/dev/null || true; fi
  if test -n "$task14_next_pid"; then wait "$task14_next_pid" 2>/dev/null || true; fi
  if test -n "$task14_backend_pid"; then wait "$task14_backend_pid" 2>/dev/null || true; fi
}
trap task14_cleanup EXIT INT TERM

cd "$task14_repo"

env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$task14_repo" \
  CURATION_DATASET_ALIASES_JSON='{"local/pnp_trash":"/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash"}' \
  CURATION_WORKSPACE=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_curation \
  CURATION_OUTPUT=/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash_cleaned \
  CURATION_BROWSER_ORIGIN=http://127.0.0.1:3000 \
  CURATION_BEARER_TOKEN="$task14_bearer_token" \
  COSMOS_BASE_URL=http://127.0.0.1:34001/v1 \
  COSMOS_MODEL=nvidia/Cosmos3-Nano \
  COSMOS_API_KEY_ENV=COSMOS_API_KEY \
  COSMOS_ENDPOINT_IDENTITY=h100-loopback-34001 \
  ISAAC_GROOT_ROOT=/home/jihun/work/Isaac-GR00T \
  COSMOS_API_KEY="$task14_cosmos_token" \
  backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
task14_backend_pid=$!

env -i PATH="$PATH" HOME="$HOME" \
  CURATION_BACKEND_URL=http://127.0.0.1:8000 \
  CURATION_BEARER_TOKEN="$task14_bearer_token" \
  NEXT_PUBLIC_DATASET_URL=http://127.0.0.1:8000/api/local-datasets \
  ./node_modules/.bin/next dev --hostname 127.0.0.1 --port 3000 &
task14_next_pid=$!

curl --retry 60 --retry-delay 1 --retry-connrefused -fsS \
  'http://127.0.0.1:3000/api/curation/summary?dataset_alias=local%2Fpnp_trash'

printf '%s\n' 'Review URL: http://127.0.0.1:3000/local/pnp_trash/episode_0?tab=annotations'
wait -n "$task14_backend_pid" "$task14_next_pid"
```

Expected: both listeners bind only to `127.0.0.1`; the summary request returns
the existing 92-episode pending workspace. This harness does not start the
Cosmos worker or Task 15.

- [ ] **Step 5: Exercise the new label in a real browser**

Open:

```text
http://127.0.0.1:3000/local/pnp_trash/episode_0?tab=annotations
```

Confirm the playback label initially matches this shape, with exactly two
digits after each decimal point (the real duration may differ from this
example):

```text
0.00 / 128.70
```

In browser developer tools, run this exact snippet to seek to 8.24 seconds and
inspect the adjacent label:

```js
const slider = document.querySelector('input[aria-label="Seek video"]');
if (!(slider instanceof HTMLInputElement)) {
  throw new Error("Playback slider not found");
}
const setValue = Object.getOwnPropertyDescriptor(
  HTMLInputElement.prototype,
  "value",
)?.set;
if (!setValue) throw new Error("Native input value setter not found");
setValue.call(slider, "8.24");
slider.dispatchEvent(new Event("input", { bubbles: true }));
const label = slider.nextElementSibling?.textContent?.trim() ?? "";
if (!/^8\.24 \/ \d+\.\d{2}$/.test(label)) {
  throw new Error(`Unexpected playback label: ${label}`);
}
label;
```

Expected: the console returns a value shaped like `8.24 / 128.70`; the video
seeks, the label remains on one line, and the annotation timeline continues to
show and manipulate source-frame boundaries exactly as before.

- [ ] **Step 6: Leave the verified result open for user approval**

Report the focused and full validation outputs and provide the review URL. Do
not merge, push, create a pull request, run the Cosmos worker, or begin Task 15
until the user approves the visible result.

After the user finishes reviewing, stop the two review services with `Ctrl-C`
in their dedicated terminal. The harness trap must reap only the two processes
it started.

## Post-Plan Integration Gate

Once the user explicitly approves the browser result, invoke the
`finishing-a-development-branch` skill. Re-verify the feature worktree and the
dirty primary checkout before choosing a safe integration path. The merge is a
separate authorized operation; this implementation plan performs no push or
pull-request creation and does not start Task 15.
