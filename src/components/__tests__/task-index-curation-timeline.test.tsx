import React, { useEffect } from "react";
import { describe, expect, mock, test } from "bun:test";
import {
  fireEvent,
  getQueriesForElement,
  render,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { TaskIndexCurationTimeline } from "../task-index-curation-timeline";
import { TimeProvider, useTime } from "../../context/time-context";
import type {
  GripDiagnostic,
  TransitionFrames,
} from "../../types/curation.types";

const timestamps = Array.from({ length: 12 }, (_, frame) => frame / 10);
const screen = getQueriesForElement(document.body);

const grip: GripDiagnostic = {
  datasetAlias: "local/pnp_trash",
  sourceEpisodeIndex: 0,
  status: "available",
  reason: null,
  side: "left",
  usableJointIndices: [22, 23, 24, 25],
  grasp: { frame: 2, timestampSeconds: 0.2, derivative: 0.8 },
  release: { frame: 8, timestampSeconds: 0.8, derivative: -0.7 },
  advisories: ["grasp differs by 2.1 s"],
  graspDeltaSeconds: 2.1,
  releaseDeltaSeconds: 0.1,
};

function SetCurrentTime({ seconds }: { seconds: number }) {
  const { seek } = useTime();
  useEffect(() => seek(seconds), [seconds, seek]);
  return null;
}

function CurrentTimeProbe() {
  const { currentTime } = useTime();
  return <output aria-label="player time">{currentTime.toFixed(1)}</output>;
}

function renderTimeline(
  transitions: TransitionFrames,
  onChange = mock(() => undefined),
  options: { currentTime?: number; disabled?: boolean } = {},
) {
  const view = render(
    <TimeProvider duration={1.2}>
      <SetCurrentTime seconds={options.currentTime ?? 0.4} />
      <CurrentTimeProbe />
      <TaskIndexCurationTimeline
        timestamps={timestamps}
        transitions={transitions}
        proposalTransitions={[2, null, 5, 7, 8, 10]}
        proposalStatuses={[
          "completed",
          "not_observed",
          "completed",
          "partial",
          "completed",
          "completed",
        ]}
        grip={grip}
        disabled={options.disabled}
        onChange={onChange}
      />
    </TimeProvider>,
  );
  return { ...view, onChange };
}

describe("TaskIndexCurationTimeline", () => {
  test("renders six controlled source-frame handles and seven exhaustive spans", () => {
    renderTimeline([1, 3, 5, 7, 9, 10]);

    const sliders = screen.getAllByRole("slider") as HTMLInputElement[];
    expect(sliders).toHaveLength(6);
    expect(sliders.map((slider) => slider.value)).toEqual([
      "1",
      "3",
      "5",
      "7",
      "9",
      "10",
    ]);
    expect(screen.getAllByTestId("curation-phase-span")).toHaveLength(7);
    expect(screen.getByText("Frames 0–0 · 0.1 s")).toBeTruthy();
    expect(screen.getByText("Frames 10–11 · 0.2 s")).toBeTruthy();
  });

  test("sets the selected boundary from current time using the source timestamps", async () => {
    const user = userEvent.setup();
    const { onChange } = renderTimeline([1, 3, 5, 7, 9, 10], undefined, {
      currentTime: 0.6,
    });

    await user.click(
      screen.getByRole("button", { name: "Select step 4 boundary" }),
    );
    await user.click(
      screen.getByRole("button", {
        name: "Set selected boundary from current frame",
      }),
    );

    expect(onChange).toHaveBeenCalledWith([1, 3, 6, 7, 9, 10]);
  });

  test("drags and keyboard-nudges exactly one source frame without crossing neighbors", async () => {
    const user = userEvent.setup();
    const { onChange } = renderTimeline([1, 3, 5, 7, 9, 10]);
    const stepFour = screen.getByRole("slider", {
      name: "Step 4 transition frame",
    });

    await user.click(stepFour);
    await user.keyboard("{ArrowLeft}");
    expect(onChange).toHaveBeenCalledWith([1, 3, 4, 7, 9, 10]);

    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenCalledWith([1, 3, 6, 7, 9, 10]);

    fireEvent.input(stepFour, { target: { value: "6" } });
    expect(onChange).toHaveBeenCalledWith([1, 3, 6, 7, 9, 10]);
  });

  test("shows human, nullable proposal, and grip markers and seeks to observed markers", async () => {
    const user = userEvent.setup();
    renderTimeline([1, 3, 5, 7, 9, 10]);

    expect(screen.getAllByText("Human")).toHaveLength(6);
    expect(screen.getByText("Proposal: NOT OBSERVED")).toBeTruthy();
    expect(screen.getByText("Grip grasp · frame 2")).toBeTruthy();
    expect(screen.getByText("Grip release · frame 8")).toBeTruthy();

    await user.click(
      screen.getByRole("button", { name: "Seek to grip release frame 8" }),
    );
    expect(screen.getByLabelText("player time").textContent).toBe("0.8");
  });

  test("supports incomplete human boundaries without fabricating spans", () => {
    renderTimeline([1, null, 5, null, 9, 10]);

    expect(
      screen
        .getAllByRole("button", { name: /Select step .* boundary/ })
        .filter((button) => button.textContent?.includes("Not set")),
    ).toHaveLength(2);
    expect(screen.queryAllByTestId("curation-phase-span")).toHaveLength(0);
    expect(
      screen.getByText(
        "Complete all six boundaries to preview seven phase spans.",
      ),
    ).toBeTruthy();
  });

  test("exposes nullable boundaries as not set and initializes them by keyboard", async () => {
    const user = userEvent.setup();
    const { onChange } = renderTimeline([1, null, 5, null, 9, 10]);
    const unsetBoundary = screen.getByRole("slider", {
      name: "Step 3 transition frame",
    });

    expect(unsetBoundary.getAttribute("aria-valuetext")).toBe("Not set");
    await user.click(unsetBoundary);
    await user.keyboard("{ArrowRight}");

    expect(onChange).toHaveBeenCalledWith([1, 2, 5, null, 9, 10]);
  });

  test("clamps set-from-player to valid bounds for a nullable boundary", async () => {
    const user = userEvent.setup();
    const { onChange } = renderTimeline([1, null, 5, null, 9, 10], undefined, {
      currentTime: 1.1,
    });

    await user.click(
      screen.getByRole("button", { name: "Select step 5 boundary" }),
    );
    await user.click(
      screen.getByRole("button", {
        name: "Set selected boundary from current frame",
      }),
    );

    expect(onChange).toHaveBeenCalledWith([1, null, 5, 8, 9, 10]);
  });

  test("disables pointer, nudge, and set-from-player controls while saving", () => {
    renderTimeline([1, 3, 5, 7, 9, 10], undefined, { disabled: true });

    expect(
      (
        screen.getByRole("slider", {
          name: "Step 2 transition frame",
        }) as HTMLInputElement
      ).disabled,
    ).toBe(true);
    expect(
      (
        screen.getByRole("button", {
          name: "Set selected boundary from current frame",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(
      (
        screen.getByRole("button", {
          name: "Move selected boundary back one frame",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });
});
