import React from "react";
import { describe, expect, test } from "bun:test";
import {
  fireEvent,
  getQueriesForElement,
  render,
} from "@testing-library/react";

import PlaybackBar, { formatPlaybackSeconds } from "../playback-bar";
import { TimeProvider, useTime } from "../../context/time-context";

const screen = getQueriesForElement(document.body);

function SeekToPreciseTime() {
  const { seek } = useTime();

  return <button onClick={() => seek(8.239)}>Seek precisely</button>;
}

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
        <SeekToPreciseTime />
        <PlaybackBar />
      </TimeProvider>,
    );

    expect(screen.getByText("0.00 / 128.70")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Seek precisely" }));

    const label = screen.getByText("8.24 / 128.70");
    expect(label).toBeTruthy();
    expect(label.className).toContain("w-28");
    expect(label.className).toContain("whitespace-nowrap");
    const slider = screen.getByRole("slider", { name: "Seek video" });
    if (!(slider instanceof HTMLInputElement)) {
      throw new Error("Seek video control is not an HTML input");
    }
    expect(slider.value).toBe("8.239");
  });
});
