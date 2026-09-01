import { describe, expect, test } from "bun:test";

import { resolveInitialEpisodeViewerTab } from "../episodeViewerTabs";

const resolve = (
  query: string,
  persisted: string | null,
  urdfAvailable = false,
) =>
  resolveInitialEpisodeViewerTab(new URLSearchParams(query), persisted, {
    urdfAvailable,
  });

describe("resolveInitialEpisodeViewerTab", () => {
  test("prefers one recognized query value over persisted state", () => {
    expect(resolve("tab=annotations", "statistics")).toBe("annotations");
    expect(resolve("tab=doctor", "episodes")).toBe("doctor");
  });

  test("rejects empty, repeated, and unknown query values", () => {
    expect(resolve("tab=", "frames")).toBe("frames");
    expect(resolve("tab=annotations&tab=doctor", "statistics")).toBe(
      "statistics",
    );
    expect(resolve("tab=annotations&tab=annotations", "frames")).toBe("frames");
    expect(resolve("tab=unknown", "frames")).toBe("frames");
  });

  test("retains the exact legacy persisted whitelist", () => {
    for (const tab of [
      "episodes",
      "annotations",
      "statistics",
      "frames",
      "insights",
      "filtering",
      "urdf",
    ] as const) {
      expect(resolve("", tab, true)).toBe(tab);
    }
    expect(resolve("", "doctor", true)).toBe("episodes");
    expect(resolve("", "unknown", true)).toBe("episodes");
  });

  test("requires URDF availability for query and persisted inputs", () => {
    expect(resolve("tab=urdf", "annotations", true)).toBe("urdf");
    expect(resolve("tab=urdf", "annotations", false)).toBe("annotations");
    expect(resolve("", "urdf", true)).toBe("urdf");
    expect(resolve("", "urdf", false)).toBe("episodes");
  });
});
