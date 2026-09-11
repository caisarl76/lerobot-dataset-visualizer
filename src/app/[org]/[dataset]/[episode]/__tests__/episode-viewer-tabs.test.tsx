import React, { type PropsWithChildren } from "react";
import { describe, expect, mock, test } from "bun:test";
import { act, render, type RenderResult } from "@testing-library/react";

let loadedFixture = episodeFixture();
const seekSpy = mock(() => {});

mock.module("next/navigation", () => ({
  useRouter: () => ({ push: mock(() => {}) }),
  useSearchParams: () => new URLSearchParams(window.location.search),
}));

mock.module("../fetch-data", () => ({
  computeColumnMinMax: () => [],
  getAdjacentEpisodesVideoInfo: async () => [],
  getEpisodeDataSafe: async () => ({ data: loadedFixture, error: null }),
  loadAllEpisodeFrameInfo: async () => null,
  loadAllEpisodeLengthsV3: async () => null,
  loadCrossEpisodeActionVariance: async () => null,
}));

mock.module("@/context/time-context", () => ({
  TimeProvider: ({ children }: PropsWithChildren) => children,
  useTime: () => ({
    currentTime: 0,
    isPlaying: false,
    seek: seekSpy,
    setIsPlaying: mock(() => {}),
  }),
}));

mock.module("@/context/flagged-episodes-context", () => ({
  FlaggedEpisodesProvider: ({ children }: PropsWithChildren) => children,
}));

mock.module("@/context/annotations-context", () => ({
  AnnotationsProvider: ({ children }: PropsWithChildren) => children,
  useAnnotations: () => ({ setEpisode: mock(() => {}) }),
}));

mock.module("@/context/curation-context", () => ({
  CurationProvider: ({ children }: PropsWithChildren) => children,
  CurationRouteSync: () => null,
  useCuration: () => {
    throw new Error("workspace leaf must remain mocked");
  },
}));

const { isTaskIndexCurationDataset } =
  await import("@/components/task-index-curation-workspace");
mock.module("@/components/task-index-curation-workspace", () => ({
  isTaskIndexCurationDataset,
  TaskIndexCurationWorkspace: () => (
    <div data-testid="task-index-workspace-probe" />
  ),
}));

mock.module("@/components/simple-videos-player", () => ({
  SimpleVideosPlayer: () => null,
}));
mock.module("@/components/playback-bar", () => ({ default: () => null }));
mock.module("@/components/side-nav", () => ({ default: () => null }));
mock.module("@/components/stats-panel", () => ({
  default: () => <div data-testid="statistics-probe" />,
}));
mock.module("@/components/overview-panel", () => ({ default: () => null }));
mock.module("@/components/loading-component", () => ({ default: () => null }));
mock.module("@/components/hf-auth-button", () => ({ default: () => null }));
mock.module("@/components/annotations-panel", () => ({
  AnnotationsPanel: () => null,
}));
mock.module("@/components/annotations-timeline", () => ({
  AnnotationsTimeline: () => null,
}));
mock.module("@/components/urdf-viewer", () => ({
  default: () => <div data-testid="urdf-probe" />,
}));
mock.module("@/components/action-insights-panel", () => ({
  default: () => null,
}));
mock.module("@/components/filtering-panel", () => ({ default: () => null }));
mock.module("@/components/data-recharts", () => ({ default: () => null }));
mock.module("@/utils/postParentMessage", () => ({
  postParentMessageWithParams: mock(() => {}),
}));
mock.module("@/utils/versionUtils", () => ({
  getDatasetVersionAndInfo: async () => ({
    version: "v2.1",
    info: { fps: 50 },
  }),
}));

const { default: EpisodeViewer } = await import("../episode-viewer");

function episodeFixture(robotType = "unitree_g1", codebaseVersion = "v2.1") {
  return {
    datasetInfo: {
      repoId: "local/pnp_trash",
      total_frames: 2,
      total_episodes: 1,
      fps: 50,
      robot_type: robotType,
      codebase_version: codebaseVersion,
      total_tasks: 1,
      dataset_size_mb: 1,
      cameras: [],
    },
    episodeId: 0,
    videosInfo: [],
    chartDataGroups: [],
    flatChartData: [],
    episodes: [0],
    ignoredColumns: [],
    duration: 1,
    task: "fixture task",
    languageAtoms: [],
    frameTimestamps: [0, 0.02],
  };
}

async function renderViewer(options: {
  query?: string;
  persisted?: string | null;
  robotType?: string;
  codebaseVersion?: string;
}): Promise<RenderResult> {
  window.history.replaceState(
    {},
    "",
    `/local/pnp_trash/episode_0${options.query ?? ""}`,
  );
  sessionStorage.clear();
  if (options.persisted !== undefined && options.persisted !== null) {
    sessionStorage.setItem("activeTab", options.persisted);
  }
  loadedFixture = episodeFixture(options.robotType, options.codebaseVersion);
  let view!: RenderResult;
  await act(async () => {
    view = render(
      <EpisodeViewer org="local" dataset="pnp_trash" episodeId={0} />,
    );
    await Promise.resolve();
    await Promise.resolve();
  });
  return view;
}

describe("EpisodeViewer initial tab wiring", () => {
  test("the documented annotations URL initializes the real viewer on task_index", async () => {
    const view = await renderViewer({ query: "?tab=annotations" });
    expect(await view.findByTestId("task-index-workspace-probe")).toBeTruthy();
  });

  test("a repeated tab query falls back to the persisted tab", async () => {
    const view = await renderViewer({
      query: "?tab=annotations&tab=doctor",
      persisted: "statistics",
    });
    expect(await view.findByTestId("statistics-probe")).toBeTruthy();
  });

  for (const authority of ["query", "persisted"] as const) {
    test(`${authority} urdf renders only when the loaded dataset supports it`, async () => {
      const eligible = await renderViewer({
        query: authority === "query" ? "?tab=urdf" : "",
        persisted: authority === "persisted" ? "urdf" : null,
        robotType: "unitree_g1",
        codebaseVersion: "v3.0",
      });
      expect(await eligible.findByTestId("urdf-probe")).toBeTruthy();
      eligible.unmount();

      const ineligible = await renderViewer({
        query: authority === "query" ? "?tab=urdf" : "",
        persisted: authority === "persisted" ? "urdf" : null,
        robotType: "unitree_g1",
        codebaseVersion: "v2.1",
      });
      expect(await ineligible.findByText("fixture task")).toBeTruthy();
      expect(ineligible.queryByTestId("urdf-probe")).toBeNull();
    });
  }
});

test("rounded URL bookmarks do not seek the current episode again", async () => {
  seekSpy.mockClear();
  const view = await renderViewer({ query: "?t=0.72" });
  expect(seekSpy).toHaveBeenCalledWith(0.72);
  seekSpy.mockClear();
  window.history.replaceState({}, "", "/local/pnp_trash/episode_0?t=0");
  await act(async () => {
    view.rerender(
      <EpisodeViewer org="local" dataset="pnp_trash" episodeId={0} />,
    );
  });
  expect(seekSpy).not.toHaveBeenCalled();
  view.unmount();
});
