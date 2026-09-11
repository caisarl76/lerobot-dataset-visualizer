import React from "react";
import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { act, fireEvent, render, waitFor } from "@testing-library/react";
import * as client from "../../utils/annotationsClient";
import {
  AnnotationsProvider,
  useAnnotations,
} from "../../context/annotations-context";
import { TimeProvider, useTime } from "../../context/time-context";
import { EpisodeExclusions } from "../episode-exclusions";

type Interval = { start_frame: number; end_frame: number };
const interval = (start_frame: number, end_frame: number): Interval => ({
  start_frame,
  end_frame,
});
const makeRun = (intervals: Interval[] = []): client.WorkflowRun => ({
  run_id: "r",
  revision: 3,
  publication_state: "draft",
  current_job_id: null,
  task_prompt: "",
  subtask_prompts: [],
  metrics: {},
  episodes: {
    "2": {
      generation_status: "done",
      issues: [],
      decision: "pending",
      excluded_intervals: intervals,
    },
    "3": {
      generation_status: "done",
      issues: [],
      decision: "pending",
      excluded_intervals: [],
    },
  },
});
let serverRun: client.WorkflowRun;

function Fixture({ episode, count }: { episode: number; count: number }) {
  const { setEpisode, addAtom } = useAnnotations();
  const { currentTime, seek, isPlaying, setIsPlaying, externalSeekVersion } =
    useTime();
  React.useEffect(
    () =>
      setEpisode(
        episode,
        { repoId: "local/demo" },
        [],
        Array.from({ length: count }, (_, i) => i / 50),
      ),
    [episode, count, setEpisode],
  );
  return (
    <>
      <button
        onClick={() =>
          addAtom({
            style: "subtask",
            role: "assistant",
            content: "edited",
            timestamp: 0,
            camera: null,
            tool_calls: null,
          })
        }
      >
        Edit annotation
      </button>
      <button onClick={() => setIsPlaying(true)}>Play</button>
      <label>
        Source clock
        <input
          aria-label="Source clock"
          type="number"
          value={currentTime}
          onInput={(event) =>
            seek(Number(event.currentTarget.value), "external")
          }
        />
      </label>
      <span data-testid="playing">{String(isPlaying)}</span>
      <span data-testid="seek-version">{externalSeekVersion}</span>
      <EpisodeExclusions duration={(count - 1) / 50} />
    </>
  );
}
const content = (episode = 2, count = 100) => (
  <AnnotationsProvider>
    <TimeProvider duration={(count - 1) / 50}>
      <Fixture episode={episode} count={count} />
    </TimeProvider>
  </AnnotationsProvider>
);
const ui = (episode = 2, count = 100) => render(content(episode, count));
type UI = ReturnType<typeof ui>;
const addButton = (view: UI) =>
  view.getByRole("button", { name: "Exclude interval" }) as HTMLButtonElement;
const setRange = (view: UI, start: number, end: number) => {
  fireEvent.input(view.getByLabelText("Start seconds"), {
    target: { value: String(start) },
  });
  fireEvent.input(view.getByLabelText("End seconds"), {
    target: { value: String(end) },
  });
};
const loaded = async (view: UI) => {
  await waitFor(() =>
    expect(
      (view.getByLabelText("End seconds") as HTMLInputElement).disabled,
    ).toBe(false),
  );
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  serverRun = makeRun();
  spyOn(client, "isAnnotateBackendEnabled").mockReturnValue(false);
  spyOn(client, "fetchWorkflow").mockImplementation(async () =>
    structuredClone(serverRun),
  );
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "old-job",
    status: "completed",
  });
  spyOn(client, "postWorkflowExclusions").mockImplementation(
    async (_alias, body) => {
      serverRun = {
        ...serverRun,
        revision: serverRun.revision + 1,
        episodes: {
          ...serverRun.episodes,
          [body.episode_index]: {
            ...serverRun.episodes[body.episode_index],
            excluded_intervals: body.excluded_intervals,
          },
        },
      };
      return structuredClone(serverRun);
    },
  );
});
afterEach(() => mock.restore());

test("0..1 seconds at 50 FPS persists exactly frames [0, 50), then shows retained duration", async () => {
  const view = ui();
  await loaded(view);
  expect(addButton(view).disabled).toBe(true);
  setRange(view, 0, 1);
  expect(addButton(view).disabled).toBe(false);
  fireEvent.click(addButton(view));
  await waitFor(() =>
    expect(client.postWorkflowExclusions).toHaveBeenCalledWith("demo", {
      episode_index: 2,
      expected_revision: 3,
      excluded_intervals: [interval(0, 50)],
    }),
  );
  await view.findByRole("button", { name: "Remove exclusion 1" });
  expect(view.getByText(/Retained: 1.000 s \(50 frames\)/)).toBeTruthy();
});

test("tail end uses actual frame count beyond the last frame timestamp", async () => {
  const view = ui(2, 101);
  await loaded(view);
  setRange(view, 1, 2.02);
  expect(addButton(view).disabled).toBe(false);
  fireEvent.click(addButton(view));
  await waitFor(() =>
    expect(client.postWorkflowExclusions).toHaveBeenCalledWith(
      "demo",
      expect.objectContaining({ excluded_intervals: [interval(50, 101)] }),
    ),
  );
});

test("empty, reversed, out-of-bounds, and entire-episode ranges cannot submit", async () => {
  const view = ui();
  await loaded(view);
  for (const [start, end] of [
    [0, 0],
    [1, 0.5],
    [-1, 1],
    [0, 3],
    [0, 2],
  ]) {
    setRange(view, start, end);
    expect(addButton(view).disabled).toBe(true);
  }
  expect(view.getByRole("alert").textContent).toContain(
    "At least one source frame must remain",
  );
  expect(client.postWorkflowExclusions).not.toHaveBeenCalled();
});

test("the union of saved and selected exclusions must retain a frame", async () => {
  serverRun = makeRun([interval(0, 50)]);
  const view = ui();
  await loaded(view);
  setRange(view, 1, 2);
  expect(addButton(view).disabled).toBe(true);
  expect(view.getByRole("alert").textContent).toContain(
    "At least one source frame must remain",
  );
  expect(client.postWorkflowExclusions).not.toHaveBeenCalled();
});

test("overlapping source selections merge before submission", async () => {
  serverRun = makeRun([interval(10, 30)]);
  const view = ui();
  await loaded(view);
  setRange(view, 0.4, 0.8);
  fireEvent.click(addButton(view));
  await waitFor(() =>
    expect(client.postWorkflowExclusions).toHaveBeenCalledWith(
      "demo",
      expect.objectContaining({ excluded_intervals: [interval(10, 40)] }),
    ),
  );
});

test("failed persistence preserves saved cuts and displays the failure", async () => {
  serverRun = makeRun([interval(0, 10)]);
  const pending = deferred<client.WorkflowRun>();
  spyOn(client, "postWorkflowExclusions").mockImplementationOnce(
    () => pending.promise,
  );
  const view = ui();
  await loaded(view);
  setRange(view, 0.4, 0.8);
  fireEvent.click(addButton(view));
  expect(view.queryByRole("button", { name: "Remove exclusion 2" })).toBeNull();
  await act(async () => pending.reject(new Error("Revision conflict")));
  await view.findByText(/Revision conflict/);
  expect(view.getByRole("button", { name: "Remove exclusion 1" })).toBeTruthy();
  expect(view.queryByRole("button", { name: "Remove exclusion 2" })).toBeNull();
  expect((view.getByLabelText("End seconds") as HTMLInputElement).value).toBe(
    "0.8",
  );
});

test("saved bars select without deleting; Remove and Clear persist explicit changes", async () => {
  serverRun = makeRun([interval(0, 10), interval(20, 30)]);
  const view = ui();
  await loaded(view);
  fireEvent.click(view.getByRole("button", { name: "Select exclusion 2" }));
  expect(client.postWorkflowExclusions).not.toHaveBeenCalled();
  expect(
    view
      .getByRole("button", { name: "Select exclusion 2" })
      .getAttribute("aria-pressed"),
  ).toBe("true");
  fireEvent.click(view.getByRole("button", { name: "Remove exclusion 2" }));
  await waitFor(() =>
    expect(client.postWorkflowExclusions).toHaveBeenCalledWith(
      "demo",
      expect.objectContaining({ excluded_intervals: [interval(0, 10)] }),
    ),
  );
  await loaded(view);
  fireEvent.click(view.getByRole("button", { name: "Clear" }));
  await waitFor(() =>
    expect(client.postWorkflowExclusions).toHaveBeenLastCalledWith(
      "demo",
      expect.objectContaining({ excluded_intervals: [] }),
    ),
  );
  await waitFor(() =>
    expect(
      view.queryByRole("button", { name: "Remove exclusion 1" }),
    ).toBeNull(),
  );
});

test("drag selection snaps source boundaries and waits for explicit submit", async () => {
  const view = ui();
  await loaded(view);
  const track = view.getByTestId("exclusion-track");
  spyOn(track, "getBoundingClientRect").mockReturnValue({
    left: 100,
    right: 500,
    top: 0,
    bottom: 20,
    width: 400,
    height: 20,
    x: 100,
    y: 0,
    toJSON: () => ({}),
  });
  fireEvent.pointerDown(track, { pointerId: 1, button: 0, clientX: 100 });
  fireEvent.pointerMove(track, { pointerId: 1, clientX: 300 });
  fireEvent.pointerUp(track, { pointerId: 1, button: 0, clientX: 300 });
  expect((view.getByLabelText("Start seconds") as HTMLInputElement).value).toBe(
    "0",
  );
  expect((view.getByLabelText("End seconds") as HTMLInputElement).value).toBe(
    "1",
  );
  expect(client.postWorkflowExclusions).not.toHaveBeenCalled();
  fireEvent.click(addButton(view));
  await waitFor(() =>
    expect(client.postWorkflowExclusions).toHaveBeenCalledWith(
      "demo",
      expect.objectContaining({ excluded_intervals: [interval(0, 50)] }),
    ),
  );
});

test("exclude lane shows the synchronized current time playhead", async () => {
  const view = ui();
  await loaded(view);
  const playhead = view.getByTestId("exclusion-playhead");
  expect(playhead.textContent).toBe("0.000 s");
  fireEvent.input(view.getByLabelText("Source clock"), {
    target: { value: "1.25" },
  });
  await waitFor(() => expect(playhead.textContent).toBe("1.250 s"));
  expect(playhead.className).toContain("middle");
});

test("preview skips prefix and middle intervals while retaining the source clock", async () => {
  serverRun = makeRun([interval(0, 10), interval(30, 40)]);
  const view = ui();
  await loaded(view);
  fireEvent.click(view.getByRole("checkbox", { name: "Preview clipping" }));
  fireEvent.click(view.getByRole("button", { name: "Play" }));
  await waitFor(() =>
    expect(
      (view.getByLabelText("Source clock") as HTMLInputElement).value,
    ).toBe("0.2"),
  );
  fireEvent.input(view.getByLabelText("Source clock"), {
    target: { value: ".65" },
  });
  await waitFor(() =>
    expect(
      (view.getByLabelText("Source clock") as HTMLInputElement).value,
    ).toBe("0.8"),
  );
  expect(view.getByTestId("playing").textContent).toBe("true");
});

test("preview stops at the final retained frame instead of repeatedly seeking the excluded tail", async () => {
  serverRun = makeRun([interval(75, 100)]);
  const view = ui();
  await loaded(view);
  fireEvent.click(view.getByRole("checkbox", { name: "Preview clipping" }));
  fireEvent.input(view.getByLabelText("Source clock"), {
    target: { value: "1.6" },
  });
  fireEvent.click(view.getByRole("button", { name: "Play" }));
  await waitFor(() =>
    expect(view.getByTestId("playing").textContent).toBe("false"),
  );
  expect((view.getByLabelText("Source clock") as HTMLInputElement).value).toBe(
    "1.48",
  );
  const version = view.getByTestId("seek-version").textContent;
  await act(async () => {});
  expect(view.getByTestId("seek-version").textContent).toBe(version);
});

test("workflow refresh uses the new revision after another control changes the run", async () => {
  const view = ui();
  await loaded(view);
  serverRun = { ...serverRun, revision: 9 };
  act(() =>
    window.dispatchEvent(new window.Event("annotation-workflow-changed")),
  );
  await loaded(view);
  setRange(view, 0, 1);
  fireEvent.click(addButton(view));
  await waitFor(() =>
    expect(client.postWorkflowExclusions).toHaveBeenCalledWith(
      "demo",
      expect.objectContaining({ expected_revision: 9 }),
    ),
  );
});

test("active job status blocks changes but a completed job ID does not", async () => {
  serverRun = { ...serverRun, current_job_id: "old-job" };
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "old-job",
    status: "running",
  });
  const view = ui();
  await view.findByText("Workflow job running…");
  expect(
    (view.getByLabelText("End seconds") as HTMLInputElement).disabled,
  ).toBe(true);
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "old-job",
    status: "completed",
  });
  act(() =>
    window.dispatchEvent(new window.Event("annotation-workflow-changed")),
  );
  await loaded(view);
  setRange(view, 0, 1);
  expect(addButton(view).disabled).toBe(false);
});

test("dirty annotation edits and deletion decisions disable clipping changes", async () => {
  const view = ui();
  await loaded(view);
  setRange(view, 0, 1);
  fireEvent.click(view.getByRole("button", { name: "Edit annotation" }));
  expect(addButton(view).disabled).toBe(true);
  view.rerender(content(3));
  serverRun.episodes["3"].decision = "delete";
  act(() =>
    window.dispatchEvent(new window.Event("annotation-workflow-changed")),
  );
  await view.findByText("This episode is marked for deletion.");
  expect(
    (view.getByLabelText("End seconds") as HTMLInputElement).disabled,
  ).toBe(true);
});

for (const result of ["resolve", "reject"] as const) {
  test(`stale save ${result} cannot update another episode or clear its pending save`, async () => {
    const first = deferred<client.WorkflowRun>();
    const second = deferred<client.WorkflowRun>();
    spyOn(client, "postWorkflowExclusions")
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    const view = ui();
    await loaded(view);
    setRange(view, 0, 1);
    fireEvent.click(addButton(view));
    view.rerender(content(3));
    await loaded(view);
    setRange(view, 0.2, 0.4);
    fireEvent.click(addButton(view));
    await act(async () => {
      if (result === "resolve") first.resolve(makeRun([interval(0, 50)]));
      else first.reject(new Error("Old episode failure"));
    });
    expect(view.queryByText(/Old episode failure/)).toBeNull();
    expect(
      view.queryByRole("button", { name: "Remove exclusion 1" }),
    ).toBeNull();
    expect(view.getByRole("status").textContent).toBe("Saving exclusions…");
    expect(addButton(view).disabled).toBe(true);
    await act(async () => second.resolve(serverRun));
  });
}

test("late workflow load is ignored after navigation", async () => {
  const pending = deferred<client.WorkflowRun>();
  spyOn(client, "fetchWorkflow").mockImplementationOnce(() => pending.promise);
  const view = ui();
  view.rerender(content(3));
  await loaded(view);
  await act(async () => pending.resolve(makeRun([interval(0, 50)])));
  expect(view.queryByRole("button", { name: "Remove exclusion 1" })).toBeNull();
});

test("successful persistence notifies workflow and clipping listeners", async () => {
  const workflowChanged = mock(() => {});
  const clippingChanged = mock(() => {});
  window.addEventListener("annotation-workflow-changed", workflowChanged);
  window.addEventListener("annotation-clipping-changed", clippingChanged);
  try {
    const view = ui();
    await loaded(view);
    setRange(view, 0, 1);
    fireEvent.click(addButton(view));
    await waitFor(() => expect(clippingChanged).toHaveBeenCalledTimes(1));
    expect(workflowChanged).toHaveBeenCalledTimes(1);
    expect(view.queryByRole("alert")).toBeNull();
  } finally {
    window.removeEventListener("annotation-workflow-changed", workflowChanged);
    window.removeEventListener("annotation-clipping-changed", clippingChanged);
  }
});

test("late saves after unmount do not notify a different workspace", async () => {
  const pending = deferred<client.WorkflowRun>();
  spyOn(client, "postWorkflowExclusions").mockImplementationOnce(
    () => pending.promise,
  );
  const changed = mock(() => {});
  window.addEventListener("annotation-clipping-changed", changed);
  try {
    const view = ui();
    await loaded(view);
    setRange(view, 0, 1);
    fireEvent.click(addButton(view));
    view.unmount();
    await act(async () => pending.resolve(makeRun([interval(0, 50)])));
    expect(changed).not.toHaveBeenCalled();
  } finally {
    window.removeEventListener("annotation-clipping-changed", changed);
  }
});

test("failed workflow refresh keeps existing cuts visible and disables mutation", async () => {
  serverRun = makeRun([interval(10, 20)]);
  const view = ui();
  await loaded(view);
  spyOn(client, "fetchWorkflow").mockRejectedValueOnce(
    new Error("Workflow offline"),
  );
  act(() =>
    window.dispatchEvent(new window.Event("annotation-workflow-changed")),
  );
  await view.findByText(/Workflow offline/);
  expect(view.getByRole("button", { name: "Remove exclusion 1" })).toBeTruthy();
  expect(
    (view.getByLabelText("End seconds") as HTMLInputElement).disabled,
  ).toBe(true);
});
