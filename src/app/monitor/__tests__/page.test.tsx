import React from "react";
import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import {
  act,
  fireEvent,
  render,
  waitFor,
  within,
} from "@testing-library/react";
import * as client from "../../../utils/monitorClient";
import MonitorPage from "../page";

const KEY = "lerobot-monitor:selected-runs:v1";
function dataset(id = "source", parents: string[] = []): client.MonitorDataset {
  return {
    id,
    name: id,
    path: `/data/${id}`,
    canonical_path: `/data/${id}`,
    state: "Ready",
    collected: 12,
    reported_collected: 12,
    episode_lengths: {},
    parent_ids: parents,
    child_ids: [],
    provenance: {
      status: parents.length ? "confirmed" : "unknown",
      sources: parents.map((x) => `/data/${x}`),
    },
    default_run_id: "one",
    diagnostics: [],
    exports: [],
    publications: [],
    runs: ["one", "two"].map((run_id) => ({
      run_id,
      updated_at: null,
      current_repo_id: "local/exact-alias",
      first_retained_episode: 3,
      detail_signature: `${run_id}-sig`,
      metrics: {
        imported: 10,
        accepted: 8,
        rejected: 2,
        pending: 0,
        retained: 8,
        reviewed: 8,
        decision_rate: 1,
        review_rate: 1,
        new_episode_ids: [10, 11],
        missing_episode_ids: [],
        changed_length_ids: [],
        accepted_frames: 100,
        accepted_seconds: 2,
        counts_complete: true,
      },
      findings: {
        unresolved: 0,
        accepted_advisory: 2,
        generation_failed: 0,
        unreadable: 0,
      },
      exclusions: { episodes: 0, frames: 0 },
      job: null,
      freshness: {
        publication_state: "not_published",
        metadata_only: true,
        source_changed: true,
        local_changes: null,
        verifiable: false,
      },
      diagnostics: [],
    })),
  };
}
function summary(datasets = [dataset()]): client.MonitorSummary {
  return {
    configured: true,
    root: "/data",
    scanned_at: "2026-09-16T01:00:00Z",
    updating: false,
    diagnostics: [],
    datasets,
  };
}
function detail(
  run_id = "one",
  signature = `${run_id}-sig`,
): client.MonitorDetail {
  return {
    dataset_id: "source",
    run_id,
    signature,
    scanned_at: null,
    updating: false,
    metrics: null,
    findings: null,
    exclusions: null,
    publications: [],
    exports: [],
    diagnostics: [
      { code: "detail", message: `Loaded ${signature}`, severity: "info" },
    ],
    prompts: null,
  };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
async function toggle(container: HTMLElement, open = true) {
  await act(async () => {
    const el = container.querySelector("details")!;
    el.open = open;
    fireEvent(el, new window.Event("toggle"));
  });
}
const nativeTimeout = globalThis.setTimeout;
let visible = true;
beforeEach(() => {
  visible = true;
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => (visible ? "visible" : "hidden"),
  });
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(summary());
  spyOn(client, "fetchMonitorDetail").mockImplementation(async (_id, run) =>
    detail(run),
  );
  spyOn(client, "checkMonitorPublication").mockResolvedValue({
    status: "match",
    checked_at: "2026-09-16T02:00:00Z",
    current_commit: "commit",
    message: null,
  });
});
afterEach(() => {
  globalThis.setTimeout = nativeTimeout;
  mock.restore();
  delete (document as unknown as Record<string, unknown>).visibilityState;
  window.history.replaceState(null, "", "/");
});
async function ready() {
  const ui = render(<MonitorPage />);
  await waitFor(() => expect(ui.getByLabelText("source")).toBeTruthy());
  return ui;
}

test("restores valid selections, preserves refresh selection and falls back when a run disappears", async () => {
  localStorage.setItem(KEY, JSON.stringify({ source: "two" }));
  const ui = await ready();
  expect(ui.getByText(/Selected run: two/)).toBeTruthy();
  await toggle(ui.container);
  await waitFor(() =>
    expect(client.fetchMonitorDetail).toHaveBeenCalledWith(
      "source",
      "two",
      expect.any(AbortSignal),
    ),
  );
  fireEvent.change(ui.getByLabelText("Annotation run for source"), {
    target: { value: "one" },
  });
  await waitFor(() =>
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual({ source: "one" }),
  );
  fireEvent.click(ui.getByRole("button", { name: "Refresh" }));
  await waitFor(() =>
    expect(client.fetchMonitorSummary).toHaveBeenCalledTimes(2),
  );
  expect(ui.getByText(/Selected run: one/)).toBeTruthy();
  const next = summary();
  next.datasets[0].runs = [next.datasets[0].runs[1]];
  next.datasets[0].default_run_id = "two";
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(next);
  fireEvent.click(ui.getByRole("button", { name: "Refresh" }));
  await waitFor(() => expect(ui.getByText(/Selected run: two/)).toBeTruthy());
});

test("ignores late detail responses after run and signature changes and aborts collapse", async () => {
  const old = deferred<client.MonitorDetail>();
  spyOn(client, "fetchMonitorDetail").mockReturnValueOnce(old.promise);
  const ui = await ready();
  await toggle(ui.container);
  await waitFor(() =>
    expect(client.fetchMonitorDetail).toHaveBeenCalledTimes(1),
  );
  const firstSignal = (client.fetchMonitorDetail as ReturnType<typeof mock>)
    .mock.calls[0][2] as AbortSignal;
  fireEvent.change(ui.getByLabelText("Annotation run for source"), {
    target: { value: "two" },
  });
  await waitFor(() => expect(ui.getByText(/Loaded two-sig/)).toBeTruthy());
  expect(firstSignal.aborted).toBe(true);
  await act(async () => old.resolve(detail()));
  expect(ui.queryByText(/Loaded one-sig/)).toBeNull();
  const pending = deferred<client.MonitorDetail>();
  spyOn(client, "fetchMonitorDetail").mockReturnValueOnce(pending.promise);
  const next = summary();
  next.datasets[0].runs[1].detail_signature = "changed";
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(next);
  fireEvent.click(ui.getByRole("button", { name: "Refresh" }));
  await waitFor(() =>
    expect(client.fetchMonitorDetail).toHaveBeenCalledTimes(3),
  );
  expect(ui.queryByText(/Loaded two-sig/)).toBeNull();
  const signal = (client.fetchMonitorDetail as ReturnType<typeof mock>).mock
    .calls[2][2] as AbortSignal;
  await toggle(ui.container, false);
  expect(signal.aborted).toBe(true);
  await act(async () => pending.resolve(detail("two", "changed")));
  expect(ui.queryByText(/Loaded changed/)).toBeNull();
});

test("polls only after completion, never overlaps, pauses hidden and refreshes visible without automatic detail or HF work", async () => {
  const callbacks: (() => void)[] = [];
  const native = globalThis.setTimeout;
  globalThis.setTimeout = ((fn: () => void, ms: number, ...args: unknown[]) =>
    ms === 30000
      ? (callbacks.push(fn), 98765)
      : native(fn, ms, ...args)) as typeof setTimeout;
  const pending = deferred<client.MonitorSummary>();
  spyOn(client, "fetchMonitorSummary").mockReturnValueOnce(pending.promise);
  const ui = render(<MonitorPage />);
  expect(callbacks.length).toBe(0);
  act(() => {
    visible = false;
    document.dispatchEvent(new window.Event("visibilitychange"));
    visible = true;
    document.dispatchEvent(new window.Event("visibilitychange"));
  });
  expect(client.fetchMonitorSummary).toHaveBeenCalledTimes(1);
  await act(async () => pending.resolve(summary()));
  await waitFor(() =>
    expect(client.fetchMonitorSummary).toHaveBeenCalledTimes(2),
  );
  expect(callbacks.length).toBe(1);
  await act(async () => callbacks.pop()!());
  expect(client.fetchMonitorSummary).toHaveBeenCalledTimes(3);
  act(() => {
    visible = false;
    document.dispatchEvent(new window.Event("visibilitychange"));
  });
  await act(async () => callbacks.pop()!());
  expect(client.fetchMonitorSummary).toHaveBeenCalledTimes(3);
  act(() => {
    visible = true;
    document.dispatchEvent(new window.Event("visibilitychange"));
  });
  await waitFor(() =>
    expect(client.fetchMonitorSummary).toHaveBeenCalledTimes(4),
  );
  expect(client.fetchMonitorDetail).not.toHaveBeenCalled();
  expect(client.checkMonitorPublication).not.toHaveBeenCalled();
  ui.unmount();
});

test("unmount aborts pending summary and detail requests", async () => {
  spyOn(client, "fetchMonitorDetail").mockReturnValue(new Promise(() => {}));
  const ui = await ready();
  await toggle(ui.container);
  await waitFor(() => expect(client.fetchMonitorDetail).toHaveBeenCalled());
  const detailSignal = (client.fetchMonitorDetail as ReturnType<typeof mock>)
    .mock.calls[0][2] as AbortSignal;
  spyOn(client, "fetchMonitorSummary").mockReturnValue(new Promise(() => {}));
  fireEvent.click(ui.getByRole("button", { name: "Refresh" }));
  const signal = (
    client.fetchMonitorSummary as ReturnType<typeof mock>
  ).mock.calls.at(-1)![1] as AbortSignal;
  ui.unmount();
  expect(signal.aborted).toBe(true);
  expect(detailSignal.aborted).toBe(true);
});

test("manual invalidation failure keeps last good snapshot and time", async () => {
  const ui = await ready();
  spyOn(client, "fetchMonitorSummary").mockRejectedValue(
    new Error("Backend unavailable"),
  );
  fireEvent.click(ui.getByRole("button", { name: "Refresh" }));
  await waitFor(() =>
    expect(ui.getByRole("alert").textContent).toContain("Backend unavailable"),
  );
  expect(client.fetchMonitorSummary).toHaveBeenLastCalledWith(
    true,
    expect.any(AbortSignal),
  );
  expect(ui.getByLabelText("source")).toBeTruthy();
  expect(ui.getByText(/2026-09-16T01:00:00Z/)).toBeTruthy();
});

test("storage denial does not block monitoring", async () => {
  spyOn(window.localStorage, "getItem").mockImplementation(() => {
    throw new Error("Denied");
  });
  spyOn(window.localStorage, "setItem").mockImplementation(() => {
    throw new Error("Denied");
  });
  const ui = await ready();
  await toggle(ui.container);
  fireEvent.change(ui.getByLabelText("Annotation run for source"), {
    target: { value: "two" },
  });
  await waitFor(() => expect(ui.getByText(/Selected run: two/)).toBeTruthy());
});

test("groups confirmed single-parent derivatives, keeps merges independent, and counts overlapping groups", async () => {
  const source = dataset(),
    child = dataset("child", ["source"]),
    merge = dataset("merge", ["source", "child"]);
  child.runs[0].metrics.reviewed = 3;
  child.runs[0].metrics.review_rate = 3 / 8;
  merge.runs = [];
  merge.default_run_id = null;
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(
    summary([source, child, merge]),
  );
  const ui = await ready();
  const groups = ui.getAllByRole("group", { name: /Collection group/ });
  expect(groups.length).toBe(2);
  expect(within(groups[0]).getByLabelText("child")).toBeTruthy();
  expect(
    within(ui.getByLabelText("Collection groups")).getByText("2"),
  ).toBeTruthy();
  expect(
    within(ui.getByLabelText("Groups awaiting review")).getByText("1"),
  ).toBeTruthy();
  expect(
    within(ui.getByLabelText("Groups with new episodes")).getByText("1"),
  ).toBeTruthy();
  expect(ui.getAllByText("8 / 8 reviewed").length).toBe(1);
  fireEvent.change(ui.getByLabelText("Status filter"), {
    target: { value: "unimported" },
  });
  expect(!!ui.queryByLabelText("source")).toBe(false);
  expect(ui.getByLabelText("merge")).toBeTruthy();
  await toggle(ui.container);
  fireEvent.click(ui.getByRole("link", { name: "source" }));
  await waitFor(() => expect(ui.getByLabelText("source")).toBeTruthy());
});

test("publication search and explicit HF checks survive row unmount and concurrent old summary", async () => {
  const data = dataset();
  data.publications = [
    {
      id: "pub",
      repo_id: "org/published",
      revision: "main",
      commit: "commit",
      url: "https://huggingface.co/datasets/org/published",
      export_path: null,
      export_available: false,
      format: null,
      instruction_mode: null,
      exported_frames: null,
      manifest_sha256: null,
      linked_run_id: "one",
      remote_check: null,
    },
  ];
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(summary([data]));
  const ui = await ready();
  await toggle(ui.container);
  await waitFor(() => expect(client.fetchMonitorDetail).toHaveBeenCalled());
  const pending = deferred<client.MonitorSummary>();
  spyOn(client, "fetchMonitorSummary").mockReturnValueOnce(pending.promise);
  fireEvent.click(ui.getByRole("button", { name: "Refresh" }));
  fireEvent.click(ui.getByRole("button", { name: "Check HF" }));
  await waitFor(() =>
    expect(ui.getByText("Matches recorded commit")).toBeTruthy(),
  );
  await act(async () => pending.resolve(summary([data])));
  fireEvent.input(ui.getByLabelText("Search datasets"), {
    target: { value: "nothing" },
  });
  expect(!!ui.queryByLabelText("source")).toBe(false);
  fireEvent.input(ui.getByLabelText("Search datasets"), {
    target: { value: "org/published" },
  });
  await toggle(ui.container);
  expect(ui.getByText("Matches recorded commit")).toBeTruthy();
  expect(client.checkMonitorPublication).toHaveBeenCalledTimes(1);
});

test("unconfigured root and empty root have explicit states", async () => {
  spyOn(client, "fetchMonitorSummary").mockResolvedValue({
    ...summary([]),
    configured: false,
    root: null,
  });
  const ui = render(<MonitorPage />);
  await waitFor(() =>
    expect(ui.getByText(/LEROBOT_MONITOR_ROOT/)).toBeTruthy(),
  );
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(summary([]));
  fireEvent.click(ui.getByRole("button", { name: "Refresh" }));
  await waitFor(() => expect(ui.getByText(/No datasets found/)).toBeTruthy());
});

test("malformed preferences fall back to the server default without persisting unknown run ids", async () => {
  localStorage.setItem(KEY, JSON.stringify({ source: "removed", other: 7 }));
  const ui = await ready();
  expect(ui.getByText(/Selected run: one/)).toBeTruthy();
  expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual({ source: "one" });
});

test("attention, reviewing and published filters use selected-run evidence without source growth diluting review", async () => {
  const clean = dataset("clean");
  clean.runs.forEach((run) => {
    run.metrics.new_episode_ids = [];
    run.freshness.source_changed = false;
  });
  const reviewing = dataset("reviewing");
  reviewing.runs[0].metrics.reviewed = 2;
  const published = dataset("published");
  published.runs = structuredClone(clean.runs);
  published.publications = [
    {
      id: "pub",
      repo_id: "org/recorded",
      revision: null,
      commit: null,
      url: null,
      export_path: null,
      export_available: false,
      format: null,
      instruction_mode: null,
      exported_frames: null,
      manifest_sha256: null,
      linked_run_id: null,
      remote_check: null,
    },
  ];
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(
    summary([dataset(), clean, reviewing, published]),
  );
  const ui = await ready();
  expect(
    within(ui.getByLabelText("source")).getByText("8 / 8 reviewed"),
  ).toBeTruthy();
  fireEvent.change(ui.getByLabelText("Status filter"), {
    target: { value: "reviewing" },
  });
  expect(!!ui.queryByLabelText("source")).toBe(false);
  expect(!!ui.queryByLabelText("reviewing")).toBe(true);
  fireEvent.change(ui.getByLabelText("Status filter"), {
    target: { value: "attention" },
  });
  expect(!!ui.queryByLabelText("clean")).toBe(false);
  expect(!!ui.queryByLabelText("source")).toBe(true);
  fireEvent.change(ui.getByLabelText("Status filter"), {
    target: { value: "published" },
  });
  expect(!!ui.queryByLabelText("published")).toBe(true);
  expect(!!ui.queryByLabelText("reviewing")).toBe(false);
});

test("cyclic and conflicting provenance never duplicates or hides datasets", async () => {
  const a = dataset("a", ["b"]),
    b = dataset("b", ["a"]),
    c = dataset("c", ["a"]);
  c.provenance.status = "conflict";
  spyOn(client, "fetchMonitorSummary").mockResolvedValue(summary([a, b, c]));
  const ui = render(<MonitorPage />);
  await waitFor(() =>
    expect(ui.getAllByRole("group", { name: /Collection group/ }).length).toBe(
      3,
    ),
  );
  for (const id of ["a", "b", "c"])
    expect(ui.getAllByLabelText(id).length).toBe(1);
});

test("backend configuration failures explain the backend connection without claiming an empty root", async () => {
  spyOn(client, "fetchMonitorSummary").mockRejectedValue(
    new Error("Annotation backend is not configured."),
  );
  const ui = render(<MonitorPage />);
  await waitFor(() =>
    expect(ui.getByRole("alert").textContent).toContain(
      "NEXT_PUBLIC_ANNOTATE_BACKEND_URL",
    ),
  );
  expect(!!ui.queryByText(/No datasets found/)).toBe(false);
});
