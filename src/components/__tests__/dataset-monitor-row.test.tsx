import React from "react";
import { afterEach, expect, mock, spyOn, test } from "bun:test";
import {
  act,
  fireEvent,
  render,
  waitFor,
  within,
} from "@testing-library/react";
import {
  DatasetMonitorRow,
  type DatasetMonitorRowProps,
} from "../dataset-monitor-row";
import type {
  MonitorDataset,
  MonitorDetail,
  RemoteCheck,
} from "../../utils/monitorClient";

afterEach(() => mock.restore());
function fixture(): DatasetMonitorRowProps {
  const dataset: MonitorDataset = {
    id: "folder",
    name: "pnp_table_260909",
    path: "/data/pnp_table_260909",
    canonical_path: "/data/pnp_table_260909",
    state: "Ready",
    collected: 87,
    reported_collected: 87,
    episode_lengths: {},
    parent_ids: [],
    child_ids: [],
    provenance: { status: "unknown", sources: [] },
    default_run_id: "run-one",
    diagnostics: [],
    runs: [
      {
        run_id: "run-one",
        updated_at: "2026-09-16T00:00:00Z",
        current_repo_id: "local/annotation-99d858fff3f3e9ff",
        first_retained_episode: 1,
        detail_signature: "signature",
        metrics: {
          imported: 87,
          accepted: 71,
          rejected: 16,
          pending: 0,
          retained: 71,
          reviewed: 71,
          review_rate: 1,
          decision_rate: 1,
          new_episode_ids: [],
          missing_episode_ids: [],
          changed_length_ids: [],
          accepted_frames: 34594,
          accepted_seconds: 691.88,
          counts_complete: true,
        },
        findings: {
          unresolved: 0,
          accepted_advisory: 66,
          generation_failed: 0,
          unreadable: 0,
        },
        exclusions: { episodes: 70, frames: 10375 },
        job: { job_id: "job", status: "completed" },
        freshness: {
          publication_state: "published",
          metadata_only: true,
          source_changed: false,
          local_changes: false,
          verifiable: true,
        },
        diagnostics: [],
      },
    ],
    publications: [
      {
        id: "publication",
        repo_id: "mncai/G1_Dex3_PickTable",
        revision: "260915",
        commit: "recorded-commit",
        url: "https://huggingface.co/datasets/mncai/G1_Dex3_PickTable/tree/260915",
        export_path: "/exports/pnp_table_260915",
        export_available: true,
        format: "groot_v21",
        instruction_mode: "subtask",
        exported_frames: 34594,
        manifest_sha256: "manifest",
        linked_run_id: "run-one",
        remote_check: null,
      },
    ],
    exports: [
      {
        id: "export",
        run_id: "run-one",
        path: "/exports/pnp_table_260915",
        available: true,
        format: "groot_v21",
        instruction_mode: "subtask",
        frames: 34594,
        seconds: 691.88,
        manifest_sha256: "manifest",
        output_repo_id: "local/export",
      },
    ],
  };
  const detail: MonitorDetail = {
    dataset_id: "folder",
    run_id: "run-one",
    signature: "signature",
    scanned_at: "2026-09-16T00:00:00Z",
    updating: false,
    metrics: dataset.runs[0].metrics,
    findings: dataset.runs[0].findings,
    exclusions: dataset.runs[0].exclusions,
    diagnostics: [],
    publications: [],
    exports: [],
    prompts: {
      eligible_episodes: 71,
      evaluated_episodes: 70,
      unknown_episode_ids: ["42"],
      retained_frames: 100,
      unlabeled_frames: 10,
      ambiguous_frames: 5,
      complete: false,
      rows: [
        {
          text: "pick apple\n",
          frames: 60,
          seconds: 1.2,
          episodes: 3,
          ratio: 0.6,
        },
        {
          text: "Unlabeled",
          frames: 25,
          seconds: 0.5,
          episodes: 2,
          ratio: 0.25,
        },
      ],
    },
  };
  return {
    dataset,
    selectedRunId: "run-one",
    detail,
    loading: false,
    error: null,
    onSelectRun: mock(() => {}),
    onExpand: mock(() => {}),
    onCheckPublication: mock(
      async () =>
        ({
          status: "match",
          checked_at: "2026-09-16T01:00:00Z",
          current_commit: "recorded-commit",
          message: "Matches",
        }) as RemoteCheck,
    ),
  };
}
function expand(container: HTMLElement) {
  const details = container.querySelector("details")!;
  details.open = true;
  fireEvent(details, new window.Event("toggle"));
}
test("closed row exposes selected scope, counts, review and first retained review navigation", () => {
  const props = fixture();
  const ui = render(<DatasetMonitorRow {...props} />);
  expect(ui.getByText("71 / 71 reviewed")).toBeTruthy();
  for (const label of [
    "Accepted",
    "Rejected",
    "Pending",
    "Review progress",
    "Not imported",
    "Collected",
    "Imported",
    "Usable duration",
  ])
    expect(ui.getByText(label)).toBeTruthy();
  expect(ui.getByText(/Selected run: run-one/)).toBeTruthy();
  expect(
    ui.getByRole("link", { name: "Open review" }).getAttribute("href"),
  ).toBe("/local/annotation-99d858fff3f3e9ff/episode_1?tab=annotations");
  expect(props.onCheckPublication).not.toHaveBeenCalled();
});
test("no run has Not imported status and a preparation link without automatic work", () => {
  const props = fixture();
  props.dataset.runs = [];
  props.selectedRunId = null;
  props.detail = null;
  const ui = render(<DatasetMonitorRow {...props} />);
  expect(ui.getAllByText("Not imported").length).toBeGreaterThan(0);
  expect(
    ui.getByRole("link", { name: "Prepare dataset" }).getAttribute("href"),
  ).toBe("/annotate?local_path=%2Fdata%2Fpnp_table_260909");
  expect(ui.queryByRole("link", { name: "Open review" })).toBeNull();
});
test("zero and unknown denominators are distinct from complete review", () => {
  const props = fixture();
  props.dataset.runs[0].metrics = {
    ...props.dataset.runs[0].metrics,
    imported: 0,
    retained: 0,
    reviewed: 0,
    review_rate: null,
    decision_rate: null,
    new_episode_ids: null,
  };
  props.detail = null;
  const ui = render(<DatasetMonitorRow {...props} />);
  expect(ui.getByText(/No retained episodes/)).toBeTruthy();
  expect(ui.getByText(/No episodes/)).toBeTruthy();
  expect(ui.queryByText("100%")).toBeNull();
  props.dataset.runs[0].metrics = {
    ...props.dataset.runs[0].metrics,
    imported: null,
    retained: null,
    reviewed: null,
  };
  ui.rerender(<DatasetMonitorRow {...props} />);
  expect(ui.getAllByText(/Unknown/).length).toBeGreaterThan(0);
  expect(ui.queryByText(/No retained episodes/)).toBeNull();
});
test("expanded details preserve literal prompts, whitespace, buckets and incomplete frame denominator", () => {
  const ui = render(<DatasetMonitorRow {...fixture()} />);
  expand(ui.container);
  const table = ui.getByRole("table", { name: "Prompt distribution" });
  expect(table.textContent).toContain("pick apple\n");
  expect(ui.getByText("pick·apple↵")).toBeTruthy();
  expect(within(table).getByText("Unlabeled")).toBeTruthy();
  expect(within(table).getByText("Unlabeled (bucket)")).toBeTruthy();
  expect(within(table).getByText("Ambiguous (bucket)")).toBeTruthy();
  expect(ui.getByText(/70 of 71 eligible episodes evaluated/)).toBeTruthy();
  expect(ui.getByText(/Unknown episode IDs: 42/)).toBeTruthy();
  expect(ui.getByText(/Denominator: 100 retained frames/)).toBeTruthy();
});
test("loading and detail errors are accessible while whole-folder publication history remains", () => {
  const props = fixture();
  props.loading = true;
  props.error = "Could not load details";
  props.detail = null;
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.getByText("Loading details…").getAttribute("role")).toBe("status");
  expect(ui.getByRole("alert").textContent).toContain("Could not load details");
  expect(
    ui.getByRole("link", { name: /Open HF/ }).getAttribute("href"),
  ).toContain("/tree/260915");
});
test("missing export is unavailable and cannot be copied; old receipt survives local changes", () => {
  const props = fixture();
  props.dataset.exports[0].available = false;
  props.dataset.publications[0].export_available = false;
  props.dataset.runs[0].freshness.local_changes = true;
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.getByText("Unpublished changes")).toBeTruthy();
  expect(ui.getAllByText(/Path unavailable/).length).toBeGreaterThan(0);
  expect(ui.queryByRole("button", { name: /Copy training path/ })).toBeNull();
  expect(ui.getByText("recorded-commit")).toBeTruthy();
});
test("publication history survives a different selected run whose detail has no receipts", () => {
  const props = fixture();
  props.dataset.runs.push({ ...props.dataset.runs[0], run_id: "run-two" });
  props.selectedRunId = "run-two";
  props.detail = {
    ...props.detail!,
    run_id: "run-two",
    publications: [],
    exports: [],
  };
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.getByRole("link", { name: /Open HF/ })).toBeTruthy();
  fireEvent.change(
    ui.getByRole("combobox", { name: "Annotation run for pnp_table_260909" }),
    {
      target: { value: "run-one" },
    },
  );
  expect(props.onSelectRun).toHaveBeenCalledWith("run-one");
  expect(props.onExpand).toHaveBeenCalledWith(true);
});
test("mismatched details never display another run's prompt distribution", () => {
  const props = fixture();
  props.detail!.run_id = "obsolete";
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.queryByRole("table", { name: "Prompt distribution" })).toBeNull();
});
test("missing review alias does not invent a review URL", () => {
  const props = fixture();
  props.dataset.runs[0].current_repo_id = null;
  const ui = render(<DatasetMonitorRow {...props} />);
  expect(ui.queryByRole("link", { name: "Open review" })).toBeNull();
  expect(ui.getByText(/Review alias unavailable/)).toBeTruthy();
});
test("source growth, missing IDs, exclusions and accepted advisories are distinct", () => {
  const props = fixture();
  Object.assign(props.dataset.runs[0].metrics, {
    new_episode_ids: [90, 91],
    missing_episode_ids: [3],
    changed_length_ids: [7],
  });
  props.dataset.runs[0].freshness.source_changed = true;
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.getByText(/Missing source IDs: 3/)).toBeTruthy();
  expect(ui.getByText(/Changed length IDs: 7/)).toBeTruthy();
  expect(ui.getByText(/Accepted advisory findings: 66/)).toBeTruthy();
  expect(ui.getByText(/Unresolved findings: 0/)).toBeTruthy();
  expect(ui.getByText(/Exclusions: 70 episodes/)).toBeTruthy();
});
test("HF verification is explicit, exposes loading, result/time, and retryable errors", async () => {
  const props = fixture();
  let resolve!: (check: RemoteCheck) => void;
  props.onCheckPublication = mock(
    () =>
      new Promise<RemoteCheck>((done) => {
        resolve = done;
      }),
  );
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  fireEvent.click(ui.getByRole("button", { name: "Check HF" }));
  expect(
    ui.getByRole("button", { name: "Checking HF…" }).hasAttribute("disabled"),
  ).toBe(true);
  await act(async () =>
    resolve({
      status: "changed",
      checked_at: "2026-09-16T01:00:00Z",
      current_commit: "new-head",
      message: "Branch advanced",
    }),
  );
  expect(ui.getByText(/Branch advanced\/changed/)).toBeTruthy();
  expect(ui.getByText(/2026-09-16T01:00:00Z/)).toBeTruthy();
  props.onCheckPublication = mock(async () => {
    throw new Error("Network unavailable");
  });
  ui.rerender(<DatasetMonitorRow {...props} />);
  fireEvent.click(ui.getByRole("button", { name: "Check HF" }));
  await waitFor(() =>
    expect(ui.getByRole("alert").textContent).toContain("Network unavailable"),
  );
  expect(ui.getByText("recorded-commit")).toBeTruthy();
});
test("unknown or unsafe HF URLs never produce active links", () => {
  const props = fixture();
  props.dataset.publications[0].url = "javascript:alert(1)";
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.queryByRole("link", { name: /Open HF/ })).toBeNull();
});

test("available training paths copy only on explicit action and use the frozen export path", async () => {
  const clipboard = spyOn(navigator.clipboard, "writeText").mockResolvedValue(
    undefined,
  );
  const props = fixture();
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(clipboard).not.toHaveBeenCalled();
  fireEvent.click(
    ui.getByRole("button", { name: "Copy training path for export export" }),
  );
  await waitFor(() =>
    expect(clipboard).toHaveBeenCalledWith("/exports/pnp_table_260915"),
  );
  expect(ui.getByText("Training path copied.").getAttribute("role")).toBe(
    "status",
  );
  expect(ui.getAllByText(/GR00T v2.1/).length).toBeGreaterThan(0);
  expect(ui.getAllByText(/34,594 frames/).length).toBeGreaterThan(0);
});
test("clipboard errors provide accessible manual-copy guidance", async () => {
  spyOn(navigator.clipboard, "writeText").mockRejectedValue(
    new Error("Denied"),
  );
  const ui = render(<DatasetMonitorRow {...fixture()} />);
  expand(ui.container);
  fireEvent.click(
    ui.getByRole("button", {
      name: "Copy training path for publication publication",
    }),
  );
  await waitFor(() =>
    expect(ui.getByRole("alert").textContent).toContain("Select and copy"),
  );
});
test("detail signature changes prevent stale prompt coverage from being displayed", () => {
  const props = fixture();
  props.detail!.signature = "old-signature";
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.queryByRole("table", { name: "Prompt distribution" })).toBeNull();
  expect(
    ui.getByText("Prompt details are not loaded for this run."),
  ).toBeTruthy();
});
test("zero first retained episode remains valid but a missing retained episode has no review link", () => {
  const props = fixture();
  props.dataset.runs[0].first_retained_episode = 0;
  const ui = render(<DatasetMonitorRow {...props} />);
  expect(
    ui.getByRole("link", { name: "Open review" }).getAttribute("href"),
  ).toContain("episode_0");
  props.dataset.runs[0].first_retained_episode = null;
  ui.rerender(<DatasetMonitorRow {...props} />);
  expect(ui.queryByRole("link", { name: "Open review" })).toBeNull();
  expect(ui.getByText("No retained episode available to open.")).toBeTruthy();
});
test("diagnostic severity and stale detail snapshot remain explicit", () => {
  const props = fixture();
  props.dataset.state = "Updating";
  props.dataset.diagnostics = [
    {
      code: "partial",
      message: "Metadata is partially written",
      severity: "warning",
    },
  ];
  props.detail!.updating = true;
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.getByText(/Metadata is partially written/).textContent).toContain(
    "Warning:",
  );
  expect(
    ui
      .getByText(/showing the last consistent detail snapshot/)
      .getAttribute("role"),
  ).toBe("status");
});

test("a cached detail cannot erase a newer summary remote check", () => {
  const props = fixture();
  props.detail!.publications = [{ ...props.dataset.publications[0] }];
  props.dataset.publications[0].remote_check = {
    status: "changed",
    checked_at: "2026-09-16T02:00:00Z",
    current_commit: "newer-head",
    message: "New check",
  };
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.getByText("Branch advanced/changed")).toBeTruthy();
});

test("normal edits preserve known unpublished state and receipt when the frozen export was removed", () => {
  const props = fixture();
  // Matches test_receipt_survives_normal_edit_that_removes_export on the backend.
  props.dataset.runs[0].freshness = {
    ...props.dataset.runs[0].freshness,
    publication_state: "unpublished_changes",
    local_changes: null,
    verifiable: false,
  };
  props.dataset.exports = [];
  props.dataset.publications[0] = {
    ...props.dataset.publications[0],
    export_path: null,
    export_available: false,
    manifest_sha256: null,
  };
  const ui = render(<DatasetMonitorRow {...props} />);
  expect(ui.getByText("Unpublished changes")).toBeTruthy();
  expect(ui.getByText("Digest comparison unverified")).toBeTruthy();
  expand(ui.container);
  expect(ui.getByText("recorded-commit")).toBeTruthy();
  expect(
    ui.getByRole("link", { name: /Open HF/ }).getAttribute("href"),
  ).toContain("/tree/260915");
  expect(ui.queryByRole("button", { name: /Copy training path/ })).toBeNull();
});

test("unknown prompt episode keys preserve both numeric strings and invalid keys", () => {
  const props = fixture();
  props.detail!.prompts!.unknown_episode_ids = ["42", "invalid-key"];
  const ui = render(<DatasetMonitorRow {...props} />);
  expand(ui.container);
  expect(ui.getByText("Unknown episode IDs: 42, invalid-key")).toBeTruthy();
});
