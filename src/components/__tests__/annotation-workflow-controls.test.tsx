import React, { useEffect } from "react";
import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { act, fireEvent, render, waitFor } from "@testing-library/react";
import * as client from "../../utils/annotationsClient";
import {
  AnnotationsProvider,
  useAnnotations,
} from "../../context/annotations-context";
import { AnnotationWorkflowControls } from "../annotation-workflow-controls";
import { AnnotationsPanel } from "../annotations-panel";
import { TimeProvider } from "../../context/time-context";
const atom = (content: string, timestamp: number) => ({
  content,
  timestamp,
  style: "subtask" as const,
  role: "assistant" as const,
  camera: null,
  tool_calls: null,
});
const metric = {
  mae_seconds: 0.5,
  eligible_episodes: 1,
  eligible_boundaries: 1,
  missing_predictions: 1,
  topology_mismatch: 2,
};
const run = () => ({
  run_id: "run1",
  revision: 3,
  review_snapshot_sha256: "sha-1",
  publication_state: "draft",
  current_job_id: null,
  task_prompt: "Pick",
  subtask_prompts: ["Reach", "Grasp"],
  metrics: { first_pass: metric, latest: metric },
  episodes: {
    "0": {
      generation_status: "generated",
      decision: "pending",
      issues: [
        {
          code: "junk",
          source: "vlm",
          severity: "warning",
          message: "Object absent",
          start: 1,
          end: 2,
        },
      ],
      review: { status: "unreviewed" },
      predictions: [
        { atoms: [atom("Reach", 0), atom("Grasp", 1)], created_at: "today" },
      ],
    },
    "1": {
      generation_status: "failed",
      decision: "delete",
      issues: [],
      review: { status: "unreviewed" },
      predictions: [],
    },
  },
});
function Editor({
  repoId,
  episode = 0,
  panel = false,
}: {
  repoId: string;
  episode?: number;
  panel?: boolean;
}) {
  const { setEpisode } = useAnnotations();
  useEffect(
    () =>
      setEpisode(
        episode,
        { repoId },
        [atom("Reach", 0), atom("Grasp", 1.5)],
        [0, 1, 2],
      ),
    [repoId, episode, setEpisode],
  );
  return panel ? (
    <AnnotationsPanel cameraKeys={[]} />
  ) : (
    <AnnotationWorkflowControls />
  );
}
const content = (
  repoId = "local/annotation-demo",
  episode = 0,
  panel = false,
) => (
  <AnnotationsProvider>
    <TimeProvider duration={2}>
      <Editor repoId={repoId} episode={episode} panel={panel} />
    </TimeProvider>
  </AnnotationsProvider>
);
const view = () => render(content());
const input = (ui: ReturnType<typeof view>, name: string, value: string) =>
  fireEvent.input(ui.getByLabelText(name), { target: { value } });
const ready = async (ui: ReturnType<typeof view>) => {
  await waitFor(() =>
    expect(
      (ui.getByRole("button", { name: "Keep" }) as HTMLButtonElement).disabled,
    ).toBe(false),
  );
};
const reviewedRun = (): client.WorkflowRun => {
  const value = run();
  value.episodes["0"].decision = "keep";
  value.episodes["0"].review.status = "reviewed";
  return value as client.WorkflowRun;
};
const frozenRun = (): client.WorkflowRun => ({
  ...reviewedRun(),
  repo_id: "team/source",
  source_format: "v2.1",
  publication_state: "exported",
  revision: 7,
  export: {
    manifest_sha256: "frozen",
    format: "rich",
    instruction_mode: "subtask",
    dataset_name: "frozen-folder",
    retained_episodes: 1,
    deleted_episodes: [1],
    retained_frames: 50,
    local_path: "/tmp/exports/frozen-folder",
    output_repo_id: "local/export-result",
    destination: {
      repo_id: "team/target",
      revision: "release/new",
      exists: false,
      revision_exists: false,
      expected_commit: null,
      private: false,
    },
    managed_changes: {
      added_or_updated: ["meta/info.json"],
      deleted: ["data/old.parquet"],
    },
  },
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
test("reload restores frozen options, output path, viewer and existing destination metadata", async () => {
  const snapshot = frozenRun();
  snapshot.export!.destination = {
    repo_id: "team/target",
    revision: "release/new",
    exists: true,
    revision_exists: true,
    expected_commit: "commit123",
    private: true,
  };
  spyOn(client, "fetchWorkflow").mockResolvedValue(snapshot);
  const ui = view();
  await ready(ui);
  expect((ui.getByLabelText("Export format") as HTMLSelectElement).value).toBe(
    "rich",
  );
  expect(
    (ui.getByLabelText("Instruction mode") as HTMLSelectElement).value,
  ).toBe("subtask");
  expect(
    (ui.getByLabelText("Dataset folder name") as HTMLInputElement).value,
  ).toBe("frozen-folder");
  expect(
    (ui.getByLabelText("HF repository (optional)") as HTMLInputElement).value,
  ).toBe("team/target");
  expect((ui.getByLabelText("Revision") as HTMLInputElement).value).toBe(
    "release/new",
  );
  expect(ui.getByText("/tmp/exports/frozen-folder")).toBeTruthy();
  expect(
    ui
      .getByRole("link", { name: "Open exported dataset" })
      .getAttribute("href"),
  ).toBe("/local/export-result/episode_0?tab=annotations");
  expect(
    ui
      .getByRole("link", { name: "Inspect destination on Hugging Face" })
      .getAttribute("href"),
  ).toBe("https://huggingface.co/datasets/team/target/tree/release%2Fnew");
  expect(
    ui.getByText(/Existing repository.*Existing branch.*Private/),
  ).toBeTruthy();
  expect(ui.getByText("commit123")).toBeTruthy();
  expect(ui.queryByRole("alert")).toBeNull();
});
test("existing public destination ignores the new-repository privacy preference when confirming publish", async () => {
  const snapshot = frozenRun();
  snapshot.export!.destination!.exists = true;
  snapshot.export!.destination!.private = false;
  const request = spyOn(client, "fetchWorkflow").mockResolvedValue(
    reviewedRun(),
  );
  const exporting = spyOn(client, "exportWorkflow").mockResolvedValue({
    job_id: "export",
    status: "queued",
  });
  const publish = spyOn(client, "publishWorkflow").mockResolvedValue({
    job_id: "publish",
    status: "queued",
  });
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "job",
    status: "completed",
  });
  const ui = view();
  await ready(ui);
  input(ui, "HF repository (optional)", "team/target");
  expect(
    (ui.getByLabelText("Create new repositories privately") as HTMLInputElement)
      .checked,
  ).toBe(true);
  request.mockResolvedValue(snapshot);
  fireEvent.click(ui.getByRole("button", { name: "Preview export" }));
  await ui.findByText(/Frozen destination:/);
  await ready(ui);
  expect(exporting).toHaveBeenCalledWith(
    "annotation-demo",
    3,
    expect.objectContaining({
      destination_repo_id: "team/target",
      destination_private: true,
    }),
  );
  expect(
    (ui.getByLabelText("Create new repositories privately") as HTMLInputElement)
      .checked,
  ).toBe(true);
  expect(ui.getByText(/Existing repository.*Public/)).toBeTruthy();
  expect(ui.queryByRole("alert")).toBeNull();
  fireEvent.click(ui.getByLabelText(/I confirm publishing/));
  const button = ui.getByRole("button", {
    name: "Update Hugging Face",
  }) as HTMLButtonElement;
  expect(button.disabled).toBe(false);
  fireEvent.click(button);
  await waitFor(() =>
    expect(publish).toHaveBeenCalledWith("annotation-demo", "frozen", 7),
  );
  expect(snapshot.export!.destination!.private).toBe(false);
});
test("bulk counts include pending retained episodes and exclude deleted episodes from review", async () => {
  const ui = view();
  await ready(ui);
  expect(
    ui.getByText("Retained: 1 · Delete: 1 · Unreviewed retained: 1"),
  ).toBeTruthy();
  expect(
    (ui.getByRole("button", { name: "Keep remaining" }) as HTMLButtonElement)
      .disabled,
  ).toBe(false);
  fireEvent.click(ui.getByLabelText(/I verified the retained prompts/));
  expect(
    (
      ui.getByRole("button", {
        name: "Mark retained reviewed",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(false);
  expect(
    (ui.getByRole("button", { name: "Preview export" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
});
test("local-only preview cannot publish to the source repository", async () => {
  const snapshot = frozenRun();
  snapshot.export!.destination = null;
  spyOn(client, "fetchWorkflow").mockResolvedValue(snapshot);
  const publish = spyOn(client, "publishWorkflow");
  const ui = view();
  await ready(ui);
  expect(ui.getByText(/Frozen destination: local export only/)).toBeTruthy();
  expect(
    (ui.getByLabelText("HF repository (optional)") as HTMLInputElement).value,
  ).toBe("");
  expect(ui.queryByLabelText(/I confirm publishing/)).toBeNull();
  const button = ui.getByRole("button", {
    name: "Update Hugging Face",
  }) as HTMLButtonElement;
  expect(button.disabled).toBe(true);
  fireEvent.click(button);
  expect(publish).not.toHaveBeenCalled();
});
test("both consents expire when revision or reviewed snapshot changes", async () => {
  let snapshot = frozenRun();
  const request = spyOn(client, "fetchWorkflow").mockImplementation(
    async () => snapshot,
  );
  const ui = view();
  await ready(ui);
  for (const patch of [
    { revision: 8 },
    { review_snapshot_sha256: "changed-snapshot" },
  ]) {
    fireEvent.click(ui.getByLabelText(/I confirm publishing/));
    fireEvent.click(ui.getByLabelText(/I verified the retained prompts/));
    expect(
      (
        ui.getByRole("button", {
          name: "Update Hugging Face",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
    snapshot = { ...snapshot, ...patch };
    const calls = request.mock.calls.length;
    act(() => {
      window.dispatchEvent(new window.Event("annotation-workflow-changed"));
    });
    await waitFor(() =>
      expect(request.mock.calls.length).toBeGreaterThan(calls),
    );
    await ready(ui);
    expect(
      (ui.getByLabelText(/I confirm publishing/) as HTMLInputElement).checked,
    ).toBe(false);
    expect(
      (ui.getByLabelText(/I verified the retained prompts/) as HTMLInputElement)
        .checked,
    ).toBe(false);
    expect(
      (
        ui.getByRole("button", {
          name: "Update Hugging Face",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  }
});
test("navigation resets export form and consent, and ignores an old preview failure", async () => {
  const pending = deferred<{ job_id: string; status: string }>();
  const original = frozenRun();
  const next = { ...reviewedRun(), run_id: "other", revision: 20 };
  spyOn(client, "fetchWorkflow").mockImplementation(async (alias) =>
    alias === "annotation-demo" ? original : next,
  );
  spyOn(client, "exportWorkflow").mockReturnValue(pending.promise);
  const ui = view();
  await ready(ui);
  fireEvent.click(ui.getByLabelText(/I confirm publishing/));
  input(ui, "Dataset folder name", "unsaved-old");
  fireEvent.click(ui.getByRole("button", { name: "Preview export" }));
  ui.rerender(content("local/other"));
  await ready(ui);
  expect(
    (ui.getByLabelText("Dataset folder name") as HTMLInputElement).value,
  ).toBe("retained-dataset");
  expect(
    (ui.getByLabelText("HF repository (optional)") as HTMLInputElement).value,
  ).toBe("");
  await act(async () => {
    pending.reject(new Error("stale preview failure"));
  });
  expect(ui.queryByText("stale preview failure")).toBeNull();
  expect(ui.queryByText(/Frozen destination:/)).toBeNull();
  expect(
    (ui.getByRole("button", { name: "Preview export" }) as HTMLButtonElement)
      .disabled,
  ).toBe(false);
});
test("an old mutation cannot replace the new dataset or unlock its pending mutation", async () => {
  const old = deferred<client.WorkflowRun>();
  const next = deferred<client.WorkflowRun>();
  const request = spyOn(client, "keepWorkflowRemaining").mockImplementation(
    (alias) => (alias === "annotation-demo" ? old.promise : next.promise),
  );
  const ui = view();
  await ready(ui);
  fireEvent.click(ui.getByRole("button", { name: "Keep remaining" }));
  ui.rerender(content("local/other"));
  await ready(ui);
  fireEvent.click(ui.getByRole("button", { name: "Keep remaining" }));
  expect(request).toHaveBeenCalledWith("other", 3);
  await act(async () => {
    old.resolve(frozenRun());
  });
  expect(ui.queryByText(/Frozen destination:/)).toBeNull();
  expect(
    (ui.getByRole("button", { name: "Keep" }) as HTMLButtonElement).disabled,
  ).toBe(true);
  await act(async () => {
    next.resolve(reviewedRun());
  });
  await ready(ui);
});
test("workflow refresh while a mutation is pending preserves the request and restores controls", async () => {
  const pending = deferred<client.WorkflowRun>();
  spyOn(client, "keepWorkflowRemaining").mockReturnValue(pending.promise);
  const ui = view();
  await ready(ui);
  fireEvent.click(ui.getByRole("button", { name: "Keep remaining" }));
  act(() => {
    window.dispatchEvent(new window.Event("annotation-workflow-changed"));
  });
  await act(async () => {
    pending.resolve(reviewedRun());
  });
  await ready(ui);
  expect(ui.getByText(/Remaining episodes kept/)).toBeTruthy();
});
test("failed preview preserves edited options and reports the error without publishing", async () => {
  spyOn(client, "fetchWorkflow").mockResolvedValue(reviewedRun());
  spyOn(client, "exportWorkflow").mockRejectedValue(
    new Error("Destination permission denied"),
  );
  const publish = spyOn(client, "publishWorkflow");
  const ui = view();
  await ready(ui);
  input(ui, "HF repository (optional)", "team/restricted");
  input(ui, "Dataset folder name", "custom-folder");
  fireEvent.click(ui.getByRole("button", { name: "Preview export" }));
  await ui.findByText("Destination permission denied");
  await ready(ui);
  expect(
    (ui.getByLabelText("Dataset folder name") as HTMLInputElement).value,
  ).toBe("custom-folder");
  expect(
    (ui.getByLabelText("HF repository (optional)") as HTMLInputElement).value,
  ).toBe("team/restricted");
  expect(publish).not.toHaveBeenCalled();
});
test("top Review and export shortcut opens and scrolls the existing controls without legacy Save dataset", async () => {
  const ui = render(content("local/annotation-demo", 0, true));
  await ready(ui);
  const details = ui.getByText("Export & publish").closest("details")!;
  const scroll = mock(() => {});
  details.scrollIntoView = scroll;
  expect(details.open).toBe(false);
  fireEvent.click(ui.getByRole("button", { name: "Review & export" }));
  await waitFor(() => expect(details.open).toBe(true));
  expect(scroll).toHaveBeenCalledWith({ block: "start", behavior: "smooth" });
  expect(ui.queryByRole("button", { name: "Save dataset" })).toBeNull();
});
beforeEach(() => {
  spyOn(client, "isAnnotateBackendEnabled").mockReturnValue(false);
  spyOn(client, "fetchWorkflow").mockResolvedValue(run() as client.WorkflowRun);
});
afterEach(() => mock.restore());
test("shows filterable queue, issue evidence and safe topology comparison", async () => {
  const ui = view();
  await ui.findByText("Object absent");
  expect(ui.getByText(/vlm.*warning/)).toBeTruthy();
  expect(
    ui.queryByRole("button", { name: "Resume unfinished episodes" }),
  ).toBeNull();
  expect(ui.getByRole("link", { name: "Episode 1" }).getAttribute("href")).toBe(
    "/local/annotation-demo/episode_1?tab=annotations",
  );
  fireEvent.change(ui.getByLabelText("Queue filter"), {
    target: { value: "flagged" },
  });
  expect(ui.queryByRole("link", { name: "Episode 1" })).toBeNull();
  expect(ui.getByText("+0.500 s")).toBeTruthy();
  expect(ui.getAllByText(/Eligible episodes: 1/)).toBeTruthy();
});
test("records reason and revision; deletion is only a reversible decision", async () => {
  const decision = spyOn(client, "postWorkflowDecision").mockResolvedValue(
    run() as client.WorkflowRun,
  );
  const ui = view();
  await ui.findByText("Object absent");
  fireEvent.input(ui.getByLabelText("Decision reason"), {
    target: { value: "No object" },
  });
  fireEvent.click(ui.getByRole("button", { name: "Delete episode" }));
  await waitFor(() =>
    expect(decision).toHaveBeenCalledWith("annotation-demo", {
      episode_index: 0,
      decision: "delete",
      reason: "No object",
      expected_revision: 3,
    }),
  );
});
test("previews the current options and requires consent to the exact frozen target before publishing", async () => {
  const snapshot = frozenRun();
  spyOn(client, "fetchWorkflow").mockResolvedValue(reviewedRun());
  const exporting = spyOn(client, "exportWorkflow").mockResolvedValue({
    job_id: "export",
    status: "queued",
  });
  const publish = spyOn(client, "publishWorkflow").mockResolvedValue({
    job_id: "publish",
    status: "queued",
  });
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "job",
    status: "completed",
  });
  const ui = view();
  await ready(ui);
  fireEvent.click(ui.getByText("Export & publish"));
  fireEvent.change(ui.getByLabelText("Export format"), {
    target: { value: "rich" },
  });
  fireEvent.change(ui.getByLabelText("Instruction mode"), {
    target: { value: "subtask" },
  });
  input(ui, "Dataset folder name", "frozen-folder");
  input(ui, "HF repository (optional)", "team/target");
  input(ui, "Revision", "release/new");
  fireEvent.click(ui.getByLabelText("Create new repositories privately"));
  spyOn(client, "fetchWorkflow").mockResolvedValue(snapshot);
  fireEvent.click(ui.getByRole("button", { name: "Preview export" }));
  await ui.findByText(/Frozen destination:/);
  await ready(ui);
  expect(exporting).toHaveBeenCalledWith("annotation-demo", 3, {
    export_format: "rich",
    instruction_mode: "subtask",
    dataset_name: "frozen-folder",
    destination_repo_id: "team/target",
    destination_revision: "release/new",
    destination_private: false,
  });
  expect(ui.getByText("team/target @ release/new")).toBeTruthy();
  expect(ui.queryByText(/Destination: team\/source/)).toBeNull();
  expect(ui.getByText(/New repository.*New branch.*Public/)).toBeTruthy();
  expect(ui.getByText(/Frozen format: Rich annotations/)).toBeTruthy();
  expect(ui.getByText(/Add\/update: 1; remove: 1/)).toBeTruthy();
  const button = ui.getByRole("button", {
    name: "Update Hugging Face",
  }) as HTMLButtonElement;
  expect(button.disabled).toBe(true);
  fireEvent.click(button);
  expect(publish).not.toHaveBeenCalled();
  fireEvent.click(
    ui.getByLabelText(
      /I confirm publishing this frozen export to team\/target @ release\/new/,
    ),
  );
  expect(button.disabled).toBe(false);
  fireEvent.click(button);
  await waitFor(() =>
    expect(publish).toHaveBeenCalledWith("annotation-demo", "frozen", 7),
  );
});

test("changed subtask topology never shows fabricated boundary deltas", async () => {
  const snapshot = run();
  snapshot.episodes["0"].predictions[0].atoms[1].content = "Different task";
  spyOn(client, "fetchWorkflow").mockResolvedValue(
    snapshot as client.WorkflowRun,
  );
  const ui = view();
  await ui.findByText(/Subtask topology changed/);
  expect(ui.queryByText("+0.500 s")).toBeNull();
});
test("review completion refreshes revision and the queue", async () => {
  const request = spyOn(client, "fetchWorkflow").mockResolvedValue(
    run() as client.WorkflowRun,
  );
  const ui = view();
  await ui.findByText("Object absent");
  const snapshot = run();
  snapshot.episodes["0"].review.status = "reviewed";
  snapshot.revision = 4;
  request.mockResolvedValue(snapshot as client.WorkflowRun);
  act(() => {
    window.dispatchEvent(new window.Event("annotation-workflow-changed"));
  });
  await ui.findByText("reviewed");
  expect(request.mock.calls.length).toBeGreaterThan(1);
});

test("a malformed episode can be marked for deletion without opening its broken viewer", async () => {
  const request = spyOn(client, "postWorkflowDecision").mockResolvedValue(
    run() as client.WorkflowRun,
  );
  const ui = view();
  await ui.findByText("Object absent");
  fireEvent.click(ui.getByRole("button", { name: "Inspect episode 1 issues" }));
  fireEvent.input(ui.getByLabelText("Decision reason"), {
    target: { value: "Unreadable video" },
  });
  fireEvent.click(ui.getByRole("button", { name: "Delete episode" }));
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith("annotation-demo", {
      episode_index: 1,
      decision: "delete",
      reason: "Unreadable video",
      expected_revision: 3,
    }),
  );
  expect(ui.queryByText("+0.500 s")).toBeNull();
});

test("bulk keep preserves deletions and review requires explicit consent plus snapshot hash", async () => {
  const snapshot = {
    ...run(),
    review_snapshot_sha256: "sha-1",
  } as client.WorkflowRun;
  const keep = spyOn(client, "keepWorkflowRemaining").mockResolvedValue(
    snapshot,
  );
  const review = spyOn(client, "reviewWorkflowRetained").mockResolvedValue(
    snapshot,
  );
  const ui = view();
  await ui.findByText("Object absent");
  fireEvent.click(ui.getByText("Export & publish"));
  expect(
    ui
      .getByRole("button", { name: "Mark retained reviewed" })
      .hasAttribute("disabled"),
  ).toBe(true);
  fireEvent.click(ui.getByRole("button", { name: "Keep remaining" }));
  await waitFor(() => expect(keep).toHaveBeenCalledWith("annotation-demo", 3));
  await ready(ui);
  fireEvent.click(ui.getByLabelText(/I verified the retained prompts/));
  fireEvent.click(ui.getByRole("button", { name: "Mark retained reviewed" }));
  await waitFor(() =>
    expect(review).toHaveBeenCalledWith("annotation-demo", {
      expected_revision: 3,
      expected_review_sha256: "sha-1",
      confirmed: true,
    }),
  );
});

test("changed options disable publish while preview sends new values and refresh preserves unsaved form", async () => {
  const snapshot = frozenRun();
  const fetching = spyOn(client, "fetchWorkflow").mockResolvedValue(snapshot);
  const exporting = spyOn(client, "exportWorkflow").mockResolvedValue({
    job_id: "export",
    status: "queued",
  });
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "export",
    status: "completed",
  });
  const publish = spyOn(client, "publishWorkflow").mockResolvedValue({
    job_id: "publish",
    status: "queued",
  });
  const ui = view();
  await ready(ui);
  fireEvent.click(ui.getByText("Export & publish"));
  fireEvent.click(ui.getByLabelText(/I confirm publishing/));
  expect(
    (
      ui.getByRole("button", {
        name: "Update Hugging Face",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(false);
  input(ui, "HF repository (optional)", "team/other");
  input(ui, "Dataset folder name", "edited-folder");
  expect(
    (
      ui.getByRole("button", {
        name: "Update Hugging Face",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
  expect(ui.getByText("team/target @ release/new")).toBeTruthy();
  expect(ui.getByRole("alert").textContent).toContain("Preview again");
  fetching.mockResolvedValue({ ...snapshot, revision: 8 });
  act(() => {
    window.dispatchEvent(new window.Event("annotation-workflow-changed"));
  });
  await ready(ui);
  expect(
    (ui.getByLabelText("HF repository (optional)") as HTMLInputElement).value,
  ).toBe("team/other");
  expect(
    (ui.getByLabelText("Dataset folder name") as HTMLInputElement).value,
  ).toBe("edited-folder");
  expect(
    (ui.getByRole("button", { name: "Preview export" }) as HTMLButtonElement)
      .disabled,
  ).toBe(false);
  fireEvent.click(ui.getByRole("button", { name: "Preview export" }));
  await waitFor(() =>
    expect(exporting).toHaveBeenCalledWith(
      "annotation-demo",
      8,
      expect.objectContaining({
        destination_repo_id: "team/other",
        dataset_name: "edited-folder",
      }),
    ),
  );
  expect(publish).not.toHaveBeenCalled();
});

test("puts episode actions first and keeps queue and export collapsed", async () => {
  const ui = view();
  await ui.findByText("Object absent");
  const root = ui.getByRole("region", { name: "Annotation workflow" });
  const deleteButton = ui.getByRole("button", { name: "Delete episode" });
  const queueSummary = ui.getByText("Review queue (2 episodes)");
  const exportSummary = ui.getByText("Export & publish");
  expect(root.textContent?.indexOf("Delete episode")).toBeLessThan(
    root.textContent?.indexOf("Review queue"),
  );
  expect(deleteButton.hasAttribute("disabled")).toBe(true);
  expect(queueSummary.parentElement?.hasAttribute("open")).toBe(false);
  expect(exportSummary.parentElement?.hasAttribute("open")).toBe(false);
  fireEvent.input(ui.getByLabelText("Decision reason"), {
    target: { value: "No object" },
  });
  expect(deleteButton.hasAttribute("disabled")).toBe(false);
});

test("links the durable checkpoint and resumes only unfinished episodes", async () => {
  const saved = {
    ...run(),
    current_repo_id: "local/annotation-checkpoint",
    example_episode_indices: [],
    episodes: {
      ...run().episodes,
      "1": { ...run().episodes["1"], decision: "pending" },
    },
  };
  spyOn(client, "fetchWorkflow").mockResolvedValue(saved as client.WorkflowRun);
  const start = spyOn(client, "createAnnotationJob").mockResolvedValue({
    job_id: "job",
    status: "queued",
  });
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "job",
    status: "completed",
  });
  const ui = view();
  const link = await ui.findByRole("link", { name: "Open current checkpoint" });
  expect(link.getAttribute("href")).toContain("local/annotation-checkpoint");
  fireEvent.click(
    ui.getByRole("button", { name: "Resume unfinished episodes" }),
  );
  await waitFor(() => expect(start).toHaveBeenCalledTimes(1));
  expect(start.mock.calls[0][0]).toMatchObject({
    repoId: "local/annotation-checkpoint",
    resume_unfinished: true,
  });
});
