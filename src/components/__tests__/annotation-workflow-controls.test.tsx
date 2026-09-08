import React, { useEffect } from "react";
import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { act, fireEvent, render, waitFor } from "@testing-library/react";
import * as client from "../../utils/annotationsClient";
import {
  AnnotationsProvider,
  useAnnotations,
} from "../../context/annotations-context";
import { AnnotationWorkflowControls } from "../annotation-workflow-controls";
const atom = (content: string, timestamp: number) => ({
  content,
  timestamp,
  style: "subtask",
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
function Editor() {
  const { setEpisode } = useAnnotations();
  useEffect(
    () =>
      setEpisode(
        0,
        { repoId: "local/annotation-demo" },
        [atom("Reach", 0), atom("Grasp", 1.5)],
        [0, 1, 2],
      ),
    [setEpisode],
  );
  return <AnnotationWorkflowControls />;
}
const view = () =>
  render(
    <AnnotationsProvider>
      <Editor />
    </AnnotationsProvider>,
  );
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
  fireEvent.click(ui.getByRole("button", { name: "Delete" }));
  await waitFor(() =>
    expect(decision).toHaveBeenCalledWith("annotation-demo", {
      episode_index: 0,
      decision: "delete",
      reason: "No object",
      expected_revision: 3,
    }),
  );
});
test("previews frozen export and publishes its exact manifest", async () => {
  const snapshot = {
    ...run(),
    publication_state: "exported",
    repo_id: "team/collection",
    source_format: "v2.1",
    export: {
      manifest_sha256: "frozen",
      managed_changes: {
        added_or_updated: ["meta/info.json"],
        deleted: ["data/old.parquet"],
      },
      retained_episodes: 1,
      deleted_episodes: 1,
    },
  };
  const exporting = spyOn(client, "exportWorkflow").mockResolvedValue({
    job_id: "export",
    status: "queued",
  });
  const publish = spyOn(client, "publishWorkflow").mockResolvedValue({
    job_id: "publish",
    status: "queued",
  });
  spyOn(client, "getAnnotationJob").mockResolvedValue({
    job_id: "export",
    status: "completed",
  });
  const ui = view();
  await ui.findByText("Object absent");
  spyOn(client, "fetchWorkflow").mockResolvedValue(
    snapshot as client.WorkflowRun,
  );
  fireEvent.click(ui.getByRole("button", { name: "Preview export" }));
  await ui.findByText(/Retained: 1; deleted: 1/);
  expect(ui.getByText(/Destination: team\/collection/)).toBeTruthy();
  expect(ui.getByText(/main format: v2.1/)).toBeTruthy();
  expect(ui.getByText(/Add\/update: 1; remove: 1/)).toBeTruthy();
  expect(exporting).toHaveBeenCalledWith("annotation-demo", 3);
  fireEvent.click(ui.getByRole("button", { name: "Update Hugging Face" }));
  await waitFor(() =>
    expect(publish).toHaveBeenCalledWith("annotation-demo", "frozen", 3),
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
  fireEvent.click(ui.getByRole("button", { name: "Delete" }));
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
