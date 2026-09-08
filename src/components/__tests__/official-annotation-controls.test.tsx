import React, { useEffect } from "react";
import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { act, fireEvent, render, waitFor } from "@testing-library/react";

const originalFetch = globalThis.fetch;
const originalUrl = process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = "http://annotation.test";
const { AnnotationsProvider, useAnnotations } =
  await import("../../context/annotations-context");
const { OfficialAnnotationControls } =
  await import("../official-annotation-controls");
const defaults = {
  plan: { enabled: true, n_task_rephrasings: 7 },
  interjections: { enabled: true },
  vqa: { enabled: true },
};
let requests: { path: string; body: Record<string, unknown> }[];
let saveResponse: () => Promise<Response>;
let jobResponse: () => Promise<Response>;
let navigate: ReturnType<typeof spyOn>;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function completed() {
  return Response.json({
    job_id: "j1",
    status: "completed",
    result: {
      repo_id: "local/annotation-output",
      output_dir: "/draft",
      first_generated_episode_index: 3,
      validation: {
        ok: true,
        errors: [],
        warnings: ["Review camera grounding"],
        episodes_checked: 1,
      },
    },
  });
}
function Editor({ episode, repoId }: { episode: number; repoId: string }) {
  const { setEpisode, addAtom } = useAnnotations();
  useEffect(() => {
    setEpisode(episode, { repoId }, [], [0, 1]);
  }, [episode, repoId, setEpisode]);
  return (
    <>
      <button
        onClick={() =>
          addAtom({
            style: "subtask",
            role: "assistant",
            content: "Reach",
            timestamp: 0,
            camera: null,
            tool_calls: null,
          })
        }
      >
        Edit
      </button>
      <OfficialAnnotationControls />
    </>
  );
}
function view(episode = 2, repoId = "org/source") {
  return (
    <AnnotationsProvider>
      <Editor episode={episode} repoId={repoId} />
    </AnnotationsProvider>
  );
}

beforeEach(() => {
  navigate = spyOn(window.location, "assign").mockImplementation(() => {});
  requests = [];
  saveResponse = async () =>
    Response.json({ path: "/draft/meta/lerobot_annotations.json" });
  jobResponse = async () => completed();
  globalThis.fetch = Object.assign(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input)).pathname;
      const body = JSON.parse(String(init?.body || "{}"));
      requests.push({ path, body });
      if (path.endsWith("/config"))
        return Response.json({ revision: "pinned", config: defaults });
      if (path.endsWith("/validate"))
        return Response.json({
          ok: false,
          errors: ["Missing speech pair"],
          warnings: ["Check memory boundary"],
          episodes_checked: 1,
        });
      if (path.endsWith("/frame_timestamps"))
        return Response.json({ timestamps: [0, 1] });
      if (path.endsWith("/atoms"))
        return init?.method === "POST"
          ? saveResponse()
          : Response.json({ atoms: [] });
      if (path.endsWith("/jobs"))
        return Response.json({ job_id: "j1", status: "queued" });
      if (path.endsWith("/jobs/j1")) return jobResponse();
      return Response.json({});
    },
    { preconnect: originalFetch.preconnect },
  );
});
afterEach(() => {
  navigate.mockRestore();
  globalThis.fetch = originalFetch;
});
afterAll(() => {
  if (originalUrl === undefined)
    delete process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
  else process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = originalUrl;
});

async function ready(ui: ReturnType<typeof render>) {
  await waitFor(() =>
    expect(
      (ui.getByRole("button", { name: "Generate draft" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false),
  );
  await act(async () => {});
}

test("saves before generation, preserves nested options and links the reviewed draft", async () => {
  const saved = deferred<Response>();
  saveResponse = () => saved.promise;
  const ui = render(view());
  await ready(ui);
  fireEvent.click(
    ui.getByRole("checkbox", {
      name: "Subtasks, plans, memory and task phrasings",
    }),
  );
  fireEvent.click(ui.getByText("Edit"));
  fireEvent.click(ui.getByRole("button", { name: "Generate draft" }));
  await waitFor(() =>
    expect(
      requests.some((r) => r.path.endsWith("/atoms") && r.body.atoms),
    ).toBe(true),
  );
  expect(requests.some((r) => r.path.endsWith("/jobs"))).toBe(false);
  await act(async () => saved.resolve(Response.json({ path: "/saved" })));
  const link = await ui.findByRole("link", {
    name: "Review generated dataset",
  });
  expect(link.getAttribute("href")).toBe("/local/annotation-output/episode_2");
  expect(navigate).toHaveBeenCalledWith("/local/annotation-output/episode_2");
  const body = requests.find((r) => r.path.endsWith("/jobs"))!.body;
  expect(body.repo_id).toBe("org/source");
  expect(body.episode_indices).toEqual([2]);
  expect(body.config).toMatchObject({
    plan: { enabled: false, n_task_rephrasings: 7 },
  });
});

test("save failure prevents generation; official validation errors and warnings are visible", async () => {
  saveResponse = async () =>
    Response.json({ detail: "disk full" }, { status: 500 });
  const ui = render(view());
  await ready(ui);
  fireEvent.click(ui.getByText("Edit"));
  fireEvent.click(ui.getByRole("button", { name: "Generate draft" }));
  await ui.findByText("Save the current episode before generating.");
  expect(requests.some((r) => r.path.endsWith("/jobs"))).toBe(false);
  fireEvent.click(ui.getByRole("button", { name: "Validate" }));
  await ui.findByText("Error: Missing speech pair");
  expect(ui.getByText("Warning: Check memory boundary")).toBeTruthy();
});

test("completion after episode navigation does not display the previous result", async () => {
  const pending = deferred<Response>();
  jobResponse = () => pending.promise;
  const ui = render(view());
  await ready(ui);
  fireEvent.click(ui.getByRole("button", { name: "Generate draft" }));
  await waitFor(() =>
    expect(requests.some((r) => r.path.endsWith("/jobs/j1"))).toBe(true),
  );
  ui.rerender(view(3));
  await ready(ui);
  await act(async () => pending.resolve(completed()));
  expect(
    ui.queryByRole("link", { name: "Review generated dataset" }),
  ).toBeNull();
  expect(ui.queryByText("Draft generated.")).toBeNull();
  expect(navigate).not.toHaveBeenCalled();
});

test("new edits during generation prevent automatic navigation", async () => {
  const pending = deferred<Response>();
  jobResponse = () => pending.promise;
  const ui = render(view());
  await ready(ui);
  fireEvent.click(ui.getByRole("button", { name: "Generate draft" }));
  await waitFor(() =>
    expect(requests.some((r) => r.path.endsWith("/jobs/j1"))).toBe(true),
  );
  fireEvent.click(ui.getByText("Edit"));
  await act(async () => pending.resolve(completed()));
  await ui.findByText(
    "Draft generated. You have new unsaved edits; use the review link after saving them.",
  );
  expect(navigate).not.toHaveBeenCalled();
  expect(
    ui.getByRole("link", { name: "Review generated dataset" }),
  ).toBeTruthy();
});

test("sends all episodes and example episode IDs", async () => {
  const ui = render(view());
  await ready(ui);
  fireEvent.click(ui.getByRole("checkbox", { name: "All episodes" }));
  fireEvent.input(ui.getByRole("textbox", { name: "Example episodes" }), {
    target: { value: "0, 3, 7" },
  });
  fireEvent.click(ui.getByRole("button", { name: "Generate draft" }));
  await ui.findByRole("link", { name: "Review generated dataset" });
  const body = requests.find((r) => r.path.endsWith("/jobs"))!.body;
  expect(body.episode_indices).toBeUndefined();
  expect(body.example_episode_indices).toEqual([0, 3, 7]);
});

test.each(["1, nope, 1", "1,,2", "01,1", "-1", "0,1,2,3,4,5"])(
  "rejects malformed example IDs %s without starting a job",
  async (value) => {
    const ui = render(view());
    await ready(ui);
    fireEvent.input(ui.getByRole("textbox", { name: "Example episodes" }), {
      target: { value },
    });
    fireEvent.click(ui.getByRole("button", { name: "Generate draft" }));
    await ui.findByText(
      "Example episodes must be up to 5 distinct comma-separated nonnegative IDs.",
    );
    expect(requests.some((r) => r.path.endsWith("/jobs"))).toBe(false);
  },
);

test("examples persist across episodes but reset for another dataset", async () => {
  const ui = render(view());
  await ready(ui);
  fireEvent.input(ui.getByRole("textbox", { name: "Example episodes" }), {
    target: { value: "0, 3, 7" },
  });
  ui.rerender(view(5));
  await ready(ui);
  expect(
    (ui.getByRole("textbox", { name: "Example episodes" }) as HTMLInputElement)
      .value,
  ).toBe("0, 3, 7");
  ui.rerender(view(5, "org/other"));
  await ready(ui);
  expect(
    (ui.getByRole("textbox", { name: "Example episodes" }) as HTMLInputElement)
      .value,
  ).toBe("");
});

test("generation from an example opens a generated target instead of the preserved example", async () => {
  const ui = render(view(2));
  await ready(ui);
  fireEvent.click(ui.getByRole("checkbox", { name: "All episodes" }));
  fireEvent.input(ui.getByRole("textbox", { name: "Example episodes" }), {
    target: { value: "2, 4" },
  });
  fireEvent.click(ui.getByRole("button", { name: "Generate draft" }));
  const link = await ui.findByRole("link", {
    name: "Review generated dataset",
  });
  expect(link.getAttribute("href")).toBe("/local/annotation-output/episode_3");
  expect(navigate).toHaveBeenCalledWith("/local/annotation-output/episode_3");
});
