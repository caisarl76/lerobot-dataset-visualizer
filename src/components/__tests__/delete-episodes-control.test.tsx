import React, { useEffect } from "react";
import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { act, fireEvent, render, waitFor } from "@testing-library/react";

const originalUrl = process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = "http://annotation.test";
const { AnnotationsProvider, useAnnotations } =
  await import("../../context/annotations-context");
const { DeleteEpisodesControl } = await import("../delete-episodes-control");
const originalFetch = globalThis.fetch;
let requests: { path: string; body: Record<string, unknown> }[];
let saveResponse: () => Promise<Response>;
let jobResponse: () => Promise<Response>;

function Editor() {
  const { setEpisode, addAtom } = useAnnotations();
  useEffect(
    () => setEpisode(2, { repoId: "org/source", revision: "draft" }, []),
    [setEpisode],
  );
  return (
    <>
      <button
        onClick={() =>
          addAtom({
            style: "subtask",
            role: "assistant",
            content: "edit",
            timestamp: 0,
            camera: null,
            tool_calls: null,
          })
        }
      >
        Edit
      </button>
      <DeleteEpisodesControl />
    </>
  );
}
function view() {
  return (
    <AnnotationsProvider>
      <Editor />
    </AnnotationsProvider>
  );
}
function completed() {
  return Response.json({
    job_id: "j1",
    status: "completed",
    result: {
      repo_id: "org/cleaned",
      output_dir: "/draft",
      first_episode_index: 0,
    },
  });
}

beforeEach(() => {
  requests = [];
  saveResponse = async () => Response.json({ path: "/saved" });
  jobResponse = async () => completed();
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input)).pathname;
    const body = JSON.parse(String(init?.body || "{}"));
    requests.push({ path, body });
    if (path.endsWith("/atoms"))
      return init?.method === "POST"
        ? saveResponse()
        : Response.json({ atoms: [] });
    if (path.endsWith("/delete-episodes"))
      return Response.json({ job_id: "j1", status: "queued" });
    if (path.endsWith("/jobs/j1")) return jobResponse();
    if (path.endsWith("/frame_timestamps"))
      return Response.json({ timestamps: [0, 1] });
    return Response.json({});
  };
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});
afterAll(() => {
  if (originalUrl === undefined)
    delete process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
  else process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = originalUrl;
});

async function open(ui: ReturnType<typeof render>) {
  fireEvent.click(ui.getByText("Delete episodes"));
  await act(async () => {});
}

test("saves before delete and links the completed cleaned draft", async () => {
  let resolveSave!: (response: Response) => void;
  saveResponse = () =>
    new Promise((resolve) => {
      resolveSave = resolve;
    });
  const ui = render(view());
  await open(ui);
  fireEvent.click(ui.getByText("Edit"));
  fireEvent.input(ui.getByRole("textbox"), { target: { value: "1, 3" } });
  fireEvent.click(ui.getByRole("button", { name: "Create cleaned draft" }));
  await waitFor(() =>
    expect(requests.some((x) => x.path.endsWith("/atoms"))).toBe(true),
  );
  expect(requests.some((x) => x.path.endsWith("/delete-episodes"))).toBe(false);
  await act(async () => resolveSave(Response.json({ path: "/saved" })));
  const link = await ui.findByRole("link", { name: "Open cleaned draft" });
  expect(link.getAttribute("href")).toBe("/org/cleaned/episode_0");
  const request = requests.find((x) => x.path.endsWith("/delete-episodes"))!;
  expect(request.body).toMatchObject({
    repo_id: "org/source",
    revision: "draft",
    episode_indices: [1, 3],
  });
});

test("rejects malformed IDs without a request", async () => {
  const ui = render(view());
  await open(ui);
  fireEvent.input(ui.getByRole("textbox"), { target: { value: "1, nope, 1" } });
  fireEvent.click(ui.getByRole("button", { name: "Create cleaned draft" }));
  await ui.findByText(
    "Episode IDs must be distinct comma-separated nonnegative integers.",
  );
  expect(requests.some((x) => x.path.endsWith("/delete-episodes"))).toBe(false);
});

test("rejects empty IDs without a request", async () => {
  const ui = render(view());
  await open(ui);
  const input = ui.getByRole("textbox") as HTMLInputElement;
  input.value = "";
  fireEvent.input(input);
  fireEvent.click(ui.getByRole("button", { name: "Create cleaned draft" }));
  await ui.findByText(
    "Episode IDs must be distinct comma-separated nonnegative integers.",
  );
  expect(requests.some((x) => x.path.endsWith("/delete-episodes"))).toBe(false);
});

test("shows a failed job response", async () => {
  jobResponse = async () =>
    Response.json({ job_id: "j1", status: "failed", error: "disk full" });
  const ui = render(view());
  await open(ui);
  fireEvent.click(ui.getByRole("button", { name: "Create cleaned draft" }));
  await ui.findByText("disk full");
});
