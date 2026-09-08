import { afterAll, afterEach, describe, expect, mock, test } from "bun:test";

const originalUrl = process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
const originalFetch = globalThis.fetch;
process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = "http://annotation.test";
const { createAnnotationJob, validateAnnotation } =
  await import("../annotationsClient");
afterEach(() => {
  globalThis.fetch = originalFetch;
});
afterAll(() => {
  if (originalUrl === undefined)
    delete process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
  else process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = originalUrl;
});

describe("annotations client", () => {
  test("converts editor identity to backend fields without sending camelCase", async () => {
    const request = mock(
      async (_input: RequestInfo | URL, _init?: RequestInit) => {
        void _input;
        void _init;
        return Response.json({ job_id: "j1", status: "queued" });
      },
    );
    globalThis.fetch = Object.assign(request, {
      preconnect: originalFetch.preconnect,
    });
    await createAnnotationJob({
      repoId: "org/data",
      localPath: null,
      revision: "revision-1",
      episode_indices: [3],
      config: { plan: { enabled: false } },
    });
    expect(JSON.parse(String(request.mock.calls[0][1]?.body))).toEqual({
      repo_id: "org/data",
      local_path: null,
      revision: "revision-1",
      episode_indices: [3],
      config: { plan: { enabled: false } },
    });
  });

  test("validates the local editor's exact atom payload and surfaces backend failures", async () => {
    const request = mock(
      async (_input: RequestInfo | URL, _init?: RequestInit) => {
        void _input;
        void _init;
        return Response.json({ detail: "Unknown episode" }, { status: 404 });
      },
    );
    globalThis.fetch = Object.assign(request, {
      preconnect: originalFetch.preconnect,
    });
    await expect(
      validateAnnotation({
        localPath: "/datasets/draft",
        episode_index: 3,
        atoms: [],
      }),
    ).rejects.toThrow("Unknown episode");
    expect(JSON.parse(String(request.mock.calls[0][1]?.body))).toEqual({
      repo_id: null,
      local_path: "/datasets/draft",
      episode_index: 3,
      atoms: [],
    });
  });
});

test("hosted client keeps every atoms, review and workflow request on the same-origin prefix", async () => {
  const previous = process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
  process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = "/api/annotation";
  const requests: { url: string; body?: Record<string, unknown> }[] = [];
  try {
    const modulePath = `../annotationsClient.ts?hosted-client-test=${Date.now()}`;
    const hosted = (await import(
      modulePath
    )) as typeof import("../annotationsClient");
    globalThis.fetch = Object.assign(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        requests.push({
          url: String(input),
          body: init?.body ? JSON.parse(String(init.body)) : undefined,
        });
        return Response.json({
          atoms: [],
          annotation_sha256: "hash",
          timestamps: [0],
          status: "unreviewed",
          job_id: "job",
          path: "/draft",
        });
      },
      { preconnect: originalFetch.preconnect },
    );
    const ident = { repoId: "local/annotation-demo" };
    expect(
      (await hosted.fetchEpisodeAtomsWithHash(0, ident)).annotation_sha256,
    ).toBe("hash");
    await hosted.saveEpisodeAtoms(0, ident, [], "hash");
    await hosted.fetchEpisodeReview(0, ident);
    await hosted.fetchFrameTimestamps(0, ident);
    await hosted.fetchAnnotationConfig();
    await hosted.fetchWorkflow("annotation-demo");
    await hosted.exportWorkflow("annotation-demo", 2);
    await hosted.publishWorkflow("annotation-demo", "manifest", 3);
    expect(
      requests.every((request) =>
        request.url.startsWith("/api/annotation/api/"),
      ),
    ).toBe(true);
    expect(requests[1].url).toBe(
      "/api/annotation/api/episodes/0/atoms?repo_id=local%2Fannotation-demo",
    );
    expect(requests[2].body?.expected_annotation_sha256).toBe("hash");
    expect(requests.at(-1)?.body).toEqual({
      manifest_sha256: "manifest",
      expected_revision: 3,
    });
  } finally {
    if (previous === undefined)
      delete process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL;
    else process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = previous;
  }
});
