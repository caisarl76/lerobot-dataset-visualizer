import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { NextRequest } from "next/server";
import { GET, HEAD, POST } from "../route";

const origin = "https://viewer.example";
const backend = "http://127.0.0.1:8123";
const token = "server-secret";
const ctx = (...path: string[]) => ({ params: Promise.resolve({ path }) });
const req = (path: string, init: RequestInit = {}) =>
  new NextRequest(`${origin}/api/annotation/${path}`, {
    ...init,
    headers: { origin, ...(init.headers ?? {}) },
  });

describe("hosted annotation proxy", () => {
  const savedFetch = globalThis.fetch;
  beforeEach(() => {
    process.env.ANNOTATION_BACKEND_URL = backend;
    process.env.ANNOTATION_BACKEND_TOKEN = token;
    process.env.ANNOTATION_BROWSER_ORIGIN = origin;
    process.env.ANNOTATION_HOSTED_PRIVATE_SPACE = "1";
  });
  afterEach(() => {
    globalThis.fetch = savedFetch;
    for (const k of [
      "ANNOTATION_BACKEND_URL",
      "ANNOTATION_BACKEND_TOKEN",
      "ANNOTATION_BROWSER_ORIGIN",
      "ANNOTATION_HOSTED_PRIVATE_SPACE",
    ])
      delete process.env[k];
    mock.restore();
  });

  test("allows read-only robot motion and rejects writes to it", async () => {
    globalThis.fetch = mock(async () =>
      Response.json({ timestamps: [0] }),
    ) as typeof fetch;
    const path = "api/episodes/1/robot-motion";
    expect((await GET(req(path), ctx(...path.split("/")))).status).toBe(200);
    expect(
      (await POST(req(path, { method: "POST" }), ctx(...path.split("/"))))
        .status,
    ).toBe(403);
  });
  test("denies unlisted config paths", async () => {
    const fetchSpy = mock(async () => Response.json({ ok: true }));
    globalThis.fetch = fetchSpy as typeof fetch;
    const response = await GET(
      req("api/config/extra"),
      ctx("api", "config", "extra"),
    );
    expect(response.status).toBe(403);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
  test("requires exact origin for mutations", async () => {
    const fetchSpy = mock(async () => Response.json({ ok: true }));
    globalThis.fetch = fetchSpy as typeof fetch;
    const response = await POST(
      new NextRequest(`${origin}/api/annotation/dataset/load`, {
        method: "POST",
        headers: {
          origin: "https://evil.example",
          "content-type": "application/json",
        },
        body: "{}",
      }),
      ctx("dataset", "load"),
    );
    expect(response.status).toBe(403);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
  test("rejects traversal and browser secrets", async () => {
    const fetchSpy = mock(async () => Response.json({ ok: true }));
    globalThis.fetch = fetchSpy as typeof fetch;
    const traversal = await GET(
      req("datasets/local/a%2Fb/resolve/main/meta/x"),
      ctx("datasets", "local", "a%2Fb", "resolve", "main", "meta", "x"),
    );
    expect(traversal.status).toBe(403);
    const secret = await POST(
      req("dataset/load", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ token: "x" }),
      }),
      ctx("dataset", "load"),
    );
    expect(secret.status).toBe(400);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
  for (const action of ["decision", "export", "publish"])
    test(`allows authenticated workflow ${action}`, async () => {
      let seen = "";
      globalThis.fetch = mock(async (url: string | URL | Request) => {
        seen = String(url);
        return Response.json({ job_id: "job" });
      }) as typeof fetch;
      const path = `api/workflow/annotation-demo/${action}`;
      const response = await POST(
        req(path, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ expected_revision: 2 }),
        }),
        ctx(...path.split("/")),
      );
      expect(response.status).toBe(200);
      expect(seen).toBe(`${backend}/${path}`);
    });
  test("blocks secrets passed through GET query parameters", async () => {
    const response = await GET(
      req("api/episodes/0/atoms?local_path=/tmp/dataset"),
      ctx("api", "episodes", "0", "atoms"),
    );
    expect(response.status).toBe(400);
  });
  test("HEAD media requests preserve range headers without a response body", async () => {
    globalThis.fetch = mock(
      async () =>
        new Response(null, {
          headers: { "content-length": "123", "accept-ranges": "bytes" },
        }),
    ) as typeof fetch;
    const path = "datasets/local/demo/resolve/main/videos/a.mp4";
    const response = await HEAD(
      req(path, { method: "HEAD" }),
      ctx(...path.split("/")),
    );
    expect(response.status).toBe(200);
    expect(await response.text()).toBe("");
    expect(response.headers.get("content-length")).toBe("123");
  });
  test("streams ranged media and sends only backend bearer", async () => {
    let seen: RequestInit | undefined;
    globalThis.fetch = mock(
      async (_url: string | URL | Request, init?: RequestInit) => {
        seen = init;
        return new Response("bytes", {
          status: 206,
          headers: {
            "content-type": "video/mp4",
            "content-range": "bytes 0-4/5",
          },
        });
      },
    ) as typeof fetch;
    const response = await GET(
      req("datasets/local/demo/resolve/main/videos/a.mp4", {
        headers: {
          range: "bytes=0-4",
          "if-range": "tag",
          authorization: "Bearer browser",
        },
      }),
      ctx("datasets", "local", "demo", "resolve", "main", "videos", "a.mp4"),
    );
    expect(response.status).toBe(206);
    expect(await response.text()).toBe("bytes");
    const headers = new Headers(seen?.headers);
    expect(headers.get("authorization")).toBe(`Bearer ${token}`);
    expect(headers.get("range")).toBe("bytes=0-4");
    expect(headers.get("if-range")).toBe("tag");
  });
});

for (const path of [
  "api/annotation/config",
  "api/episodes/0/frame_timestamps",
  "api/workflow/annotation-demo",
]) {
  test(`hosted client GET ${path} keeps backend api prefix`, async () => {
    process.env.ANNOTATION_BACKEND_URL = backend;
    process.env.ANNOTATION_BACKEND_TOKEN = token;
    process.env.ANNOTATION_BROWSER_ORIGIN = origin;
    process.env.ANNOTATION_HOSTED_PRIVATE_SPACE = "1";
    const previous = globalThis.fetch;
    let seen = "";
    try {
      globalThis.fetch = mock(async (url: string | URL | Request) => {
        seen = String(url);
        return Response.json({});
      }) as typeof fetch;
      const response = await GET(req(path), ctx(...path.split("/")));
      expect(response.status).toBe(200);
      expect(seen).toBe(`${backend}/${path}`);
    } finally {
      globalThis.fetch = previous;
      for (const key of [
        "ANNOTATION_BACKEND_URL",
        "ANNOTATION_BACKEND_TOKEN",
        "ANNOTATION_BROWSER_ORIGIN",
        "ANNOTATION_HOSTED_PRIVATE_SPACE",
      ])
        delete process.env[key];
    }
  });
}
