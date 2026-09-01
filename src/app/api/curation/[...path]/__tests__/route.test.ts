import {
  afterEach,
  beforeEach,
  describe,
  expect,
  mock as bunMock,
  test,
} from "bun:test";
import { NextRequest } from "next/server";

import { GET, OPTIONS, PATCH, POST, dynamic, runtime } from "../route";

const BACKEND_URL = "http://127.0.0.1:8000";
const SERVER_TOKEN = "server-curation-secret";
const BROWSER_ORIGIN = "http://127.0.0.1:3000";
const BROWSER_HOST = "127.0.0.1:3000";
const MUTATION_MARKER = "same-origin";
const nativeFetch = globalThis.fetch;
const mock = Object.assign(
  function typedFetchMock<
    T extends (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => Promise<Response>,
  >(implementation: T) {
    return Object.assign(bunMock(implementation), {
      preconnect: nativeFetch.preconnect,
    });
  },
  { restore: bunMock.restore },
);

type RouteHandler = typeof GET;

interface BrowserMetadata {
  host?: string | null;
  origin?: string | null;
  fetchSite?: string | null;
  marker?: string | null;
  forwardedHost?: string | null;
  forwardedProto?: string | null;
}

function request(
  path: string,
  init: ConstructorParameters<typeof NextRequest>[1] = {},
  metadata: BrowserMetadata = {},
): NextRequest {
  const headers = new Headers(init.headers);
  const mutation = init.method === "POST" || init.method === "PATCH";
  const values = {
    host: metadata.host === undefined ? BROWSER_HOST : metadata.host,
    origin:
      metadata.origin === undefined
        ? mutation
          ? BROWSER_ORIGIN
          : null
        : metadata.origin,
    "sec-fetch-site":
      metadata.fetchSite === undefined
        ? mutation
          ? "same-origin"
          : null
        : metadata.fetchSite,
    "x-curation-request":
      metadata.marker === undefined ? null : metadata.marker,
    "x-forwarded-host":
      metadata.forwardedHost === undefined ? null : metadata.forwardedHost,
    "x-forwarded-proto":
      metadata.forwardedProto === undefined ? null : metadata.forwardedProto,
  };
  for (const [name, value] of Object.entries(values)) {
    if (value !== null) headers.set(name, value);
  }
  return new NextRequest(`${BROWSER_ORIGIN}/api/curation/${path}`, {
    ...init,
    headers,
  });
}

function context(...path: string[]) {
  return { params: Promise.resolve({ path }) };
}

async function call(
  handler: RouteHandler,
  path: string[],
  init: ConstructorParameters<typeof NextRequest>[1] = {},
  metadata: BrowserMetadata = {},
) {
  return handler(
    request(
      `${path.join("/")}?dataset_alias=local%2Fpnp_trash`,
      init,
      metadata,
    ),
    context(...path),
  );
}

describe("curation same-origin proxy", () => {
  const originalFetch = globalThis.fetch;
  const originalTimeout = AbortSignal.timeout;
  const originalError = console.error;

  beforeEach(() => {
    process.env.CURATION_BACKEND_URL = BACKEND_URL;
    process.env.CURATION_BEARER_TOKEN = SERVER_TOKEN;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    AbortSignal.timeout = originalTimeout;
    console.error = originalError;
    delete process.env.CURATION_BACKEND_URL;
    delete process.env.CURATION_BEARER_TOKEN;
    mock.restore();
  });

  test("declares a dynamic Node route", () => {
    expect(runtime).toBe("nodejs");
    expect(dynamic).toBe("force-dynamic");
  });

  test("OPTIONS is a nonpermissive 405 response", () => {
    const fetchSpy = mock(async () => Response.json({ unexpected: true }));
    globalThis.fetch = fetchSpy as typeof fetch;

    const response = OPTIONS();

    expect(response.status).toBe(405);
    const permissiveCorsHeaders: string[] = [];
    response.headers.forEach((_value, name) => {
      if (name.startsWith("access-control-allow-")) {
        permissiveCorsHeaders.push(name);
      }
    });
    expect(permissiveCorsHeaders).toEqual([]);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("GET preserves path, query, status, JSON, and disables caching", async () => {
    let observedUrl = "";
    let observedInit: RequestInit | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        observedUrl = String(input);
        observedInit = init;
        return Response.json(
          { counts: { pending: 92 }, prompt_template_version: "pnp-trash-v1" },
          { status: 207 },
        );
      },
    ) as typeof fetch;

    const response = await call(GET, ["summary"]);

    expect(observedUrl).toBe(
      `${BACKEND_URL}/api/curation/summary?dataset_alias=local%2Fpnp_trash`,
    );
    expect(observedInit?.method).toBe("GET");
    expect(observedInit?.cache).toBe("no-store");
    expect(response.status).toBe(207);
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(response.headers.get("content-type")).toContain("application/json");
    expect(await response.json()).toEqual({
      counts: { pending: 92 },
      prompt_template_version: "pnp-trash-v1",
    });
  });

  for (const [method, handler] of [
    ["POST", POST],
    ["PATCH", PATCH],
  ] as const) {
    test(`${method} preserves the JSON body and content type`, async () => {
      const body = JSON.stringify({
        dataset_alias: "local/pnp_trash",
        expected_revision: 3,
      });
      let observedInit: RequestInit | undefined;
      globalThis.fetch = mock(
        async (_input: string | URL | Request, init?: RequestInit) => {
          observedInit = init;
          return Response.json({ revision: 4 });
        },
      ) as typeof fetch;

      const response = await call(
        handler,
        ["episodes", "7", "draft"],
        {
          method,
          headers: {
            "content-type": "application/json; charset=utf-8",
            authorization: "Bearer browser-supplied-token",
          },
          body,
        },
        {
          marker: MUTATION_MARKER,
          forwardedHost: "attacker.example:3000",
          forwardedProto: "https",
        },
      );

      expect(observedInit?.method).toBe(method);
      const upstreamHeaders = new Headers(observedInit?.headers);
      expect(upstreamHeaders.get("content-type")).toBe(
        "application/json; charset=utf-8",
      );
      expect(upstreamHeaders.get("authorization")).toBe(
        `Bearer ${SERVER_TOKEN}`,
      );
      const upstreamHeaderNames: string[] = [];
      upstreamHeaders.forEach((_value, name) => upstreamHeaderNames.push(name));
      expect(upstreamHeaderNames.sort()).toEqual([
        "authorization",
        "content-type",
      ]);
      expect(upstreamHeaders.has("x-curation-request")).toBe(false);
      expect(observedInit?.body).toBe(body);
      expect(await response.json()).toEqual({ revision: 4 });
    });
  }

  test("discards client authorization and injects only the server bearer", async () => {
    const observed: { authorization: string | null } = { authorization: null };
    globalThis.fetch = mock(
      async (_input: string | URL | Request, init?: RequestInit) => {
        observed.authorization = new Headers(init?.headers).get(
          "authorization",
        );
        return Response.json({ ok: true });
      },
    ) as typeof fetch;

    await GET(
      request("summary", {
        headers: { authorization: "Bearer hf-or-client-token" },
      }),
      context("summary"),
    );

    expect(observed.authorization).toBe(`Bearer ${SERVER_TOKEN}`);
  });

  test("rejects non-JSON mutation bodies before contacting the backend", async () => {
    const fetchSpy = mock(async () => Response.json({ unexpected: true }));
    globalThis.fetch = fetchSpy as typeof fetch;

    const response = await POST(
      request("batches", {
        method: "POST",
        headers: { "content-type": "text/plain" },
        body: "local/pnp_trash",
      }),
      context("batches"),
    );

    expect(response.status).toBe(415);
    expect(await response.json()).toEqual({
      error: "json_content_type_required",
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("preserves an absent body for the no-body cancellation POST", async () => {
    let observedInit: RequestInit | undefined;
    globalThis.fetch = mock(
      async (_input: string | URL | Request, init?: RequestInit) => {
        observedInit = init;
        return Response.json({
          job_id: "job-1",
          state: "cancelled",
          changed: true,
        });
      },
    ) as typeof fetch;

    const response = await POST(
      request("batches/job-1/cancel", { method: "POST" }),
      context("batches", "job-1", "cancel"),
    );

    expect(response.status).toBe(200);
    expect(observedInit?.body).toBeUndefined();
    expect(new Headers(observedInit?.headers).has("content-type")).toBe(false);
  });

  test("accepts an absent-origin marker fallback for JSON and bodyless mutations", async () => {
    const observed: RequestInit[] = [];
    globalThis.fetch = mock(
      async (_input: string | URL | Request, init?: RequestInit) => {
        observed.push(init ?? {});
        return Response.json(
          observed.length === 1
            ? { revision: 4 }
            : { job_id: "job-1", state: "cancelled", changed: true },
        );
      },
    ) as typeof fetch;

    const draft = await PATCH(
      request(
        "episodes/7/draft",
        {
          method: "PATCH",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ expected_revision: 3 }),
        },
        {
          origin: null,
          fetchSite: null,
          marker: MUTATION_MARKER,
        },
      ),
      context("episodes", "7", "draft"),
    );
    const cancel = await POST(
      request(
        "batches/job-1/cancel",
        { method: "POST" },
        {
          origin: null,
          fetchSite: null,
          marker: MUTATION_MARKER,
        },
      ),
      context("batches", "job-1", "cancel"),
    );

    expect(draft.status).toBe(200);
    expect(cancel.status).toBe(200);
    expect(observed[1]?.body).toBeUndefined();
    expect(new Headers(observed[1]?.headers).has("content-type")).toBe(false);
  });

  test("rejects every untrusted browser shape before upstream access", async () => {
    const fetchSpy = mock(async () => Response.json({ unexpected: true }));
    globalThis.fetch = fetchSpy as typeof fetch;
    const rejected: Array<[string, "GET" | "POST" | "PATCH", BrowserMetadata]> =
      [
        ["missing Host", "GET", { host: null }],
        ["empty Host", "GET", { host: "" }],
        ["duplicate Host", "GET", { host: `${BROWSER_HOST}, ${BROWSER_HOST}` }],
        ["scheme-bearing Host", "GET", { host: BROWSER_ORIGIN }],
        ["wrong Host", "GET", { host: "localhost:3000" }],
        ["wrong Host port", "GET", { host: "127.0.0.1:3001" }],
        [
          "DNS rebinding Host and Origin",
          "GET",
          {
            host: "attacker.example:3000",
            origin: "http://attacker.example:3000",
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "forwarded authority cannot replace Host",
          "GET",
          {
            host: "attacker.example:3000",
            forwardedHost: BROWSER_HOST,
            forwardedProto: "http",
          },
        ],
        [
          "cross-origin GET",
          "GET",
          {
            origin: "http://attacker.example:3000",
            fetchSite: "cross-site",
          },
        ],
        [
          "empty Origin",
          "POST",
          { origin: "", fetchSite: "same-origin", marker: MUTATION_MARKER },
        ],
        [
          "null Origin",
          "POST",
          {
            origin: "null",
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "malformed Origin",
          "POST",
          {
            origin: "not-an-origin",
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "duplicate Origin",
          "POST",
          {
            origin: `${BROWSER_ORIGIN}, ${BROWSER_ORIGIN}`,
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "wrong Origin scheme",
          "POST",
          {
            origin: "https://127.0.0.1:3000",
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "wrong Origin host",
          "POST",
          {
            origin: "http://localhost:3000",
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "wrong Origin port",
          "POST",
          {
            origin: "http://127.0.0.1:3001",
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "same-site Fetch-Site",
          "POST",
          { origin: null, fetchSite: "same-site", marker: MUTATION_MARKER },
        ],
        [
          "cross-site Fetch-Site",
          "POST",
          { origin: null, fetchSite: "cross-site", marker: MUTATION_MARKER },
        ],
        [
          "none Fetch-Site",
          "POST",
          { origin: null, fetchSite: "none", marker: MUTATION_MARKER },
        ],
        [
          "empty Fetch-Site",
          "POST",
          { origin: null, fetchSite: "", marker: MUTATION_MARKER },
        ],
        [
          "duplicate Fetch-Site",
          "POST",
          {
            origin: null,
            fetchSite: "same-origin, same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "empty marker",
          "POST",
          { origin: BROWSER_ORIGIN, fetchSite: "same-origin", marker: "" },
        ],
        [
          "wrong marker",
          "POST",
          {
            origin: BROWSER_ORIGIN,
            fetchSite: "same-origin",
            marker: "cross-origin",
          },
        ],
        ["wrong marker on GET", "GET", { marker: "cross-origin" }],
        [
          "duplicate marker",
          "POST",
          {
            origin: BROWSER_ORIGIN,
            fetchSite: "same-origin",
            marker: `${MUTATION_MARKER}, ${MUTATION_MARKER}`,
          },
        ],
        [
          "mutation without Origin or marker",
          "POST",
          { origin: null, fetchSite: "same-origin", marker: null },
        ],
        [
          "marker cannot rescue a wrong Origin",
          "POST",
          {
            origin: "http://attacker.example:3000",
            fetchSite: "same-origin",
            marker: MUTATION_MARKER,
          },
        ],
        [
          "valid Origin cannot rescue a wrong marker",
          "PATCH",
          {
            origin: BROWSER_ORIGIN,
            fetchSite: "same-origin",
            marker: "wrong",
          },
        ],
      ];

    for (const [name, method, metadata] of rejected) {
      const handler = method === "GET" ? GET : method === "POST" ? POST : PATCH;
      const response = await handler(
        request("summary", { method }, metadata),
        context("summary"),
      );
      expect({ name, status: response.status }).toEqual({ name, status: 403 });
      expect({ name, body: await response.json() }).toEqual({
        name,
        body: { error: "curation_cross_origin_forbidden" },
      });
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("ignores forwarded authority when the direct authority is trusted", async () => {
    const fetchSpy = mock(async () => Response.json({ ok: true }));
    globalThis.fetch = fetchSpy as typeof fetch;

    const response = await GET(
      request(
        "summary",
        {},
        {
          forwardedHost: "attacker.example:3000",
          forwardedProto: "https",
        },
      ),
      context("summary"),
    );

    expect(response.status).toBe(200);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  test("rejects a cross-origin bodyless cancellation without contacting upstream", async () => {
    const fetchSpy = mock(async () => Response.json({ unexpected: true }));
    globalThis.fetch = fetchSpy as typeof fetch;

    const response = await POST(
      request(
        "batches/job-1/cancel",
        { method: "POST" },
        {
          origin: "https://attacker.example",
          fetchSite: "cross-site",
        },
      ),
      context("batches", "job-1", "cancel"),
    );

    expect(response.status).toBe(403);
    expect(await response.json()).toEqual({
      error: "curation_cross_origin_forbidden",
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("rejects path escapes and asset-like paths without proxying them", async () => {
    const fetchSpy = mock(async () => Response.json({ unexpected: true }));
    globalThis.fetch = fetchSpy as typeof fetch;

    for (const path of [
      [
        "..",
        "local-datasets",
        "local",
        "pnp_trash",
        "resolve",
        "main",
        "meta",
        "info.json",
      ],
      ["%2e%2e", "local-datasets"],
      ["episodes/0/video"],
      ["episodes\\0\\video"],
      [""],
    ]) {
      const response = await GET(request("summary"), context(...path));
      expect(response.status).toBe(403);
      expect(await response.json()).toEqual({
        error: "curation_path_forbidden",
      });
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("returns 503 without exposing which server configuration is missing", async () => {
    delete process.env.CURATION_BEARER_TOKEN;
    const response = await GET(request("summary"), context("summary"));
    const body = await response.text();

    expect(response.status).toBe(503);
    expect(JSON.parse(body)).toEqual({ error: "curation_proxy_unavailable" });
    expect(body).not.toContain(BACKEND_URL);
  });

  test("uses a 120-second timeout and maps timeout failures to 504", async () => {
    let timeoutMs: number | undefined;
    AbortSignal.timeout = ((ms: number) => {
      timeoutMs = ms;
      return new AbortController().signal;
    }) as typeof AbortSignal.timeout;
    globalThis.fetch = mock(async () => {
      throw new DOMException("timed out", "TimeoutError");
    }) as typeof fetch;

    const response = await GET(request("summary"), context("summary"));

    expect(timeoutMs).toBe(120_000);
    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({
      error: "curation_backend_timeout",
    });
  });

  test("keeps the timeout active while parsing the upstream JSON body", async () => {
    globalThis.fetch = mock(async () => {
      const response = Response.json({ late: true });
      response.json = mock(async () => {
        throw new DOMException("body timed out", "TimeoutError");
      });
      return response;
    }) as typeof fetch;

    const response = await GET(request("summary"), context("summary"));

    expect(response.status).toBe(504);
    expect(await response.json()).toEqual({
      error: "curation_backend_timeout",
    });
  });

  for (const [kind, location] of [
    ["same-origin", `${BACKEND_URL}/api/curation/summary`],
    ["cross-origin", `https://redirect.invalid/${SERVER_TOKEN}`],
  ] as const) {
    test(`rejects a ${kind} upstream redirect without following or reflecting it`, async () => {
      let calls = 0;
      let redirectMode: RequestRedirect | undefined;
      globalThis.fetch = mock(
        async (_input: string | URL | Request, init?: RequestInit) => {
          calls += 1;
          redirectMode = init?.redirect;
          return Response.json(
            { redirect: location },
            { status: 302, headers: { location } },
          );
        },
      ) as typeof fetch;

      const response = await GET(request("summary"), context("summary"));
      const body = await response.text();

      expect(calls).toBe(1);
      expect(redirectMode).toBe("manual");
      expect(response.status).toBe(502);
      expect(JSON.parse(body)).toEqual({ error: "curation_backend_redirect" });
      expect(body).not.toContain(BACKEND_URL);
      expect(body).not.toContain(SERVER_TOKEN);
      expect(response.headers.has("location")).toBe(false);
    });
  }

  test("maps connection and invalid upstream JSON failures to generic 502 responses", async () => {
    console.error = mock(() => {
      throw new Error("the proxy must not log connection details or secrets");
    });
    globalThis.fetch = mock(async () => {
      throw new Error(`${BACKEND_URL} refused ${SERVER_TOKEN}`);
    }) as typeof fetch;
    const connectionResponse = await GET(
      request("summary"),
      context("summary"),
    );
    expect(connectionResponse.status).toBe(502);
    expect(await connectionResponse.json()).toEqual({
      error: "curation_backend_unreachable",
    });

    globalThis.fetch = mock(
      async () =>
        // Deliberately include sensitive-looking text; it must never be reflected.
        new Response(`${BACKEND_URL} ${SERVER_TOKEN}`, {
          headers: { "content-type": "text/plain" },
        }),
    ) as typeof fetch;
    const invalidResponse = await GET(request("summary"), context("summary"));
    const invalidBody = await invalidResponse.text();
    expect(invalidResponse.status).toBe(502);
    expect(JSON.parse(invalidBody)).toEqual({
      error: "curation_backend_invalid_response",
    });
    expect(invalidBody).not.toContain(SERVER_TOKEN);
  });
});
