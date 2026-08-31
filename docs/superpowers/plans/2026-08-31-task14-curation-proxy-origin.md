# Task 14 Curation Proxy Origin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the real loopback browser able to use the bearer-backed curation proxy without trusting arbitrary hostnames or granting cross-origin access.

**Architecture:** The existing browser client helper will mark every mutation with one fixed non-safelisted header, including bodyless cancellation. The existing Next.js catch-all route will enforce a frozen `127.0.0.1:3000` authority on every request, accept mutations through either the exact trusted origin or an absent-origin marker fallback, reject all malformed metadata before upstream access, and answer preflight with a nonpermissive 405. Upstream headers remain constructed from scratch.

**Tech Stack:** TypeScript, Next.js 15 route handlers, Fetch `Headers`/`Request` APIs, Bun test, curl-based loopback smoke testing.

---

### Task 1: Mark every browser mutation centrally

**Files:**

- Modify: `src/utils/__tests__/curationClient.test.ts`
- Modify: `src/utils/curationClient.ts:1076-1100`

- [ ] **Step 1: Write the failing client contract test**

Extend the imports with `applyEpisodeProposal`, `approveEpisodeReject`, and
`reopenEpisode`. Add a test that invokes every public mutation through the real
shared helper, allows each intentionally malformed response to fail decoding,
and inspects the outgoing request:

```ts
test("marks every curation mutation, including bodyless cancellation, but not reads", async () => {
  const mutations: Array<[string, () => Promise<unknown>]> = [
    [
      "workspace open",
      () => openCurationWorkspace("local/pnp_trash", "curator"),
    ],
    [
      "draft save",
      () =>
        saveEpisodeDraft("local/pnp_trash", 0, 0, "curator", {
          transitionFrames: [1, 2, 3, 4, 5, 6],
        }),
    ],
    [
      "proposal apply",
      () => applyEpisodeProposal("local/pnp_trash", 0, 0, "curator"),
    ],
    [
      "keep approval",
      () => approveEpisodeKeep("local/pnp_trash", 0, 0, "curator", "reviewer"),
    ],
    [
      "reject approval",
      () =>
        approveEpisodeReject(
          "local/pnp_trash",
          0,
          0,
          "curator",
          "reviewer",
          "bad grasp",
        ),
    ],
    ["episode reopen", () => reopenEpisode("local/pnp_trash", 0, 0, "curator")],
    ["batch start", () => startCurationBatch("local/pnp_trash")],
    [
      "batch retry",
      () =>
        retryCurationBatch(
          "job-1",
          { failureStates: ["manual_only"] },
          "local/pnp_trash",
        ),
    ],
    ["batch cancel", () => cancelCurationBatch("job-1")],
  ];

  for (const [name, operation] of mutations) {
    let observedInit: RequestInit | undefined;
    globalThis.fetch = mock(
      async (_input: string | URL | Request, init?: RequestInit) => {
        observedInit = init;
        return Response.json({ malformed: name });
      },
    ) as typeof fetch;

    await expect(operation()).rejects.toMatchObject({
      code: "invalid_response",
    });
    const headers = new Headers(observedInit?.headers);
    expect(headers.get("x-curation-request")).toBe("same-origin");
    if (name === "batch cancel") {
      expect(observedInit?.body).toBeUndefined();
      expect(headers.has("content-type")).toBe(false);
    }
  }

  let readHeaders = new Headers();
  globalThis.fetch = mock(
    async (_input: string | URL | Request, init?: RequestInit) => {
      readHeaders = new Headers(init?.headers);
      return Response.json({ malformed: "summary" });
    },
  ) as typeof fetch;
  await expect(fetchCurationSummary("local/pnp_trash")).rejects.toMatchObject({
    code: "invalid_response",
  });
  expect(readHeaders.has("x-curation-request")).toBe(false);
});
```

- [ ] **Step 2: Run the focused client test and verify RED**

Run:

```bash
bun test src/utils/__tests__/curationClient.test.ts
```

Expected: the new test fails because mutation requests do not contain
`X-Curation-Request`.

- [ ] **Step 3: Add the marker in the shared request helper**

Define fixed header constants beside `requestJson`, resolve the method once,
and add the marker only for non-GET methods:

```ts
const CURATION_REQUEST_HEADER = "x-curation-request";
const CURATION_REQUEST_VALUE = "same-origin";

async function requestJson<T>(
  path: string,
  decoder: Decoder<T>,
  options: {
    method?: "GET" | "POST" | "PATCH";
    body?: unknown;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  const method = options.method ?? "GET";
  const headers = new Headers({ accept: "application/json" });
  if (method !== "GET") {
    headers.set(CURATION_REQUEST_HEADER, CURATION_REQUEST_VALUE);
  }
  const init: RequestInit = {
    method,
    cache: "no-store",
    credentials: "same-origin",
    signal: options.signal,
    headers,
  };
```

Replace only the existing header/`RequestInit` construction through its closing
brace with this block. Leave the immediately following existing
`if (options.body !== undefined)` branch and the fetch/error/decoder branches
byte-for-byte unchanged. The body branch therefore remains conditional on a
real body and cancellation stays bodyless.

- [ ] **Step 4: Run the focused suite and verify GREEN**

Run:

```bash
bun test src/utils/__tests__/curationClient.test.ts
bunx tsc -p tsconfig.test.json --noEmit
```

Expected: all client tests and the test compiler pass.

- [ ] **Step 5: Commit the client slice**

```bash
git add src/utils/curationClient.ts src/utils/__tests__/curationClient.test.ts
git commit -m "fix: mark curation browser mutations"
```

### Task 2: Enforce the frozen proxy authority and preflight boundary

**Files:**

- Modify: `src/app/api/curation/[...path]/__tests__/route.test.ts`
- Modify: `src/app/api/curation/[...path]/route.ts:1-155`

- [ ] **Step 1: Write the failing proxy security tests**

Import `OPTIONS` from the route. Replace the request helper with this
metadata-aware helper so ordinary existing calls receive the trusted direct
Host and existing mutations receive exact trusted origin metadata, while
security cases can explicitly omit or corrupt each header:

```ts
const BROWSER_ORIGIN = "http://127.0.0.1:3000";
const BROWSER_HOST = "127.0.0.1:3000";
const MUTATION_MARKER = "same-origin";

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
```

Add tests covering these contracts:

```ts
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
      {
        method: "POST",
      },
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
});
```

The explicit third argument makes both calls exercise the reproduced
absent-origin path rather than the helper defaults.

Add this rejected-request table. It covers missing, empty, comma-joined,
malformed, and mismatched authority metadata; the DNS-rebinding shape;
forwarded-header substitution; every disallowed Fetch-Site value; missing
mutation evidence; and all invalid supplied marker forms:

```ts
test("rejects every untrusted browser shape before upstream access", async () => {
  const fetchSpy = mock(async () => Response.json({ unexpected: true }));
  globalThis.fetch = fetchSpy as typeof fetch;
  const rejected: Array<[string, "GET" | "POST" | "PATCH", BrowserMetadata]> = [
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
    ["empty Origin", "GET", { origin: "" }],
    ["null Origin", "GET", { origin: "null" }],
    ["malformed Origin", "GET", { origin: "not-an-origin" }],
    [
      "duplicate Origin",
      "GET",
      { origin: `${BROWSER_ORIGIN}, ${BROWSER_ORIGIN}` },
    ],
    ["wrong Origin scheme", "GET", { origin: "https://127.0.0.1:3000" }],
    ["wrong Origin host", "GET", { origin: "http://localhost:3000" }],
    ["wrong Origin port", "GET", { origin: "http://127.0.0.1:3001" }],
    ["same-site Fetch-Site", "GET", { fetchSite: "same-site" }],
    ["cross-site Fetch-Site", "GET", { fetchSite: "cross-site" }],
    ["none Fetch-Site", "GET", { fetchSite: "none" }],
    ["empty Fetch-Site", "GET", { fetchSite: "" }],
    ["duplicate Fetch-Site", "GET", { fetchSite: "same-origin, same-origin" }],
    ["empty marker", "GET", { marker: "" }],
    ["wrong marker", "GET", { marker: "cross-origin" }],
    [
      "duplicate marker",
      "GET",
      { marker: `${MUTATION_MARKER}, ${MUTATION_MARKER}` },
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
      { origin: BROWSER_ORIGIN, fetchSite: "same-origin", marker: "wrong" },
    ],
  ];

  for (const [name, method, metadata] of rejected) {
    const handler = method === "GET" ? GET : method === "POST" ? POST : PATCH;
    const response = await handler(
      request("summary", { method }, metadata),
      context("summary"),
    );
    expect(response.status, name).toBe(403);
    expect(await response.json(), name).toEqual({
      error: "curation_cross_origin_forbidden",
    });
  }
  expect(fetchSpy).not.toHaveBeenCalled();
});
```

For every rejected table row the final assertion proves no request reached the
upstream mock.

Add a positive case showing exact direct Host/Origin remain authoritative even
when `X-Forwarded-Host` and `X-Forwarded-Proto` contain attacker values:

```ts
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
```

Extend the successful mutation assertion to prove the upstream headers contain
only the server authorization and validated content type, with no marker,
Host, Origin, Fetch-Site, forwarded authority, or browser authorization:

```ts
const upstreamHeaders = new Headers(observedInit?.headers);
expect([...upstreamHeaders.keys()].sort()).toEqual([
  "authorization",
  "content-type",
]);
expect(upstreamHeaders.get("authorization")).toBe(`Bearer ${SERVER_TOKEN}`);
expect(upstreamHeaders.has("x-curation-request")).toBe(false);
```

Finally, exercise the exported handler itself:

```ts
test("OPTIONS is a nonpermissive 405 response", async () => {
  const fetchSpy = mock(async () => Response.json({ unexpected: true }));
  globalThis.fetch = fetchSpy as typeof fetch;

  const response = OPTIONS();

  expect(response.status).toBe(405);
  for (const name of response.headers.keys()) {
    expect(name.startsWith("access-control-allow-")).toBe(false);
  }
  expect(fetchSpy).not.toHaveBeenCalled();
});
```

- [ ] **Step 2: Run the route suite and verify RED**

Run:

```bash
bun test 'src/app/api/curation/[...path]/__tests__/route.test.ts'
```

Expected: the suite fails because the marker fallback, strict authority gate,
GET validation, and explicit `OPTIONS` handler do not yet exist.

- [ ] **Step 3: Implement the minimal centralized proxy gate**

Replace `isSameOriginMutation` with constants and one method-aware validator:

```ts
const TRUSTED_BROWSER_ORIGIN = "http://127.0.0.1:3000";
const TRUSTED_BROWSER_HOST = "127.0.0.1:3000";
const CURATION_REQUEST_HEADER = "x-curation-request";
const CURATION_REQUEST_VALUE = "same-origin";

function isTrustedBrowserRequest(
  request: NextRequest,
  method: "GET" | "POST" | "PATCH",
): boolean {
  const host = request.headers.get("host");
  const origin = request.headers.get("origin");
  const fetchSite = request.headers.get("sec-fetch-site");
  const marker = request.headers.get(CURATION_REQUEST_HEADER);

  if (host !== TRUSTED_BROWSER_HOST) return false;
  if (origin !== null && origin !== TRUSTED_BROWSER_ORIGIN) return false;
  if (fetchSite !== null && fetchSite !== "same-origin") return false;
  if (marker !== null && marker !== CURATION_REQUEST_VALUE) return false;
  if (method === "GET") return true;
  return (
    origin === TRUSTED_BROWSER_ORIGIN ||
    (origin === null && marker === CURATION_REQUEST_VALUE)
  );
}
```

Call this at the first line of `forward` for every method. Do not read
`request.nextUrl.origin`, `X-Forwarded-Host`, or `X-Forwarded-Proto`. Retain
the fresh upstream `Headers` construction so the marker cannot be forwarded.

Add the explicit preflight handler:

```ts
export function OPTIONS(): Response {
  return new Response(null, {
    status: 405,
    headers: {
      allow: "GET, POST, PATCH",
      "cache-control": "no-store",
    },
  });
}
```

- [ ] **Step 4: Run focused security verification and verify GREEN**

Run:

```bash
bun test 'src/app/api/curation/[...path]/__tests__/route.test.ts'
bun test src/utils/__tests__/curationClient.test.ts
bunx tsc -p tsconfig.test.json --noEmit
```

Expected: all focused tests and the test compiler pass.

- [ ] **Step 5: Commit the proxy slice**

```bash
git add 'src/app/api/curation/[...path]/route.ts' \
  'src/app/api/curation/[...path]/__tests__/route.test.ts'
git commit -m "fix: harden curation proxy origin checks"
```

### Task 3: Re-run Task 14 regression and live browser/security gates

**Files:**

- Verify: `src/**`
- Verify: `docs/pnp-trash-curation-runbook.md`
- Preserve: approved source and resumable workspace

- [ ] **Step 1: Run full frontend validation**

Run:

```bash
bun run type-check
bun run lint
bun run format:check
bun test
git diff --check
git status --short
```

Expected: 235 or more tests pass, type checks and formatting pass, lint has at
most the three already documented hook warnings, and the worktree is clean
after any formatting-only commit required by the checks.

- [ ] **Step 2: Prove real Next.js OPTIONS dispatch is nonpermissive**

Against the running loopback Next.js service, run:

```bash
curl -sS -i -X OPTIONS \
  -H 'Origin: http://attacker.example:3000' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: X-Curation-Request' \
  http://127.0.0.1:3000/api/curation/workspaces/open
```

Expected: status 405 and no `Access-Control-Allow-Origin`,
`Access-Control-Allow-Methods`, or `Access-Control-Allow-Headers` response.

- [ ] **Step 3: Prove rejected live requests never reach FastAPI**

Hash the backend log, send a DNS-rebinding-shaped `GET` and a marked mutation
with attacker origin to the real Next.js route, then hash it again:

```bash
sha256sum /tmp/pnp-task14-backend-45d1cce.log
curl -sS -o /tmp/pnp-task14-rebind-get.json -w '%{http_code}\n' \
  -H 'Host: attacker.example:3000' \
  -H 'Origin: http://attacker.example:3000' \
  -H 'Sec-Fetch-Site: same-origin' \
  http://127.0.0.1:3000/api/curation/summary
curl -sS -o /tmp/pnp-task14-rebind-post.json -w '%{http_code}\n' \
  -X POST \
  -H 'Host: 127.0.0.1:3000' \
  -H 'Origin: http://attacker.example:3000' \
  -H 'Sec-Fetch-Site: same-origin' \
  -H 'X-Curation-Request: same-origin' \
  http://127.0.0.1:3000/api/curation/batches/job-1/cancel
sha256sum /tmp/pnp-task14-backend-45d1cce.log
```

Expected: both status codes are 403, both bounded bodies equal
`{"error":"curation_cross_origin_forbidden"}`, and the log hash is unchanged.
Then send a valid direct Host request with hostile forwarded authority headers:

```bash
curl -sS -o /tmp/pnp-task14-forwarded-get.json -w '%{http_code}\n' \
  -H 'Host: 127.0.0.1:3000' \
  -H 'X-Forwarded-Host: attacker.example:3000' \
  -H 'X-Forwarded-Proto: https' \
  'http://127.0.0.1:3000/api/curation/summary?dataset_alias=local%2Fpnp_trash'
```

Expected: status 200, proving forwarded metadata is ignored rather than
substituted for the trusted direct Host.

- [ ] **Step 4: Re-run the real documented browser smoke**

Open
`http://127.0.0.1:3000/local/pnp_trash/episode_0?tab=annotations` in the
existing private headless browser. Verify the workspace opens without
`curation_cross_origin_forbidden`, the seven-phase task-index UI and metadata
charts render, the video is seekable, local assets load directly from port
8000, and the DOM/scripts/requests reveal neither bearer secret nor Cosmos
endpoint configuration.

- [ ] **Step 5: Preserve evidence and stop temporary processes**

Confirm the approved source manifest and bytes remain unchanged, the cleaned
output remains absent, and no batch/worker/model proposal was created. Preserve
the resumable curation workspace, stop the private Chrome, Next/FastAPI, and SSH
tunnel sessions, remove only their exact temporary Chrome profile after its
process exits, and report the Task 14 results before any merge, push, or pull
request.
