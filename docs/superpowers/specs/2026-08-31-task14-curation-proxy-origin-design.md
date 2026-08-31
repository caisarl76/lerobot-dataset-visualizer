# Task 14 Curation Proxy Origin Design

## Context

The Task 14 browser smoke reaches the viewer at
`http://127.0.0.1:3000`, but the first curation mutation returns
`curation_cross_origin_forbidden`. In the live Next.js development runtime,
`request.nextUrl.origin` is normalized to `http://localhost:3000`, while the
browser request uses the contracted `127.0.0.1` authority. The observed
same-origin browser `POST` also omits both `Origin` and `Sec-Fetch-Site`, so the
current proxy gate has no trustworthy signal it accepts.

The proxy must solve that compatibility bug without trusting attacker-chosen
hostnames. In particular, comparing arbitrary matching `Host` and `Origin`
values would permit a DNS-rebinding hostname to reach bearer-backed curation
operations.

## Trusted Browser Authority

The Next.js proxy will freeze the Task 14 browser contract to these constants:

- origin: `http://127.0.0.1:3000`;
- authority: `127.0.0.1:3000`;
- mutation marker: `X-Curation-Request: same-origin`.

Every `GET`, `POST`, and `PATCH` request must contain exactly the trusted
`Host` value. Missing, empty, malformed, comma-joined, differently cased,
wrong-host, wrong-port, and scheme-bearing values are rejected. The proxy will
ignore `X-Forwarded-Host` and `X-Forwarded-Proto`; neither can replace or alter
the direct `Host` check.

When `Origin` is present, it must exactly equal the trusted origin. Empty,
`null`, malformed, comma-joined, wrong-scheme, wrong-host, and wrong-port
values are rejected. A genuinely absent `Origin` is permitted for `GET` and
may be permitted for a mutation only through the marker fallback described
below.

When `Sec-Fetch-Site` is present, it must exactly equal `same-origin`.
`same-site`, `cross-site`, `none`, empty, and comma-joined values are rejected.
Its absence is permitted because the reproduced browser mutation omits it.

These checks apply before configuration, path handling, body reads, or any
upstream request. Every rejection returns a bounded 403 JSON response and
must be proven not to contact the FastAPI backend.

## Mutation Marker Fallback

The shared curation client request helper will add
`X-Curation-Request: same-origin` to every non-`GET` operation. This includes
the bodyless batch-cancellation `POST`; it will not add a synthetic request
body or content type. `GET` operations do not receive the marker.

For `POST` and `PATCH`, the proxy accepts either:

1. the exact trusted `Origin`; or
2. a genuinely absent `Origin` plus the exact marker.

An invalid or mismatched `Origin` is never rescued by the marker. Whenever the
marker header is supplied, its value must be exactly `same-origin`; empty,
wrong, or comma-joined values are rejected even when `Origin` is valid or the
method is `GET`.

A cross-origin script cannot send this non-safelisted marker without a CORS
preflight. The route will explicitly answer `OPTIONS` with status 405 and no
`Access-Control-Allow-*` headers, so the browser receives no permission to
send the marked request.

## Upstream Boundary

The proxy will continue to construct upstream headers from scratch. It sends
only the server-held bearer and, for a real JSON body, the validated content
type. It never forwards `Host`, `Origin`, `Sec-Fetch-Site`, forwarded authority
headers, the marker, a browser authorization header, or other client headers.

This preserves the existing three-variable Next.js boundary:
`CURATION_BACKEND_URL`, `CURATION_BEARER_TOKEN`, and
`NEXT_PUBLIC_LOCAL_DATASET_BASE_URL`. The already configured, non-secret
`CURATION_BROWSER_ORIGIN` remains FastAPI-only; no new Next.js environment
coupling is introduced.

## Verification

Test-first route coverage will prove:

- valid trusted-authority `GET` and exact-origin mutations still proxy;
- the absent-origin marker fallback works for both JSON and bodyless
  mutations;
- every invalid Host, Origin, Fetch-Site, and marker case returns 403 before
  upstream access, including DNS-rebinding and forwarded-header attempts;
- a valid direct Host is authoritative even when forwarded authority headers
  are attacker-controlled;
- the marker and all other browser-only headers are absent upstream;
- the explicit `OPTIONS` handler returns 405 with no
  `Access-Control-Allow-*` header.

Client tests will exercise every public mutation through the shared request
helper and prove that each carries the exact marker, including bodyless
cancellation. They will also prove representative `GET` operations do not
carry it.

After focused and full frontend validation, the live Task 14 gate will use the
real Next.js dispatcher to confirm `OPTIONS` returns 405 without permissive
CORS headers, the documented annotations URL opens its workspace without a
403, local assets remain direct, and no secret or Cosmos configuration reaches
the browser. No Task 15 worker, batch, model proposal, or review action is in
scope.
