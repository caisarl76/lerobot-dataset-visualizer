import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const UPSTREAM_TIMEOUT_MS = 120_000;
const JSON_CONTENT_TYPE = /^application\/(?:[a-z0-9.+-]*\+)?json(?:\s*;|$)/i;
const FORBIDDEN_ASSET_SEGMENTS = new Set([
  "asset",
  "assets",
  "local-datasets",
  "parquet",
  "video",
  "videos",
]);

type RouteContext = { params: Promise<{ path: string[] }> };

function json(payload: unknown, status: number): Response {
  return Response.json(payload, {
    status,
    headers: { "cache-control": "no-store" },
  });
}

function configuration(): { backend: URL; token: string } | null {
  const rawBackend = process.env.CURATION_BACKEND_URL;
  const token = process.env.CURATION_BEARER_TOKEN;
  if (!rawBackend || !token) return null;

  try {
    const backend = new URL(rawBackend);
    if (!["http:", "https:"].includes(backend.protocol)) return null;
    if (backend.username || backend.password || backend.search || backend.hash)
      return null;
    backend.pathname = `${backend.pathname.replace(/\/+$/, "")}/`;
    return { backend, token };
  } catch {
    return null;
  }
}

function safePath(path: string[]): string[] | null {
  if (path.length === 0) return null;
  const safe: string[] = [];
  for (const segment of path) {
    let decoded: string;
    try {
      decoded = decodeURIComponent(segment);
    } catch {
      return null;
    }
    const normalized = decoded.toLowerCase();
    if (
      decoded.length === 0 ||
      decoded === "." ||
      decoded === ".." ||
      decoded.includes("/") ||
      decoded.includes("\\") ||
      decoded.includes("\0") ||
      FORBIDDEN_ASSET_SEGMENTS.has(normalized) ||
      /\.(?:mp4|parquet)$/i.test(decoded)
    ) {
      return null;
    }
    safe.push(decoded);
  }
  return safe;
}

function isJsonContentType(value: string | null): boolean {
  return value !== null && JSON_CONTENT_TYPE.test(value);
}

function isSameOriginMutation(request: NextRequest): boolean {
  const origin = request.headers.get("origin");
  const fetchSite = request.headers.get("sec-fetch-site");
  if (origin !== null && origin !== request.nextUrl.origin) return false;
  if (fetchSite !== null && fetchSite !== "same-origin") return false;
  return origin === request.nextUrl.origin || fetchSite === "same-origin";
}

async function forward(
  request: NextRequest,
  context: RouteContext,
  method: "GET" | "POST" | "PATCH",
): Promise<Response> {
  if (method !== "GET" && !isSameOriginMutation(request)) {
    return json({ error: "curation_cross_origin_forbidden" }, 403);
  }

  const config = configuration();
  if (config === null)
    return json({ error: "curation_proxy_unavailable" }, 503);

  const { path } = await context.params;
  const segments = safePath(path);
  if (segments === null) return json({ error: "curation_path_forbidden" }, 403);

  const headers = new Headers({ authorization: `Bearer ${config.token}` });
  let body: string | undefined;
  if (method !== "GET" && request.body !== null) {
    const contentType = request.headers.get("content-type");
    if (!isJsonContentType(contentType)) {
      return json({ error: "json_content_type_required" }, 415);
    }
    headers.set("content-type", contentType!);
    body = await request.text();
  }

  const upstream = new URL(
    `api/curation/${segments.map(encodeURIComponent).join("/")}`,
    config.backend,
  );
  upstream.search = request.nextUrl.search;

  const timeoutSignal = AbortSignal.timeout(UPSTREAM_TIMEOUT_MS);
  const upstreamSignal = AbortSignal.any([request.signal, timeoutSignal]);
  try {
    const response = await fetch(upstream, {
      method,
      headers,
      body,
      cache: "no-store",
      redirect: "manual",
      signal: upstreamSignal,
    });
    if (response.status >= 300 && response.status < 400) {
      return json({ error: "curation_backend_redirect" }, 502);
    }
    if (!isJsonContentType(response.headers.get("content-type"))) {
      return json({ error: "curation_backend_invalid_response" }, 502);
    }
    return json(await response.json(), response.status);
  } catch (error) {
    if (
      (error instanceof Error && error.name === "TimeoutError") ||
      (timeoutSignal.aborted &&
        timeoutSignal.reason instanceof Error &&
        timeoutSignal.reason.name === "TimeoutError")
    ) {
      return json({ error: "curation_backend_timeout" }, 504);
    }
    if (error instanceof SyntaxError) {
      return json({ error: "curation_backend_invalid_response" }, 502);
    }
    return json({ error: "curation_backend_unreachable" }, 502);
  }
}

export function GET(
  request: NextRequest,
  context: RouteContext,
): Promise<Response> {
  return forward(request, context, "GET");
}

export function POST(
  request: NextRequest,
  context: RouteContext,
): Promise<Response> {
  return forward(request, context, "POST");
}

export function PATCH(
  request: NextRequest,
  context: RouteContext,
): Promise<Response> {
  return forward(request, context, "PATCH");
}
