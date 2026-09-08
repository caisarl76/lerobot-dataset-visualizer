import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const MAX_BODY = 1024 * 1024;
const JSON_TYPE = /^application\/(?:[a-z0-9.+-]*\+)?json(?:\s*;|$)/i;
const SECRET_KEYS = new Set([
  "local_path",
  "output_dir",
  "token",
  "hf_token",
  "access_token",
  "api_key",
  "api_base",
  "serve_command",
]);
type Context = { params: Promise<{ path: string[] }> };

const fail = (error: string, status: number) =>
  Response.json(
    { error },
    { status, headers: { "cache-control": "no-store" } },
  );
function config() {
  if (process.env.ANNOTATION_HOSTED_PRIVATE_SPACE !== "1") return null;
  const raw = process.env.ANNOTATION_BACKEND_URL,
    token = process.env.ANNOTATION_BACKEND_TOKEN,
    origin = process.env.ANNOTATION_BROWSER_ORIGIN;
  if (!raw || !token || !origin) return null;
  try {
    const url = new URL(raw);
    const loopback = ["127.0.0.1", "localhost", "::1"].includes(url.hostname);
    if (
      (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash
    )
      return null;
    url.pathname = `${url.pathname.replace(/\/+$/, "")}/`;
    return { url, token, origin };
  } catch {
    return null;
  }
}
function paths(raw: string[]) {
  const out: string[] = [];
  for (const part of raw) {
    let value: string;
    try {
      value = decodeURIComponent(part);
    } catch {
      return null;
    }
    if (!value || value === "." || value === ".." || /[\\/\0]/.test(value))
      return null;
    out.push(value);
  }
  return out.length ? out : null;
}
function allowed(path: string[], method: string) {
  const key = path.join("/");
  if (method === "GET" || method === "HEAD") {
    if (["api/health", "api/annotation/config"].includes(key)) return true;
    if (
      /^api\/annotation\/jobs\/[0-9a-f]{32}$/.test(key) ||
      /^api\/workflow\/[A-Za-z0-9_-]+$/.test(key)
    )
      return true;
    if (/^api\/episodes\/\d+\/(atoms|review|frame_timestamps)$/.test(key))
      return true;
    if (
      /^datasets\/local\/[A-Za-z0-9_-]+\/resolve\/main\/(meta|data|videos)\/.+/.test(
        key,
      )
    )
      return true;
  }
  if (method === "POST")
    return (
      [
        "api/dataset/load",
        "api/annotation/prepare",
        "api/annotation/jobs",
        "api/annotation/validate",
      ].includes(key) ||
      /^api\/episodes\/\d+\/(atoms|review)$/.test(key) ||
      /^api\/workflow\/[A-Za-z0-9_-]+\/(decision|export|publish)$/.test(key)
    );
  return false;
}
function forbidden(value: unknown, key?: string): boolean {
  if (key && SECRET_KEYS.has(key) && !(key === "local_path" && value === null))
    return true;
  if (Array.isArray(value)) return value.some((item) => forbidden(item));
  if (value && typeof value === "object")
    return Object.entries(value).some(([k, v]) => forbidden(v, k));
  return false;
}
async function handle(
  request: NextRequest,
  context: Context,
  method: "GET" | "HEAD" | "POST",
) {
  const cfg = config();
  if (!cfg) return fail("annotation_proxy_unavailable", 503);
  if (method === "POST" && request.headers.get("origin") !== cfg.origin)
    return fail("annotation_origin_forbidden", 403);
  const rawPath = paths(await context.params.then((p) => p.path));
  const path =
    rawPath && !["api", "datasets"].includes(rawPath[0])
      ? ["api", ...rawPath]
      : rawPath;
  if (!path || !allowed(path, method))
    return fail("annotation_path_forbidden", 403);
  for (const key of request.nextUrl.searchParams.keys())
    if (SECRET_KEYS.has(key)) return fail("annotation_secret_forbidden", 400);
  const headers = new Headers({ authorization: `Bearer ${cfg.token}` });
  for (const name of [
    "range",
    "if-range",
    "if-none-match",
    "if-modified-since",
  ]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  let body: string | undefined;
  if (method === "POST") {
    const contentType = request.headers.get("content-type");
    if (!contentType || !JSON_TYPE.test(contentType))
      return fail("json_content_type_required", 415);
    body = await request.text();
    if (new TextEncoder().encode(body).byteLength > MAX_BODY)
      return fail("annotation_body_too_large", 413);
    try {
      const parsed = JSON.parse(body);
      if (forbidden(parsed)) return fail("annotation_secret_forbidden", 400);
    } catch {
      return fail("invalid_json", 400);
    }
    headers.set("content-type", contentType);
  }
  const upstream = new URL(path.map(encodeURIComponent).join("/"), cfg.url);
  upstream.search = request.nextUrl.search;
  try {
    const response = await fetch(upstream, {
      method,
      headers,
      body,
      cache: "no-store",
      redirect: "manual",
      signal: AbortSignal.timeout(120000),
    });
    if (
      response.status >= 300 &&
      response.status < 400 &&
      response.status !== 304
    )
      return fail("annotation_backend_redirect", 502);
    const out = new Headers();
    for (const name of [
      "content-type",
      "content-length",
      "content-range",
      "etag",
      "last-modified",
      "accept-ranges",
      "cache-control",
    ]) {
      const value = response.headers.get(name);
      if (value) out.set(name, value);
    }
    if (path[0] === "api") out.set("cache-control", "no-store");
    return new Response(
      method === "HEAD" || response.status === 304 ? null : response.body,
      {
        status: response.status,
        headers: out,
      },
    );
  } catch {
    return fail("annotation_backend_unreachable", 502);
  }
}
export function GET(request: NextRequest, context: Context) {
  return handle(request, context, "GET");
}
export function HEAD(request: NextRequest, context: Context) {
  return handle(request, context, "HEAD");
}
export function POST(request: NextRequest, context: Context) {
  return handle(request, context, "POST");
}
