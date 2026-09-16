import { afterEach, expect, mock, spyOn, test } from "bun:test";
import * as annotations from "../annotationsClient";
import {
  fetchMonitorSummary,
  fetchMonitorDetail,
  checkMonitorPublication,
} from "../monitorClient";

afterEach(() => mock.restore());
function backend(base: string | null) {
  spyOn(annotations, "getAnnotateBackendUrl").mockReturnValue(base);
  return spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify({ configured: true })),
  );
}
test("absolute backend URLs resolve the API from the origin and forward abort", async () => {
  const fetch = backend("http://localhost:7861/base/");
  const controller = new AbortController();
  await fetchMonitorSummary(true, controller.signal);
  expect(fetch.mock.calls[0][0]).toBe(
    "http://localhost:7861/api/monitor?refresh=true",
  );
  expect(fetch.mock.calls[0][1]?.signal).toBe(controller.signal);
});
test("relative bases preserve the proxy prefix and encode dataset/run identities", async () => {
  const fetch = backend("/api/annotation/");
  await fetchMonitorDetail("folder /?", "run /&?", undefined);
  expect(fetch.mock.calls[0][0]).toBe(
    "/api/annotation/api/monitor/datasets/folder%20%2F%3F?run_id=run%20%2F%26%3F",
  );
});
test("summary default does not force refresh", async () => {
  const fetch = backend("/backend");
  await fetchMonitorSummary();
  expect(fetch.mock.calls[0][0]).toBe("/backend/api/monitor");
});
test("publication check returns RemoteCheck directly and explicitly POSTs only its opaque id", async () => {
  const fetch = backend("/backend");
  const result = {
    status: "match" as const,
    checked_at: "now",
    current_commit: "sha",
    message: "Match",
  };
  fetch.mockResolvedValue(new Response(JSON.stringify(result)));
  const controller = new AbortController();
  expect(await checkMonitorPublication("pub /?", controller.signal)).toEqual(
    result,
  );
  expect(fetch.mock.calls[0][0]).toBe(
    "/backend/api/monitor/publications/pub%20%2F%3F/check",
  );
  expect(fetch.mock.calls[0][1]?.method).toBe("POST");
  expect(fetch.mock.calls[0][1]?.signal).toBe(controller.signal);
  expect(fetch.mock.calls[0][1]?.body).toBeUndefined();
});
test("unconfigured backend is distinct from transport failures", async () => {
  const fetch = backend(null);
  await expect(fetchMonitorSummary()).rejects.toThrow(
    "Annotation backend is not configured.",
  );
  expect(fetch).not.toHaveBeenCalled();
  spyOn(annotations, "getAnnotateBackendUrl").mockReturnValue("/backend");
  fetch.mockRejectedValue(new TypeError("Failed to fetch"));
  await expect(fetchMonitorSummary()).rejects.toThrow("Failed to fetch");
});
test("HTTP detail messages are preserved with status and empty bodies have a fallback", async () => {
  const fetch = backend("/backend");
  fetch.mockResolvedValueOnce(
    new Response(JSON.stringify({ detail: "Collection root unavailable" }), {
      status: 503,
    }),
  );
  await expect(fetchMonitorSummary()).rejects.toThrow(
    "Collection root unavailable",
  );
  fetch.mockResolvedValueOnce(new Response("", { status: 404 }));
  await expect(fetchMonitorDetail("missing", "run")).rejects.toThrow("404");
});
test("aborts remain identifiable to the caller", async () => {
  const fetch = backend("/backend");
  const abort = new DOMException("Aborted", "AbortError");
  fetch.mockRejectedValueOnce(abort);
  try {
    await fetchMonitorSummary();
    throw new Error("Expected abort");
  } catch (error) {
    expect(error).toBe(abort);
  }
});
