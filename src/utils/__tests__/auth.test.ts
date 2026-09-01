import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { AUTH_STORAGE_KEY, authHeaders, proxyHfUrl } from "@/utils/auth";

const protectedHf =
  "https://huggingface.co/datasets/acme/private/resolve/main/meta/info.json";
const localAsset =
  "http://127.0.0.1:8000/api/local-datasets/local/pnp_trash/resolve/main/meta/info.json";
const localVideo =
  "http://127.0.0.1:8000/api/local-datasets/local/pnp_trash/resolve/main/videos/observation.images.ego/chunk-000/file-000.mp4";

const originalWindowDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  "window",
);

function installSentinelToken() {
  const values = new Map<string, string>();
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => values.set(key, value),
      },
    },
  });
  window.localStorage.setItem(
    AUTH_STORAGE_KEY,
    JSON.stringify({ accessToken: "sentinel-hf-token" }),
  );
}

beforeEach(installSentinelToken);

afterEach(() => {
  if (originalWindowDescriptor) {
    Object.defineProperty(globalThis, "window", originalWindowDescriptor);
  } else {
    delete (globalThis as { window?: unknown }).window;
  }
});

describe("destination-aware Hugging Face authentication", () => {
  test("only attaches the token to exact HTTPS huggingface.co destinations", () => {
    expect(authHeaders(protectedHf)).toEqual({
      Authorization: "Bearer sentinel-hf-token",
    });
    expect(authHeaders(localAsset)).toEqual({});
    expect(authHeaders("https://huggingface.co.evil.test/file")).toEqual({});
    expect(authHeaders("http://huggingface.co/file")).toEqual({});
    expect(authHeaders("/api/curation/summary")).toEqual({});
    expect(authHeaders("not a url")).toEqual({});
  });

  test("does not proxy local or lookalike video URLs", () => {
    expect(proxyHfUrl(protectedHf)).toBe(
      "/api/proxy/datasets/acme/private/resolve/main/meta/info.json",
    );
    expect(proxyHfUrl(localVideo)).toBe(localVideo);
    expect(proxyHfUrl("https://huggingface.co.evil.test/video.mp4")).toBe(
      "https://huggingface.co.evil.test/video.mp4",
    );
    expect(proxyHfUrl("http://huggingface.co/video.mp4")).toBe(
      "http://huggingface.co/video.mp4",
    );
  });
});
