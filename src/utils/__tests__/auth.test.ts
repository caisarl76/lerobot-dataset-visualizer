import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  AUTH_STORAGE_KEY,
  authHeaders,
  proxyHfUrl,
  resolveDatasetFetchUrl,
} from "@/utils/auth";

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

test("hosted server reads resolve draft assets with server-only authentication", () => {
  const saved = {
    url: process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL,
    backend: process.env.ANNOTATION_BACKEND_URL,
    token: process.env.ANNOTATION_BACKEND_TOKEN,
  };
  process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = "/api/annotation";
  process.env.ANNOTATION_BACKEND_URL = "https://backend.example";
  process.env.ANNOTATION_BACKEND_TOKEN = "server-only-secret";
  const path =
    "/api/annotation/datasets/local/annotation-demo/resolve/main/meta/info.json";
  try {
    expect(resolveDatasetFetchUrl(path)).toBe(path);
    expect(
      authHeaders(
        "https://backend.example/datasets/local/annotation-demo/resolve/main/meta/info.json",
      ),
    ).toEqual({});
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: undefined,
    });
    const resolved = resolveDatasetFetchUrl(path);
    expect(resolved).toBe(
      "https://backend.example/datasets/local/annotation-demo/resolve/main/meta/info.json",
    );
    expect(authHeaders(resolved)).toEqual({
      Authorization: "Bearer server-only-secret",
    });
    expect(
      authHeaders(
        "https://backend.example.evil/datasets/local/annotation-demo/resolve/main/meta/info.json",
      ),
    ).toEqual({});
    expect(resolveDatasetFetchUrl(protectedHf)).toBe(protectedHf);
  } finally {
    for (const [key, value] of Object.entries({
      NEXT_PUBLIC_ANNOTATE_BACKEND_URL: saved.url,
      ANNOTATION_BACKEND_URL: saved.backend,
      ANNOTATION_BACKEND_TOKEN: saved.token,
    })) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});
