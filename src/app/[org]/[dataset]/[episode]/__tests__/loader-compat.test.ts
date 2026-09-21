import { describe, expect, test } from "bun:test";
import type {
  EpisodeData,
  EpisodeFramesData,
  EpisodeLengthStats,
} from "../fetch-data";

// Tiny, uncompressed parquet fixtures: two metadata rows and four data rows.
// Data indices start at 100 to exercise shared-file/global index conversion.
const fixtures = {
  metadata:
    "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",
  data: "UEFSMRUAFYABFYABLBUIFQAVBhUGHAAAAAIAAAAIAAIAAAAIBAkAAABhbm5vdGF0b3IJAAAAYW5ub3RhdG9yCQAAAGFubm90YXRvcgkAAABhbm5vdGF0b3IVABWQARWQASwVCBUAFQYVBhwAAAACAAAACAACAAAACAQLAAAAbW92ZSBzYWZlbHkLAAAAbW92ZSBzYWZlbHkLAAAAbW92ZSBzYWZlbHkLAAAAbW92ZSBzYWZlbHkVABVwFXAsFQgVABUGFQYcAAAAAgAAAAgAAgAAAAgEBwAAAHN1YnRhc2sHAAAAc3VidGFzawcAAABzdWJ0YXNrBwAAAHN1YnRhc2sVABVMFUwsFQgVABUGFQYcAAAAAgAAAAgBZAAAAAAAAABlAAAAAAAAAGYAAAAAAAAAZwAAAAAAAAAVABVMFUwsFQgVABUGFQYcAAAAAgAAAAgBAAAAAAAAAACamZmZmZm5PwAAAAAAAAAAmpmZmZmZuT8VABVMFUwsFQgVABUGFQYcAAAAAgAAAAgBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVABVMFUwsFQgVABUGFQYcAAAAAgAAAAgBBAAAAG1vdmUEAAAAbW92ZQQAAABtb3ZlBAAAAG1vdmUVABWYARWYASwVEBUAFQYVBhwAAAACAAAAA6oCAAAAEAMAAAAAAADwPwAAAAAAAABAAAAAAAAAAEAAAAAAAAAIQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAgQAAAAAAAACJAFQAVmAEVmAEsFRAVABUGFQYcAAAAAgAAAAOqAgAAABADAAAAAAAACEAAAAAAAAAQQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAYQAAAAAAAABxAAAAAAAAAJEAAAAAAAAAmQBUEGfwRNQAYBnNjaGVtYRUOADUCGBNsYW5ndWFnZV9wZXJzaXN0ZW50FQIVBkw8AAAANQQYBGxpc3QVAgA1AhgHZWxlbWVudBUGABUMJQIYBHJvbGUlAEwcAAAAFQwlAhgHY29udGVudCUATBwAAAAVDCUCGAVzdHlsZSUATBwAAAAVBCUCGAVpbmRleAAVCiUCGAl0aW1lc3RhbXAAFQQlAhgKdGFza19pbmRleAAVDCUCGBRsYW5ndWFnZV9pbnN0cnVjdGlvbiUATBwAAAA1AhgGYWN0aW9uFQIVBkw8AAAANQQYBGxpc3QVAgAVCiUCGAdlbGVtZW50ADUCGBFvYnNlcnZhdGlvbi5zdGF0ZRUCFQZMPAAAADUEGARsaXN0FQIAFQolAhgHZWxlbWVudAAWCBkcGZwmABwVDBklBgAZSBNsYW5ndWFnZV9wZXJzaXN0ZW50BGxpc3QHZWxlbWVudARyb2xlFQAWCBaqARaqASYISRwVABUAFQIAPBZIGSYIABlWAAAAAAgAAAAmABwVDBklBgAZSBNsYW5ndWFnZV9wZXJzaXN0ZW50BGxpc3QHZWxlbWVudAdjb250ZW50FQAWCBa6ARa6ASayAUkcFQAVABUCADwWWBkmCAAZVgAAAAAIAAAAJgAcFQwZJQYAGUgTbGFuZ3VhZ2VfcGVyc2lzdGVudARsaXN0B2VsZW1lbnQFc3R5bGUVABYIFpYBFpYBJuwCSRwVABUAFQIAPBY4GSYIABlWAAAAAAgAAAAmABwVBBklBgAZGAVpbmRleBUAFggWchZyJoIESRwVABUAFQIAPCkGGSYACAAAACYAHBUKGSUGABkYCXRpbWVzdGFtcBUAFggWchZyJvQESRwVABUAFQIAPCkGGSYACAAAACYAHBUEGSUGABkYCnRhc2tfaW5kZXgVABYIFnIWcibmBUkcFQAVABUCADwpBhkmAAgAAAAmABwVDBklBgAZGBRsYW5ndWFnZV9pbnN0cnVjdGlvbhUAFggWchZyJtgGSRwVABUAFQIAPBYgGQYZJgAIAAAAJgAcFQoZJQYAGTgGYWN0aW9uBGxpc3QHZWxlbWVudBUAFhAWwgEWwgEmygdJHBUAFQAVAgA8KSYICBlGAAAAEAAAACYAHBUKGSUGABk4EW9ic2VydmF0aW9uLnN0YXRlBGxpc3QHZWxlbWVudBUAFhAWwgEWwgEmjAlJHBUAFQAVAgA8KSYICBlGAAAAEAAAABbGChYIJggWxgoAKCBwYXJxdWV0LWNwcC1hcnJvdyB2ZXJzaW9uIDI1LjAuMRmcHAAAHAAAHAAAHAAAHAAAHAAAHAAAHAAAHAAAAPYDAABQQVIx",
  v2data:
    "UEFSMRUAFUwVTCwVBBUAFQYVBhwAAAACAAAABAACAAAABAQJAAAAYW5ub3RhdG9yCQAAAGFubm90YXRvchUAFVQVVCwVBBUAFQYVBhwAAAACAAAABAACAAAABAQLAAAAbW92ZSBzYWZlbHkLAAAAbW92ZSBzYWZlbHkVABVEFUQsFQQVABUGFQYcAAAAAgAAAAQAAgAAAAQEBwAAAHN1YnRhc2sHAAAAc3VidGFzaxUAFSwVLCwVBBUAFQYVBhwAAAACAAAABAFmAAAAAAAAAGcAAAAAAAAAFQAVLBUsLBUEFQAVBhUGHAAAAAIAAAAEAQAAAAAAAAAAmpmZmZmZuT8VABUsFSwsFQQVABUGFQYcAAAAAgAAAAQBAAAAAAAAAAAAAAAAAAAAABUAFSwVLCwVBBUAFQYVBhwAAAACAAAABAEEAAAAbW92ZQQAAABtb3ZlFQAVWBVYLBUIFQAVBhUGHAAAAAIAAAADCgIAAAAIAwAAAAAAABBAAAAAAAAAFEAAAAAAAAAgQAAAAAAAACJAFQAVWBVYLBUIFQAVBhUGHAAAAAIAAAADCgIAAAAIAwAAAAAAABhAAAAAAAAAHEAAAAAAAAAkQAAAAAAAACZAFQQZ/BE1ABgGc2NoZW1hFQ4ANQIYE2xhbmd1YWdlX3BlcnNpc3RlbnQVAhUGTDwAAAA1BBgEbGlzdBUCADUCGAdlbGVtZW50FQYAFQwlAhgEcm9sZSUATBwAAAAVDCUCGAdjb250ZW50JQBMHAAAABUMJQIYBXN0eWxlJQBMHAAAABUEJQIYBWluZGV4ABUKJQIYCXRpbWVzdGFtcAAVBCUCGAp0YXNrX2luZGV4ABUMJQIYFGxhbmd1YWdlX2luc3RydWN0aW9uJQBMHAAAADUCGAZhY3Rpb24VAhUGTDwAAAA1BBgEbGlzdBUCABUKJQIYB2VsZW1lbnQANQIYEW9ic2VydmF0aW9uLnN0YXRlFQIVBkw8AAAANQQYBGxpc3QVAgAVCiUCGAdlbGVtZW50ABYEGRwZnCYAHBUMGSUGABlIE2xhbmd1YWdlX3BlcnNpc3RlbnQEbGlzdAdlbGVtZW50BHJvbGUVABYEFnIWciYISRwVABUAFQIAPBYkGSYEABlWAAAAAAQAAAAmABwVDBklBgAZSBNsYW5ndWFnZV9wZXJzaXN0ZW50BGxpc3QHZWxlbWVudAdjb250ZW50FQAWBBZ6FnomekkcFQAVABUCADwWLBkmBAAZVgAAAAAEAAAAJgAcFQwZJQYAGUgTbGFuZ3VhZ2VfcGVyc2lzdGVudARsaXN0B2VsZW1lbnQFc3R5bGUVABYEFmoWaib0AUkcFQAVABUCADwWHBkmBAAZVgAAAAAEAAAAJgAcFQQZJQYAGRgFaW5kZXgVABYEFlIWUibeAkkcFQAVABUCADwpBhkmAAQAAAAmABwVChklBgAZGAl0aW1lc3RhbXAVABYEFlIWUiawA0kcFQAVABUCADwpBhkmAAQAAAAmABwVBBklBgAZGAp0YXNrX2luZGV4FQAWBBZSFlImggRJHBUAFQAVAgA8KQYZJgAEAAAAJgAcFQwZJQYAGRgUbGFuZ3VhZ2VfaW5zdHJ1Y3Rpb24VABYEFlIWUibUBEkcFQAVABUCADwWEBkGGSYABAAAACYAHBUKGSUGABk4BmFjdGlvbgRsaXN0B2VsZW1lbnQVABYIFn4WfiamBUkcFQAVABUCADwpJgQEGUYAAAAIAAAAJgAcFQoZJQYAGTgRb2JzZXJ2YXRpb24uc3RhdGUEbGlzdAdlbGVtZW50FQAWCBZ+Fn4mpAZJHBUAFQAVAgA8KSYEBBlGAAAACAAAABaaBxYEJggWmgcAKCBwYXJxdWV0LWNwcC1hcnJvdyB2ZXJzaW9uIDI1LjAuMRmcHAAAHAAAHAAAHAAAHAAAHAAAHAAAHAAAHAAAAOsDAABQQVIx",
  singleMetadata:
    "UEFSMRUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgGxaN46AAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgEAAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgEDAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgFmAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgFoAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgECAAAAAAAAABUAFSgVKCwVAhUAFQYVBhwAAAACAAAAAgACAAAAAgMEAAAAbW92ZRUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgEBAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgEFAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgGamZmZmZnJPxUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgGamZmZmZnZPxUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgEBAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgEFAAAAAAAAABUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgFmZmZmZmYkQBUAFRwVHCwVAhUAFQYVBhwAAAACAAAAAgHNzMzMzMwkQBUEGfwSNQAYBnNjaGVtYRUeABUEJQIYDWVwaXNvZGVfaW5kZXgAFQQlAhgQZGF0YS9jaHVua19pbmRleAAVBCUCGA9kYXRhL2ZpbGVfaW5kZXgAFQQlAhgSZGF0YXNldF9mcm9tX2luZGV4ABUEJQIYEGRhdGFzZXRfdG9faW5kZXgAFQQlAhgGbGVuZ3RoADUCGAV0YXNrcxUCFQZMPAAAADUEGARsaXN0FQIAFQwlAhgHZWxlbWVudCUATBwAAAAVBCUCGBl2aWRlb3MvY2FtZXJhL2NodW5rX2luZGV4ABUEJQIYGHZpZGVvcy9jYW1lcmEvZmlsZV9pbmRleAAVCiUCGBx2aWRlb3MvY2FtZXJhL2Zyb21fdGltZXN0YW1wABUKJQIYGnZpZGVvcy9jYW1lcmEvdG9fdGltZXN0YW1wABUEJQIYGHZpZGVvcy9kZXB0aC9jaHVua19pbmRleAAVBCUCGBd2aWRlb3MvZGVwdGgvZmlsZV9pbmRleAAVCiUCGBt2aWRlb3MvZGVwdGgvZnJvbV90aW1lc3RhbXAAFQolAhgZdmlkZW9zL2RlcHRoL3RvX3RpbWVzdGFtcAAWAhkcGfwPJgAcFQQZJQYAGRgNZXBpc29kZV9pbmRleBUAFgIWQhZCJghJHBUAFQAVAgA8KQYZJgACAAAAJgAcFQQZJQYAGRgQZGF0YS9jaHVua19pbmRleBUAFgIWQhZCJkpJHBUAFQAVAgA8KQYZJgACAAAAJgAcFQQZJQYAGRgPZGF0YS9maWxlX2luZGV4FQAWAhZCFkImjAFJHBUAFQAVAgA8KQYZJgACAAAAJgAcFQQZJQYAGRgSZGF0YXNldF9mcm9tX2luZGV4FQAWAhZCFkImzgFJHBUAFQAVAgA8KQYZJgACAAAAJgAcFQQZJQYAGRgQZGF0YXNldF90b19pbmRleBUAFgIWQhZCJpACSRwVABUAFQIAPCkGGSYAAgAAACYAHBUEGSUGABkYBmxlbmd0aBUAFgIWQhZCJtICSRwVABUAFQIAPCkGGSYAAgAAACYAHBUMGSUGABk4BXRhc2tzBGxpc3QHZWxlbWVudBUAFgIWThZOJpQDSRwVABUAFQIAPBYIGSYCABlGAAAAAgAAACYAHBUEGSUGABkYGXZpZGVvcy9jYW1lcmEvY2h1bmtfaW5kZXgVABYCFkIWQibiA0kcFQAVABUCADwpBhkmAAIAAAAmABwVBBklBgAZGBh2aWRlb3MvY2FtZXJhL2ZpbGVfaW5kZXgVABYCFkIWQiakBEkcFQAVABUCADwpBhkmAAIAAAAmABwVChklBgAZGBx2aWRlb3MvY2FtZXJhL2Zyb21fdGltZXN0YW1wFQAWAhZCFkIm5gRJHBUAFQAVAgA8KQYZJgACAAAAJgAcFQoZJQYAGRgadmlkZW9zL2NhbWVyYS90b190aW1lc3RhbXAVABYCFkIWQiaoBUkcFQAVABUCADwpBhkmAAIAAAAmABwVBBklBgAZGBh2aWRlb3MvZGVwdGgvY2h1bmtfaW5kZXgVABYCFkIWQibqBUkcFQAVABUCADwpBhkmAAIAAAAmABwVBBklBgAZGBd2aWRlb3MvZGVwdGgvZmlsZV9pbmRleBUAFgIWQhZCJqwGSRwVABUAFQIAPCkGGSYAAgAAACYAHBUKGSUGABkYG3ZpZGVvcy9kZXB0aC9mcm9tX3RpbWVzdGFtcBUAFgIWQhZCJu4GSRwVABUAFQIAPCkGGSYAAgAAACYAHBUKGSUGABkYGXZpZGVvcy9kZXB0aC90b190aW1lc3RhbXAVABYCFkIWQiawB0kcFQAVABUCADwpBhkmAAIAAAAW6gcWAiYIFuoHACggcGFycXVldC1jcHAtYXJyb3cgdmVyc2lvbiAyNS4wLjEZ/A8cAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAcAAAA0gUAAFBBUjE=",
};

interface Result {
  episode: EpisodeData;
  flat: Record<string, number>[];
  frames: EpisodeFramesData;
  lengths: EpisodeLengthStats | null;
  adjacent: { episodeId: number; videosInfo: EpisodeData["videosInfo"] }[];
  variance: { numEpisodes: number } | null;
  requests: {
    url: string;
    method: string;
    authorization: string | null;
    range: string | null;
  }[];
}

function loadScenario(options: {
  version: string;
  repoId?: string;
  base?: string;
  annotation?: boolean;
  browser?: boolean;
  shards?: number;
}): Result {
  // Isolate module-level environment settings and caches without replacing the
  // package or parquet reader. Only HTTP is faked, including HEAD/range requests.
  const script = `
    const options = ${JSON.stringify(options)};
    const fixtures = ${JSON.stringify(fixtures)};
    const requests = [];
    if (options.browser) globalThis.window = {
      localStorage: { getItem: () => JSON.stringify({ accessToken: "hf-test-token" }) }
    };
    const repoId = options.repoId ?? "org/robot";
    const v3 = options.version.startsWith("v3");
    const episodeId = options.shards ? options.shards - 1 : 1;
    const info = {
      codebase_version: options.version, total_episodes: options.shards ?? 2, total_frames: (options.shards ?? 2) * 2,
      total_tasks: 1, fps: 10, chunks_size: 1000, robot_type: "test",
      data_path: v3 ? "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet" : "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
      video_path: v3 ? "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4" : "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
      features: {
        camera: { dtype: "video", shape: [10, 20, 3], names: null },
        depth: { dtype: "video", shape: [10, 20, 1], names: null },
        action: { dtype: "float64", shape: [2], names: ["a", "b"] },
        "observation.state": { dtype: "float64", shape: [2], names: ["a", "b"] }
      }
    };
    globalThis.fetch = async (input, init) => {
      const url = String(input);
      const headers = new Headers(init?.headers);
      const range = headers.get("range");
      const method = init?.method ?? "GET";
      requests.push({ url, method, range, authorization: headers.get("authorization") });
      const path = url.split("/resolve/main/")[1];
      let body;
      if (path === "meta/info.json") body = Buffer.from(JSON.stringify(info));
      else if (path === "meta/episodes.jsonl" && !repoId.startsWith("local/")) body = Buffer.from(
        [0, 1].map(i => JSON.stringify({ episode_index: i, length: 2, tasks: ["move"] })).join("\\n") + "\\n"
      );
      else if (options.shards && path?.startsWith("meta/episodes/")) {
        const shardMatch = /^meta\\/episodes\\/chunk-000\\/file-(\\d+)\\.parquet$/.exec(path);
        if (!shardMatch || Number(shardMatch[1]) >= options.shards) return new Response(null, { status: 404 });
        body = Buffer.from(fixtures.singleMetadata, "base64");
        // Uncompressed plain int64, no statistics: replacing this unique value
        // gives each one-row shard its real episode index without mocking reads.
        const sentinel = Buffer.alloc(8); sentinel.writeBigInt64LE(987654321n);
        body.writeBigInt64LE(BigInt(shardMatch[1]), body.indexOf(sentinel));
      }
      else if (path === "meta/episodes/chunk-000/file-000.parquet") body = Buffer.from(fixtures.metadata, "base64");
      else if (path?.startsWith("data/")) body = Buffer.from(v3 ? fixtures.data : fixtures.v2data, "base64");
      else return new Response(null, { status: 404 });
      const size = body.length;
      if (method === "HEAD") return new Response(null, { headers: { "Content-Length": String(size) } });
      if (range) {
        const parts = /^bytes=(\\d*)-(\\d*)$/.exec(range);
        let start = parts[1] ? Number(parts[1]) : Math.max(0, size - Number(parts[2]));
        let end = parts[1] && parts[2] ? Math.min(size - 1, Number(parts[2])) : size - 1;
        return new Response(body.subarray(start, end + 1), { status: 206, headers: {
          "Content-Range": "bytes " + start + "-" + end + "/" + size,
          "Content-Length": String(end - start + 1)
        }});
      }
      return new Response(body, { headers: { "Content-Length": String(size) } });
    };
    const loader = await import("./src/app/[org]/[dataset]/[episode]/fetch-data.ts");
    const [org, dataset] = repoId.split("/");
    const episode = await loader.getEpisodeData(org, dataset, episodeId);
    const flat = await loader.loadEpisodeFlatChartData(repoId, options.version, info, episodeId);
    const frames = await loader.loadAllEpisodeFrameInfo(repoId, options.version, info);
    const lengths = v3 ? await loader.loadAllEpisodeLengthsV3(repoId, options.version, info.fps) : null;
    const adjacent = await loader.getAdjacentEpisodesVideoInfo(org, dataset, episodeId, 1);
    const variance = v3 ? await loader.loadCrossEpisodeActionVariance(repoId, options.version, info, info.fps) : null;
    console.log("RESULT:" + JSON.stringify({ episode, flat, frames, lengths, adjacent, variance, requests }));
  `;
  const env = { ...process.env };
  for (const key of [
    "DATASET_URL",
    "NEXT_PUBLIC_DATASET_URL",
    "NEXT_PUBLIC_ANNOTATE_BACKEND_URL",
    "ANNOTATION_BACKEND_URL",
    "ANNOTATION_BACKEND_TOKEN",
    "EPISODES",
  ])
    delete env[key];
  if (options.base) {
    env.NEXT_PUBLIC_DATASET_URL = options.base;
    env.DATASET_URL = "https://unused.example/datasets";
  }
  if (options.annotation) {
    env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL = "/api/annotations";
    env.ANNOTATION_BACKEND_URL = "http://backend.internal:7861";
    env.ANNOTATION_BACKEND_TOKEN = "backend-secret";
  }
  const child = Bun.spawnSync([process.execPath, "-e", script], {
    cwd: process.cwd(),
    env,
    stdout: "pipe",
    stderr: "pipe",
  });
  expect(child.exitCode, child.stderr.toString()).toBe(0);
  const output = child.stdout
    .toString()
    .split("\n")
    .find((line) => line.startsWith("RESULT:"));
  expect(output, child.stdout.toString()).toBeDefined();
  return JSON.parse(output!.slice("RESULT:".length));
}

function expectCharts(result: Result) {
  expect(result.episode.flatChartData).toHaveLength(2);
  expect(result.episode.flatChartData[0]["action | a"]).toBe(4);
  expect(result.flat).toEqual(result.episode.flatChartData);
  expect(result.episode.frameTimestamps).toEqual([0, 0.1]);
  expect(
    result.episode.languageAtoms?.some(
      (atom) => atom.content === "move safely",
    ),
  ).toBe(true);
}

function expectV3(result: Result) {
  expectCharts(result);
  expect(
    result.episode.videosInfo.map((v) => [v.segmentStart, v.segmentEnd]),
  ).toEqual([
    [0.2, 0.4],
    [10.2, 10.4],
  ]);
  expect(result.frames.framesByCamera.camera).toHaveLength(2);
  expect(result.frames.framesByCamera.depth[1].firstFrameTime).toBe(10.2);
  expect(result.lengths?.allEpisodeLengths).toEqual([
    { episodeIndex: 0, frames: 2, lengthSeconds: 0.2 },
    { episodeIndex: 1, frames: 2, lengthSeconds: 0.2 },
  ]);
  expect(result.adjacent).toHaveLength(1);
  expect(result.adjacent[0].videosInfo[0].isSegmented).toBe(true);
  expect(result.variance?.numEpisodes).toBe(2);
}

describe("merged episode loader compatibility", () => {
  test("loads episodes and statistics beyond the package's 64-file index cap", () => {
    const result = loadScenario({ version: "v3.0", shards: 65 });
    expectCharts(result);
    expect(result.episode.episodeId).toBe(64);
    expect(result.frames.framesByCamera.camera).toHaveLength(65);
    expect(result.frames.framesByCamera.camera.at(-1)?.episodeIndex).toBe(64);
    expect(result.lengths?.allEpisodeLengths).toHaveLength(65);
    expect(result.variance?.numEpisodes).toBe(65);
  });

  test("keeps v3.1 annotation reads authenticated and media URLs same-origin", () => {
    const result = loadScenario({
      version: "v3.1",
      repoId: "local/annotation-test",
      annotation: true,
    });
    expectV3(result);
    expect(result.episode.videosInfo[0].url).toBe(
      "/api/annotations/datasets/local/annotation-test/resolve/main/videos/camera/chunk-001/file-005.mp4",
    );
    expect(
      result.requests.every(
        (r) =>
          r.url.startsWith(
            "http://backend.internal:7861/datasets/local/annotation-test/",
          ) && r.authorization === "Bearer backend-secret",
      ),
    ).toBe(true);
    expect(JSON.stringify({ ...result, requests: [] })).not.toContain(
      "backend-secret",
    );
    expect(JSON.stringify({ ...result, requests: [] })).not.toContain(
      "backend.internal",
    );
  });

  test("keeps local v2 charts and videos working without episodes.jsonl", () => {
    const result = loadScenario({
      version: "v2.1",
      repoId: "local/annotation-test",
      annotation: true,
    });
    expectCharts(result);
    expect(result.episode.videosInfo[0].url).toEndWith(
      "videos/chunk-000/camera/episode_000001.mp4",
    );
    expect(result.requests.some((r) => r.url.includes("meta/episodes"))).toBe(
      false,
    );
  });

  test("keeps local v3.0 annotation datasets on the authenticated reader", () => {
    const result = loadScenario({
      version: "v3.0",
      repoId: "local/annotation-test",
      annotation: true,
    });
    expectV3(result);
    expect(
      result.requests.every(
        (r) =>
          r.url.startsWith(
            "http://backend.internal:7861/datasets/local/annotation-test/",
          ) && r.authorization === "Bearer backend-secret",
      ),
    ).toBe(true);
  });

  test("honors arbitrary roots for remote v3.0 metadata and media", () => {
    const result = loadScenario({
      version: "v3.0",
      base: "https://mirror.example/custom-root",
      browser: true,
    });
    expectV3(result);
    expect(
      result.requests.every(
        (r) =>
          r.url.startsWith("https://mirror.example/custom-root/org/robot/") &&
          r.authorization === null,
      ),
    ).toBe(true);
    expect(
      result.episode.videosInfo.every((v) =>
        v.url.startsWith("https://mirror.example/custom-root/org/robot/"),
      ),
    ).toBe(true);
  });

  test("keeps remote v3.1 segmentation, statistics, and overview", () => {
    const result = loadScenario({ version: "v3.1" });
    expectV3(result);
    expect(
      result.requests.every((r) =>
        r.url.startsWith("https://huggingface.co/datasets/org/robot/"),
      ),
    ).toBe(true);
  });

  test("uses the actual package ranged index reader and HF auth for remote v3.0", () => {
    const result = loadScenario({ version: "v3.0", browser: true });
    expectV3(result);
    expect(
      result.requests.some(
        (r) =>
          r.url.endsWith("meta/episodes/chunk-000/file-000.parquet") &&
          r.range === "bytes=-65536",
      ),
    ).toBe(true);
    expect(
      result.requests.every((r) => r.authorization === "Bearer hf-test-token"),
    ).toBe(true);
  });

  test("keeps remote v2 video loading independent of capped episode indexes", () => {
    const result = loadScenario({ version: "v2.0" });
    expectCharts(result);
    expect(
      result.requests.some((r) => r.url.endsWith("meta/episodes.jsonl")),
    ).toBe(false);
    expect(result.episode.videosInfo[0].isSegmented).toBeUndefined();
  });

  test("honors arbitrary NEXT_PUBLIC_DATASET_URL roots for legacy local datasets", () => {
    const result = loadScenario({
      version: "v2.1",
      repoId: "local/legacy",
      base: "http://localhost:9000/legacy-root",
      browser: true,
    });
    expectCharts(result);
    expect(
      result.requests.every(
        (r) =>
          r.url.startsWith("http://localhost:9000/legacy-root/local/legacy/") &&
          r.authorization === null,
      ),
    ).toBe(true);
  });

  test("honors custom Hub endpoints without sending HF credentials", () => {
    const result = loadScenario({
      version: "v3.0",
      base: "https://mirror.example/datasets",
      browser: true,
    });
    expectV3(result);
    expect(
      result.requests.every(
        (r) =>
          r.url.startsWith("https://mirror.example/datasets/org/robot/") &&
          r.authorization === null,
      ),
    ).toBe(true);
    expect(
      result.requests.some(
        (r) =>
          r.url.endsWith("meta/episodes/chunk-000/file-000.parquet") &&
          r.range === "bytes=-65536",
      ),
    ).toBe(true);
  });
});
