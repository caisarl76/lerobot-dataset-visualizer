/**
 * Client for the FastAPI annotation backend in `backend/`.
 *
 * The backend URL is configured via the `NEXT_PUBLIC_ANNOTATE_BACKEND_URL`
 * env var so it can be statically substituted by Next.js. When unset, all
 * annotation write paths are disabled and the UI falls back to sessionStorage
 * for read/edit only.
 */

import type { LanguageAtom } from "../types/language.types";

const ENV_URL = (() => {
  const v =
    typeof process !== "undefined"
      ? process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL
      : undefined;
  return (v || "").trim() || null;
})();

export function isAnnotateBackendEnabled(): boolean {
  return !!ENV_URL;
}

export function getAnnotateBackendUrl(): string | null {
  return ENV_URL;
}

interface DatasetIdent {
  repoId?: string | null;
  localPath?: string | null;
  revision?: string | null;
}

export interface EpisodeReview {
  status: "reviewed" | "unreviewed";
  annotation_sha256: string;
  reviewed_at: string | null;
  prediction_available: boolean;
  exclusions_sha256?: string;
}

export class AnnotationRequestError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "AnnotationRequestError";
  }
}

export type OfficialAnnotationConfig = Record<string, unknown>;
export type AnnotationValidation = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  episodes_checked: number;
};

async function annotationRequest(path: string, init?: RequestInit) {
  if (!ENV_URL)
    throw new Error(
      "Annotation backend unavailable. Set NEXT_PUBLIC_ANNOTATE_BACKEND_URL.",
    );
  const res = await fetch(resolveAnnotationUrl(path), {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `annotation backend: ${res.status}`,
    );
  return res.json();
}

function resolveAnnotationUrl(path: string): string {
  if (!ENV_URL) throw new Error("Annotate backend not configured");
  if (ENV_URL.startsWith("/"))
    return `${ENV_URL.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
  return new URL(path, ENV_URL).toString();
}

export async function fetchAnnotationConfig() {
  return annotationRequest("/api/annotation/config") as Promise<{
    revision: string;
    config: OfficialAnnotationConfig;
  }>;
}
function annotationBody(body: Record<string, unknown>) {
  const { repoId, localPath, ...payload } = body;
  return {
    ...payload,
    repo_id: payload.repo_id ?? repoId ?? null,
    local_path: payload.local_path ?? localPath ?? null,
  };
}

export async function createAnnotationJob(body: Record<string, unknown>) {
  return annotationRequest("/api/annotation/jobs", {
    method: "POST",
    body: JSON.stringify(annotationBody(body)),
  }) as Promise<{ job_id: string; status: string }>;
}
export async function deleteAnnotationEpisodes(body: Record<string, unknown>) {
  return annotationRequest("/api/annotation/delete-episodes", {
    method: "POST",
    body: JSON.stringify(annotationBody(body)),
  }) as Promise<{ job_id: string; status: string }>;
}
export async function getAnnotationJob(jobId: string) {
  return annotationRequest(
    `/api/annotation/jobs/${encodeURIComponent(jobId)}`,
  ) as Promise<{
    job_id: string;
    status: string;
    error?: string;
    result?: {
      output_dir: string;
      repo_id: string;
      first_episode_index?: number;
      first_generated_episode_index?: number;
      validation: AnnotationValidation;
    };
  }>;
}
export async function validateAnnotation(body: Record<string, unknown>) {
  return annotationRequest("/api/annotation/validate", {
    method: "POST",
    body: JSON.stringify(annotationBody(body)),
  }) as Promise<AnnotationValidation>;
}
export async function prepareAnnotationDataset(body: Record<string, unknown>) {
  return annotationRequest("/api/annotation/prepare", {
    method: "POST",
    body: JSON.stringify(annotationBody(body)),
  }) as Promise<{ job_id: string; status: string }>;
}

function buildUrl(path: string, ident: DatasetIdent): string {
  if (!ENV_URL) throw new Error("Annotate backend not configured");
  const url = new URL(
    resolveAnnotationUrl(path),
    "http://annotation-relative.invalid",
  );
  if (ident.repoId) url.searchParams.set("repo_id", ident.repoId);
  if (ident.revision) url.searchParams.set("revision", ident.revision);
  if (ident.localPath) url.searchParams.set("local_path", ident.localPath);
  return ENV_URL.startsWith("/")
    ? `${url.pathname}${url.search}`
    : url.toString();
}

export type WorkflowMetric = {
  mae_seconds: number | null;
  eligible_episodes: number;
  eligible_boundaries: number;
  missing_predictions: number;
  topology_mismatch: number;
};
export type WorkflowEpisode = {
  generation_status: string;
  issues: {
    code: string;
    source: string;
    severity: string;
    message: string;
    start?: number | null;
    end?: number | null;
  }[];
  decision: string;
  decision_reason?: string | null;
  review?: { status: string };
  predictions?: { atoms: LanguageAtom[]; created_at: string }[];
  deltas?: number[];
  excluded_intervals?: { start_frame: number; end_frame: number }[];
};
export type WorkflowRun = {
  repo_id?: string | null;
  source_format?: string;
  current_repo_id?: string;
  example_episode_indices?: number[];
  run_id: string;
  revision: number;
  review_snapshot_sha256?: string;
  publication_state: string;
  current_job_id: string | null;
  task_prompt: string;
  subtask_prompts: string[];
  episodes: Record<string, WorkflowEpisode>;
  metrics: { first_pass?: WorkflowMetric; latest?: WorkflowMetric };
  publication?: {
    main_commit?: string;
    rich_commit?: string;
    rich_revision?: string;
    main_url?: string;
    rich_url?: string;
    raw_url?: string;
    urls?: { main?: string; rich?: string; raw?: string };
  };
  transition_detection?: {
    episodes_changed: number;
    episodes_considered: number;
    skipped_reason?: string;
  };
  export?: {
    manifest_sha256: string;
    retained_episodes: number;
    deleted_episodes: number | number[];
    managed_changes?: { added_or_updated: string[]; deleted: string[] };
    validation?: AnnotationValidation;
    format?: "groot_v21" | "rich";
    instruction_mode?: "task" | "subtask";
    dataset_name?: string;
    retained_frames?: number;
    local_path?: string;
    output_repo_id?: string;
    destination?: {
      repo_id: string;
      revision: string;
      exists: boolean;
      revision_exists?: boolean;
      expected_commit?: string | null;
      private?: boolean;
    } | null;
  };
};

export async function detectWorkflowTransitions(
  alias: string,
  body: { expected_revision: number; include_reviewed: boolean },
): Promise<WorkflowRun> {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}/detect-transition-pauses`,
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  ) as Promise<WorkflowRun>;
}

export async function fetchWorkflow(alias: string): Promise<WorkflowRun> {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}`,
  ) as Promise<WorkflowRun>;
}
export async function postWorkflowExclusions(
  alias: string,
  body: {
    episode_index: number;
    expected_revision: number;
    excluded_intervals: { start_frame: number; end_frame: number }[];
  },
): Promise<WorkflowRun> {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}/exclusions`,
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  ) as Promise<WorkflowRun>;
}
export async function postWorkflowDecision(
  alias: string,
  body: {
    episode_index: number;
    decision: "keep" | "delete" | "pending";
    reason?: string;
    expected_revision: number;
  },
) {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}/decision`,
    { method: "POST", body: JSON.stringify(body) },
  ) as Promise<WorkflowRun>;
}
export async function keepWorkflowRemaining(
  alias: string,
  expected_revision: number,
) {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}/keep-remaining`,
    {
      method: "POST",
      body: JSON.stringify({ expected_revision }),
    },
  ) as Promise<WorkflowRun>;
}
export async function reviewWorkflowRetained(
  alias: string,
  body: {
    expected_revision: number;
    expected_review_sha256: string;
    confirmed: true;
  },
) {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}/review-retained`,
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  ) as Promise<WorkflowRun>;
}
export type WorkflowExportOptions = {
  export_format: "groot_v21" | "rich";
  instruction_mode: "task" | "subtask";
  dataset_name: string;
  destination_repo_id: string | null;
  destination_revision: string;
  destination_private: boolean;
};
export async function exportWorkflow(
  alias: string,
  expected_revision: number,
  options?: WorkflowExportOptions,
) {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}/export`,
    { method: "POST", body: JSON.stringify({ expected_revision, ...options }) },
  );
}
export async function publishWorkflow(
  alias: string,
  manifest_sha256: string,
  expected_revision: number,
) {
  return annotationRequest(
    `/api/workflow/${encodeURIComponent(alias)}/publish`,
    {
      method: "POST",
      body: JSON.stringify({ manifest_sha256, expected_revision }),
    },
  );
}

export async function pingBackend(): Promise<boolean> {
  if (!ENV_URL) return false;
  try {
    const res = await fetch(resolveAnnotationUrl("/api/health"));
    return res.ok;
  } catch {
    return false;
  }
}

export async function loadDataset(
  ident: DatasetIdent,
): Promise<{ ok: boolean }> {
  if (!ENV_URL) return { ok: false };
  const res = await fetch(resolveAnnotationUrl("/api/dataset/load"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_id: ident.repoId || null,
      revision: ident.revision || null,
      local_path: ident.localPath || null,
    }),
  });
  return { ok: res.ok };
}

export async function fetchEpisodeAtomsWithHash(
  episodeId: number,
  ident: DatasetIdent,
): Promise<{ atoms: LanguageAtom[]; annotation_sha256?: string }> {
  if (!ENV_URL) return { atoms: [] };
  await loadDataset(ident);
  const res = await fetch(buildUrl(`/api/episodes/${episodeId}/atoms`, ident));
  if (!res.ok)
    throw new AnnotationRequestError(res.status, `fetch atoms: ${res.status}`);
  const data = await res.json();
  return { atoms: data.atoms || [], annotation_sha256: data.annotation_sha256 };
}
export async function fetchEpisodeAtoms(
  episodeId: number,
  ident: DatasetIdent,
): Promise<LanguageAtom[]> {
  return (await fetchEpisodeAtomsWithHash(episodeId, ident)).atoms;
}

export async function saveEpisodeAtoms(
  episodeId: number,
  ident: DatasetIdent,
  atoms: LanguageAtom[],
  expectedAnnotationSha256?: string,
): Promise<{ path: string | null; annotation_sha256?: string }> {
  if (!ENV_URL) return { path: null };
  const res = await fetch(
    resolveAnnotationUrl(`/api/episodes/${episodeId}/atoms`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        episode_index: episodeId,
        repo_id: ident.repoId || null,
        local_path: ident.localPath || null,
        revision: ident.revision || null,
        atoms,
        ...(expectedAnnotationSha256
          ? { expected_annotation_sha256: expectedAnnotationSha256 }
          : {}),
      }),
    },
  );
  if (!res.ok) {
    const text = await res.text().catch(() => `${res.status}`);
    throw new AnnotationRequestError(
      res.status,
      text || `save atoms: ${res.status}`,
    );
  }
  const data = (await res.json().catch(() => ({}))) as {
    path?: string | null;
    annotation_sha256?: string;
  };
  return {
    path: data.path ?? null,
    ...(data.annotation_sha256
      ? { annotation_sha256: data.annotation_sha256 }
      : {}),
  };
}

export async function fetchEpisodeReview(
  episodeId: number,
  ident: DatasetIdent,
): Promise<EpisodeReview> {
  const res = await fetch(buildUrl(`/api/episodes/${episodeId}/review`, ident));
  if (!res.ok)
    throw new AnnotationRequestError(res.status, `review: ${res.status}`);
  return (await res.json()) as EpisodeReview;
}

export async function setEpisodeReview(
  episodeId: number,
  ident: DatasetIdent,
  reviewed: boolean,
  annotationSha256: string,
  exclusionsSha256?: string,
): Promise<EpisodeReview> {
  const res = await fetch(
    buildUrl(`/api/episodes/${episodeId}/review`, ident),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_id: ident.repoId || null,
        local_path: ident.localPath || null,
        revision: ident.revision || null,
        episode_index: episodeId,
        reviewed,
        annotation_sha256: annotationSha256,
        ...(exclusionsSha256
          ? { expected_exclusions_sha256: exclusionsSha256 }
          : {}),
      }),
    },
  );
  if (!res.ok) {
    const message =
      (await res.text().catch(() => "")) || `review: ${res.status}`;
    throw new AnnotationRequestError(res.status, message);
  }
  const review = (await res.json()) as EpisodeReview;
  if (typeof window !== "undefined")
    window.dispatchEvent(new window.Event("annotation-workflow-changed"));
  return review;
}

export async function fetchFrameTimestamps(
  episodeId: number,
  ident: DatasetIdent,
): Promise<number[]> {
  if (!ENV_URL) return [];
  const res = await fetch(
    buildUrl(`/api/episodes/${episodeId}/frame_timestamps`, ident),
  );
  if (!res.ok) return [];
  const data = (await res.json()) as { timestamps?: number[] };
  return data.timestamps || [];
}

export async function exportDataset(
  ident: DatasetIdent,
  outputDir?: string | null,
  copyVideos = false,
): Promise<{
  output_dir: string;
  persistent_rows: number;
  event_rows: number;
}> {
  if (!ENV_URL) throw new Error("Annotate backend not configured");
  const res = await fetch(resolveAnnotationUrl("/api/export"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_id: ident.repoId || null,
      revision: ident.revision || null,
      local_path: ident.localPath || null,
      output_dir: outputDir || null,
      copy_videos: !!copyVideos,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => `${res.status}`);
    throw new Error(text || `export: ${res.status}`);
  }
  return res.json();
}

export interface PushToHubResult {
  ok: boolean;
  repo_id: string;
  url: string;
  message: string;
}

export async function pushToHub(
  ident: DatasetIdent,
  hfToken: string,
  pushInPlace: boolean,
  newRepoId: string | null,
  privateRepo: boolean,
  commitMessage: string,
): Promise<PushToHubResult> {
  if (!ENV_URL) throw new Error("Annotate backend not configured");
  const res = await fetch(resolveAnnotationUrl("/api/push_to_hub"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_id: ident.repoId || null,
      revision: ident.revision || null,
      local_path: ident.localPath || null,
      hf_token: hfToken,
      push_in_place: pushInPlace,
      new_repo_id: newRepoId || null,
      private: privateRepo,
      commit_message: commitMessage,
    }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => `${res.status}`);
    throw new Error(text || `push: ${res.status}`);
  }
  return res.json();
}

export interface RobotMotion {
  joint_names: string[];
  timestamps: number[];
  positions: number[][];
  root_orientations: number[][] | null;
}

export async function fetchRobotMotion(
  episodeId: number,
  ident: DatasetIdent,
  signal: AbortSignal,
): Promise<RobotMotion> {
  const response = await fetch(
    buildUrl(`/api/episodes/${episodeId}/robot-motion`, ident),
    { signal },
  );
  if (!response.ok)
    throw new Error(
      `Robot motion unavailable (${response.status}). This replay requires named, measured joint states.`,
    );
  return response.json();
}
