import { getAnnotateBackendUrl } from "./annotationsClient";

export interface MonitorDiagnostic {
  code: string;
  message: string;
  severity: "info" | "warning" | "error";
}
export interface MonitorMetrics {
  imported: number | null;
  accepted: number | null;
  rejected: number | null;
  pending: number | null;
  retained: number | null;
  reviewed: number | null;
  decision_rate: number | null;
  review_rate: number | null;
  new_episode_ids: number[] | null;
  missing_episode_ids: number[] | null;
  changed_length_ids: number[] | null;
  accepted_frames: number | null;
  accepted_seconds: number | null;
  counts_complete: boolean;
}
export interface MonitorFindings {
  unresolved: number;
  accepted_advisory: number;
  generation_failed: number;
  unreadable: number;
}
export interface MonitorExclusions {
  episodes: number | null;
  frames: number | null;
}
export interface MonitorRun {
  run_id: string;
  updated_at: string | null;
  current_repo_id: string | null;
  first_retained_episode: number | null;
  detail_signature: string;
  metrics: MonitorMetrics;
  findings: MonitorFindings;
  exclusions: MonitorExclusions;
  job: { job_id: string; status: string; error?: string | null } | null;
  freshness: {
    publication_state: string;
    metadata_only: true;
    source_changed: boolean | null;
    local_changes: boolean | null;
    verifiable: boolean;
  };
  diagnostics: MonitorDiagnostic[];
}
export interface RemoteCheck {
  status: "match" | "changed" | "missing" | "access_denied" | "unavailable";
  checked_at: string;
  current_commit: string | null;
  message: string | null;
}
export interface MonitorPublication {
  id: string;
  repo_id: string;
  revision: string | null;
  commit: string | null;
  url: string | null;
  export_path: string | null;
  export_available: boolean;
  format: string | null;
  instruction_mode: string | null;
  exported_frames: number | null;
  manifest_sha256: string | null;
  linked_run_id: string | null;
  remote_check: RemoteCheck | null;
}
export interface MonitorExport {
  id: string;
  run_id: string | null;
  path: string | null;
  available: boolean;
  format: string | null;
  instruction_mode: string | null;
  frames: number | null;
  seconds: number | null;
  manifest_sha256: string | null;
  output_repo_id: string | null;
}
export interface MonitorPrompts {
  eligible_episodes: number;
  evaluated_episodes: number;
  unknown_episode_ids: number[];
  retained_frames: number;
  unlabeled_frames: number;
  ambiguous_frames: number;
  complete: boolean;
  rows: {
    text: string;
    frames: number;
    seconds: number | null;
    episodes: number;
    ratio: number | null;
  }[];
  diagnostics?: MonitorDiagnostic[];
}
export interface MonitorDataset {
  id: string;
  name: string;
  path: string;
  canonical_path: string;
  state: string;
  collected: number | null;
  reported_collected: number | null;
  episode_lengths: Record<string, number>;
  parent_ids: string[];
  child_ids: string[];
  provenance: {
    status: "confirmed" | "unknown" | "conflict";
    sources: string[];
  };
  default_run_id: string | null;
  runs: MonitorRun[];
  publications: MonitorPublication[];
  exports: MonitorExport[];
  diagnostics: MonitorDiagnostic[];
}
export interface MonitorSummary {
  configured: boolean;
  root: string | null;
  scanned_at: string | null;
  updating: boolean;
  diagnostics: MonitorDiagnostic[];
  datasets: MonitorDataset[];
}
export interface MonitorDetail {
  dataset_id: string;
  run_id: string | null;
  signature: string | null;
  scanned_at: string | null;
  updating: boolean;
  metrics: MonitorMetrics | null;
  findings: MonitorFindings | null;
  exclusions: MonitorExclusions | null;
  prompts: MonitorPrompts | null;
  publications: MonitorPublication[];
  exports: MonitorExport[];
  diagnostics: MonitorDiagnostic[];
}

export class MonitorRequestError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "MonitorRequestError";
  }
}
function requestUrl(path: string): string {
  const base = getAnnotateBackendUrl();
  if (!base) throw new Error("Annotation backend is not configured.");
  return base.startsWith("/")
    ? `${base.replace(/\/$/, "")}${path}`
    : new URL(path, base).toString();
}
async function request<T>(
  path: string,
  signal?: AbortSignal,
  method = "GET",
): Promise<T> {
  const response = await fetch(requestUrl(path), {
    method,
    signal,
    cache: "no-store",
  });
  if (!response.ok) {
    let message = await response.text().catch(() => "");
    try {
      const body: unknown = JSON.parse(message);
      if (
        body &&
        typeof body === "object" &&
        "detail" in body &&
        typeof body.detail === "string"
      )
        message = body.detail;
    } catch {
      /* Non-JSON error responses remain readable. */
    }
    throw new MonitorRequestError(
      response.status,
      message || `Monitor request failed (${response.status}).`,
    );
  }
  return response.json() as Promise<T>;
}
export function fetchMonitorSummary(
  refresh = false,
  signal?: AbortSignal,
): Promise<MonitorSummary> {
  return request(`/api/monitor${refresh ? "?refresh=true" : ""}`, signal);
}
export function fetchMonitorDetail(
  datasetId: string,
  runId: string,
  signal?: AbortSignal,
): Promise<MonitorDetail> {
  return request(
    `/api/monitor/datasets/${encodeURIComponent(datasetId)}?run_id=${encodeURIComponent(runId)}`,
    signal,
  );
}
export function checkMonitorPublication(
  publicationId: string,
  signal?: AbortSignal,
): Promise<RemoteCheck> {
  return request(
    `/api/monitor/publications/${encodeURIComponent(publicationId)}/check`,
    signal,
    "POST",
  );
}
