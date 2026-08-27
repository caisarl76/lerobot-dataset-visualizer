import type {
  ActiveProposal,
  AttemptState,
  AuditDistribution,
  BatchConfiguration,
  BatchState,
  BatchStatus,
  CosmosModelResponse,
  CosmosPhase,
  CosmosSegment,
  CurationAudit,
  EpisodeCuration,
  EpisodeDraftPatch,
  GripDiagnostic,
  GripPoint,
  Hand,
  ReviewState,
  TransitionFrames,
  TurnDirection,
  WorkspaceOpened,
  WorkspaceSummary,
} from "../types/curation.types";

export type CurationClientErrorCode = "http_error" | "invalid_response";
export type CurationErrorPayload = Record<string, unknown>;

export class CurationClientError extends Error {
  constructor(
    readonly code: CurationClientErrorCode,
    readonly status: number,
    readonly errorPayload?: CurationErrorPayload,
  ) {
    super(
      code === "invalid_response"
        ? "curation service returned an invalid response"
        : `curation request failed with status ${status}`,
    );
    this.name = "CurationClientError";
  }
}

/** @deprecated Prefer CurationClientError. */
export const CurationHttpError = CurationClientError;

type Decoder<T> = (value: unknown) => T;

function invalid(): never {
  throw new TypeError("invalid response");
}

function hasOwn(record: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function object(
  value: unknown,
  required: readonly string[],
  optional: readonly string[] = [],
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    return invalid();
  const record: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) record[key] = item;
  const allowed = new Set([...required, ...optional]);
  if (
    !required.every((key) => hasOwn(record, key)) ||
    Object.keys(record).some((key) => !allowed.has(key))
  )
    return invalid();
  return record;
}

function string(value: unknown, allowEmpty = false): string {
  if (typeof value !== "string" || (!allowEmpty && value.trim().length === 0))
    return invalid();
  return value;
}

function nullableString(value: unknown): string | null {
  return value === null ? null : string(value);
}

function nullableStringAllowEmpty(value: unknown): string | null {
  return value === null ? null : string(value, true);
}

function boolean(value: unknown): boolean {
  if (typeof value !== "boolean") return invalid();
  return value;
}

function finite(value: unknown, minimum = 0): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum)
    return invalid();
  return value;
}

function integer(value: unknown, minimum = 0): number {
  const result = finite(value, minimum);
  if (!Number.isInteger(result)) return invalid();
  return result;
}

function nullableFinite(value: unknown): number | null {
  return value === null ? null : finite(value);
}

function nullableInteger(value: unknown): number | null {
  return value === null ? null : integer(value);
}

function array<T>(value: unknown, decoder: Decoder<T>): T[] {
  if (!Array.isArray(value)) return invalid();
  return value.map((item) => decoder(item));
}

function reviewState(value: unknown): ReviewState {
  switch (value) {
    case "pending":
    case "draft":
    case "approved_keep":
    case "approved_reject":
      return value;
    default:
      return invalid();
  }
}

function hand(value: unknown): Hand {
  if (value === "left" || value === "right") return value;
  return invalid();
}

function nullableHand(value: unknown): Hand | null {
  return value === null ? null : hand(value);
}

function turnDirection(value: unknown): TurnDirection {
  if (value === "left" || value === "right") return value;
  return invalid();
}

function nullableTurnDirection(value: unknown): TurnDirection | null {
  return value === null ? null : turnDirection(value);
}

function batchState(value: unknown): BatchState {
  switch (value) {
    case "queued":
    case "running":
    case "cancel_requested":
    case "completed":
    case "completed_with_failures":
    case "cancelled":
    case "failed":
      return value;
    default:
      return invalid();
  }
}

function attemptState(value: unknown): AttemptState {
  switch (value) {
    case "queued":
    case "leased":
    case "requesting":
    case "succeeded":
    case "manual_only":
    case "retryable":
    case "cancelled":
      return value;
    default:
      return invalid();
  }
}

function sha256(value: unknown): string {
  const result = string(value);
  if (!/^[0-9a-f]{64}$/.test(result)) return invalid();
  return result;
}

function strings(value: unknown): string[] {
  return array(value, (item) => string(item));
}

function timestamps(value: unknown, expectedLength: number): number[] {
  const result = array(value, finite);
  if (
    result.length !== expectedLength ||
    result.some((item, index) => index > 0 && item <= result[index - 1])
  )
    return invalid();
  return result;
}

function transitionFrames(
  value: unknown,
  sourceLength?: number,
): TransitionFrames {
  if (!Array.isArray(value) || value.length !== 6) return invalid();
  const values = value.map(nullableInteger);
  let previous = 0;
  for (const frame of values) {
    if (frame === null) continue;
    if (
      frame <= previous ||
      frame <= 0 ||
      (sourceLength !== undefined && frame >= sourceLength)
    )
      return invalid();
    previous = frame;
  }
  return [values[0], values[1], values[2], values[3], values[4], values[5]];
}

function proposalTransitionFrames(
  value: unknown,
  sourceLength: number,
): TransitionFrames {
  if (!Array.isArray(value) || value.length !== 6) return invalid();
  const frames = value.map(nullableInteger);
  if (frames.some((frame) => frame !== null && frame >= sourceLength))
    return invalid();
  return [frames[0], frames[1], frames[2], frames[3], frames[4], frames[5]];
}

function cosmosStep(value: unknown): 1 | 2 | 3 | 4 | 5 | 6 | 7 {
  switch (value) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      return value;
    default:
      return invalid();
  }
}

function cosmosSegment<
  Step extends 1 | 2 | 3 | 4 | 5 | 6 | 7,
  Phase extends CosmosPhase,
>(value: unknown, step: Step, phase: Phase): CosmosSegment<Step, Phase> {
  const segment = object(value, [
    "step",
    "phase",
    "status",
    "start_s",
    "end_s",
    "confidence",
    "caption",
    "evidence",
  ]);
  if (segment.step !== step || segment.phase !== phase) return invalid();
  const caption = string(segment.caption);
  if (segment.status === "not_observed") {
    if (
      segment.start_s !== null ||
      segment.end_s !== null ||
      segment.confidence !== null ||
      segment.evidence !== null
    )
      return invalid();
    return {
      step,
      phase,
      status: "not_observed",
      start_s: null,
      end_s: null,
      confidence: null,
      caption,
      evidence: null,
    };
  }
  if (segment.status !== "completed" && segment.status !== "partial")
    return invalid();
  const start = finite(segment.start_s);
  const end = finite(segment.end_s);
  const confidence = finite(segment.confidence);
  if (end <= start || confidence > 1) return invalid();
  return {
    step,
    phase,
    status: segment.status,
    start_s: start,
    end_s: end,
    confidence,
    caption,
    evidence: string(segment.evidence),
  };
}

function cosmosResponse(value: unknown): CosmosModelResponse {
  const record = object(value, [
    "schema_version",
    "episode_complete",
    "segments",
    "missing_steps",
    "uncertainties",
  ]);
  if (record.schema_version !== 2 || !Array.isArray(record.segments))
    return invalid();
  if (record.segments.length !== 7) return invalid();
  const segments: CosmosModelResponse["segments"] = [
    cosmosSegment(record.segments[0], 1, "approach_brown_table"),
    cosmosSegment(record.segments[1], 2, "pick_up_object"),
    cosmosSegment(record.segments[2], 3, "turn_to_find_black_trash_bin"),
    cosmosSegment(record.segments[3], 4, "approach_black_trash_bin"),
    cosmosSegment(record.segments[4], 5, "lean_down_to_black_trash_bin"),
    cosmosSegment(record.segments[5], 6, "drop_object_into_black_trash_bin"),
    cosmosSegment(record.segments[6], 7, "stand_straight"),
  ];
  const starts = segments.flatMap((segment) =>
    segment.start_s === null ? [] : [segment.start_s],
  );
  if (starts.some((start, index) => index > 0 && start <= starts[index - 1]))
    return invalid();
  const missing = array(record.missing_steps, cosmosStep);
  const expectedMissing = segments
    .filter((segment) => segment.status !== "completed")
    .map((segment) => segment.step);
  if (
    missing.length !== expectedMissing.length ||
    missing.some((step, index) => step !== expectedMissing[index])
  )
    return invalid();
  const complete = boolean(record.episode_complete);
  const uncertainties = strings(record.uncertainties);
  if (
    complete !== (missing.length === 0) ||
    (!complete && uncertainties.length === 0)
  )
    return invalid();
  return {
    schema_version: 2,
    episode_complete: complete,
    segments,
    missing_steps: missing,
    uncertainties,
  };
}

function proposal(value: unknown, sourceLength: number): ActiveProposal | null {
  if (value === null) return null;
  const record = object(value, [
    "id",
    "attempt_id",
    "model_response",
    "transition_frames",
    "warnings",
    "created_at",
  ]);
  return {
    id: string(record.id),
    attemptId: string(record.attempt_id),
    modelResponse: cosmosResponse(record.model_response),
    transitionFrames: proposalTransitionFrames(
      record.transition_frames,
      sourceLength,
    ),
    warnings: strings(record.warnings),
    createdAt: string(record.created_at),
  };
}

function episode(value: unknown): EpisodeCuration {
  const record = object(value, [
    "dataset_alias",
    "source_episode_index",
    "source_length",
    "timestamps",
    "decision",
    "active_proposal",
    "prompt_preview",
    "warnings",
    "revision",
    "approval_locked",
    "reviewer",
    "approval_revision",
    "approved_at",
  ]);
  const sourceLength = integer(record.source_length, 1);
  const decisionRecord = object(record.decision, [
    "review_state",
    "object_name",
    "pickup_hand",
    "turn_direction",
    "transition_frames",
    "rejection_reason",
    "prompt_template_sha256",
  ]);
  const state = reviewState(decisionRecord.review_state);
  const objectName = nullableString(decisionRecord.object_name);
  const pickupHand = nullableHand(decisionRecord.pickup_hand);
  const direction = nullableTurnDirection(decisionRecord.turn_direction);
  const transitions = transitionFrames(
    decisionRecord.transition_frames,
    sourceLength,
  );
  const rejectionReason = nullableStringAllowEmpty(
    decisionRecord.rejection_reason,
  );
  const locked = boolean(record.approval_locked);
  const reviewer = nullableString(record.reviewer);
  const revision = integer(record.revision);
  const approvalRevision = nullableInteger(record.approval_revision);
  const approvedAt = nullableString(record.approved_at);
  const approved = state === "approved_keep" || state === "approved_reject";
  if (
    locked !== approved ||
    (approved &&
      (reviewer === null ||
        approvalRevision !== revision ||
        approvedAt === null)) ||
    (!approved &&
      (reviewer !== null || approvalRevision !== null || approvedAt !== null))
  )
    return invalid();
  if (
    state === "approved_keep" &&
    (objectName === null ||
      pickupHand === null ||
      direction === null ||
      transitions.some((frame) => frame === null))
  )
    return invalid();
  let promptPreview: string[] | null = null;
  if (record.prompt_preview !== null) {
    promptPreview = strings(record.prompt_preview);
    if (promptPreview.length !== 7) return invalid();
  }
  if (state === "approved_keep" && promptPreview === null) return invalid();
  return {
    datasetAlias: string(record.dataset_alias),
    sourceEpisodeIndex: integer(record.source_episode_index),
    sourceLength,
    timestamps: timestamps(record.timestamps, sourceLength),
    decision: {
      reviewState: state,
      objectName,
      pickupHand,
      turnDirection: direction,
      transitionFrames: transitions,
      rejectionReason,
      promptTemplateSha256: sha256(decisionRecord.prompt_template_sha256),
    },
    activeProposal: proposal(record.active_proposal, sourceLength),
    promptPreview,
    warnings: strings(record.warnings),
    revision,
    approvalLocked: locked,
    reviewer,
    approvalRevision,
    approvedAt,
  };
}

function exactCountRecord<T extends string>(
  value: unknown,
  keys: readonly T[],
  requireEvery: boolean,
): Record<T, number> | Partial<Record<T, number>> {
  const record = object(value, requireEvery ? keys : [], keys);
  const result: Partial<Record<T, number>> = {};
  for (const key of keys) {
    if (hasOwn(record, key)) result[key] = integer(record[key]);
  }
  return result;
}

function workspace(value: unknown): WorkspaceOpened {
  const record = object(value, [
    "dataset_alias",
    "source_fingerprint",
    "prompt_template_version",
    "prompt_template_sha256",
    "episode_count",
  ]);
  return {
    datasetAlias: string(record.dataset_alias),
    sourceFingerprint: sha256(record.source_fingerprint),
    promptTemplateVersion: string(record.prompt_template_version),
    promptTemplateSha256: sha256(record.prompt_template_sha256),
    episodeCount: integer(record.episode_count),
  };
}

const REVIEW_STATES: readonly ReviewState[] = [
  "pending",
  "draft",
  "approved_keep",
  "approved_reject",
];
const BATCH_STATES: readonly BatchState[] = [
  "queued",
  "running",
  "cancel_requested",
  "completed",
  "completed_with_failures",
  "cancelled",
  "failed",
];
const ATTEMPT_STATES: readonly AttemptState[] = [
  "queued",
  "leased",
  "requesting",
  "succeeded",
  "manual_only",
  "retryable",
  "cancelled",
];

function summary(value: unknown): WorkspaceSummary {
  const record = object(value, [
    "dataset_alias",
    "episode_count",
    "counts",
    "prompt_template_version",
    "prompt_template_sha256",
  ]);
  const counts = exactCountRecord(record.counts, REVIEW_STATES, true);
  return {
    datasetAlias: string(record.dataset_alias),
    episodeCount: integer(record.episode_count),
    counts: {
      pending: counts.pending ?? invalid(),
      draft: counts.draft ?? invalid(),
      approved_keep: counts.approved_keep ?? invalid(),
      approved_reject: counts.approved_reject ?? invalid(),
    },
    promptTemplateVersion: string(record.prompt_template_version),
    promptTemplateSha256: sha256(record.prompt_template_sha256),
  };
}

function configuration(value: unknown): BatchConfiguration {
  const record = object(
    value,
    [
      "schema_version",
      "dataset_alias",
      "dataset_id",
      "source_path",
      "source_manifest_sha256",
      "source_fps",
      "episode_indices",
      "prompt",
      "cosmos",
      "sampling",
      "worker",
      "transport",
      "limits",
    ],
    ["parent_job_id"],
  );
  if (record.schema_version !== 1) return invalid();
  const episodes = array(record.episode_indices, integer);
  if (
    episodes.length === 0 ||
    episodes.some((item, index) => index > 0 && item <= episodes[index - 1])
  )
    return invalid();
  const prompt = object(record.prompt, ["version", "sha256"]);
  const cosmos = object(record.cosmos, [
    "base_url",
    "model",
    "api_key_env",
    "endpoint_identity",
  ]);
  const sampling = object(record.sampling, ["target_fps"]);
  const worker = object(record.worker, [
    "concurrency",
    "lease_seconds",
    "heartbeat_seconds",
  ]);
  const transport = object(record.transport, [
    "timeout_seconds",
    "initial_attempts",
    "repair_attempts",
  ]);
  const limits = object(record.limits, [
    "maximum_duration_seconds",
    "maximum_sampled_frames",
    "maximum_payload_bytes",
  ]);
  if (
    sampling.target_fps !== 2 ||
    worker.concurrency !== 1 ||
    worker.lease_seconds !== 180 ||
    worker.heartbeat_seconds !== 15 ||
    transport.timeout_seconds !== 120 ||
    transport.initial_attempts !== 2 ||
    transport.repair_attempts !== 1
  )
    return invalid();
  const parent = hasOwn(record, "parent_job_id")
    ? string(record.parent_job_id)
    : undefined;
  return {
    schema_version: 1,
    dataset_alias: string(record.dataset_alias),
    dataset_id: integer(record.dataset_id, 1),
    source_path: string(record.source_path),
    source_manifest_sha256: sha256(record.source_manifest_sha256),
    source_fps: finite(record.source_fps, Number.MIN_VALUE),
    episode_indices: episodes,
    ...(parent === undefined ? {} : { parent_job_id: parent }),
    prompt: { version: string(prompt.version), sha256: sha256(prompt.sha256) },
    cosmos: {
      base_url: string(cosmos.base_url),
      model: string(cosmos.model),
      api_key_env: string(cosmos.api_key_env),
      endpoint_identity: string(cosmos.endpoint_identity),
    },
    sampling: { target_fps: 2 },
    worker: { concurrency: 1, lease_seconds: 180, heartbeat_seconds: 15 },
    transport: {
      timeout_seconds: 120,
      initial_attempts: 2,
      repair_attempts: 1,
    },
    limits: {
      maximum_duration_seconds: finite(
        limits.maximum_duration_seconds,
        Number.MIN_VALUE,
      ),
      maximum_sampled_frames: integer(limits.maximum_sampled_frames, 1),
      maximum_payload_bytes: integer(limits.maximum_payload_bytes, 1),
    },
  };
}

function batch(value: unknown): BatchStatus {
  const record = object(value, [
    "job_id",
    "parent_job_id",
    "state",
    "configuration",
    "counts",
    "episodes",
    "lease",
    "current_episode",
    "cancel_requested",
    "created_at",
    "updated_at",
    "errors",
    "active_proposal_coverage",
  ]);
  const counts = exactCountRecord(record.counts, ATTEMPT_STATES, false);
  const leaseRecord =
    record.lease === null
      ? null
      : object(record.lease, ["owner", "expires_at"]);
  return {
    jobId: string(record.job_id),
    parentJobId: nullableString(record.parent_job_id),
    state: batchState(record.state),
    configuration: configuration(record.configuration),
    counts,
    episodes: array(record.episodes, (item) => {
      const attempt = object(item, [
        "attempt_id",
        "attempt_number",
        "source_episode_index",
        "state",
      ]);
      return {
        attemptId: string(attempt.attempt_id),
        attemptNumber: integer(attempt.attempt_number),
        sourceEpisodeIndex: integer(attempt.source_episode_index),
        state: attemptState(attempt.state),
      };
    }),
    lease:
      leaseRecord === null
        ? null
        : {
            owner: string(leaseRecord.owner),
            expiresAt: string(leaseRecord.expires_at),
          },
    currentEpisode: nullableInteger(record.current_episode),
    cancelRequested: boolean(record.cancel_requested),
    createdAt: string(record.created_at),
    updatedAt: string(record.updated_at),
    errors: array(record.errors, (item) => {
      const error = object(item, [
        "attempt_id",
        "source_episode_index",
        "error_class",
        "error_summary",
      ]);
      return {
        attemptId: string(error.attempt_id),
        sourceEpisodeIndex: integer(error.source_episode_index),
        errorClass: nullableString(error.error_class),
        errorSummary: nullableString(error.error_summary),
      };
    }),
    activeProposalCoverage: integer(record.active_proposal_coverage),
  };
}

function gripPoint(value: unknown): GripPoint {
  const record = object(value, ["frame", "timestamp_s", "derivative"]);
  return {
    frame: integer(record.frame),
    timestampSeconds: finite(record.timestamp_s),
    derivative: finite(record.derivative, -Number.MAX_VALUE),
  };
}

function grip(value: unknown): GripDiagnostic {
  const record = object(value, [
    "dataset_alias",
    "source_episode_index",
    "status",
    "side",
    "reason",
    "usable_joint_indices",
    "grasp",
    "release",
    "advisories",
    "grasp_delta_s",
    "release_delta_s",
  ]);
  const usableJointIndices = array(record.usable_joint_indices, integer);
  if (
    usableJointIndices.some(
      (index, position) =>
        position > 0 && index <= usableJointIndices[position - 1],
    )
  )
    return invalid();
  const base = {
    datasetAlias: string(record.dataset_alias),
    sourceEpisodeIndex: integer(record.source_episode_index),
    usableJointIndices,
    advisories: strings(record.advisories),
    graspDeltaSeconds: nullableFinite(record.grasp_delta_s),
    releaseDeltaSeconds: nullableFinite(record.release_delta_s),
  };
  if (record.status === "available") {
    if (record.reason !== null) return invalid();
    return {
      ...base,
      status: "available",
      side: hand(record.side),
      reason: null,
      grasp: gripPoint(record.grasp),
      release: gripPoint(record.release),
    };
  }
  if (
    record.status !== "unavailable" ||
    record.grasp !== null ||
    record.release !== null
  )
    return invalid();
  return {
    ...base,
    status: "unavailable",
    side: nullableHand(record.side),
    reason: string(record.reason),
    grasp: null,
    release: null,
  };
}

function distribution(value: unknown): AuditDistribution {
  const record = object(value, [
    "count",
    "min_s",
    "p25_s",
    "median_s",
    "p75_s",
    "max_s",
  ]);
  const count = integer(record.count);
  const values = [
    nullableFinite(record.min_s),
    nullableFinite(record.p25_s),
    nullableFinite(record.median_s),
    nullableFinite(record.p75_s),
    nullableFinite(record.max_s),
  ];
  if (
    (count === 0 && values.some((item) => item !== null)) ||
    (count > 0 &&
      (values.some((item) => item === null) ||
        values.some((item, index) => {
          const previous = values[index - 1];
          return (
            index > 0 && item !== null && previous !== null && item < previous
          );
        })))
  )
    return invalid();
  return {
    count,
    minSeconds: values[0],
    p25Seconds: values[1],
    medianSeconds: values[2],
    p75Seconds: values[3],
    maxSeconds: values[4],
  };
}

function stringRecord<T>(
  value: unknown,
  decoder: Decoder<T>,
): Record<string, T> {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    return invalid();
  const result: Record<string, T> = {};
  for (const [key, item] of Object.entries(value)) result[key] = decoder(item);
  return result;
}

function audit(value: unknown): CurationAudit {
  const record = object(value, [
    "dataset_alias",
    "review_state_counts",
    "cosmos",
    "transition_time_distributions",
    "phase_duration_distributions",
    "boundary_errors",
    "grip_disagreements",
    "grip_unavailable",
    "unreadable_files",
    "contact_sheet_issues",
    "source_fingerprint",
  ]);
  const cosmos = object(record.cosmos, [
    "job_state_counts",
    "attempt_state_counts",
    "proposal_state_counts",
    "proposal_result_counts",
    "attempt_error_class_counts",
  ]);
  const fingerprint = object(record.source_fingerprint, [
    "expected_sha256",
    "current_matches",
  ]);
  const reviewCounts = exactCountRecord(
    record.review_state_counts,
    REVIEW_STATES,
    true,
  );
  return {
    datasetAlias: string(record.dataset_alias),
    reviewStateCounts: {
      pending: reviewCounts.pending ?? invalid(),
      draft: reviewCounts.draft ?? invalid(),
      approved_keep: reviewCounts.approved_keep ?? invalid(),
      approved_reject: reviewCounts.approved_reject ?? invalid(),
    },
    cosmos: {
      jobStateCounts: exactCountRecord(
        cosmos.job_state_counts,
        BATCH_STATES,
        false,
      ),
      attemptStateCounts: exactCountRecord(
        cosmos.attempt_state_counts,
        ATTEMPT_STATES,
        false,
      ),
      proposalStateCounts: exactCountRecord(
        cosmos.proposal_state_counts,
        ["active", "superseded"],
        false,
      ),
      proposalResultCounts: exactCountRecord(
        cosmos.proposal_result_counts,
        ["complete", "incomplete", "invalid"],
        false,
      ),
      attemptErrorClassCounts: stringRecord(
        cosmos.attempt_error_class_counts,
        integer,
      ),
    },
    transitionTimeDistributions: stringRecord(
      record.transition_time_distributions,
      distribution,
    ),
    phaseDurationDistributions: stringRecord(
      record.phase_duration_distributions,
      distribution,
    ),
    boundaryErrors: array(record.boundary_errors, (item) => {
      const error = object(item, ["source_episode_index", "code"], ["step"]);
      const step = hasOwn(error, "step") ? integer(error.step, 1) : undefined;
      if (step !== undefined && step > 7) return invalid();
      return {
        sourceEpisodeIndex: integer(error.source_episode_index),
        code: string(error.code),
        ...(step === undefined ? {} : { step }),
      };
    }),
    gripDisagreements: array(record.grip_disagreements, (item) => {
      const disagreement = object(item, [
        "source_episode_index",
        "warnings",
        "grasp_delta_s",
        "release_delta_s",
      ]);
      return {
        sourceEpisodeIndex: integer(disagreement.source_episode_index),
        warnings: strings(disagreement.warnings),
        graspDeltaSeconds: nullableFinite(disagreement.grasp_delta_s),
        releaseDeltaSeconds: nullableFinite(disagreement.release_delta_s),
      };
    }),
    gripUnavailable: array(record.grip_unavailable, (item) => {
      const unavailable = object(item, ["source_episode_index", "reason"]);
      return {
        sourceEpisodeIndex: integer(unavailable.source_episode_index),
        reason: string(unavailable.reason),
      };
    }),
    unreadableFiles: array(record.unreadable_files, (item) => {
      const unreadable = object(item, [
        "source_episode_index",
        "asset_kind",
        "reason",
      ]);
      return {
        sourceEpisodeIndex: nullableInteger(unreadable.source_episode_index),
        assetKind: string(unreadable.asset_kind),
        reason: string(unreadable.reason),
      };
    }),
    contactSheetIssues: array(record.contact_sheet_issues, (item) => {
      const issue = object(
        item,
        ["kind", "source_episode_index", "reason"],
        ["proposal_id", "approval_revision"],
      );
      if (
        (issue.kind !== "proposal" && issue.kind !== "final") ||
        (issue.reason !== "missing" &&
          issue.reason !== "conflict" &&
          issue.reason !== "incomplete")
      )
        return invalid();
      const hasProposal = hasOwn(issue, "proposal_id");
      const hasApproval = hasOwn(issue, "approval_revision");
      if (
        (issue.kind === "proposal" && (!hasProposal || hasApproval)) ||
        (issue.kind === "final" && (hasProposal || !hasApproval))
      )
        return invalid();
      return {
        kind: issue.kind,
        sourceEpisodeIndex: integer(issue.source_episode_index),
        reason: issue.reason,
        ...(hasProposal ? { proposalId: string(issue.proposal_id) } : {}),
        ...(hasApproval
          ? { approvalRevision: integer(issue.approval_revision) }
          : {}),
      };
    }),
    sourceFingerprint: {
      expectedSha256: sha256(fingerprint.expected_sha256),
      currentMatches: boolean(fingerprint.current_matches),
    },
  };
}

function sanitizedJson(value: unknown, depth = 0): unknown {
  if (depth > 8) return invalid();
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "string") {
    if (value.length > 4096) return invalid();
    return value;
  }
  if (typeof value === "number") return finite(value, -Number.MAX_VALUE);
  if (Array.isArray(value)) {
    if (value.length > 256) return invalid();
    return value.map((item) => sanitizedJson(item, depth + 1));
  }
  if (typeof value !== "object") return invalid();
  const entries = Object.entries(value);
  if (entries.length > 256) return invalid();
  const result: Record<string, unknown> = {};
  for (const [key, item] of entries)
    result[key] = sanitizedJson(item, depth + 1);
  return result;
}

function errorPayload(value: unknown): CurationErrorPayload {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const errorEntry = Object.entries(value).find(([key]) => key === "error");
    if (errorEntry?.[1] === "revision_conflict") {
      // The current episode can be large. It is reloaded independently below
      // the context layer, so retain only this bounded discriminator.
      return { error: "revision_conflict" };
    }
  }
  const sanitized = sanitizedJson(value);
  if (
    sanitized === null ||
    typeof sanitized !== "object" ||
    Array.isArray(sanitized)
  )
    return invalid();
  const record: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(sanitized)) record[key] = item;
  const error = record.error;
  const detail = record.detail;
  const nestedError =
    detail !== null && typeof detail === "object" && !Array.isArray(detail)
      ? Object.entries(detail).find(([key]) => key === "error")?.[1]
      : undefined;
  if (
    !(typeof error === "string" && error.length > 0) &&
    !(typeof detail === "string" && detail.length > 0) &&
    !(typeof nestedError === "string" && nestedError.length > 0)
  )
    return invalid();
  return record;
}

function curationUrl(path: string): string {
  if (!path.startsWith("/api/curation/") || path.includes("\\"))
    throw new TypeError("curation client paths must stay under /api/curation/");
  return new URL(path, window.location.origin).toString();
}

async function requestJson<T>(
  path: string,
  decoder: Decoder<T>,
  options: {
    method?: "GET" | "POST" | "PATCH";
    body?: unknown;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  const headers = new Headers({ accept: "application/json" });
  const init: RequestInit = {
    method: options.method ?? "GET",
    cache: "no-store",
    credentials: "same-origin",
    signal: options.signal,
    headers,
  };
  if (options.body !== undefined) {
    headers.set("content-type", "application/json");
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(curationUrl(path), init);
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new CurationClientError("invalid_response", response.status || 502);
  }
  if (!response.ok) {
    try {
      throw new CurationClientError(
        "http_error",
        response.status,
        errorPayload(payload),
      );
    } catch (error) {
      if (error instanceof CurationClientError) throw error;
      throw new CurationClientError("invalid_response", response.status);
    }
  }
  try {
    return decoder(payload);
  } catch {
    throw new CurationClientError("invalid_response", response.status);
  }
}

function aliasQuery(datasetAlias: string): string {
  return `dataset_alias=${encodeURIComponent(datasetAlias)}`;
}

function matching<T>(
  decoder: Decoder<T>,
  predicate: (decoded: T) => boolean,
): Decoder<T> {
  return (value) => {
    const decoded = decoder(value);
    if (!predicate(decoded)) return invalid();
    return decoded;
  };
}

export function openCurationWorkspace(
  datasetAlias: string,
  actor: string,
  signal?: AbortSignal,
): Promise<WorkspaceOpened> {
  return requestJson(
    "/api/curation/workspaces/open",
    matching(workspace, (decoded) => decoded.datasetAlias === datasetAlias),
    {
      method: "POST",
      body: { dataset_alias: datasetAlias, actor },
      signal,
    },
  );
}

export function fetchCurationSummary(
  datasetAlias: string,
  signal?: AbortSignal,
): Promise<WorkspaceSummary> {
  return requestJson(
    `/api/curation/summary?${aliasQuery(datasetAlias)}`,
    matching(summary, (decoded) => decoded.datasetAlias === datasetAlias),
    {
      signal,
    },
  );
}

export function fetchEpisodeCuration(
  datasetAlias: string,
  sourceEpisodeIndex: number,
  signal?: AbortSignal,
): Promise<EpisodeCuration> {
  return requestJson(
    `/api/curation/episodes/${sourceEpisodeIndex}?${aliasQuery(datasetAlias)}`,
    matching(
      episode,
      (decoded) =>
        decoded.datasetAlias === datasetAlias &&
        decoded.sourceEpisodeIndex === sourceEpisodeIndex,
    ),
    { signal },
  );
}

interface MutationIdentity {
  dataset_alias: string;
  expected_revision: number;
  actor: string;
}

function episodeMutation(
  path: string,
  body: MutationIdentity & Record<string, unknown>,
  datasetAlias: string,
  sourceEpisodeIndex: number,
  method: "POST" | "PATCH" = "POST",
  signal?: AbortSignal,
): Promise<EpisodeCuration> {
  return requestJson(
    path,
    matching(
      episode,
      (decoded) =>
        decoded.datasetAlias === datasetAlias &&
        decoded.sourceEpisodeIndex === sourceEpisodeIndex,
    ),
    { method, body, signal },
  );
}

export function saveEpisodeDraft(
  datasetAlias: string,
  sourceEpisodeIndex: number,
  revision: number,
  actor: string,
  patch: EpisodeDraftPatch,
  signal?: AbortSignal,
): Promise<EpisodeCuration> {
  return episodeMutation(
    `/api/curation/episodes/${sourceEpisodeIndex}/draft`,
    {
      dataset_alias: datasetAlias,
      expected_revision: revision,
      actor,
      object_name: patch.objectName,
      pickup_hand: patch.pickupHand,
      turn_direction: patch.turnDirection,
      transition_frames: patch.transitionFrames,
    },
    datasetAlias,
    sourceEpisodeIndex,
    "PATCH",
    signal,
  );
}

export function applyEpisodeProposal(
  datasetAlias: string,
  sourceEpisodeIndex: number,
  revision: number,
  actor: string,
  signal?: AbortSignal,
): Promise<EpisodeCuration> {
  return episodeMutation(
    `/api/curation/episodes/${sourceEpisodeIndex}/apply-proposal`,
    { dataset_alias: datasetAlias, expected_revision: revision, actor },
    datasetAlias,
    sourceEpisodeIndex,
    "POST",
    signal,
  );
}

export function approveEpisodeKeep(
  datasetAlias: string,
  sourceEpisodeIndex: number,
  revision: number,
  actor: string,
  reviewer: string,
  signal?: AbortSignal,
): Promise<EpisodeCuration> {
  return episodeMutation(
    `/api/curation/episodes/${sourceEpisodeIndex}/approve-keep`,
    {
      dataset_alias: datasetAlias,
      expected_revision: revision,
      actor,
      reviewer,
    },
    datasetAlias,
    sourceEpisodeIndex,
    "POST",
    signal,
  );
}

export function approveEpisodeReject(
  datasetAlias: string,
  sourceEpisodeIndex: number,
  revision: number,
  actor: string,
  reviewer: string,
  reason?: string | null,
  signal?: AbortSignal,
): Promise<EpisodeCuration> {
  return episodeMutation(
    `/api/curation/episodes/${sourceEpisodeIndex}/approve-reject`,
    {
      dataset_alias: datasetAlias,
      expected_revision: revision,
      actor,
      reviewer,
      reason,
    },
    datasetAlias,
    sourceEpisodeIndex,
    "POST",
    signal,
  );
}

export function reopenEpisode(
  datasetAlias: string,
  sourceEpisodeIndex: number,
  revision: number,
  actor: string,
  signal?: AbortSignal,
): Promise<EpisodeCuration> {
  return episodeMutation(
    `/api/curation/episodes/${sourceEpisodeIndex}/reopen`,
    { dataset_alias: datasetAlias, expected_revision: revision, actor },
    datasetAlias,
    sourceEpisodeIndex,
    "POST",
    signal,
  );
}

export function fetchGripDiagnostic(
  datasetAlias: string,
  sourceEpisodeIndex: number,
  signal?: AbortSignal,
): Promise<GripDiagnostic> {
  return requestJson(
    `/api/curation/episodes/${sourceEpisodeIndex}/grip?${aliasQuery(datasetAlias)}`,
    matching(
      grip,
      (decoded) =>
        decoded.datasetAlias === datasetAlias &&
        decoded.sourceEpisodeIndex === sourceEpisodeIndex,
    ),
    { signal },
  );
}

export function fetchCurationAudit(
  datasetAlias: string,
  signal?: AbortSignal,
): Promise<CurationAudit> {
  return requestJson(
    `/api/curation/audit?${aliasQuery(datasetAlias)}`,
    matching(audit, (decoded) => decoded.datasetAlias === datasetAlias),
    { signal },
  );
}

export function startCurationBatch(
  datasetAlias: string,
  episodeIndices?: number[],
  signal?: AbortSignal,
): Promise<BatchStatus> {
  return requestJson(
    "/api/curation/batches",
    matching(
      batch,
      (decoded) => decoded.configuration.dataset_alias === datasetAlias,
    ),
    {
      method: "POST",
      body: {
        dataset_alias: datasetAlias,
        ...(episodeIndices === undefined
          ? {}
          : { episode_indices: episodeIndices }),
      },
      signal,
    },
  );
}

export function fetchCurationBatch(
  jobId: string,
  signal?: AbortSignal,
): Promise<BatchStatus> {
  return requestJson(
    `/api/curation/batches/${jobId}`,
    matching(batch, (decoded) => decoded.jobId === jobId),
    { signal },
  );
}

export function retryCurationBatch(
  jobId: string,
  selection: {
    episodeIndices?: number[];
    failureStates?: Array<
      Extract<AttemptState, "manual_only" | "retryable" | "cancelled">
    >;
  },
  signal?: AbortSignal,
): Promise<BatchStatus> {
  return requestJson(
    `/api/curation/batches/${jobId}/retry`,
    matching(
      batch,
      (decoded) =>
        decoded.parentJobId === jobId &&
        decoded.configuration.parent_job_id === jobId,
    ),
    {
      method: "POST",
      body: {
        ...(selection.episodeIndices === undefined
          ? {}
          : { episode_indices: selection.episodeIndices }),
        ...(selection.failureStates === undefined
          ? {}
          : { failure_states: selection.failureStates }),
      },
      signal,
    },
  );
}

export function cancelCurationBatch(
  jobId: string,
  signal?: AbortSignal,
): Promise<{
  job_id: string;
  state: "cancel_requested" | "cancelled";
  changed: boolean;
}> {
  return requestJson(
    `/api/curation/batches/${jobId}/cancel`,
    matching(
      (value) => {
        const record = object(value, ["job_id", "state", "changed"]);
        if (record.state !== "cancel_requested" && record.state !== "cancelled")
          return invalid();
        return {
          job_id: string(record.job_id),
          state: record.state,
          changed: boolean(record.changed),
        };
      },
      (decoded) => decoded.job_id === jobId,
    ),
    { method: "POST", signal },
  );
}
