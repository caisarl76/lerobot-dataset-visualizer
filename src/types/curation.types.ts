export type ReviewState =
  | "pending"
  | "draft"
  | "approved_keep"
  | "approved_reject";
export type Hand = "left" | "right";
export type TurnDirection = "left" | "right";
export type TransitionFrames = [
  number | null,
  number | null,
  number | null,
  number | null,
  number | null,
  number | null,
];

export type CosmosPhase =
  | "approach_brown_table"
  | "pick_up_object"
  | "turn_to_find_black_trash_bin"
  | "approach_black_trash_bin"
  | "lean_down_to_black_trash_bin"
  | "drop_object_into_black_trash_bin"
  | "stand_straight";

interface CosmosSegmentBase<
  Step extends 1 | 2 | 3 | 4 | 5 | 6 | 7,
  Phase extends CosmosPhase,
> {
  step: Step;
  phase: Phase;
  caption: string;
}

export type CosmosSegment<
  Step extends 1 | 2 | 3 | 4 | 5 | 6 | 7 = 1 | 2 | 3 | 4 | 5 | 6 | 7,
  Phase extends CosmosPhase = CosmosPhase,
> =
  | (CosmosSegmentBase<Step, Phase> & {
      status: "completed" | "partial";
      start_s: number;
      end_s: number;
      confidence: number;
      evidence: string;
    })
  | (CosmosSegmentBase<Step, Phase> & {
      status: "not_observed";
      start_s: null;
      end_s: null;
      confidence: null;
      evidence: null;
    });

export interface CosmosModelResponse {
  schema_version: 2;
  episode_complete: boolean;
  segments: [
    CosmosSegment<1, "approach_brown_table">,
    CosmosSegment<2, "pick_up_object">,
    CosmosSegment<3, "turn_to_find_black_trash_bin">,
    CosmosSegment<4, "approach_black_trash_bin">,
    CosmosSegment<5, "lean_down_to_black_trash_bin">,
    CosmosSegment<6, "drop_object_into_black_trash_bin">,
    CosmosSegment<7, "stand_straight">,
  ];
  missing_steps: Array<1 | 2 | 3 | 4 | 5 | 6 | 7>;
  uncertainties: string[];
}

export interface ActiveProposal {
  id: string;
  attemptId: string;
  modelResponse: CosmosModelResponse;
  transitionFrames: TransitionFrames;
  warnings: string[];
  createdAt: string;
}

export interface CurationDecision {
  reviewState: ReviewState;
  objectName: string | null;
  pickupHand: Hand | null;
  turnDirection: TurnDirection | null;
  transitionFrames: TransitionFrames;
  rejectionReason: string | null;
  promptTemplateSha256: string | null;
}

export interface EpisodeCuration {
  datasetAlias: string;
  sourceEpisodeIndex: number;
  sourceLength: number;
  timestamps: number[];
  decision: CurationDecision;
  activeProposal: ActiveProposal | null;
  /** The backend-generated strings are the only prompt templates used by the UI. */
  promptPreview: string[] | null;
  warnings: string[];
  revision: number;
  approvalLocked: boolean;
  reviewer: string | null;
  approvalRevision: number | null;
  approvedAt: string | null;
}

export interface WorkspaceSummary {
  datasetAlias: string;
  episodeCount: number;
  counts: Record<ReviewState, number>;
  promptTemplateVersion: string;
  promptTemplateSha256: string;
}

export interface WorkspaceOpened {
  datasetAlias: string;
  sourceFingerprint: string;
  promptTemplateVersion: string;
  promptTemplateSha256: string;
  episodeCount: number;
}

export type BatchState =
  | "queued"
  | "running"
  | "cancel_requested"
  | "completed"
  | "completed_with_failures"
  | "cancelled"
  | "failed";

export type AttemptState =
  | "queued"
  | "leased"
  | "requesting"
  | "succeeded"
  | "manual_only"
  | "retryable"
  | "cancelled";

export interface BatchAttemptSummary {
  attemptId: string;
  attemptNumber: number;
  sourceEpisodeIndex: number;
  state: AttemptState;
}

export interface BatchErrorSummary {
  attemptId: string;
  sourceEpisodeIndex: number;
  errorClass: string | null;
  errorSummary: string | null;
}

export interface BatchLease {
  owner: string;
  expiresAt: string;
}

export interface BatchConfiguration {
  schema_version: 1;
  dataset_alias: string;
  dataset_id: number;
  source_path: string;
  source_manifest_sha256: string;
  source_fps: number;
  episode_indices: number[];
  parent_job_id?: string;
  prompt: { version: string; sha256: string };
  cosmos: {
    base_url: string;
    model: string;
    api_key_env: string;
    endpoint_identity: string;
  };
  sampling: { target_fps: 2 };
  worker: { concurrency: 1; lease_seconds: 180; heartbeat_seconds: 15 };
  transport: { timeout_seconds: 120; initial_attempts: 2; repair_attempts: 1 };
  limits: {
    maximum_duration_seconds: number;
    maximum_sampled_frames: number;
    maximum_payload_bytes: number;
  };
}

export interface BatchStatus {
  jobId: string;
  parentJobId: string | null;
  state: BatchState;
  configuration: BatchConfiguration;
  counts: Partial<Record<AttemptState, number>>;
  episodes: BatchAttemptSummary[];
  lease: BatchLease | null;
  currentEpisode: number | null;
  cancelRequested: boolean;
  createdAt: string;
  updatedAt: string;
  errors: BatchErrorSummary[];
  activeProposalCoverage: number;
}

export interface GripPoint {
  frame: number;
  timestampSeconds: number;
  derivative: number;
}

interface GripDiagnosticBase {
  datasetAlias: string;
  sourceEpisodeIndex: number;
  side: Hand | null;
  usableJointIndices: number[];
  advisories: string[];
  graspDeltaSeconds: number | null;
  releaseDeltaSeconds: number | null;
}

export type GripDiagnostic =
  | (GripDiagnosticBase & {
      status: "available";
      reason: null;
      side: Hand;
      grasp: GripPoint;
      release: GripPoint;
    })
  | (GripDiagnosticBase & {
      status: "unavailable";
      reason: string;
      grasp: null;
      release: null;
    });

export interface AuditDistribution {
  count: number;
  minSeconds: number | null;
  p25Seconds: number | null;
  medianSeconds: number | null;
  p75Seconds: number | null;
  maxSeconds: number | null;
}

export interface CurationAudit {
  datasetAlias: string;
  reviewStateCounts: Record<ReviewState, number>;
  cosmos: {
    jobStateCounts: Partial<Record<BatchState, number>>;
    attemptStateCounts: Partial<Record<AttemptState, number>>;
    proposalStateCounts: Partial<Record<"active" | "superseded", number>>;
    proposalResultCounts: Partial<
      Record<"complete" | "incomplete" | "invalid", number>
    >;
    attemptErrorClassCounts: Record<string, number>;
  };
  transitionTimeDistributions: Record<string, AuditDistribution>;
  phaseDurationDistributions: Record<string, AuditDistribution>;
  boundaryErrors: Array<{
    sourceEpisodeIndex: number;
    code: string;
    step?: number;
  }>;
  gripDisagreements: Array<{
    sourceEpisodeIndex: number;
    warnings: string[];
    graspDeltaSeconds: number | null;
    releaseDeltaSeconds: number | null;
  }>;
  gripUnavailable: Array<{ sourceEpisodeIndex: number; reason: string }>;
  unreadableFiles: Array<{
    sourceEpisodeIndex: number | null;
    assetKind: string;
    reason: string;
  }>;
  contactSheetIssues: Array<{
    kind: "proposal" | "final";
    sourceEpisodeIndex: number;
    reason: "missing" | "conflict" | "incomplete";
    proposalId?: string;
    approvalRevision?: number;
  }>;
  sourceFingerprint: { expectedSha256: string; currentMatches: boolean };
}

export interface EpisodeDraftPatch {
  objectName?: string | null;
  pickupHand?: Hand | null;
  turnDirection?: TurnDirection | null;
  transitionFrames?: TransitionFrames;
}

export interface CurationConflictPrompt {
  kind: "revision_conflict";
  message: string;
  current: EpisodeCuration;
}
