"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from "react";

import type {
  AttemptState,
  BatchStatus,
  CurationAudit,
  CurationConflictPrompt,
  EpisodeCuration,
  EpisodeDraftPatch,
  GripDiagnostic,
  WorkspaceSummary,
} from "../types/curation.types";
import {
  CurationClientError,
  applyEpisodeProposal,
  approveEpisodeKeep,
  approveEpisodeReject,
  cancelCurationBatch,
  fetchCurationAudit,
  fetchCurationBatch,
  fetchCurationSummary,
  fetchEpisodeCuration,
  fetchGripDiagnostic,
  openCurationWorkspace,
  reopenEpisode,
  retryCurationBatch,
  saveEpisodeDraft,
  startCurationBatch,
} from "../utils/curationClient";

const TERMINAL_BATCH_STATES = new Set<BatchStatus["state"]>([
  "completed",
  "completed_with_failures",
  "cancelled",
  "failed",
]);

export interface CurationState {
  workspaceGeneration: number;
  workspaceDatasetAlias: string;
  workspaceReady: boolean;
  selectedEpisodeIndex: number;
  summary: WorkspaceSummary | null;
  episode: EpisodeCuration | null;
  grip: GripDiagnostic | null;
  audit: CurationAudit | null;
  batch: BatchStatus | null;
  loading: boolean;
  saving: boolean;
  error: string | null;
  warning: string | null;
  conflict: CurationConflictPrompt | null;
  episodeOperationToken: number;
  batchOperationToken: number;
  batchOperationPending: boolean;
  batchPollGeneration: number;
  auditOperationToken: number;
}

type WorkspaceIdentity = { workspaceGeneration: number };
type EpisodeIdentity = WorkspaceIdentity & {
  sourceEpisodeIndex: number;
  token: number;
};

type CurationAction =
  | {
      type: "workspace_reset";
      workspaceGeneration: number;
      datasetAlias: string;
      initialEpisodeIndex: number;
      episodeToken: number;
      batchToken: number;
      pollGeneration: number;
      auditToken: number;
    }
  | ({ type: "workspace_ready" } & WorkspaceIdentity)
  | ({ type: "workspace_error"; message: string } & WorkspaceIdentity)
  | ({ type: "select_episode" } & EpisodeIdentity)
  | ({ type: "episode_loading" } & EpisodeIdentity)
  | ({ type: "episode_loaded"; episode: EpisodeCuration } & EpisodeIdentity)
  | ({ type: "episode_saving" } & EpisodeIdentity)
  | ({ type: "episode_saved"; episode: EpisodeCuration } & EpisodeIdentity)
  | ({ type: "grip_loading" } & EpisodeIdentity)
  | ({ type: "grip_loaded"; grip: GripDiagnostic } & EpisodeIdentity)
  | ({ type: "conflict"; current: EpisodeCuration } & EpisodeIdentity)
  | ({ type: "episode_error"; message: string } & EpisodeIdentity)
  | ({ type: "edit_draft"; patch: EpisodeDraftPatch } & EpisodeIdentity)
  | ({ type: "summary_loaded"; summary: WorkspaceSummary } & WorkspaceIdentity)
  | ({ type: "warning"; message: string } & WorkspaceIdentity)
  | ({ type: "audit_started"; token: number } & WorkspaceIdentity)
  | ({
      type: "audit_loaded";
      token: number;
      audit: CurationAudit;
    } & WorkspaceIdentity)
  | ({
      type: "audit_error";
      token: number;
      message: string;
    } & WorkspaceIdentity)
  | ({
      type: "batch_operation_started";
      token: number;
      pollGeneration: number;
    } & WorkspaceIdentity)
  | ({
      type: "batch_loaded";
      token: number;
      batch: BatchStatus;
    } & WorkspaceIdentity)
  | ({
      type: "batch_error";
      token: number;
      message: string;
    } & WorkspaceIdentity)
  | ({
      type: "batch_poll_started";
      jobId: string;
      generation: number;
    } & WorkspaceIdentity)
  | ({
      type: "batch_polled";
      jobId: string;
      generation: number;
      batch: BatchStatus;
    } & WorkspaceIdentity)
  | ({
      type: "batch_poll_warning";
      jobId: string;
      generation: number;
      message: string;
    } & WorkspaceIdentity)
  | { type: "error"; message: string }
  | { type: "clear_conflict" };

function matchesEpisode(state: CurationState, identity: EpisodeIdentity) {
  return (
    state.workspaceGeneration === identity.workspaceGeneration &&
    state.selectedEpisodeIndex === identity.sourceEpisodeIndex &&
    state.episodeOperationToken === identity.token
  );
}

function matchesWorkspace(
  state: CurationState,
  identity: WorkspaceIdentity,
): boolean {
  return state.workspaceGeneration === identity.workspaceGeneration;
}

export function curationReducer(
  state: CurationState,
  action: CurationAction,
): CurationState {
  switch (action.type) {
    case "workspace_reset":
      return {
        workspaceGeneration: action.workspaceGeneration,
        workspaceDatasetAlias: action.datasetAlias,
        workspaceReady: false,
        selectedEpisodeIndex: action.initialEpisodeIndex,
        summary: null,
        episode: null,
        grip: null,
        audit: null,
        batch: null,
        loading: true,
        saving: false,
        error: null,
        warning: null,
        conflict: null,
        episodeOperationToken: action.episodeToken,
        batchOperationToken: action.batchToken,
        batchOperationPending: false,
        batchPollGeneration: action.pollGeneration,
        auditOperationToken: action.auditToken,
      };
    case "workspace_ready":
      if (!matchesWorkspace(state, action)) return state;
      return { ...state, workspaceReady: true };
    case "workspace_error":
      if (!matchesWorkspace(state, action)) return state;
      return { ...state, loading: false, saving: false, error: action.message };
    case "select_episode":
      if (!matchesWorkspace(state, action)) return state;
      return {
        ...state,
        selectedEpisodeIndex: action.sourceEpisodeIndex,
        episodeOperationToken: action.token,
        episode: null,
        grip: null,
        loading: true,
        saving: false,
        error: null,
        conflict: null,
      };
    case "episode_loading":
      if (
        !matchesWorkspace(state, action) ||
        state.selectedEpisodeIndex !== action.sourceEpisodeIndex
      )
        return state;
      return {
        ...state,
        episodeOperationToken: action.token,
        loading: true,
        saving: false,
        error: null,
      };
    case "episode_loaded":
      if (!matchesEpisode(state, action)) return state;
      return {
        ...state,
        episode: action.episode,
        grip: null,
        loading: false,
        saving: false,
        error: null,
      };
    case "episode_saving":
      if (
        !matchesWorkspace(state, action) ||
        state.selectedEpisodeIndex !== action.sourceEpisodeIndex
      )
        return state;
      return {
        ...state,
        episodeOperationToken: action.token,
        saving: true,
        error: null,
        warning: null,
        conflict: null,
      };
    case "episode_saved":
      if (!matchesEpisode(state, action)) return state;
      return {
        ...state,
        episode: action.episode,
        grip: null,
        saving: false,
        error: null,
        conflict: null,
      };
    case "grip_loading":
      if (
        !matchesWorkspace(state, action) ||
        state.selectedEpisodeIndex !== action.sourceEpisodeIndex
      )
        return state;
      return {
        ...state,
        episodeOperationToken: action.token,
        grip: null,
        error: null,
      };
    case "grip_loaded":
      if (!matchesEpisode(state, action)) return state;
      return { ...state, grip: action.grip };
    case "conflict":
      if (!matchesEpisode(state, action)) return state;
      return {
        ...state,
        episode: action.current,
        grip: null,
        loading: false,
        saving: false,
        error: null,
        conflict: {
          kind: "revision_conflict",
          message:
            "This episode changed elsewhere. Reloaded the current revision; reconcile before saving.",
          current: action.current,
        },
      };
    case "episode_error":
      if (!matchesEpisode(state, action)) return state;
      return {
        ...state,
        loading: false,
        saving: false,
        error: action.message,
      };
    case "edit_draft":
      if (
        !matchesWorkspace(state, action) ||
        state.selectedEpisodeIndex !== action.sourceEpisodeIndex ||
        state.episode === null ||
        state.episode.approvalLocked
      )
        return state;
      return {
        ...state,
        episodeOperationToken: action.token,
        grip: null,
        episode: {
          ...state.episode,
          decision: { ...state.episode.decision, ...action.patch },
          // Prompt strings come only from EpisodeCuration.promptPreview.
          promptPreview: null,
        },
        error: null,
        conflict: null,
      };
    case "summary_loaded":
      if (!matchesWorkspace(state, action)) return state;
      return { ...state, summary: action.summary };
    case "warning":
      if (!matchesWorkspace(state, action)) return state;
      return { ...state, warning: action.message };
    case "audit_started":
      if (!matchesWorkspace(state, action)) return state;
      return { ...state, auditOperationToken: action.token };
    case "audit_loaded":
      if (
        !matchesWorkspace(state, action) ||
        state.auditOperationToken !== action.token
      )
        return state;
      return { ...state, audit: action.audit };
    case "audit_error":
      if (
        !matchesWorkspace(state, action) ||
        state.auditOperationToken !== action.token
      )
        return state;
      return { ...state, error: action.message };
    case "batch_operation_started":
      if (!matchesWorkspace(state, action)) return state;
      return {
        ...state,
        batchOperationToken: action.token,
        batchOperationPending: true,
        batchPollGeneration: action.pollGeneration,
        error: null,
      };
    case "batch_loaded":
      if (
        !matchesWorkspace(state, action) ||
        state.batchOperationToken !== action.token
      )
        return state;
      return { ...state, batch: action.batch, batchOperationPending: false };
    case "batch_error":
      if (
        !matchesWorkspace(state, action) ||
        state.batchOperationToken !== action.token
      )
        return state;
      return { ...state, batchOperationPending: false, error: action.message };
    case "batch_poll_started":
      if (
        !matchesWorkspace(state, action) ||
        state.batch?.jobId !== action.jobId
      )
        return state;
      return { ...state, batchPollGeneration: action.generation };
    case "batch_polled":
      if (
        !matchesWorkspace(state, action) ||
        state.batch?.jobId !== action.jobId ||
        state.batchPollGeneration !== action.generation
      )
        return state;
      return { ...state, batch: action.batch };
    case "batch_poll_warning":
      if (
        !matchesWorkspace(state, action) ||
        state.batch?.jobId !== action.jobId ||
        state.batchPollGeneration !== action.generation
      )
        return state;
      return { ...state, warning: action.message };
    case "error":
      return { ...state, loading: false, saving: false, error: action.message };
    case "clear_conflict":
      return { ...state, conflict: null };
  }
}

interface CurationContextValue extends CurationState {
  selectEpisode: (sourceEpisodeIndex: number) => void;
  updateDraft: (patch: EpisodeDraftPatch) => void;
  saveDraft: () => Promise<void>;
  applyProposal: () => Promise<void>;
  reopen: () => Promise<void>;
  approveKeepAndNext: () => Promise<void>;
  approveRejectAndNext: (reason?: string | null) => Promise<void>;
  startBatch: (episodeIndices?: number[]) => Promise<void>;
  cancelBatch: () => Promise<void>;
  retryBatch: (selection: {
    episodeIndices?: number[];
    failureStates?: Array<
      Extract<AttemptState, "manual_only" | "retryable" | "cancelled">
    >;
  }) => Promise<void>;
  refreshGrip: () => Promise<void>;
  refreshAudit: () => Promise<void>;
  clearConflict: () => void;
}

const CurationContext = createContext<CurationContextValue | null>(null);

export interface CurationProviderProps {
  children: React.ReactNode;
  datasetAlias: string;
  actor: string;
  reviewer: string;
  episodeIndices: number[];
  initialEpisodeIndex: number;
  pollIntervalMs?: number;
}

function errorMessage(error: unknown): string {
  if (error instanceof CurationClientError) {
    const code = error.errorPayload?.error;
    return typeof code === "string"
      ? code
      : error.code === "invalid_response"
        ? "Curation service returned an invalid response"
        : `Curation request failed (${error.status})`;
  }
  return error instanceof Error ? error.message : "Unknown curation error";
}

function isAbort(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

export function waitForPoll(
  milliseconds: number,
  signal: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    };
    const timer = window.setTimeout(
      () => {
        signal.removeEventListener("abort", onAbort);
        resolve();
      },
      Math.max(1, milliseconds),
    );
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export function CurationProvider({
  children,
  datasetAlias,
  actor,
  reviewer,
  episodeIndices,
  initialEpisodeIndex,
  pollIntervalMs = 1_000,
}: CurationProviderProps) {
  const [state, dispatch] = useReducer(curationReducer, {
    workspaceGeneration: 0,
    workspaceDatasetAlias: datasetAlias,
    workspaceReady: false,
    selectedEpisodeIndex: initialEpisodeIndex,
    summary: null,
    episode: null,
    grip: null,
    audit: null,
    batch: null,
    loading: true,
    saving: false,
    error: null,
    warning: null,
    conflict: null,
    episodeOperationToken: 0,
    batchOperationToken: 0,
    batchOperationPending: false,
    batchPollGeneration: 0,
    auditOperationToken: 0,
  });

  const actorRef = useRef(actor);
  actorRef.current = actor;
  const initialEpisodeIndexRef = useRef(initialEpisodeIndex);
  initialEpisodeIndexRef.current = initialEpisodeIndex;
  const workspaceGenerationRef = useRef(0);
  const selectedEpisodeRef = useRef(initialEpisodeIndex);
  const episodeTokenRef = useRef(0);
  const episodeControllerRef = useRef<AbortController | null>(null);
  const batchTokenRef = useRef(0);
  const batchControllerRef = useRef<AbortController | null>(null);
  const pollGenerationRef = useRef(0);
  const pollControllerRef = useRef<AbortController | null>(null);
  const auditTokenRef = useRef(0);
  const auditControllerRef = useRef<AbortController | null>(null);

  const beginEpisodeOperation = useCallback(
    (sourceEpisodeIndex: number, kind: "loading" | "saving" | "grip") => {
      episodeControllerRef.current?.abort();
      const controller = new AbortController();
      episodeControllerRef.current = controller;
      const token = ++episodeTokenRef.current;
      const workspaceGeneration = workspaceGenerationRef.current;
      const type =
        kind === "loading"
          ? "episode_loading"
          : kind === "saving"
            ? "episode_saving"
            : "grip_loading";
      dispatch({ type, workspaceGeneration, sourceEpisodeIndex, token });
      return { sourceEpisodeIndex, token, workspaceGeneration, controller };
    },
    [],
  );

  const episodeIsCurrent = useCallback(
    (identity: EpisodeIdentity) =>
      workspaceGenerationRef.current === identity.workspaceGeneration &&
      episodeTokenRef.current === identity.token &&
      selectedEpisodeRef.current === identity.sourceEpisodeIndex,
    [],
  );

  const refreshSummary = useCallback(
    async (workspaceGeneration: number, signal?: AbortSignal) => {
      const summary = await fetchCurationSummary(datasetAlias, signal);
      dispatch({ type: "summary_loaded", workspaceGeneration, summary });
    },
    [datasetAlias],
  );

  useEffect(() => {
    const controller = new AbortController();
    const workspaceGeneration = ++workspaceGenerationRef.current;
    const routeEpisodeIndex = initialEpisodeIndexRef.current;
    selectedEpisodeRef.current = routeEpisodeIndex;
    episodeControllerRef.current?.abort();
    auditControllerRef.current?.abort();
    batchControllerRef.current?.abort();
    pollControllerRef.current?.abort();
    const episodeToken = ++episodeTokenRef.current;
    const auditToken = ++auditTokenRef.current;
    const batchToken = ++batchTokenRef.current;
    const pollGeneration = ++pollGenerationRef.current;
    dispatch({
      type: "workspace_reset",
      workspaceGeneration,
      datasetAlias,
      initialEpisodeIndex: routeEpisodeIndex,
      episodeToken,
      auditToken,
      batchToken,
      pollGeneration,
    });
    void (async () => {
      try {
        await openCurationWorkspace(
          datasetAlias,
          actorRef.current,
          controller.signal,
        );
        await refreshSummary(workspaceGeneration, controller.signal);
        dispatch({ type: "workspace_ready", workspaceGeneration });
      } catch (error) {
        if (!isAbort(error))
          dispatch({
            type: "workspace_error",
            workspaceGeneration,
            message: errorMessage(error),
          });
      }
    })();
    return () => controller.abort();
  }, [datasetAlias, refreshSummary]);

  useEffect(() => {
    if (
      state.workspaceDatasetAlias !== datasetAlias ||
      selectedEpisodeRef.current === initialEpisodeIndex
    )
      return;
    episodeControllerRef.current?.abort();
    selectedEpisodeRef.current = initialEpisodeIndex;
    const token = ++episodeTokenRef.current;
    dispatch({
      type: "select_episode",
      workspaceGeneration: workspaceGenerationRef.current,
      sourceEpisodeIndex: initialEpisodeIndex,
      token,
    });
  }, [datasetAlias, initialEpisodeIndex, state.workspaceDatasetAlias]);

  useEffect(() => {
    if (!state.workspaceReady || state.workspaceDatasetAlias !== datasetAlias)
      return;
    const sourceEpisodeIndex = state.selectedEpisodeIndex;
    const operation = beginEpisodeOperation(sourceEpisodeIndex, "loading");
    void fetchEpisodeCuration(
      datasetAlias,
      sourceEpisodeIndex,
      operation.controller.signal,
    )
      .then((episode) =>
        dispatch({
          type: "episode_loaded",
          workspaceGeneration: operation.workspaceGeneration,
          sourceEpisodeIndex,
          token: operation.token,
          episode,
        }),
      )
      .catch((error: unknown) => {
        if (!isAbort(error))
          dispatch({
            type: "episode_error",
            workspaceGeneration: operation.workspaceGeneration,
            sourceEpisodeIndex,
            token: operation.token,
            message: errorMessage(error),
          });
      });
    return () => operation.controller.abort();
  }, [
    beginEpisodeOperation,
    datasetAlias,
    state.selectedEpisodeIndex,
    state.workspaceDatasetAlias,
    state.workspaceReady,
  ]);

  const polledJobId = state.batch?.jobId;
  const polledBatchState = state.batch?.state;
  useEffect(() => {
    if (
      polledJobId === undefined ||
      polledBatchState === undefined ||
      state.batchOperationPending ||
      TERMINAL_BATCH_STATES.has(polledBatchState)
    )
      return;
    const jobId = polledJobId;
    const workspaceGeneration = state.workspaceGeneration;
    const generation = ++pollGenerationRef.current;
    const controller = new AbortController();
    pollControllerRef.current = controller;
    dispatch({
      type: "batch_poll_started",
      workspaceGeneration,
      jobId,
      generation,
    });
    void (async () => {
      while (!controller.signal.aborted) {
        try {
          await waitForPoll(pollIntervalMs, controller.signal);
          const next = await fetchCurationBatch(jobId, controller.signal);
          dispatch({
            type: "batch_polled",
            workspaceGeneration,
            jobId,
            generation,
            batch: next,
          });
          if (TERMINAL_BATCH_STATES.has(next.state)) return;
        } catch (error) {
          if (isAbort(error)) return;
          dispatch({
            type: "batch_poll_warning",
            workspaceGeneration,
            jobId,
            generation,
            message: `Batch status refresh failed: ${errorMessage(error)}`,
          });
        }
      }
    })();
    return () => {
      controller.abort();
      if (pollControllerRef.current === controller)
        pollControllerRef.current = null;
    };
  }, [
    pollIntervalMs,
    polledBatchState,
    polledJobId,
    state.batchOperationPending,
    state.workspaceGeneration,
  ]);

  const selectEpisode = useCallback(
    (sourceEpisodeIndex: number) => {
      if (!episodeIndices.includes(sourceEpisodeIndex)) {
        dispatch({
          type: "error",
          message: "Episode is outside the configured review set.",
        });
        return;
      }
      if (selectedEpisodeRef.current === sourceEpisodeIndex) return;
      episodeControllerRef.current?.abort();
      selectedEpisodeRef.current = sourceEpisodeIndex;
      const token = ++episodeTokenRef.current;
      dispatch({
        type: "select_episode",
        workspaceGeneration: workspaceGenerationRef.current,
        sourceEpisodeIndex,
        token,
      });
    },
    [episodeIndices],
  );

  const updateDraft = useCallback((patch: EpisodeDraftPatch) => {
    const sourceEpisodeIndex = selectedEpisodeRef.current;
    episodeControllerRef.current?.abort();
    const token = ++episodeTokenRef.current;
    dispatch({
      type: "edit_draft",
      workspaceGeneration: workspaceGenerationRef.current,
      sourceEpisodeIndex,
      token,
      patch,
    });
  }, []);

  const runMutation = useCallback(
    async (
      operation: (
        episode: EpisodeCuration,
        signal: AbortSignal,
      ) => Promise<EpisodeCuration>,
    ) => {
      const current = state.episode;
      if (current === null) {
        dispatch({ type: "error", message: "No episode is loaded." });
        return null;
      }
      const sourceEpisodeIndex = current.sourceEpisodeIndex;
      const pending = beginEpisodeOperation(sourceEpisodeIndex, "saving");
      try {
        const saved = await operation(current, pending.controller.signal);
        if (!episodeIsCurrent(pending)) return null;
        dispatch({
          type: "episode_saved",
          workspaceGeneration: pending.workspaceGeneration,
          sourceEpisodeIndex,
          token: pending.token,
          episode: saved,
        });
        try {
          await refreshSummary(
            pending.workspaceGeneration,
            pending.controller.signal,
          );
        } catch (summaryError) {
          if (!isAbort(summaryError) && episodeIsCurrent(pending)) {
            dispatch({
              type: "warning",
              workspaceGeneration: pending.workspaceGeneration,
              message: `Review saved, but summary refresh failed: ${errorMessage(summaryError)}`,
            });
          }
        }
        return episodeIsCurrent(pending) ? saved : null;
      } catch (error) {
        if (isAbort(error) || !episodeIsCurrent(pending)) return null;
        if (
          error instanceof CurationClientError &&
          error.status === 409 &&
          error.errorPayload?.error === "revision_conflict"
        ) {
          try {
            const currentEpisode = await fetchEpisodeCuration(
              datasetAlias,
              sourceEpisodeIndex,
              pending.controller.signal,
            );
            dispatch({
              type: "conflict",
              workspaceGeneration: pending.workspaceGeneration,
              sourceEpisodeIndex,
              token: pending.token,
              current: currentEpisode,
            });
          } catch (reloadError) {
            if (!isAbort(reloadError))
              dispatch({
                type: "episode_error",
                workspaceGeneration: pending.workspaceGeneration,
                sourceEpisodeIndex,
                token: pending.token,
                message: errorMessage(reloadError),
              });
          }
        } else {
          dispatch({
            type: "episode_error",
            workspaceGeneration: pending.workspaceGeneration,
            sourceEpisodeIndex,
            token: pending.token,
            message: errorMessage(error),
          });
        }
        return null;
      }
    },
    [
      beginEpisodeOperation,
      datasetAlias,
      episodeIsCurrent,
      refreshSummary,
      state.episode,
    ],
  );

  const saveDraft = useCallback(async () => {
    if (state.episode?.approvalLocked) {
      dispatch({
        type: "error",
        message: "Reopen this approved episode before editing.",
      });
      return;
    }
    await runMutation((current, signal) =>
      saveEpisodeDraft(
        datasetAlias,
        current.sourceEpisodeIndex,
        current.revision,
        actor,
        {
          objectName: current.decision.objectName,
          pickupHand: current.decision.pickupHand,
          turnDirection: current.decision.turnDirection,
          transitionFrames: current.decision.transitionFrames,
        },
        signal,
      ),
    );
  }, [actor, datasetAlias, runMutation, state.episode]);

  const applyProposal = useCallback(async () => {
    if (state.episode?.approvalLocked) {
      dispatch({
        type: "error",
        message: "Reopen this approved episode before applying a proposal.",
      });
      return;
    }
    await runMutation((episode, signal) =>
      applyEpisodeProposal(
        datasetAlias,
        episode.sourceEpisodeIndex,
        episode.revision,
        actor,
        signal,
      ),
    );
  }, [actor, datasetAlias, runMutation, state.episode]);

  const reopen = useCallback(async () => {
    await runMutation((episode, signal) =>
      reopenEpisode(
        datasetAlias,
        episode.sourceEpisodeIndex,
        episode.revision,
        actor,
        signal,
      ),
    );
  }, [actor, datasetAlias, runMutation]);

  const goToNext = useCallback(
    (sourceEpisodeIndex: number) => {
      const position = episodeIndices.indexOf(sourceEpisodeIndex);
      const next = position < 0 ? undefined : episodeIndices[position + 1];
      if (next !== undefined) selectEpisode(next);
    },
    [episodeIndices, selectEpisode],
  );

  const approveKeepAndNext = useCallback(async () => {
    const saved = await runMutation((episode, signal) =>
      approveEpisodeKeep(
        datasetAlias,
        episode.sourceEpisodeIndex,
        episode.revision,
        actor,
        reviewer,
        signal,
      ),
    );
    if (saved !== null) goToNext(saved.sourceEpisodeIndex);
  }, [actor, datasetAlias, goToNext, reviewer, runMutation]);

  const approveRejectAndNext = useCallback(
    async (reason?: string | null) => {
      const saved = await runMutation((episode, signal) =>
        approveEpisodeReject(
          datasetAlias,
          episode.sourceEpisodeIndex,
          episode.revision,
          actor,
          reviewer,
          reason,
          signal,
        ),
      );
      if (saved !== null) goToNext(saved.sourceEpisodeIndex);
    },
    [actor, datasetAlias, goToNext, reviewer, runMutation],
  );

  const beginBatchOperation = useCallback(() => {
    batchControllerRef.current?.abort();
    pollControllerRef.current?.abort();
    const controller = new AbortController();
    batchControllerRef.current = controller;
    const token = ++batchTokenRef.current;
    const pollGeneration = ++pollGenerationRef.current;
    const workspaceGeneration = workspaceGenerationRef.current;
    dispatch({
      type: "batch_operation_started",
      workspaceGeneration,
      token,
      pollGeneration,
    });
    return { workspaceGeneration, token, controller };
  }, []);

  const startBatch = useCallback(
    async (selected?: number[]) => {
      const operation = beginBatchOperation();
      try {
        const batch = await startCurationBatch(
          datasetAlias,
          selected,
          operation.controller.signal,
        );
        dispatch({
          type: "batch_loaded",
          workspaceGeneration: operation.workspaceGeneration,
          token: operation.token,
          batch,
        });
      } catch (error) {
        if (!isAbort(error))
          dispatch({
            type: "batch_error",
            workspaceGeneration: operation.workspaceGeneration,
            token: operation.token,
            message: errorMessage(error),
          });
      }
    },
    [beginBatchOperation, datasetAlias],
  );

  const cancelBatch = useCallback(async () => {
    if (state.batch === null) {
      dispatch({ type: "error", message: "No batch is selected." });
      return;
    }
    const jobId = state.batch.jobId;
    const operation = beginBatchOperation();
    try {
      await cancelCurationBatch(jobId, operation.controller.signal);
      if (batchTokenRef.current !== operation.token) return;
      const current = await fetchCurationBatch(
        jobId,
        operation.controller.signal,
      );
      dispatch({
        type: "batch_loaded",
        workspaceGeneration: operation.workspaceGeneration,
        token: operation.token,
        batch: current,
      });
    } catch (error) {
      if (!isAbort(error))
        dispatch({
          type: "batch_error",
          workspaceGeneration: operation.workspaceGeneration,
          token: operation.token,
          message: errorMessage(error),
        });
    }
  }, [beginBatchOperation, state.batch]);

  const retryBatch = useCallback(
    async (selection: {
      episodeIndices?: number[];
      failureStates?: Array<
        Extract<AttemptState, "manual_only" | "retryable" | "cancelled">
      >;
    }) => {
      if (state.batch === null) {
        dispatch({ type: "error", message: "No batch is selected." });
        return;
      }
      const jobId = state.batch.jobId;
      const operation = beginBatchOperation();
      try {
        const child = await retryCurationBatch(
          jobId,
          selection,
          operation.controller.signal,
        );
        dispatch({
          type: "batch_loaded",
          workspaceGeneration: operation.workspaceGeneration,
          token: operation.token,
          batch: child,
        });
      } catch (error) {
        if (!isAbort(error))
          dispatch({
            type: "batch_error",
            workspaceGeneration: operation.workspaceGeneration,
            token: operation.token,
            message: errorMessage(error),
          });
      }
    },
    [beginBatchOperation, state.batch],
  );

  const refreshGrip = useCallback(async () => {
    const current = state.episode;
    if (current === null) return;
    const sourceEpisodeIndex = current.sourceEpisodeIndex;
    const operation = beginEpisodeOperation(sourceEpisodeIndex, "grip");
    try {
      const grip = await fetchGripDiagnostic(
        datasetAlias,
        sourceEpisodeIndex,
        operation.controller.signal,
      );
      dispatch({
        type: "grip_loaded",
        workspaceGeneration: operation.workspaceGeneration,
        sourceEpisodeIndex,
        token: operation.token,
        grip,
      });
    } catch (error) {
      if (!isAbort(error))
        dispatch({
          type: "episode_error",
          workspaceGeneration: operation.workspaceGeneration,
          sourceEpisodeIndex,
          token: operation.token,
          message: errorMessage(error),
        });
    }
  }, [beginEpisodeOperation, datasetAlias, state.episode]);

  const refreshAudit = useCallback(async () => {
    auditControllerRef.current?.abort();
    const controller = new AbortController();
    auditControllerRef.current = controller;
    const token = ++auditTokenRef.current;
    const workspaceGeneration = workspaceGenerationRef.current;
    dispatch({ type: "audit_started", workspaceGeneration, token });
    try {
      const audit = await fetchCurationAudit(datasetAlias, controller.signal);
      dispatch({ type: "audit_loaded", workspaceGeneration, token, audit });
    } catch (error) {
      if (!isAbort(error))
        dispatch({
          type: "audit_error",
          workspaceGeneration,
          token,
          message: errorMessage(error),
        });
    }
  }, [datasetAlias]);

  useEffect(
    () => () => {
      episodeControllerRef.current?.abort();
      batchControllerRef.current?.abort();
      pollControllerRef.current?.abort();
      auditControllerRef.current?.abort();
    },
    [],
  );

  const clearConflict = useCallback(
    () => dispatch({ type: "clear_conflict" }),
    [],
  );

  const value = useMemo<CurationContextValue>(
    () => ({
      ...state,
      selectEpisode,
      updateDraft,
      saveDraft,
      applyProposal,
      reopen,
      approveKeepAndNext,
      approveRejectAndNext,
      startBatch,
      cancelBatch,
      retryBatch,
      refreshGrip,
      refreshAudit,
      clearConflict,
    }),
    [
      applyProposal,
      approveKeepAndNext,
      approveRejectAndNext,
      cancelBatch,
      clearConflict,
      refreshAudit,
      refreshGrip,
      reopen,
      retryBatch,
      saveDraft,
      selectEpisode,
      startBatch,
      state,
      updateDraft,
    ],
  );

  return (
    <CurationContext.Provider value={value}>
      {children}
    </CurationContext.Provider>
  );
}

export function useCuration(): CurationContextValue {
  const context = useContext(CurationContext);
  if (context === null)
    throw new Error("useCuration must be used within CurationProvider");
  return context;
}
