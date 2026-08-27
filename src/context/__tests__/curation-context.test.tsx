import {
  afterEach,
  beforeEach,
  describe,
  expect,
  mock as bunMock,
  test,
} from "bun:test";
import { act, render, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";

import {
  CurationProvider,
  useCuration,
  waitForPoll,
} from "../curation-context";

const DATASET_ALIAS = "local/pnp_trash";
const PROMPT_HASH = "a".repeat(64);
const SOURCE_HASH = "b".repeat(64);
const originalFetch = globalThis.fetch;
const mock = Object.assign(
  function typedFetchMock<
    T extends (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => Promise<Response>,
  >(implementation: T) {
    return Object.assign(bunMock(implementation), {
      preconnect: originalFetch.preconnect,
    });
  },
  { restore: bunMock.restore },
);

function json(body: unknown, status = 200): Response {
  return Response.json(body, { status });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

const PHASES = [
  "approach_brown_table",
  "pick_up_object",
  "turn_to_find_black_trash_bin",
  "approach_black_trash_bin",
  "lean_down_to_black_trash_bin",
  "drop_object_into_black_trash_bin",
  "stand_straight",
];

function cosmosResponse(proposalTransitions: Array<number | null>) {
  const missing = proposalTransitions.flatMap((frame, index) =>
    frame === null ? [index + 2] : [],
  );
  return {
    schema_version: 2,
    episode_complete: missing.length === 0,
    segments: PHASES.map((phase, index) => {
      const observed = index === 0 || proposalTransitions[index - 1] !== null;
      return observed
        ? {
            step: index + 1,
            phase,
            status: "completed",
            start_s: index / 10,
            end_s: (index + 1) / 10,
            confidence: 0.9,
            caption: `step ${index + 1}`,
            evidence: "visible",
          }
        : {
            step: index + 1,
            phase,
            status: "not_observed",
            start_s: null,
            end_s: null,
            confidence: null,
            caption: `step ${index + 1}`,
            evidence: null,
          };
    }),
    missing_steps: missing,
    uncertainties: missing.length === 0 ? [] : ["step not observed"],
  };
}

function batchConfiguration(parentJobId?: string) {
  return {
    schema_version: 1,
    dataset_alias: DATASET_ALIAS,
    dataset_id: 1,
    source_path: "/datasets/pnp_trash",
    source_manifest_sha256: SOURCE_HASH,
    source_fps: 30,
    episode_indices: [0, 1],
    ...(parentJobId === undefined ? {} : { parent_job_id: parentJobId }),
    prompt: { version: "pnp-trash-v1", sha256: PROMPT_HASH },
    cosmos: {
      base_url: "https://cosmos.internal/v1",
      model: "cosmos3-nano",
      api_key_env: "COSMOS_API_KEY",
      endpoint_identity: "h100-cosmos",
    },
    sampling: { target_fps: 2 },
    worker: { concurrency: 1, lease_seconds: 180, heartbeat_seconds: 15 },
    transport: {
      timeout_seconds: 120,
      initial_attempts: 2,
      repair_attempts: 1,
    },
    limits: {
      maximum_duration_seconds: 300,
      maximum_sampled_frames: 600,
      maximum_payload_bytes: 1000000,
    },
  };
}

function wireBatch(
  jobId: string,
  state:
    | "queued"
    | "running"
    | "cancel_requested"
    | "completed"
    | "cancelled" = "queued",
  parentJobId: string | null = null,
) {
  return {
    job_id: jobId,
    parent_job_id: parentJobId,
    state,
    configuration: batchConfiguration(parentJobId ?? undefined),
    counts:
      state === "completed"
        ? { succeeded: 2 }
        : state === "cancelled"
          ? { cancelled: 2 }
          : { queued: 2 },
    episodes: [],
    lease: null,
    current_episode: null,
    cancel_requested: state === "cancel_requested" || state === "cancelled",
    created_at: "2026-08-27T00:00:00Z",
    updated_at: "2026-08-27T00:00:00Z",
    errors: [],
    active_proposal_coverage: state === "completed" ? 2 : 0,
  };
}

function wireAudit(currentMatches: boolean) {
  return {
    dataset_alias: DATASET_ALIAS,
    review_state_counts: summary.counts,
    cosmos: {
      job_state_counts: {},
      attempt_state_counts: {},
      proposal_state_counts: {},
      proposal_result_counts: {},
      attempt_error_class_counts: {},
    },
    transition_time_distributions: {},
    phase_duration_distributions: {},
    boundary_errors: [],
    grip_disagreements: [],
    grip_unavailable: [],
    unreadable_files: [],
    contact_sheet_issues: [],
    source_fingerprint: {
      expected_sha256: SOURCE_HASH,
      current_matches: currentMatches,
    },
  };
}

function wireEpisode(
  sourceEpisodeIndex: number,
  options: {
    state?: "pending" | "draft" | "approved_keep" | "approved_reject";
    revision?: number;
    objectName?: string | null;
    transitions?: Array<number | null>;
    promptPreview?: string[] | null;
    proposalTransitions?: Array<number | null> | null;
    rejectionReason?: string | null;
    datasetAlias?: string;
  } = {},
) {
  const state = options.state ?? "pending";
  const revision = options.revision ?? 0;
  const locked = state === "approved_keep" || state === "approved_reject";
  const proposalTransitions = options.proposalTransitions;
  return {
    dataset_alias: options.datasetAlias ?? DATASET_ALIAS,
    source_episode_index: sourceEpisodeIndex,
    source_length: 8,
    timestamps: Array.from({ length: 8 }, (_, frame) => frame / 10),
    decision: {
      review_state: state,
      object_name: options.objectName ?? null,
      pickup_hand: options.objectName ? "left" : null,
      turn_direction: options.objectName ? "right" : null,
      transition_frames: options.transitions ?? [
        null,
        null,
        null,
        null,
        null,
        null,
      ],
      rejection_reason:
        options.rejectionReason === undefined
          ? state === "approved_reject"
            ? "wrong episode"
            : null
          : options.rejectionReason,
      prompt_template_sha256: PROMPT_HASH,
    },
    active_proposal:
      proposalTransitions === undefined || proposalTransitions === null
        ? null
        : {
            id: "proposal-1",
            attempt_id: "attempt-1",
            model_response: cosmosResponse(proposalTransitions),
            transition_frames: proposalTransitions,
            warnings: [],
            created_at: "2026-08-27T00:00:00Z",
          },
    prompt_preview:
      options.promptPreview ??
      (state === "approved_keep"
        ? ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        : null),
    warnings: [],
    revision,
    approval_locked: locked,
    reviewer: locked ? "reviewer" : null,
    approval_revision: locked ? revision : null,
    approved_at: locked ? "2026-08-27T00:00:00Z" : null,
  };
}

const summary = {
  dataset_alias: DATASET_ALIAS,
  episode_count: 2,
  counts: { pending: 2, draft: 0, approved_keep: 0, approved_reject: 0 },
  prompt_template_version: "pnp-trash-v1",
  prompt_template_sha256: PROMPT_HASH,
};

function wrapper(
  options: { initialEpisodeIndex?: number; pollIntervalMs?: number } = {},
) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <CurationProvider
        datasetAlias={DATASET_ALIAS}
        actor="curator"
        reviewer="reviewer"
        episodeIndices={[0, 1]}
        initialEpisodeIndex={options.initialEpisodeIndex ?? 0}
        pollIntervalMs={options.pollIntervalMs ?? 10}
      >
        {children}
      </CurationProvider>
    );
  };
}

function bootstrapResponse(request: Request): Response | null {
  const url = new URL(request.url);
  if (url.pathname.endsWith("/workspaces/open")) {
    return json({
      dataset_alias: DATASET_ALIAS,
      source_fingerprint: SOURCE_HASH,
      prompt_template_version: "pnp-trash-v1",
      prompt_template_sha256: PROMPT_HASH,
      episode_count: 2,
    });
  }
  if (url.pathname.endsWith("/summary")) return json(summary);
  return null;
}

beforeEach(() => {
  globalThis.fetch = originalFetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  mock.restore();
});

describe("CurationProvider", () => {
  test("removes a completed poll delay's abort listener", async () => {
    const controller = new AbortController();
    const signal = controller.signal;
    const nativeRemove = signal.removeEventListener.bind(signal);
    const removeListener = bunMock(
      (...args: Parameters<AbortSignal["removeEventListener"]>) =>
        nativeRemove(...args),
    );
    signal.removeEventListener = removeListener;

    await waitForPoll(1, signal);

    expect(removeListener).toHaveBeenCalledTimes(1);
    expect(removeListener.mock.calls[0]?.[0]).toBe("abort");
  });

  test("opens the workspace before requesting its first episode", async () => {
    let releaseWorkspace!: () => void;
    const workspaceReleased = new Promise<void>((resolve) => {
      releaseWorkspace = resolve;
    });
    let workspaceOpen = false;
    let episodeRequestedBeforeOpen = false;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        if (path.endsWith("/workspaces/open")) {
          await workspaceReleased;
          workspaceOpen = true;
          return json({
            dataset_alias: DATASET_ALIAS,
            source_fingerprint: SOURCE_HASH,
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: PROMPT_HASH,
            episode_count: 2,
          });
        }
        if (path.endsWith("/summary")) return json(summary);
        if (path.endsWith("/episodes/0")) {
          episodeRequestedBeforeOpen = !workspaceOpen;
          return json(wireEpisode(0));
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(episodeRequestedBeforeOpen).toBe(false);
    releaseWorkspace();
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
  });

  test("fences an old workspace open and summary when dataset identity changes", async () => {
    const nextAlias = "local/pnp_trash_b";
    const oldOpen = deferred<Response>();
    const oldSummary = deferred<Response>();
    let oldOpenSignal: AbortSignal | undefined;
    let oldSummarySignal: AbortSignal | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const url = new URL(request.url);
        if (url.pathname.endsWith("/workspaces/open")) {
          const body = await request.json();
          if (
            body !== null &&
            typeof body === "object" &&
            "dataset_alias" in body &&
            body.dataset_alias === DATASET_ALIAS
          ) {
            oldOpenSignal = request.signal;
            return oldOpen.promise;
          }
          return json({
            dataset_alias: nextAlias,
            source_fingerprint: SOURCE_HASH,
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: PROMPT_HASH,
            episode_count: 2,
          });
        }
        if (url.pathname.endsWith("/summary")) {
          if (url.searchParams.get("dataset_alias") === DATASET_ALIAS) {
            oldSummarySignal = request.signal;
            return oldSummary.promise;
          }
          return json({ ...summary, dataset_alias: nextAlias });
        }
        if (url.pathname.endsWith("/episodes/1"))
          return json(wireEpisode(1, { datasetAlias: nextAlias }));
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const observed: { current: ReturnType<typeof useCuration> | null } = {
      current: null,
    };
    function Probe() {
      observed.current = useCuration();
      return null;
    }
    const ui = (datasetAlias: string, initialEpisodeIndex: number) => (
      <CurationProvider
        datasetAlias={datasetAlias}
        actor="curator"
        reviewer="reviewer"
        episodeIndices={[0, 1]}
        initialEpisodeIndex={initialEpisodeIndex}
      >
        <Probe />
      </CurationProvider>
    );

    const view = render(ui(DATASET_ALIAS, 0));
    await waitFor(() => expect(oldOpenSignal).toBeDefined());
    act(() => view.rerender(ui(nextAlias, 1)));
    await waitFor(() =>
      expect(observed.current?.episode?.datasetAlias).toBe(nextAlias),
    );

    await act(async () => {
      oldOpen.resolve(
        json({
          dataset_alias: DATASET_ALIAS,
          source_fingerprint: SOURCE_HASH,
          prompt_template_version: "pnp-trash-v1",
          prompt_template_sha256: PROMPT_HASH,
          episode_count: 2,
        }),
      );
      await Promise.resolve();
    });
    await waitFor(() => expect(oldSummarySignal).toBeDefined());
    await act(async () => {
      oldSummary.resolve(json(summary));
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    expect(oldOpenSignal?.aborted).toBe(true);
    expect(oldSummarySignal?.aborted).toBe(true);
    expect(observed.current?.selectedEpisodeIndex).toBe(1);
    expect(observed.current?.summary?.datasetAlias).toBe(nextAlias);
    expect(observed.current?.episode?.datasetAlias).toBe(nextAlias);
  });

  test("resets workspace state and fences old episode, audit, and batch results", async () => {
    const nextAlias = "local/pnp_trash_b";
    const oldEpisode = deferred<Response>();
    const oldAudit = deferred<Response>();
    const oldBatch = deferred<Response>();
    let oldEpisodeSignal: AbortSignal | undefined;
    let oldAuditSignal: AbortSignal | undefined;
    let oldBatchSignal: AbortSignal | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const url = new URL(request.url);
        if (url.pathname.endsWith("/workspaces/open")) {
          const body = await request.json();
          const alias =
            body !== null &&
            typeof body === "object" &&
            "dataset_alias" in body &&
            typeof body.dataset_alias === "string"
              ? body.dataset_alias
              : DATASET_ALIAS;
          return json({
            dataset_alias: alias,
            source_fingerprint: SOURCE_HASH,
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: PROMPT_HASH,
            episode_count: 2,
          });
        }
        if (url.pathname.endsWith("/summary")) {
          const alias = url.searchParams.get("dataset_alias") ?? DATASET_ALIAS;
          return json({ ...summary, dataset_alias: alias });
        }
        if (request.method === "GET" && url.pathname.endsWith("/episodes/0")) {
          const alias = url.searchParams.get("dataset_alias") ?? DATASET_ALIAS;
          return json(wireEpisode(0, { datasetAlias: alias }));
        }
        if (request.method === "GET" && url.pathname.endsWith("/episodes/1")) {
          oldEpisodeSignal = request.signal;
          return oldEpisode.promise;
        }
        if (url.pathname.endsWith("/audit")) {
          oldAuditSignal = request.signal;
          return oldAudit.promise;
        }
        if (request.method === "POST" && url.pathname.endsWith("/batches")) {
          oldBatchSignal = request.signal;
          return oldBatch.promise;
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const observed: { current: ReturnType<typeof useCuration> | null } = {
      current: null,
    };
    function Probe() {
      observed.current = useCuration();
      return null;
    }
    const ui = (datasetAlias: string, initialEpisodeIndex: number) => (
      <CurationProvider
        datasetAlias={datasetAlias}
        actor="curator"
        reviewer="reviewer"
        episodeIndices={[0, 1]}
        initialEpisodeIndex={initialEpisodeIndex}
      >
        <Probe />
      </CurationProvider>
    );

    const view = render(ui(DATASET_ALIAS, 0));
    await waitFor(() =>
      expect(observed.current?.episode?.sourceEpisodeIndex).toBe(0),
    );
    act(() => {
      void observed.current?.refreshAudit();
      void observed.current?.startBatch();
      observed.current?.selectEpisode(1);
    });
    await waitFor(() => {
      expect(oldEpisodeSignal).toBeDefined();
      expect(oldAuditSignal).toBeDefined();
      expect(oldBatchSignal).toBeDefined();
    });

    act(() => view.rerender(ui(nextAlias, 0)));
    expect(observed.current?.selectedEpisodeIndex).toBe(0);
    expect(observed.current?.summary).toBeNull();
    expect(observed.current?.episode).toBeNull();
    expect(observed.current?.audit).toBeNull();
    expect(observed.current?.batch).toBeNull();
    await waitFor(() =>
      expect(observed.current?.episode?.datasetAlias).toBe(nextAlias),
    );

    await act(async () => {
      oldEpisode.resolve(json(wireEpisode(1)));
      oldAudit.resolve(json(wireAudit(true)));
      oldBatch.resolve(json(wireBatch("old-job", "running")));
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    expect(oldEpisodeSignal?.aborted).toBe(true);
    expect(oldAuditSignal?.aborted).toBe(true);
    expect(oldBatchSignal?.aborted).toBe(true);
    expect(observed.current?.selectedEpisodeIndex).toBe(0);
    expect(observed.current?.summary?.datasetAlias).toBe(nextAlias);
    expect(observed.current?.episode?.datasetAlias).toBe(nextAlias);
    expect(observed.current?.audit).toBeNull();
    expect(observed.current?.batch).toBeNull();
    expect(observed.current?.conflict).toBeNull();
  });

  test("syncs a same-dataset route episode without resetting workspace state or polling", async () => {
    const targetEpisode = deferred<Response>();
    let targetSignal: AbortSignal | undefined;
    let workspaceOpenCount = 0;
    let summaryCount = 0;
    let pollCount = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const url = new URL(request.url);
        if (url.pathname.endsWith("/workspaces/open")) {
          workspaceOpenCount += 1;
          return json({
            dataset_alias: DATASET_ALIAS,
            source_fingerprint: SOURCE_HASH,
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: PROMPT_HASH,
            episode_count: 2,
          });
        }
        if (url.pathname.endsWith("/summary")) {
          summaryCount += 1;
          return json(summary);
        }
        if (request.method === "GET" && url.pathname.endsWith("/episodes/0"))
          return json(wireEpisode(0));
        if (request.method === "GET" && url.pathname.endsWith("/episodes/1")) {
          targetSignal = request.signal;
          return targetEpisode.promise;
        }
        if (request.method === "GET" && url.pathname.endsWith("/audit"))
          return json(wireAudit(true));
        if (request.method === "POST" && url.pathname.endsWith("/batches"))
          return json(wireBatch("job-route", "running"), 201);
        if (
          request.method === "GET" &&
          url.pathname.endsWith("/batches/job-route")
        ) {
          pollCount += 1;
          return json(wireBatch("job-route", "running"));
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const observed: { current: ReturnType<typeof useCuration> | null } = {
      current: null,
    };
    function Probe() {
      observed.current = useCuration();
      return null;
    }
    const ui = (initialEpisodeIndex: number) => (
      <CurationProvider
        datasetAlias={DATASET_ALIAS}
        actor="curator"
        reviewer="reviewer"
        episodeIndices={[0, 1]}
        initialEpisodeIndex={initialEpisodeIndex}
        pollIntervalMs={5}
      >
        <Probe />
      </CurationProvider>
    );

    const view = render(ui(0));
    await waitFor(() =>
      expect(observed.current?.episode?.sourceEpisodeIndex).toBe(0),
    );
    await act(async () => {
      await Promise.all([
        observed.current?.refreshAudit(),
        observed.current?.startBatch(),
      ]);
    });
    await waitFor(() => {
      expect(observed.current?.audit).not.toBeNull();
      expect(observed.current?.batch?.jobId).toBe("job-route");
      expect(pollCount).toBeGreaterThan(0);
    });
    const preservedSummary = observed.current?.summary;
    const preservedAudit = observed.current?.audit;

    act(() => view.rerender(ui(1)));
    await waitFor(() => expect(targetSignal).toBeDefined());
    expect(observed.current?.selectedEpisodeIndex).toBe(1);
    expect(observed.current?.episode).toBeNull();
    expect(observed.current?.summary).toBe(preservedSummary);
    expect(observed.current?.audit).toBe(preservedAudit);
    expect(observed.current?.batch?.jobId).toBe("job-route");
    const pollsAfterRouteChange = pollCount;
    await waitFor(() =>
      expect(pollCount).toBeGreaterThan(pollsAfterRouteChange),
    );

    await act(async () => {
      targetEpisode.resolve(json(wireEpisode(1)));
      await Promise.resolve();
    });
    await waitFor(() =>
      expect(observed.current?.episode?.sourceEpisodeIndex).toBe(1),
    );

    expect(targetSignal?.aborted).toBe(false);
    expect(workspaceOpenCount).toBe(1);
    expect(summaryCount).toBe(1);
    expect(observed.current?.summary).toBe(preservedSummary);
    expect(observed.current?.audit).toBe(preservedAudit);
    expect(observed.current?.batch?.jobId).toBe("job-route");
  });

  test("loads review state and saves a locally edited draft through optimistic revisioning", async () => {
    const initial = wireEpisode(0);
    const saved = wireEpisode(0, {
      state: "draft",
      revision: 1,
      objectName: "crumpled can",
      transitions: [1, 2, 3, 4, 5, 6],
      promptPreview: ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    });
    let savedBody: unknown;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (
          request.method === "GET" &&
          new URL(request.url).pathname.endsWith("/episodes/0")
        ) {
          return json(initial);
        }
        if (request.method === "PATCH") {
          savedBody = await request.json();
          return json(saved);
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(0),
    );

    act(() => {
      view.result.current.updateDraft({
        objectName: "crumpled can",
        pickupHand: "left",
        turnDirection: "right",
        transitionFrames: [1, 2, 3, 4, 5, 6],
      });
    });
    expect(view.result.current.episode?.promptPreview).toBeNull();
    await act(async () => view.result.current.saveDraft());

    expect(savedBody).toEqual({
      dataset_alias: DATASET_ALIAS,
      expected_revision: 0,
      actor: "curator",
      object_name: "crumpled can",
      pickup_hand: "left",
      turn_direction: "right",
      transition_frames: [1, 2, 3, 4, 5, 6],
    });
    expect(view.result.current.episode?.revision).toBe(1);
    expect(view.result.current.episode?.promptPreview).toEqual(
      saved.prompt_preview,
    );
  });

  test("reloads the current episode and prompts reconciliation after a 409", async () => {
    const initial = wireEpisode(0, {
      state: "draft",
      revision: 1,
      objectName: "can",
    });
    const remote = wireEpisode(0, {
      state: "draft",
      revision: 2,
      objectName: "bottle",
    });
    let episodeGets = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (
          request.method === "GET" &&
          new URL(request.url).pathname.endsWith("/episodes/0")
        ) {
          episodeGets += 1;
          return json(episodeGets === 1 ? initial : remote);
        }
        if (request.method === "PATCH") {
          return json(
            {
              error: "revision_conflict",
              episode: {
                ...remote,
                source_length: 300,
                timestamps: Array.from(
                  { length: 300 },
                  (_, index) => index / 30,
                ),
              },
            },
            409,
          );
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode?.revision).toBe(1));
    act(() => view.result.current.updateDraft({ objectName: "box" }));
    await act(async () => view.result.current.saveDraft());

    await waitFor(() => expect(view.result.current.episode?.revision).toBe(2));
    expect(episodeGets).toBe(2);
    expect(view.result.current.conflict?.kind).toBe("revision_conflict");
    expect(view.result.current.conflict?.message).toContain("reconcile");
  });

  test("does not mislabel non-revision 409 responses as optimistic conflicts", async () => {
    const draft = wireEpisode(0, {
      state: "draft",
      revision: 1,
      objectName: "can",
    });
    let episodeGets = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET") {
          episodeGets += 1;
          return json(draft);
        }
        if (new URL(request.url).pathname.endsWith("/reopen")) {
          return json(
            { error: "review_state_conflict", current_state: "draft" },
            409,
          );
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode?.revision).toBe(1));
    await act(async () => view.result.current.reopen());

    expect(view.result.current.conflict).toBeNull();
    expect(view.result.current.error).toBe("review_state_conflict");
    expect(episodeGets).toBe(1);
  });

  test("applies a nullable Cosmos proposal as an editable draft", async () => {
    const initial = wireEpisode(0, {
      proposalTransitions: [1, 2, null, 4, null, 6],
    });
    const applied = wireEpisode(0, {
      state: "draft",
      revision: 1,
      transitions: [1, 2, null, 4, null, 6],
      proposalTransitions: [1, 2, null, 4, null, 6],
    });
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET") return json(initial);
        if (new URL(request.url).pathname.endsWith("/apply-proposal"))
          return json(applied);
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() =>
      expect(view.result.current.episode?.activeProposal?.id).toBe(
        "proposal-1",
      ),
    );
    await act(async () => view.result.current.applyProposal());

    expect(view.result.current.error).toBeNull();
    expect(view.result.current.episode?.decision.reviewState).toBe("draft");
    expect(view.result.current.episode?.decision.transitionFrames).toEqual([
      1,
      2,
      null,
      4,
      null,
      6,
    ]);
    expect(view.result.current.episode?.approvalLocked).toBe(false);
  });

  test("loads an incomplete proposal with zero and duplicate snapped transitions", async () => {
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (new URL(request.url).pathname.endsWith("/episodes/0"))
          return json(
            wireEpisode(0, {
              proposalTransitions: [0, 0, null, 4, null, 4],
            }),
          );
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() =>
      expect(view.result.current.episode?.activeProposal?.id).toBe(
        "proposal-1",
      ),
    );

    expect(
      view.result.current.episode?.activeProposal?.transitionFrames,
    ).toEqual([0, 0, null, 4, null, 4]);
    expect(view.result.current.error).toBeNull();
  });

  test("locks approved records until reopen succeeds", async () => {
    const approved = wireEpisode(0, {
      state: "approved_keep",
      revision: 2,
      objectName: "can",
      transitions: [1, 2, 3, 4, 5, 6],
    });
    const reopened = wireEpisode(0, {
      state: "draft",
      revision: 3,
      objectName: "can",
      transitions: [1, 2, 3, 4, 5, 6],
    });
    let mutationCount = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET") return json(approved);
        mutationCount += 1;
        if (new URL(request.url).pathname.endsWith("/reopen"))
          return json(reopened);
        throw new Error("approved records must not be mutated");
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() =>
      expect(view.result.current.episode?.approvalLocked).toBe(true),
    );
    act(() =>
      view.result.current.updateDraft({ objectName: "silently ignored" }),
    );
    await act(async () => view.result.current.saveDraft());
    expect(view.result.current.episode?.decision.objectName).toBe("can");
    expect(view.result.current.error).toContain("Reopen");
    expect(mutationCount).toBe(0);

    await act(async () => view.result.current.reopen());
    expect(view.result.current.episode?.approvalLocked).toBe(false);
    expect(view.result.current.episode?.revision).toBe(3);
    expect(mutationCount).toBe(1);
  });

  test("polls an active batch and stops once it reaches a terminal state", async () => {
    const states = ["queued", "running", "completed"] as const;
    let pollCount = 0;
    const batch = (state: (typeof states)[number]) => ({
      job_id: "job-1",
      parent_job_id: null,
      state,
      configuration: batchConfiguration(),
      counts: state === "completed" ? { succeeded: 2 } : { queued: 2 },
      episodes: [],
      lease: null,
      current_episode: null,
      cancel_requested: false,
      created_at: "2026-08-27T00:00:00Z",
      updated_at: "2026-08-27T00:00:00Z",
      errors: [],
      active_proposal_coverage: state === "completed" ? 2 : 0,
    });
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET" && path.endsWith("/episodes/0"))
          return json(wireEpisode(0));
        if (request.method === "POST" && path.endsWith("/batches"))
          return json(batch(states[0]), 201);
        if (request.method === "GET" && path.endsWith("/batches/job-1")) {
          pollCount += 1;
          return json(batch(states[Math.min(pollCount, 2)]));
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), {
      wrapper: wrapper({ pollIntervalMs: 5 }),
    });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    await act(async () => view.result.current.startBatch());
    await waitFor(() =>
      expect(view.result.current.batch?.state).toBe("completed"),
    );
    const terminalPollCount = pollCount;
    await new Promise((resolve) => setTimeout(resolve, 30));

    expect(terminalPollCount).toBe(2);
    expect(pollCount).toBe(terminalPollCount);
  });

  test("loads typed grip/audit state and supports batch cancel and retry controls", async () => {
    const batch = (jobId: string, state: "queued" | "cancelled") => ({
      job_id: jobId,
      parent_job_id: jobId === "job-2" ? "job-1" : null,
      state,
      configuration: {
        ...batchConfiguration(),
        ...(jobId === "job-2" ? { parent_job_id: "job-1" } : {}),
      },
      counts: state === "queued" ? { queued: 1 } : { cancelled: 1 },
      episodes: [],
      lease: null,
      current_episode: null,
      cancel_requested: state === "cancelled",
      created_at: "2026-08-27T00:00:00Z",
      updated_at: "2026-08-27T00:00:00Z",
      errors: [],
      active_proposal_coverage: 0,
    });
    const audit = {
      dataset_alias: DATASET_ALIAS,
      review_state_counts: summary.counts,
      cosmos: {
        job_state_counts: { cancelled: 1 },
        attempt_state_counts: { cancelled: 1 },
        proposal_state_counts: {},
        proposal_result_counts: {},
        attempt_error_class_counts: {},
      },
      transition_time_distributions: {},
      phase_duration_distributions: {},
      boundary_errors: [],
      grip_disagreements: [],
      grip_unavailable: [],
      unreadable_files: [],
      contact_sheet_issues: [
        {
          kind: "final",
          source_episode_index: 0,
          reason: "incomplete",
          approval_revision: 1,
        },
      ],
      source_fingerprint: {
        expected_sha256: SOURCE_HASH,
        current_matches: true,
      },
    };
    const grip = {
      dataset_alias: DATASET_ALIAS,
      source_episode_index: 0,
      status: "available",
      side: "left",
      reason: null,
      usable_joint_indices: [0, 1, 2],
      grasp: { frame: 2, timestamp_s: 0.2, derivative: -0.4 },
      release: { frame: 5, timestamp_s: 0.5, derivative: 0.5 },
      advisories: [],
      grasp_delta_s: 0,
      release_delta_s: 0.1,
    };
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET" && path.endsWith("/episodes/0"))
          return json(wireEpisode(0));
        if (request.method === "GET" && path.endsWith("/episodes/0/grip"))
          return json(grip);
        if (request.method === "GET" && path.endsWith("/audit"))
          return json(audit);
        if (request.method === "POST" && path.endsWith("/batches"))
          return json(batch("job-1", "queued"), 201);
        if (
          request.method === "POST" &&
          path.endsWith("/batches/job-1/cancel")
        ) {
          expect(await request.text()).toBe("");
          return json({ job_id: "job-1", state: "cancelled", changed: true });
        }
        if (request.method === "GET" && path.endsWith("/batches/job-1")) {
          return json(batch("job-1", "cancelled"));
        }
        if (
          request.method === "POST" &&
          path.endsWith("/batches/job-1/retry")
        ) {
          expect(await request.json()).toEqual({
            failure_states: ["manual_only"],
          });
          return json(batch("job-2", "queued"), 201);
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), {
      wrapper: wrapper({ pollIntervalMs: 10_000 }),
    });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    await act(async () => {
      await Promise.all([
        view.result.current.refreshGrip(),
        view.result.current.refreshAudit(),
      ]);
    });
    expect(view.result.current.grip?.status).toBe("available");
    expect(view.result.current.grip?.grasp?.frame).toBe(2);
    expect(view.result.current.audit?.sourceFingerprint.currentMatches).toBe(
      true,
    );
    expect(view.result.current.audit?.contactSheetIssues[0]?.reason).toBe(
      "incomplete",
    );

    await act(async () => view.result.current.startBatch([0]));
    await act(async () => view.result.current.cancelBatch());
    expect(view.result.current.batch?.state).toBe("cancelled");
    await act(async () =>
      view.result.current.retryBatch({ failureStates: ["manual_only"] }),
    );
    expect(view.result.current.batch?.jobId).toBe("job-2");
    expect(view.result.current.batch?.parentJobId).toBe("job-1");
  });

  test("aborts a stale episode request when navigation selects another episode", async () => {
    let staleAborted = false;
    let staleStarted = false;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (path.endsWith("/episodes/1")) return json(wireEpisode(1));
        if (path.endsWith("/episodes/0")) {
          staleStarted = true;
          return new Promise<Response>((_resolve, reject) => {
            request.signal.addEventListener("abort", () => {
              staleAborted = true;
              reject(new DOMException("aborted", "AbortError"));
            });
          });
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(staleStarted).toBe(true));
    act(() => view.result.current.selectEpisode(1));
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );

    expect(staleAborted).toBe(true);
    expect(view.result.current.selectedEpisodeIndex).toBe(1);
  });

  test("selecting the current episode is a no-op that preserves the loaded record", async () => {
    let episodeGets = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (new URL(request.url).pathname.endsWith("/episodes/0")) {
          episodeGets += 1;
          return json(wireEpisode(0));
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    const loaded = view.result.current.episode;
    act(() => view.result.current.selectEpisode(0));
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(episodeGets).toBe(1);
    expect(view.result.current.episode).toBe(loaded);
    expect(view.result.current.loading).toBe(false);
  });

  test("approve-and-next locks the saved episode then selects the next configured episode", async () => {
    const draft = wireEpisode(0, {
      state: "draft",
      revision: 1,
      objectName: "can",
      transitions: [1, 2, 3, 4, 5, 6],
    });
    const approved = wireEpisode(0, {
      state: "approved_keep",
      revision: 2,
      objectName: "can",
      transitions: [1, 2, 3, 4, 5, 6],
    });
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET" && path.endsWith("/episodes/0"))
          return json(draft);
        if (request.method === "GET" && path.endsWith("/episodes/1"))
          return json(wireEpisode(1));
        if (request.method === "POST" && path.endsWith("/approve-keep"))
          return json(approved);
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() =>
      expect(view.result.current.episode?.decision.reviewState).toBe("draft"),
    );
    await act(async () => view.result.current.approveKeepAndNext());
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );

    expect(view.result.current.selectedEpisodeIndex).toBe(1);
  });

  test("accepts an empty backend-valid rejection reason and advances", async () => {
    const draft = wireEpisode(0, { state: "draft", revision: 1 });
    const rejected = wireEpisode(0, {
      state: "approved_reject",
      revision: 2,
      rejectionReason: "",
    });
    let submittedReason: unknown;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET" && path.endsWith("/episodes/0"))
          return json(draft);
        if (request.method === "GET" && path.endsWith("/episodes/1"))
          return json(wireEpisode(1));
        if (path.endsWith("/approve-reject")) {
          const body = await request.json();
          if (body !== null && typeof body === "object" && "reason" in body)
            submittedReason = body.reason;
          return json(rejected);
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode?.revision).toBe(1));
    await act(async () => view.result.current.approveRejectAndNext(""));
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );

    expect(submittedReason).toBe("");
    expect(view.result.current.error).toBeNull();
  });

  test("ignores and aborts a save result that completes after episode navigation", async () => {
    const pendingSave = deferred<Response>();
    let mutationSignal: AbortSignal | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET" && path.endsWith("/episodes/0"))
          return json(wireEpisode(0));
        if (request.method === "GET" && path.endsWith("/episodes/1"))
          return json(wireEpisode(1));
        if (request.method === "PATCH") {
          mutationSignal = request.signal;
          return pendingSave.promise;
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    let saving!: Promise<void>;
    act(() => {
      saving = view.result.current.saveDraft();
    });
    await waitFor(() => expect(view.result.current.saving).toBe(true));
    act(() => view.result.current.selectEpisode(1));
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );
    pendingSave.resolve(json(wireEpisode(0, { state: "draft", revision: 1 })));
    await act(async () => saving);

    expect(mutationSignal?.aborted).toBe(true);
    expect(view.result.current.selectedEpisodeIndex).toBe(1);
    expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1);
  });

  test("fences stale approval and conflict reload results by operation token and episode identity", async () => {
    const pendingApproval = deferred<Response>();
    const pendingConflictReload = deferred<Response>();
    let getZero = 0;
    let phase: "approval" | "conflict" = "approval";
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (request.method === "GET" && path.endsWith("/episodes/1"))
          return json(wireEpisode(1));
        if (request.method === "GET" && path.endsWith("/episodes/0")) {
          getZero += 1;
          if (getZero <= 2) {
            return json(
              wireEpisode(0, {
                state: "draft",
                revision: 1,
                objectName: "can",
                transitions: [1, 2, 3, 4, 5, 6],
              }),
            );
          }
          return pendingConflictReload.promise;
        }
        if (path.endsWith("/approve-keep")) return pendingApproval.promise;
        if (request.method === "PATCH" && phase === "conflict")
          return json({ error: "revision_conflict" }, 409);
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode?.revision).toBe(1));
    let approval!: Promise<void>;
    act(() => {
      approval = view.result.current.approveKeepAndNext();
    });
    await waitFor(() => expect(view.result.current.saving).toBe(true));
    act(() => view.result.current.selectEpisode(1));
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );
    pendingApproval.resolve(
      json(
        wireEpisode(0, {
          state: "approved_keep",
          revision: 2,
          objectName: "can",
          transitions: [1, 2, 3, 4, 5, 6],
        }),
      ),
    );
    await act(async () => approval);
    expect(view.result.current.selectedEpisodeIndex).toBe(1);

    act(() => view.result.current.selectEpisode(0));
    await waitFor(() => expect(view.result.current.episode?.revision).toBe(1));
    phase = "conflict";
    let saving!: Promise<void>;
    act(() => {
      saving = view.result.current.saveDraft();
    });
    await waitFor(() => expect(getZero).toBe(3));
    act(() => view.result.current.selectEpisode(1));
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );
    pendingConflictReload.resolve(
      json(wireEpisode(0, { state: "draft", revision: 2, objectName: "box" })),
    );
    await act(async () => saving);
    expect(view.result.current.selectedEpisodeIndex).toBe(1);
    expect(view.result.current.conflict).toBeNull();
  });

  test("clears grip on decision changes and ignores a stale grip response", async () => {
    const pendingGrip = deferred<Response>();
    let gripSignal: AbortSignal | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (path.endsWith("/episodes/0/grip")) {
          gripSignal = request.signal;
          return pendingGrip.promise;
        }
        if (path.endsWith("/episodes/0")) return json(wireEpisode(0));
        if (path.endsWith("/episodes/1")) return json(wireEpisode(1));
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    let refreshing!: Promise<void>;
    act(() => {
      refreshing = view.result.current.refreshGrip();
    });
    act(() => view.result.current.updateDraft({ pickupHand: "right" }));
    expect(view.result.current.grip).toBeNull();
    act(() => view.result.current.selectEpisode(1));
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );
    pendingGrip.resolve(
      json({
        dataset_alias: DATASET_ALIAS,
        source_episode_index: 0,
        status: "unavailable",
        side: "right",
        reason: "not_found",
        usable_joint_indices: [],
        grasp: null,
        release: null,
        advisories: [],
        grasp_delta_s: null,
        release_delta_s: null,
      }),
    );
    await act(async () => refreshing);
    expect(gripSignal?.aborted).toBe(true);
    expect(view.result.current.grip).toBeNull();
  });

  test("continues polling after a transient failure and ignores an old poll after batch replacement", async () => {
    const stalePoll = deferred<Response>();
    let jobOnePolls = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (path.endsWith("/episodes/0")) return json(wireEpisode(0));
        if (request.method === "POST" && path.endsWith("/batches"))
          return json(wireBatch("job-1"), 201);
        if (request.method === "POST" && path.endsWith("/job-1/retry"))
          return json(wireBatch("job-2", "queued", "job-1"), 201);
        if (request.method === "GET" && path.endsWith("/batches/job-1")) {
          jobOnePolls += 1;
          if (jobOnePolls === 1) throw new TypeError("temporary disconnect");
          return stalePoll.promise;
        }
        if (request.method === "GET" && path.endsWith("/batches/job-2"))
          return json(wireBatch("job-2", "completed", "job-1"));
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), {
      wrapper: wrapper({ pollIntervalMs: 5 }),
    });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    await act(async () => view.result.current.startBatch());
    await waitFor(() => expect(jobOnePolls).toBe(2));
    await act(async () =>
      view.result.current.retryBatch({ episodeIndices: [0] }),
    );
    await waitFor(() => expect(view.result.current.batch?.jobId).toBe("job-2"));
    stalePoll.resolve(json(wireBatch("job-1", "completed")));
    await waitFor(() =>
      expect(view.result.current.batch?.state).toBe("completed"),
    );
    expect(view.result.current.batch?.jobId).toBe("job-2");
  });

  test("does not let a delayed same-job poll overwrite a terminal cancellation", async () => {
    const delayedPoll = deferred<Response>();
    const delayedCancel = deferred<Response>();
    let jobStatusCalls = 0;
    let pollSignal: AbortSignal | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (path.endsWith("/episodes/0")) return json(wireEpisode(0));
        if (request.method === "POST" && path.endsWith("/batches"))
          return json(wireBatch("job-1", "running"), 201);
        if (path.endsWith("/job-1/cancel")) return delayedCancel.promise;
        if (request.method === "GET" && path.endsWith("/batches/job-1")) {
          jobStatusCalls += 1;
          if (jobStatusCalls === 1) {
            pollSignal = request.signal;
            return delayedPoll.promise;
          }
          return json(wireBatch("job-1", "cancelled"));
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), {
      wrapper: wrapper({ pollIntervalMs: 5 }),
    });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    await act(async () => view.result.current.startBatch());
    await waitFor(() => expect(jobStatusCalls).toBe(1));
    let cancellation!: Promise<void>;
    act(() => {
      cancellation = view.result.current.cancelBatch();
    });
    expect(pollSignal?.aborted).toBe(true);
    delayedCancel.resolve(
      json({ job_id: "job-1", state: "cancelled", changed: true }),
    );
    await act(async () => cancellation);
    expect(view.result.current.batch?.state).toBe("cancelled");

    await act(async () => {
      delayedPoll.resolve(json(wireBatch("job-1", "running")));
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    expect(pollSignal?.aborted).toBe(true);
    expect(view.result.current.batch?.state).toBe("cancelled");
  });

  test("aborts and ignores an old batch retry when cancellation replaces it", async () => {
    const pendingRetry = deferred<Response>();
    let retrySignal: AbortSignal | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (path.endsWith("/episodes/0")) return json(wireEpisode(0));
        if (request.method === "POST" && path.endsWith("/batches"))
          return json(wireBatch("job-1"), 201);
        if (path.endsWith("/job-1/retry")) {
          retrySignal = request.signal;
          return pendingRetry.promise;
        }
        if (path.endsWith("/job-1/cancel"))
          return json({ job_id: "job-1", state: "cancelled", changed: true });
        if (request.method === "GET" && path.endsWith("/batches/job-1"))
          return json(wireBatch("job-1", "cancelled"));
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), {
      wrapper: wrapper({ pollIntervalMs: 10_000 }),
    });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    await act(async () => view.result.current.startBatch());
    let retrying!: Promise<void>;
    act(() => {
      retrying = view.result.current.retryBatch({ episodeIndices: [0] });
    });
    await act(async () => view.result.current.cancelBatch());
    pendingRetry.resolve(json(wireBatch("job-2", "queued", "job-1")));
    await act(async () => retrying);

    expect(retrySignal?.aborted).toBe(true);
    expect(view.result.current.batch?.jobId).toBe("job-1");
    expect(view.result.current.batch?.state).toBe("cancelled");
  });

  test("aborts and ignores an old cancellation when retry replaces it", async () => {
    const pendingCancel = deferred<Response>();
    let cancelSignal: AbortSignal | undefined;
    let oldStatusFetches = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (path.endsWith("/episodes/0")) return json(wireEpisode(0));
        if (request.method === "POST" && path.endsWith("/batches"))
          return json(wireBatch("job-1"), 201);
        if (path.endsWith("/job-1/cancel")) {
          cancelSignal = request.signal;
          return pendingCancel.promise;
        }
        if (path.endsWith("/job-1/retry"))
          return json(wireBatch("job-2", "queued", "job-1"), 201);
        if (request.method === "GET" && path.endsWith("/batches/job-1")) {
          oldStatusFetches += 1;
          return json(wireBatch("job-1", "cancelled"));
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), {
      wrapper: wrapper({ pollIntervalMs: 10_000 }),
    });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    await act(async () => view.result.current.startBatch());
    let cancelling!: Promise<void>;
    act(() => {
      cancelling = view.result.current.cancelBatch();
    });
    await act(async () =>
      view.result.current.retryBatch({ failureStates: ["cancelled"] }),
    );
    pendingCancel.resolve(
      json({ job_id: "job-1", state: "cancelled", changed: true }),
    );
    await act(async () => cancelling);

    expect(cancelSignal?.aborted).toBe(true);
    expect(oldStatusFetches).toBe(0);
    expect(view.result.current.batch?.jobId).toBe("job-2");
  });

  test("aborts and ignores an old audit refresh", async () => {
    const staleAudit = deferred<Response>();
    let auditCalls = 0;
    let staleSignal: AbortSignal | undefined;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        const bootstrap = bootstrapResponse(request);
        if (bootstrap) return bootstrap;
        if (path.endsWith("/episodes/0")) return json(wireEpisode(0));
        if (path.endsWith("/audit")) {
          auditCalls += 1;
          if (auditCalls === 1) {
            staleSignal = request.signal;
            return staleAudit.promise;
          }
          return json(wireAudit(false));
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode).not.toBeNull());
    let firstRefresh!: Promise<void>;
    act(() => {
      firstRefresh = view.result.current.refreshAudit();
    });
    await act(async () => view.result.current.refreshAudit());
    staleAudit.resolve(json(wireAudit(true)));
    await act(async () => firstRefresh);

    expect(staleSignal?.aborted).toBe(true);
    expect(view.result.current.audit?.sourceFingerprint.currentMatches).toBe(
      false,
    );
  });

  test("keeps successful approval and advances when only summary refresh fails", async () => {
    let summaryCalls = 0;
    globalThis.fetch = mock(
      async (input: string | URL | Request, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        if (path.endsWith("/workspaces/open"))
          return bootstrapResponse(request) ?? json({});
        if (path.endsWith("/summary")) {
          summaryCalls += 1;
          if (summaryCalls > 1) throw new TypeError("summary unavailable");
          return json(summary);
        }
        if (path.endsWith("/episodes/0") && request.method === "GET")
          return json(
            wireEpisode(0, {
              state: "draft",
              revision: 1,
              objectName: "can",
              transitions: [1, 2, 3, 4, 5, 6],
            }),
          );
        if (path.endsWith("/approve-keep"))
          return json(
            wireEpisode(0, {
              state: "approved_keep",
              revision: 2,
              objectName: "can",
              transitions: [1, 2, 3, 4, 5, 6],
            }),
          );
        if (path.endsWith("/episodes/1")) return json(wireEpisode(1));
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      },
    ) as typeof fetch;

    const view = renderHook(() => useCuration(), { wrapper: wrapper() });
    await waitFor(() => expect(view.result.current.episode?.revision).toBe(1));
    await act(async () => view.result.current.approveKeepAndNext());
    await waitFor(() =>
      expect(view.result.current.episode?.sourceEpisodeIndex).toBe(1),
    );
    expect(view.result.current.warning).toContain("summary refresh failed");
    expect(view.result.current.error).toBeNull();
  });
});
