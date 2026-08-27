import { afterEach, describe, expect, mock as bunMock, test } from "bun:test";

import {
  CurationClientError,
  approveEpisodeKeep,
  cancelCurationBatch,
  fetchCurationAudit,
  fetchCurationBatch,
  fetchCurationSummary,
  fetchEpisodeCuration,
  fetchGripDiagnostic,
  openCurationWorkspace,
  retryCurationBatch,
  saveEpisodeDraft,
  startCurationBatch,
} from "../curationClient";

const originalFetch = globalThis.fetch;
const SHA = "a".repeat(64);
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

function realBatchWithAttemptZero() {
  return {
    job_id: "job-1",
    parent_job_id: null,
    state: "running",
    configuration: {
      schema_version: 1,
      dataset_alias: "local/pnp_trash",
      dataset_id: 1,
      source_path: "/datasets/pnp_trash",
      source_manifest_sha256: SHA,
      source_fps: 30,
      episode_indices: [0],
      prompt: { version: "pnp-trash-v1", sha256: SHA },
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
        maximum_payload_bytes: 1_000_000,
      },
    },
    counts: { requesting: 1 },
    episodes: [
      {
        attempt_id: "attempt-1",
        attempt_number: 0,
        source_episode_index: 0,
        state: "requesting",
      },
    ],
    lease: { owner: "worker-1", expires_at: "2026-08-27T00:03:00Z" },
    current_episode: 0,
    cancel_requested: false,
    created_at: "2026-08-27T00:00:00Z",
    updated_at: "2026-08-27T00:00:01Z",
    errors: [],
    active_proposal_coverage: 0,
  };
}

function pendingEpisode(datasetAlias: string, sourceEpisodeIndex: number) {
  return {
    dataset_alias: datasetAlias,
    source_episode_index: sourceEpisodeIndex,
    source_length: 8,
    timestamps: Array.from({ length: 8 }, (_, index) => index / 30),
    decision: {
      review_state: "pending",
      object_name: null,
      pickup_hand: null,
      turn_direction: null,
      transition_frames: [null, null, null, null, null, null],
      rejection_reason: null,
      prompt_template_sha256: SHA,
    },
    active_proposal: null,
    prompt_preview: null,
    warnings: [],
    revision: 0,
    approval_locked: false,
    reviewer: null,
    approval_revision: null,
    approved_at: null,
  };
}

function staleApprovedEpisode(
  datasetAlias: string,
  sourceEpisodeIndex: number,
) {
  return {
    ...pendingEpisode(datasetAlias, sourceEpisodeIndex),
    decision: {
      review_state: "approved_keep",
      object_name: "can",
      pickup_hand: "left",
      turn_direction: "right",
      transition_frames: [1, 2, 3, 4, 5, 6],
      rejection_reason: null,
      prompt_template_sha256: SHA,
    },
    prompt_preview: ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    warnings: ["approval_revision_mismatch"],
    revision: 3,
    approval_locked: true,
    reviewer: "reviewer",
    approval_revision: 2,
    approved_at: "2026-08-27T00:00:00Z",
  };
}

function emptyAudit(datasetAlias: string) {
  return {
    dataset_alias: datasetAlias,
    review_state_counts: {
      pending: 1,
      draft: 0,
      approved_keep: 0,
      approved_reject: 0,
    },
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
    source_fingerprint: { expected_sha256: SHA, current_matches: true },
  };
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  mock.restore();
});

describe("curation client runtime contracts", () => {
  test("rejects malformed successful payloads from every response decoder without retaining raw data", async () => {
    const operations: Array<[string, () => Promise<unknown>]> = [
      ["workspace", () => openCurationWorkspace("local/pnp_trash", "curator")],
      ["summary", () => fetchCurationSummary("local/pnp_trash")],
      ["episode", () => fetchEpisodeCuration("local/pnp_trash", 0)],
      [
        "save",
        () =>
          saveEpisodeDraft("local/pnp_trash", 0, 0, "curator", {
            transitionFrames: [1, 2, 3, 4, 5, 6],
          }),
      ],
      [
        "approval",
        () =>
          approveEpisodeKeep("local/pnp_trash", 0, 1, "curator", "reviewer"),
      ],
      ["batch start", () => startCurationBatch("local/pnp_trash")],
      ["batch status", () => fetchCurationBatch("job-1")],
      [
        "batch retry",
        () =>
          retryCurationBatch(
            "job-1",
            { failureStates: ["manual_only"] },
            "local/pnp_trash",
          ),
      ],
      ["batch cancel", () => cancelCurationBatch("job-1")],
      ["grip", () => fetchGripDiagnostic("local/pnp_trash", 0)],
      ["audit", () => fetchCurationAudit("local/pnp_trash")],
    ];

    for (const [name, operation] of operations) {
      const secretRawPayload = {
        malformed: name,
        secret: "must-not-be-retained",
      };
      globalThis.fetch = mock(async () =>
        Response.json(secretRawPayload),
      ) as typeof fetch;
      let caught: unknown;
      try {
        await operation();
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(CurationClientError);
      expect(caught).toMatchObject({
        name: "CurationClientError",
        code: "invalid_response",
      });
      expect(JSON.stringify(caught)).not.toContain("must-not-be-retained");
      expect(caught && typeof caught === "object" && "payload" in caught).toBe(
        false,
      );
    }
  });

  test("decodes bounded JSON error payloads and rejects malformed error bodies", async () => {
    globalThis.fetch = mock(async () =>
      Response.json(
        {
          error: "review_state_conflict",
          current_state: "draft",
          operation: "approval_reopened",
        },
        { status: 409 },
      ),
    ) as typeof fetch;

    await expect(
      fetchEpisodeCuration("local/pnp_trash", 0),
    ).rejects.toMatchObject({
      name: "CurationClientError",
      code: "http_error",
      status: 409,
      errorPayload: {
        error: "review_state_conflict",
        current_state: "draft",
        operation: "approval_reopened",
      },
    });

    globalThis.fetch = mock(async () =>
      Response.json({ error: 7 }, { status: 422 }),
    ) as typeof fetch;
    await expect(
      fetchEpisodeCuration("local/pnp_trash", 0),
    ).rejects.toMatchObject({
      name: "CurationClientError",
      code: "invalid_response",
      status: 422,
    });
  });

  test("retains only a safe active-batch recovery id from a 409", async () => {
    globalThis.fetch = mock(async () =>
      Response.json(
        {
          error: "active_batch_exists",
          job_id: "job-existing",
          secret: "must-not-be-retained",
        },
        { status: 409 },
      ),
    ) as typeof fetch;

    let caught: unknown;
    try {
      await startCurationBatch("local/pnp_trash");
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(CurationClientError);
    expect(caught).toMatchObject({
      code: "http_error",
      status: 409,
      activeBatchJobId: "job-existing",
      errorPayload: {
        error: "active_batch_exists",
        job_id: "job-existing",
      },
    });
    expect(JSON.stringify(caught)).not.toContain("must-not-be-retained");

    globalThis.fetch = mock(async () =>
      Response.json(
        { error: "active_batch_exists", job_id: "../escape" },
        { status: 409 },
      ),
    ) as typeof fetch;
    await expect(startCurationBatch("local/pnp_trash")).rejects.toMatchObject({
      code: "invalid_response",
      status: 409,
      activeBatchJobId: null,
    });
  });

  test("accepts a locked approval whose approval revision is stale", async () => {
    globalThis.fetch = mock(async () =>
      Response.json(staleApprovedEpisode("local/pnp_trash", 0)),
    ) as typeof fetch;

    const decoded = await fetchEpisodeCuration("local/pnp_trash", 0);

    expect(decoded.approvalLocked).toBe(true);
    expect(decoded.revision).toBe(3);
    expect(decoded.approvalRevision).toBe(2);
    expect(decoded.warnings).toContain("approval_revision_mismatch");
  });

  test("rejects unsafe job identifiers across batch and cancellation contracts", async () => {
    globalThis.fetch = mock(async () =>
      Response.json({ ...realBatchWithAttemptZero(), job_id: "../job" }),
    ) as typeof fetch;
    await expect(startCurationBatch("local/pnp_trash")).rejects.toMatchObject({
      code: "invalid_response",
    });

    globalThis.fetch = mock(async () =>
      Response.json({
        ...realBatchWithAttemptZero(),
        job_id: "job-child",
        parent_job_id: "../parent",
        configuration: {
          ...realBatchWithAttemptZero().configuration,
          parent_job_id: "../parent",
        },
      }),
    ) as typeof fetch;
    await expect(fetchCurationBatch("job-child")).rejects.toMatchObject({
      code: "invalid_response",
    });

    globalThis.fetch = mock(async () =>
      Response.json({
        ...realBatchWithAttemptZero(),
        job_id: "../child",
        parent_job_id: "job-parent",
        configuration: {
          ...realBatchWithAttemptZero().configuration,
          parent_job_id: "job-parent",
        },
      }),
    ) as typeof fetch;
    await expect(
      retryCurationBatch(
        "job-parent",
        { episodeIndices: [0] },
        "local/pnp_trash",
      ),
    ).rejects.toMatchObject({ code: "invalid_response" });

    let cancellationFetches = 0;
    globalThis.fetch = mock(async () => {
      cancellationFetches += 1;
      return Response.json({
        job_id: "../escape",
        state: "cancelled",
        changed: true,
      });
    }) as typeof fetch;
    await expect(cancelCurationBatch("../escape")).rejects.toMatchObject({
      code: "invalid_response",
    });
    expect(cancellationFetches).toBe(0);
  });

  test("decodes a real nonempty batch whose first attempt number is zero", async () => {
    globalThis.fetch = mock(async () =>
      Response.json(realBatchWithAttemptZero()),
    ) as typeof fetch;

    const batch = await fetchCurationBatch("job-1");

    expect(batch.episodes).toHaveLength(1);
    expect(batch.episodes[0]?.attemptNumber).toBe(0);
    expect(batch.episodes[0]?.state).toBe("requesting");
  });

  test("bounds a revision-conflict envelope without retaining its embedded episode", async () => {
    globalThis.fetch = mock(async () =>
      Response.json(
        {
          error: "revision_conflict",
          episode: {
            timestamps: Array.from({ length: 300 }, (_, index) => index / 30),
            secret: "embedded-episode-must-not-be-retained",
          },
        },
        { status: 409 },
      ),
    ) as typeof fetch;

    let caught: unknown;
    try {
      await fetchEpisodeCuration("local/pnp_trash", 0);
    } catch (error) {
      caught = error;
    }
    expect(caught).toMatchObject({
      code: "http_error",
      status: 409,
      errorPayload: { error: "revision_conflict" },
    });
    expect(JSON.stringify(caught)).not.toContain(
      "embedded-episode-must-not-be-retained",
    );
  });

  test("rejects decoded payloads whose identity does not match the request", async () => {
    const expectInvalid = async (
      operation: () => Promise<unknown>,
      response: unknown,
    ) => {
      globalThis.fetch = mock(async () =>
        Response.json(response),
      ) as typeof fetch;
      await expect(operation()).rejects.toMatchObject({
        code: "invalid_response",
      });
    };

    await expectInvalid(
      () => openCurationWorkspace("local/pnp_trash", "curator"),
      {
        dataset_alias: "local/other",
        source_fingerprint: SHA,
        prompt_template_version: "pnp-trash-v1",
        prompt_template_sha256: SHA,
        episode_count: 1,
      },
    );
    await expectInvalid(() => fetchCurationSummary("local/pnp_trash"), {
      dataset_alias: "local/other",
      episode_count: 1,
      counts: {
        pending: 1,
        draft: 0,
        approved_keep: 0,
        approved_reject: 0,
      },
      prompt_template_version: "pnp-trash-v1",
      prompt_template_sha256: SHA,
    });
    await expectInvalid(
      () => fetchEpisodeCuration("local/pnp_trash", 0),
      pendingEpisode("local/other", 0),
    );
    await expectInvalid(
      () =>
        saveEpisodeDraft("local/pnp_trash", 0, 0, "curator", {
          transitionFrames: [1, 2, 3, 4, 5, 6],
        }),
      pendingEpisode("local/pnp_trash", 1),
    );
    await expectInvalid(() => fetchGripDiagnostic("local/pnp_trash", 0), {
      dataset_alias: "local/pnp_trash",
      source_episode_index: 1,
      status: "unavailable",
      side: null,
      reason: "pickup_hand_unselected",
      usable_joint_indices: [],
      grasp: null,
      release: null,
      advisories: [],
      grasp_delta_s: null,
      release_delta_s: null,
    });
    await expectInvalid(
      () => fetchCurationAudit("local/pnp_trash"),
      emptyAudit("local/other"),
    );

    const batch = realBatchWithAttemptZero();
    await expectInvalid(() => startCurationBatch("local/pnp_trash"), {
      ...batch,
      configuration: {
        ...batch.configuration,
        dataset_alias: "local/other",
      },
    });
    await expectInvalid(() => fetchCurationBatch("job-1"), {
      ...batch,
      job_id: "job-other",
    });
    await expectInvalid(
      () =>
        retryCurationBatch(
          "job-parent",
          { episodeIndices: [0] },
          "local/pnp_trash",
        ),
      {
        ...batch,
        job_id: "job-child",
        parent_job_id: "job-other-parent",
        configuration: {
          ...batch.configuration,
          parent_job_id: "job-other-parent",
        },
      },
    );
    await expectInvalid(() => cancelCurationBatch("job-1"), {
      job_id: "job-other",
      state: "cancelled",
      changed: true,
    });
  });
});
