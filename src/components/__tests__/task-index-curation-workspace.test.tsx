import React, { useCallback, useState } from "react";
import { describe, expect, mock, test } from "bun:test";
import { getQueriesForElement, render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  isTaskIndexCurationDataset,
  TaskIndexCurationWorkspace,
  TaskIndexCurationWorkspaceView,
  type CurationWorkspaceController,
} from "../task-index-curation-workspace";
import {
  CurationProvider,
  CurationRouteSync,
  useCuration,
} from "../../context/curation-context";
import { TimeProvider } from "../../context/time-context";
import type {
  BatchConfiguration,
  CurationAudit,
  EpisodeCuration,
  EpisodeDraftPatch,
} from "../../types/curation.types";

const HASH = "a".repeat(64);
const SOURCE_HASH = "b".repeat(64);
const originalFetch = globalThis.fetch;
const screen = getQueriesForElement(document.body);
const phases = [
  "approach_brown_table",
  "pick_up_object",
  "turn_to_find_black_trash_bin",
  "approach_black_trash_bin",
  "lean_down_to_black_trash_bin",
  "drop_object_into_black_trash_bin",
  "stand_straight",
] as const;

function episode(
  options: Partial<EpisodeCuration> & {
    state?: EpisodeCuration["decision"]["reviewState"];
    valid?: boolean;
  } = {},
): EpisodeCuration {
  const { state = "draft", valid = true, ...episodeOverrides } = options;
  const locked = state === "approved_keep" || state === "approved_reject";
  return {
    datasetAlias: "local/pnp_trash",
    sourceEpisodeIndex: 4,
    sourceLength: 12,
    timestamps: Array.from({ length: 12 }, (_, frame) => frame / 10),
    decision: {
      reviewState: state,
      objectName: valid ? "plastic bottle" : null,
      pickupHand: valid ? "left" : null,
      turnDirection: valid ? "right" : null,
      transitionFrames: valid
        ? [1, 3, 5, 7, 9, 10]
        : [null, null, null, null, null, null],
      rejectionReason: state === "approved_reject" ? "wrong task" : null,
      promptTemplateSha256: HASH,
    },
    activeProposal: {
      id: "proposal-1",
      attemptId: "attempt-1",
      transitionFrames: [2, null, 5, 7, 8, 10],
      warnings: ["step_3_not_observed"],
      createdAt: "2026-08-27T00:00:00Z",
      modelResponse: {
        schema_version: 2,
        episode_complete: false,
        segments: phases.map((phase, index) =>
          index === 2
            ? {
                step: index + 1,
                phase,
                status: "not_observed",
                start_s: null,
                end_s: null,
                confidence: null,
                caption: `step ${index + 1}`,
                evidence: null,
              }
            : {
                step: index + 1,
                phase,
                status: "completed",
                start_s: index / 10,
                end_s: (index + 1) / 10,
                confidence: 0.9,
                caption: `step ${index + 1}`,
                evidence: "visible",
              },
        ) as NonNullable<
          EpisodeCuration["activeProposal"]
        >["modelResponse"]["segments"],
        missing_steps: [3],
        uncertainties: ["step 3 was not observed"],
      },
    },
    promptPreview: valid
      ? [
          "approach the brown table",
          "pick up the plastic bottle from the table with the left hand",
          "turn right to find the black trash bin",
          "approach the black trash bin while holding the plastic bottle",
          "lean down to the black trash bin",
          "drop the plastic bottle into the black trash bin",
          "go to a standing straight pose",
        ]
      : null,
    warnings: valid ? [] : ["draft_transition_frames_invalid"],
    revision: 3,
    approvalLocked: locked,
    reviewer: locked ? "reviewer" : null,
    approvalRevision: locked ? 3 : null,
    approvedAt: locked ? "2026-08-27T00:00:00Z" : null,
    ...episodeOverrides,
  };
}

function wireEpisode(
  sourceEpisodeIndex: number,
  objectName: string,
  revision = 3,
) {
  return {
    dataset_alias: "local/pnp_trash",
    source_episode_index: sourceEpisodeIndex,
    source_length: 12,
    timestamps: Array.from({ length: 12 }, (_, frame) => frame / 10),
    decision: {
      review_state: "draft",
      object_name: objectName,
      pickup_hand: "left",
      turn_direction: "right",
      transition_frames: [1, 3, 5, 7, 9, 10],
      rejection_reason: null,
      prompt_template_sha256: HASH,
    },
    active_proposal: null,
    prompt_preview: ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    warnings: [],
    revision,
    approval_locked: false,
    reviewer: null,
    approval_revision: null,
    approved_at: null,
  };
}

function installWorkspaceBackend(
  episodes: Record<number, unknown>,
  options: { approveKeep?: () => Promise<Response> } = {},
) {
  globalThis.fetch = Object.assign(
    mock(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      const path = new URL(request.url).pathname;
      if (path.endsWith("/workspaces/open")) {
        return Response.json({
          dataset_alias: "local/pnp_trash",
          source_fingerprint: SOURCE_HASH,
          prompt_template_version: "pnp-trash-v1",
          prompt_template_sha256: HASH,
          episode_count: Object.keys(episodes).length,
        });
      }
      if (path.endsWith("/summary")) {
        return Response.json({
          dataset_alias: "local/pnp_trash",
          episode_count: Object.keys(episodes).length,
          counts: {
            pending: 0,
            draft: Object.keys(episodes).length,
            approved_keep: 0,
            approved_reject: 0,
          },
          prompt_template_version: "pnp-trash-v1",
          prompt_template_sha256: HASH,
        });
      }
      const match = path.match(/\/episodes\/(\d+)$/);
      if (request.method === "GET" && match !== null) {
        const response = episodes[Number(match[1])];
        if (response !== undefined) return Response.json(response);
      }
      if (
        request.method === "POST" &&
        path.endsWith("/approve-keep") &&
        options.approveKeep !== undefined
      ) {
        return options.approveKeep();
      }
      throw new Error(`unexpected request ${request.method} ${request.url}`);
    }),
    { preconnect: originalFetch.preconnect },
  ) as typeof fetch;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function WorkspaceToggleHarness() {
  const [visible, setVisible] = useState(true);
  return (
    <>
      <button type="button" onClick={() => setVisible((value) => !value)}>
        {visible ? "Hide workspace" : "Show workspace"}
      </button>
      {visible && <TaskIndexCurationWorkspace />}
    </>
  );
}

function EpisodeSwitchHarness() {
  const curation = useCuration();
  return (
    <>
      <button type="button" onClick={() => curation.selectEpisode(4)}>
        Select episode 4
      </button>
      <button type="button" onClick={() => curation.selectEpisode(8)}>
        Select episode 8
      </button>
      <TaskIndexCurationWorkspace />
    </>
  );
}

function PersistentRouteSyncHarness() {
  const curation = useCuration();
  const [workspaceVisible, setWorkspaceVisible] = useState(true);
  const [routeEpisode, setRouteEpisode] = useState(4);
  const [navigationRequests, setNavigationRequests] = useState<number[]>([]);
  const navigate = useCallback((sourceEpisodeIndex: number) => {
    setNavigationRequests((current) => [...current, sourceEpisodeIndex]);
  }, []);
  const requestedEpisode = navigationRequests.at(-1);
  return (
    <>
      <CurationRouteSync
        currentRouteEpisode={routeEpisode}
        onEpisodeNavigate={navigate}
      />
      <output aria-label="Video episode">{routeEpisode}</output>
      <output aria-label="Navigation requests">
        {navigationRequests.join(",") || "none"}
      </output>
      <button
        type="button"
        onClick={() => setWorkspaceVisible((visible) => !visible)}
      >
        {workspaceVisible ? "Hide workspace" : "Show workspace"}
      </button>
      <button
        type="button"
        disabled={requestedEpisode === undefined}
        onClick={() => {
          if (requestedEpisode !== undefined) setRouteEpisode(requestedEpisode);
        }}
      >
        Complete route transition
      </button>
      {workspaceVisible && (
        <TaskIndexCurationWorkspace currentRouteEpisode={routeEpisode} />
      )}
      <output aria-label="Provider episode">
        {curation.episode?.sourceEpisodeIndex ?? "loading"}
      </output>
    </>
  );
}

function RouteLagHarness() {
  const curation = useCuration();
  const [routeEpisode, setRouteEpisode] = useState(4);
  const [requestedEpisode, setRequestedEpisode] = useState<number | null>(null);
  const recordNavigation = useCallback(
    (sourceEpisodeIndex: number) => setRequestedEpisode(sourceEpisodeIndex),
    [],
  );
  return (
    <>
      <CurationRouteSync
        currentRouteEpisode={routeEpisode}
        onEpisodeNavigate={recordNavigation}
      />
      <output aria-label="Video episode">{routeEpisode}</output>
      <output aria-label="Provider episode">
        {curation.episode?.sourceEpisodeIndex ?? "loading"}
      </output>
      <output aria-label="Requested route">{requestedEpisode ?? "none"}</output>
      <button type="button" onClick={() => curation.selectEpisode(8)}>
        Select next episode
      </button>
      <button
        type="button"
        onClick={() => setRouteEpisode(curation.selectedEpisodeIndex)}
      >
        Complete route transition
      </button>
      <TaskIndexCurationWorkspace currentRouteEpisode={routeEpisode} />
    </>
  );
}

function RouteLeadingContent({
  routeEpisode,
  onEpisodeNavigate,
}: {
  routeEpisode: number;
  onEpisodeNavigate: (sourceEpisodeIndex: number) => void;
}) {
  const curation = useCuration();
  return (
    <>
      <CurationRouteSync
        currentRouteEpisode={routeEpisode}
        onEpisodeNavigate={onEpisodeNavigate}
      />
      <output aria-label="Provider episode">
        {curation.episode?.sourceEpisodeIndex ?? "loading"}
      </output>
      <TaskIndexCurationWorkspace currentRouteEpisode={routeEpisode} />
    </>
  );
}

function RouteLeadingHarness() {
  const [routeEpisode, setRouteEpisode] = useState(4);
  const [requestedEpisode, setRequestedEpisode] = useState<number | null>(null);
  const recordNavigation = useCallback(
    (sourceEpisodeIndex: number) => setRequestedEpisode(sourceEpisodeIndex),
    [],
  );
  return (
    <>
      <output aria-label="Video episode">{routeEpisode}</output>
      <output aria-label="Requested route">{requestedEpisode ?? "none"}</output>
      <button type="button" onClick={() => setRouteEpisode(8)}>
        Navigate video route
      </button>
      <CurationProvider
        datasetAlias="local/pnp_trash"
        actor="curator"
        reviewer="reviewer"
        episodeIndices={[4, 8]}
        initialEpisodeIndex={routeEpisode}
      >
        <RouteLeadingContent
          routeEpisode={routeEpisode}
          onEpisodeNavigate={recordNavigation}
        />
      </CurationProvider>
    </>
  );
}

function batchConfiguration(): BatchConfiguration {
  return {
    schema_version: 1,
    dataset_alias: "local/pnp_trash",
    dataset_id: 1,
    source_path: "/datasets/pnp_trash",
    source_manifest_sha256: "b".repeat(64),
    source_fps: 30,
    episode_indices: [4, 8],
    prompt: { version: "pnp-trash-v1", sha256: HASH },
    cosmos: {
      base_url: "https://cosmos.internal/v1",
      model: "cosmos3-nano",
      api_key_env: "COSMOS_API_KEY",
      endpoint_identity: "h100-cosmos",
    },
    sampling: { target_fps: 2 },
    worker: {
      concurrency: 1,
      lease_seconds: 180,
      heartbeat_seconds: 15,
    },
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
  };
}

const audit: CurationAudit = {
  datasetAlias: "local/pnp_trash",
  reviewStateCounts: {
    pending: 3,
    draft: 2,
    approved_keep: 4,
    approved_reject: 1,
  },
  cosmos: {
    jobStateCounts: { completed_with_failures: 1 },
    attemptStateCounts: { succeeded: 8, manual_only: 2 },
    proposalStateCounts: { active: 9, superseded: 1 },
    proposalResultCounts: { complete: 8, incomplete: 1, invalid: 0 },
    attemptErrorClassCounts: { Timeout: 2 },
  },
  transitionTimeDistributions: {
    step_2_start: {
      count: 8,
      minSeconds: 0.1,
      p25Seconds: 0.2,
      medianSeconds: 0.3,
      p75Seconds: 0.4,
      maxSeconds: 0.5,
    },
  },
  phaseDurationDistributions: {
    step_1_duration: {
      count: 8,
      minSeconds: 0.1,
      p25Seconds: 0.2,
      medianSeconds: 0.3,
      p75Seconds: 0.4,
      maxSeconds: 0.5,
    },
  },
  boundaryErrors: [{ sourceEpisodeIndex: 7, code: "transition_order" }],
  gripDisagreements: [
    {
      sourceEpisodeIndex: 4,
      warnings: ["grasp_delta_exceeds_2s"],
      graspDeltaSeconds: 2.1,
      releaseDeltaSeconds: 0.2,
    },
  ],
  gripUnavailable: [{ sourceEpisodeIndex: 6, reason: "flat_signal" }],
  unreadableFiles: [],
  contactSheetIssues: [
    {
      kind: "proposal",
      sourceEpisodeIndex: 9,
      reason: "missing",
      proposalId: "p9",
    },
  ],
  sourceFingerprint: { expectedSha256: "b".repeat(64), currentMatches: true },
};

function controller(
  overrides: Partial<CurationWorkspaceController> = {},
): CurationWorkspaceController {
  return {
    workspaceGeneration: 1,
    workspaceDatasetAlias: "local/pnp_trash",
    workspaceReady: true,
    selectedEpisodeIndex: 4,
    summary: {
      datasetAlias: "local/pnp_trash",
      episodeCount: 10,
      counts: { pending: 3, draft: 2, approved_keep: 4, approved_reject: 1 },
      promptTemplateVersion: "pnp-trash-v1",
      promptTemplateSha256: HASH,
    },
    episode: episode(),
    grip: null,
    audit: null,
    batch: null,
    loading: false,
    saving: false,
    error: null,
    warning: null,
    conflict: null,
    episodeOperationToken: 1,
    batchOperationToken: 1,
    batchOperationPending: false,
    batchPollGeneration: 1,
    auditOperationToken: 1,
    reviewIntents: {
      4: {
        choice: "keep",
        rejectionReason: "",
        dirty: false,
        authoritativeLocked: false,
      },
    },
    savedObjectSuggestions: [],
    reviewIntent: { choice: "keep", rejectionReason: "" },
    selectEpisode: mock(() => undefined),
    updateDraft: mock(() => undefined),
    updateReviewIntent: mock(() => undefined),
    saveDraft: mock(async () => undefined),
    applyProposal: mock(async () => undefined),
    reopen: mock(async () => undefined),
    approveKeepAndNext: mock(async () => undefined),
    approveRejectAndNext: mock(async () => undefined),
    startBatch: mock(async () => undefined),
    cancelBatch: mock(async () => undefined),
    retryBatch: mock(async () => undefined),
    refreshGrip: mock(async () => undefined),
    refreshAudit: mock(async () => undefined),
    clearConflict: mock(() => undefined),
    ...overrides,
  };
}

function ControlledWorkspaceView({
  curation,
  objectSuggestions,
}: {
  curation: CurationWorkspaceController;
  objectSuggestions?: string[];
}) {
  const [reviewIntent, setReviewIntent] = useState(curation.reviewIntent);
  return (
    <TaskIndexCurationWorkspaceView
      curation={{
        ...curation,
        reviewIntent,
        updateReviewIntent: (patch) => {
          curation.updateReviewIntent(patch);
          setReviewIntent((current) => ({ ...current, ...patch }));
        },
      }}
      objectSuggestions={objectSuggestions}
    />
  );
}

function renderWorkspace(value: CurationWorkspaceController) {
  return render(
    <TimeProvider duration={1.2}>
      <ControlledWorkspaceView
        curation={value}
        objectSuggestions={["aluminum can", "paper cup"]}
      />
    </TimeProvider>,
  );
}

function ReviewRevisionHarness({ operation }: { operation: "save" | "apply" }) {
  const base = controller();
  const [currentEpisode, setCurrentEpisode] = useState(base.episode!);
  const completeOrdinaryMutation = async () => {
    setCurrentEpisode((current) => ({
      ...current,
      revision: current.revision + 1,
      decision: {
        ...current.decision,
        reviewState: "draft",
        rejectionReason: null,
      },
    }));
  };
  return (
    <TimeProvider duration={1.2}>
      <ControlledWorkspaceView
        curation={{
          ...base,
          episode: currentEpisode,
          saveDraft:
            operation === "save" ? completeOrdinaryMutation : base.saveDraft,
          applyProposal:
            operation === "apply"
              ? completeOrdinaryMutation
              : base.applyProposal,
        }}
      />
    </TimeProvider>
  );
}

describe("TaskIndexCurationWorkspace", () => {
  test("enables task_index by default only for exact local pnp_trash v2.1", () => {
    expect(isTaskIndexCurationDataset("local/pnp_trash", "v2.1")).toBe(true);
    expect(isTaskIndexCurationDataset("local/pnp_trash", "v3.1")).toBe(false);
    expect(isTaskIndexCurationDataset("local/other", "v2.1")).toBe(false);
    expect(isTaskIndexCurationDataset("org/pnp_trash", "v2.1")).toBe(false);
  });

  test("edits object free text, suggestions, hand, turn, and boundaries", async () => {
    const user = userEvent.setup();
    const base = controller();
    const updateDraft = mock((patch: EpisodeDraftPatch) => {
      void patch;
    });
    function EditableHarness() {
      const [currentEpisode, setCurrentEpisode] = useState(base.episode!);
      const update = (patch: EpisodeDraftPatch) => {
        updateDraft(patch);
        setCurrentEpisode((current) => ({
          ...current,
          decision: { ...current.decision, ...patch },
          promptPreview: null,
        }));
      };
      return (
        <TimeProvider duration={1.2}>
          <TaskIndexCurationWorkspaceView
            curation={{ ...base, episode: currentEpisode, updateDraft: update }}
            objectSuggestions={["aluminum can", "paper cup"]}
          />
        </TimeProvider>
      );
    }
    render(<EditableHarness />);

    const objectInput = screen.getByRole("combobox", { name: "Object name" });
    expect(objectInput.getAttribute("list")).toBe(
      "curation-object-suggestions",
    );
    expect(
      document.querySelector(
        '#curation-object-suggestions option[value="aluminum can"]',
      ),
    ).toBeTruthy();
    await user.clear(objectInput);
    await user.type(objectInput, "cardboard box");
    expect(updateDraft).toHaveBeenLastCalledWith({
      objectName: "cardboard box",
    });

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Pickup hand" }),
      "right",
    );
    expect(updateDraft).toHaveBeenLastCalledWith({ pickupHand: "right" });
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Turn direction" }),
      "left",
    );
    expect(updateDraft).toHaveBeenLastCalledWith({ turnDirection: "left" });

    const stepFour = screen.getByRole("slider", {
      name: "Step 4 transition frame",
    });
    await user.click(stepFour);
    await user.keyboard("{ArrowRight}");
    expect(updateDraft).toHaveBeenLastCalledWith({
      transitionFrames: [1, 3, 6, 7, 9, 10],
    });
  });

  test("applies proposals only as drafts and never renders hidden reasoning", async () => {
    const user = userEvent.setup();
    const applyProposal = mock(async () => undefined);
    const value = controller({ applyProposal });
    renderWorkspace(value);

    expect(document.body.textContent?.includes("<think>")).toBe(false);
    expect(screen.getByText("Step 3 · NOT OBSERVED")).toBeTruthy();
    await user.click(
      screen.getByRole("button", { name: "Apply Cosmos proposal to draft" }),
    );
    expect(applyProposal).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("button", { name: "Approve keep and next episode" }),
    ).toBeTruthy();
  });

  test("saves drafts and approves keep or reject with the rejection reason", async () => {
    const user = userEvent.setup();
    const saveDraft = mock(async () => undefined);
    const approveKeepAndNext = mock(async () => undefined);
    const approveRejectAndNext = mock(async () => undefined);
    renderWorkspace(
      controller({ saveDraft, approveKeepAndNext, approveRejectAndNext }),
    );

    await user.click(screen.getByRole("button", { name: "Save draft" }));
    expect(saveDraft).toHaveBeenCalledTimes(1);
    await user.click(
      screen.getByRole("button", { name: "Approve keep and next episode" }),
    );
    expect(approveKeepAndNext).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("radio", { name: "Reject episode" }));
    await user.type(
      screen.getByRole("textbox", { name: "Rejection reason" }),
      "object missed bin",
    );
    await user.click(
      screen.getByRole("button", { name: "Approve reject and next episode" }),
    );
    expect(approveRejectAndNext).toHaveBeenCalledWith("object missed bin");
  });

  for (const operation of ["save", "apply"] as const) {
    test(`preserves a local reject choice and reason after ordinary ${operation}`, async () => {
      const user = userEvent.setup();
      render(<ReviewRevisionHarness operation={operation} />);

      await user.click(screen.getByRole("radio", { name: "Reject episode" }));
      await user.type(
        screen.getByRole("textbox", { name: "Rejection reason" }),
        "object missed bin",
      );
      await user.click(
        operation === "save"
          ? screen.getByRole("button", { name: "Save draft" })
          : screen.getByRole("button", {
              name: "Apply Cosmos proposal to draft",
            }),
      );

      await waitFor(() =>
        expect(
          (
            screen.getByRole("radio", {
              name: "Reject episode",
            }) as HTMLInputElement
          ).checked,
        ).toBe(true),
      );
      expect(
        (
          screen.getByRole("textbox", {
            name: "Rejection reason",
          }) as HTMLTextAreaElement
        ).value,
      ).toBe("object missed bin");
    });
  }

  test("keeps reject intent when the workspace unmounts and remounts", async () => {
    installWorkspaceBackend({ 4: wireEpisode(4, "plastic bottle") });
    try {
      const user = userEvent.setup();
      render(
        <TimeProvider duration={1.2}>
          <CurationProvider
            datasetAlias="local/pnp_trash"
            actor="curator"
            reviewer="reviewer"
            episodeIndices={[4]}
            initialEpisodeIndex={4}
          >
            <WorkspaceToggleHarness />
          </CurationProvider>
        </TimeProvider>,
      );

      await user.click(
        await screen.findByRole("radio", { name: "Reject episode" }),
      );
      await user.type(
        screen.getByRole("textbox", { name: "Rejection reason" }),
        "object missed bin",
      );
      await user.click(screen.getByRole("button", { name: "Hide workspace" }));
      expect(
        screen.queryByRole("radio", { name: "Reject episode" }),
      ).toBeNull();
      await user.click(screen.getByRole("button", { name: "Show workspace" }));

      expect(
        (await screen.findByRole("radio", {
          name: "Reject episode",
        })) as HTMLInputElement,
      ).toHaveProperty("checked", true);
      expect(
        (
          screen.getByRole("textbox", {
            name: "Rejection reason",
          }) as HTMLTextAreaElement
        ).value,
      ).toBe("object missed bin");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("keeps dirty reject intent after navigating away and back", async () => {
    installWorkspaceBackend({
      4: wireEpisode(4, "plastic bottle"),
      8: wireEpisode(8, "aluminum can"),
    });
    try {
      const user = userEvent.setup();
      render(
        <TimeProvider duration={1.2}>
          <CurationProvider
            datasetAlias="local/pnp_trash"
            actor="curator"
            reviewer="reviewer"
            episodeIndices={[4, 8]}
            initialEpisodeIndex={4}
          >
            <EpisodeSwitchHarness />
          </CurationProvider>
        </TimeProvider>,
      );

      await user.click(
        await screen.findByRole("radio", { name: "Reject episode" }),
      );
      await user.type(
        screen.getByRole("textbox", { name: "Rejection reason" }),
        "object missed bin",
      );
      await user.click(
        screen.getByRole("button", { name: "Select episode 8" }),
      );
      await screen.findByRole("heading", { name: "Episode 8 · draft" });
      await user.click(
        screen.getByRole("button", { name: "Select episode 4" }),
      );
      await screen.findByRole("heading", { name: "Episode 4 · draft" });

      expect(
        screen.getByRole("radio", { name: "Reject episode" }),
      ).toHaveProperty("checked", true);
      expect(
        (
          screen.getByRole("textbox", {
            name: "Rejection reason",
          }) as HTMLTextAreaElement
        ).value,
      ).toBe("object missed bin");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("navigates once when approve-next resolves with the workspace hidden", async () => {
    const approval = deferred<Response>();
    const approvedDraft = wireEpisode(4, "plastic bottle", 4);
    const approved = {
      ...approvedDraft,
      decision: {
        ...approvedDraft.decision,
        review_state: "approved_keep",
      },
      approval_locked: true,
      reviewer: "reviewer",
      approval_revision: 4,
      approved_at: "2026-08-27T00:00:00Z",
    };
    installWorkspaceBackend(
      {
        4: wireEpisode(4, "plastic bottle"),
        8: wireEpisode(8, "aluminum can"),
      },
      { approveKeep: () => approval.promise },
    );
    try {
      const user = userEvent.setup();
      render(
        <TimeProvider duration={1.2}>
          <CurationProvider
            datasetAlias="local/pnp_trash"
            actor="curator"
            reviewer="reviewer"
            episodeIndices={[4, 8]}
            initialEpisodeIndex={4}
          >
            <PersistentRouteSyncHarness />
          </CurationProvider>
        </TimeProvider>,
      );

      await user.click(
        await screen.findByRole("button", {
          name: "Approve keep and next episode",
        }),
      );
      await user.click(screen.getByRole("button", { name: "Hide workspace" }));
      approval.resolve(Response.json(approved));

      await waitFor(() =>
        expect(screen.getByLabelText("Provider episode").textContent).toBe("8"),
      );
      expect(screen.getByLabelText("Navigation requests").textContent).toBe(
        "8",
      );
      expect(screen.queryByTestId("task-index-curation-workspace")).toBeNull();

      await user.click(screen.getByRole("button", { name: "Show workspace" }));
      expect(screen.getByText("Loading episode 8…")).toBeTruthy();
      expect(screen.getByLabelText("Navigation requests").textContent).toBe(
        "8",
      );
      await user.click(
        screen.getByRole("button", { name: "Complete route transition" }),
      );
      expect(
        await screen.findByRole("heading", { name: "Episode 8 · draft" }),
      ).toBeTruthy();
      expect(screen.getByLabelText("Navigation requests").textContent).toBe(
        "8",
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("keeps only server-snapshot object suggestions across an unmount", async () => {
    const storageKey = "lerobot-curation:pnp-trash:object-suggestions";
    window.localStorage.removeItem(storageKey);
    installWorkspaceBackend({ 4: wireEpisode(4, "plastic bottle") });
    try {
      const user = userEvent.setup();
      render(
        <TimeProvider duration={1.2}>
          <CurationProvider
            datasetAlias="local/pnp_trash"
            actor="curator"
            reviewer="reviewer"
            episodeIndices={[4]}
            initialEpisodeIndex={4}
          >
            <WorkspaceToggleHarness />
          </CurationProvider>
        </TimeProvider>,
      );

      const objectInput = await screen.findByRole("combobox", {
        name: "Object name",
      });
      await user.clear(objectInput);
      await user.type(objectInput, "can");
      await user.click(screen.getByRole("button", { name: "Hide workspace" }));
      await user.click(screen.getByRole("button", { name: "Show workspace" }));
      await screen.findByRole("combobox", { name: "Object name" });

      await waitFor(() => {
        const history: unknown = JSON.parse(
          window.localStorage.getItem(storageKey) ?? "[]",
        );
        expect(history).toContain("plastic bottle");
        expect(history).not.toContain("can");
        expect(history).not.toContain("c");
        expect(history).not.toContain("ca");
      });
      expect(
        document.querySelector(
          '#curation-object-suggestions option[value="plastic bottle"]',
        ),
      ).toBeTruthy();
      expect(
        document.querySelector(
          '#curation-object-suggestions option[value="can"]',
        ),
      ).toBeNull();
    } finally {
      globalThis.fetch = originalFetch;
      window.localStorage.removeItem(storageKey);
    }
  });

  test("hides next-episode controls until the video route catches up", async () => {
    installWorkspaceBackend({
      4: wireEpisode(4, "plastic bottle"),
      8: wireEpisode(8, "aluminum can"),
    });
    try {
      const user = userEvent.setup();
      render(
        <TimeProvider duration={1.2}>
          <CurationProvider
            datasetAlias="local/pnp_trash"
            actor="curator"
            reviewer="reviewer"
            episodeIndices={[4, 8]}
            initialEpisodeIndex={4}
          >
            <RouteLagHarness />
          </CurationProvider>
        </TimeProvider>,
      );

      await screen.findByRole("combobox", { name: "Object name" });
      await user.click(
        screen.getByRole("button", { name: "Select next episode" }),
      );
      await waitFor(() =>
        expect(screen.getByLabelText("Provider episode").textContent).toBe("8"),
      );
      expect(screen.getByLabelText("Video episode").textContent).toBe("4");
      expect(screen.getByLabelText("Requested route").textContent).toBe("8");
      expect(
        screen.queryByRole("combobox", { name: "Object name" }),
      ).toBeNull();
      expect(
        screen.queryByRole("radio", { name: "Reject episode" }),
      ).toBeNull();
      expect(screen.getByText("Loading episode 8…")).toBeTruthy();

      await user.click(
        screen.getByRole("button", { name: "Complete route transition" }),
      );
      expect(
        await screen.findByRole("heading", { name: "Episode 8 · draft" }),
      ).toBeTruthy();
      expect(
        screen.getByRole("combobox", { name: "Object name" }),
      ).toBeTruthy();
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("does not navigate backward when the video route leads provider state", async () => {
    installWorkspaceBackend({
      4: wireEpisode(4, "plastic bottle"),
      8: wireEpisode(8, "aluminum can"),
    });
    try {
      const user = userEvent.setup();
      render(
        <TimeProvider duration={1.2}>
          <RouteLeadingHarness />
        </TimeProvider>,
      );

      await screen.findByRole("combobox", { name: "Object name" });
      await user.click(
        screen.getByRole("button", { name: "Navigate video route" }),
      );
      await waitFor(() =>
        expect(screen.getByLabelText("Provider episode").textContent).toBe("8"),
      );

      expect(screen.getByLabelText("Video episode").textContent).toBe("8");
      expect(screen.getByLabelText("Requested route").textContent).toBe("none");
      expect(
        await screen.findByRole("heading", { name: "Episode 8 · draft" }),
      ).toBeTruthy();
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("persists only bounded server-saved object suggestions, not typing prefixes", async () => {
    const storageKey = "lerobot-curation:pnp-trash:object-suggestions";
    window.localStorage.setItem(
      storageKey,
      JSON.stringify(
        Array.from({ length: 30 }, (_, index) => `saved object ${index}`),
      ),
    );
    const wireEpisode = (revision: number, objectName: string) => ({
      dataset_alias: "local/pnp_trash",
      source_episode_index: 4,
      source_length: 12,
      timestamps: Array.from({ length: 12 }, (_, frame) => frame / 10),
      decision: {
        review_state: "draft",
        object_name: objectName,
        pickup_hand: "left",
        turn_direction: "right",
        transition_frames: [1, 3, 5, 7, 9, 10],
        rejection_reason: null,
        prompt_template_sha256: HASH,
      },
      active_proposal: null,
      prompt_preview: ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
      warnings: [],
      revision,
      approval_locked: false,
      reviewer: null,
      approval_revision: null,
      approved_at: null,
    });
    globalThis.fetch = Object.assign(
      mock(async (input: RequestInfo | URL, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        if (path.endsWith("/workspaces/open")) {
          return Response.json({
            dataset_alias: "local/pnp_trash",
            source_fingerprint: SOURCE_HASH,
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: HASH,
            episode_count: 1,
          });
        }
        if (path.endsWith("/summary")) {
          return Response.json({
            dataset_alias: "local/pnp_trash",
            episode_count: 1,
            counts: {
              pending: 0,
              draft: 1,
              approved_keep: 0,
              approved_reject: 0,
            },
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: HASH,
          });
        }
        if (request.method === "GET" && path.endsWith("/episodes/4"))
          return Response.json(wireEpisode(3, "plastic bottle"));
        if (request.method === "PATCH" && path.endsWith("/draft"))
          return Response.json(wireEpisode(4, "can"));
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      }),
      { preconnect: originalFetch.preconnect },
    ) as typeof fetch;

    try {
      const user = userEvent.setup();
      render(
        <TimeProvider duration={1.2}>
          <CurationProvider
            datasetAlias="local/pnp_trash"
            actor="curator"
            reviewer="reviewer"
            episodeIndices={[4]}
            initialEpisodeIndex={4}
          >
            <TaskIndexCurationWorkspace />
          </CurationProvider>
        </TimeProvider>,
      );
      const objectInput = await screen.findByRole("combobox", {
        name: "Object name",
      });
      await user.clear(objectInput);
      await user.type(objectInput, "can");
      await user.click(screen.getByRole("button", { name: "Save draft" }));

      await waitFor(() => {
        const history: unknown = JSON.parse(
          window.localStorage.getItem(storageKey) ?? "[]",
        );
        expect(history).toBeArray();
        expect(history).toContain("plastic bottle");
        expect(history).toContain("can");
        expect(history).not.toContain("c");
        expect(history).not.toContain("ca");
        expect((history as unknown[]).length).toBeLessThanOrEqual(25);
      });
    } finally {
      globalThis.fetch = originalFetch;
      window.localStorage.removeItem(storageKey);
    }
  });

  test("keeps approval disabled until the last server-saved record is valid", () => {
    renderWorkspace(controller({ episode: episode({ valid: false }) }));

    expect(
      (
        screen.getByRole("button", {
          name: "Approve keep and next episode",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(screen.getByText("draft_transition_frames_invalid")).toBeTruthy();
  });

  test("locks approved episodes until reopen succeeds and reports stale approval", async () => {
    const user = userEvent.setup();
    const reopen = mock(async () => undefined);
    const refreshGrip = mock(async () => undefined);
    const locked = episode({ state: "approved_keep" });
    locked.approvalRevision = 2;
    locked.warnings = ["approval_revision_mismatch"];
    renderWorkspace(controller({ episode: locked, reopen, refreshGrip }));

    expect(
      (
        screen.getByRole("combobox", {
          name: "Object name",
        }) as HTMLInputElement
      ).disabled,
    ).toBe(true);
    expect(
      screen.getByText(
        "Approval is stale at revision 2; current revision is 3.",
      ),
    ).toBeTruthy();
    const gripButton = screen.getByRole("button", {
      name: "Refresh grip diagnostic",
    }) as HTMLButtonElement;
    expect(gripButton.disabled).toBe(false);
    await user.click(gripButton);
    expect(refreshGrip).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "Reopen episode" }));
    expect(reopen).toHaveBeenCalledTimes(1);
  });

  test("renders a stale approval warning decoded through the real provider boundary", async () => {
    globalThis.fetch = Object.assign(
      mock(async (input: RequestInfo | URL, init?: RequestInit) => {
        const request = new Request(input, init);
        const path = new URL(request.url).pathname;
        if (path.endsWith("/workspaces/open")) {
          return Response.json({
            dataset_alias: "local/pnp_trash",
            source_fingerprint: SOURCE_HASH,
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: HASH,
            episode_count: 1,
          });
        }
        if (path.endsWith("/summary")) {
          return Response.json({
            dataset_alias: "local/pnp_trash",
            episode_count: 1,
            counts: {
              pending: 0,
              draft: 0,
              approved_keep: 1,
              approved_reject: 0,
            },
            prompt_template_version: "pnp-trash-v1",
            prompt_template_sha256: HASH,
          });
        }
        if (path.endsWith("/episodes/4")) {
          return Response.json({
            dataset_alias: "local/pnp_trash",
            source_episode_index: 4,
            source_length: 12,
            timestamps: Array.from({ length: 12 }, (_, frame) => frame / 10),
            decision: {
              review_state: "approved_keep",
              object_name: "plastic bottle",
              pickup_hand: "left",
              turn_direction: "right",
              transition_frames: [1, 3, 5, 7, 9, 10],
              rejection_reason: null,
              prompt_template_sha256: HASH,
            },
            active_proposal: null,
            prompt_preview: ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
            warnings: ["approval_revision_mismatch"],
            revision: 3,
            approval_locked: true,
            reviewer: "reviewer",
            approval_revision: 2,
            approved_at: "2026-08-27T00:00:00Z",
          });
        }
        throw new Error(`unexpected request ${request.method} ${request.url}`);
      }),
      { preconnect: originalFetch.preconnect },
    ) as typeof fetch;

    try {
      render(
        <TimeProvider duration={1.2}>
          <CurationProvider
            datasetAlias="local/pnp_trash"
            actor="curator"
            reviewer="reviewer"
            episodeIndices={[4]}
            initialEpisodeIndex={4}
          >
            <TaskIndexCurationWorkspace />
          </CurationProvider>
        </TimeProvider>,
      );

      await waitFor(() =>
        expect(
          screen.getByText(
            "Approval is stale at revision 2; current revision is 3.",
          ),
        ).toBeTruthy(),
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("stops on revision conflict and exposes the current server revision", () => {
    const current = episode({ revision: 8 });
    renderWorkspace(
      controller({
        episode: current,
        conflict: {
          kind: "revision_conflict",
          message: "Reloaded current revision; reconcile before saving.",
          current,
        },
      }),
    );

    expect(screen.getByRole("alert").textContent).toContain(
      "Server revision 8",
    );
    expect(
      (screen.getByRole("button", { name: "Save draft" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  test("disables edit and grip controls while a mutation is saving", () => {
    renderWorkspace(controller({ saving: true }));

    expect(
      (
        screen.getByRole("combobox", {
          name: "Object name",
        }) as HTMLInputElement
      ).disabled,
    ).toBe(true);
    expect(
      (
        screen.getByRole("button", {
          name: "Refresh grip diagnostic",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Save draft" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  test("creates, displays, cancels, and retries a Cosmos batch", async () => {
    const user = userEvent.setup();
    const startBatch = mock(async () => undefined);
    const cancelBatch = mock(async () => undefined);
    const retryBatch = mock(async () => undefined);
    const base = controller({ startBatch, cancelBatch, retryBatch });
    const view = renderWorkspace(base);

    await user.click(
      screen.getByRole("button", { name: "Create Cosmos batch" }),
    );
    expect(startBatch).toHaveBeenCalledTimes(1);

    const runningBatch: NonNullable<CurationWorkspaceController["batch"]> = {
      jobId: "job-1",
      parentJobId: null,
      state: "running",
      configuration: batchConfiguration(),
      counts: { succeeded: 7, requesting: 1 },
      episodes: [],
      lease: null,
      currentEpisode: 8,
      cancelRequested: false,
      createdAt: "2026-08-27T00:00:00Z",
      updatedAt: "2026-08-27T00:00:01Z",
      errors: [],
      activeProposalCoverage: 7,
    };
    view.rerender(
      <TimeProvider duration={1.2}>
        <TaskIndexCurationWorkspaceView
          curation={{
            ...base,
            batch: runningBatch,
          }}
        />
      </TimeProvider>,
    );
    expect(screen.getByText("running")).toBeTruthy();
    expect(screen.getByText("Current episode 8")).toBeTruthy();
    await user.click(
      screen.getByRole("button", { name: "Cancel Cosmos batch" }),
    );
    expect(cancelBatch).toHaveBeenCalledTimes(1);

    view.rerender(
      <TimeProvider duration={1.2}>
        <TaskIndexCurationWorkspaceView
          curation={{
            ...base,
            batch: { ...runningBatch, state: "completed_with_failures" },
          }}
        />
      </TimeProvider>,
    );
    await user.click(
      screen.getByRole("button", { name: "Retry failed episodes" }),
    );
    expect(retryBatch).toHaveBeenCalledWith({
      failureStates: ["manual_only", "retryable", "cancelled"],
    });
  });

  test("shows audit counts, distributions, and integrity warnings", async () => {
    const user = userEvent.setup();
    const refreshAudit = mock(async () => undefined);
    renderWorkspace(controller({ audit, refreshAudit }));

    expect(screen.getByText("step_2_start")).toBeTruthy();
    expect(screen.getByText("completed_with_failures: 1")).toBeTruthy();
    expect(screen.getByText("manual_only: 2")).toBeTruthy();
    expect(
      screen.getAllByText(
        (_, element) =>
          element?.tagName === "LI" &&
          element.textContent?.includes("median 0.3 s") === true,
      ),
    ).toHaveLength(2);
    expect(screen.getByText("transition_order · episode 7")).toBeTruthy();
    expect(screen.getByText("grasp_delta_exceeds_2s · episode 4")).toBeTruthy();
    await user.click(
      screen.getByRole("button", { name: "Refresh curation audit" }),
    );
    expect(refreshAudit).toHaveBeenCalledTimes(1);
  });
});
