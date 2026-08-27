"use client";

import { useMemo } from "react";

import { useCuration } from "../context/curation-context";
import type { EpisodeCuration, ReviewState } from "../types/curation.types";
import { validateTransitionFrames } from "../utils/curationTimeline";
import { CurationAuditView } from "./curation-audit";
import { CurationBatchStatusView } from "./curation-batch-status";
import { TaskIndexCurationTimeline } from "./task-index-curation-timeline";

const MAX_OBJECT_SUGGESTIONS = 25;

export type CurationWorkspaceController = ReturnType<typeof useCuration>;

export function isTaskIndexCurationDataset(
  repoId: string,
  codebaseVersion: string,
): boolean {
  return repoId === "local/pnp_trash" && codebaseVersion === "v2.1";
}

function readableState(state: ReviewState): string {
  return state.replace(/_/g, " ");
}

function normalizedSuggestions(values: readonly string[]): string[] {
  const unique: string[] = [];
  for (const value of values) {
    const normalized = value.trim();
    if (!normalized) continue;
    const previous = unique.indexOf(normalized);
    if (previous >= 0) unique.splice(previous, 1);
    unique.push(normalized);
  }
  return unique.slice(-MAX_OBJECT_SUGGESTIONS);
}

function keepIsServerValid(episode: EpisodeCuration): boolean {
  const decision = episode.decision;
  return (
    decision.reviewState === "draft" &&
    decision.objectName !== null &&
    decision.objectName.trim().length > 0 &&
    decision.pickupHand !== null &&
    decision.turnDirection !== null &&
    validateTransitionFrames(decision.transitionFrames, episode.sourceLength)
      .valid &&
    episode.promptPreview?.length === 7 &&
    !episode.warnings.some(
      (warning) =>
        warning.startsWith("draft_") ||
        warning.startsWith("approval_") ||
        warning === "source_fingerprint_mismatch",
    )
  );
}

export interface TaskIndexCurationWorkspaceProps {
  currentRouteEpisode?: number;
  objectSuggestions?: string[];
}

export function TaskIndexCurationWorkspace({
  currentRouteEpisode,
  objectSuggestions = [],
}: TaskIndexCurationWorkspaceProps) {
  const curation = useCuration();

  if (
    currentRouteEpisode !== undefined &&
    curation.selectedEpisodeIndex !== currentRouteEpisode
  ) {
    return (
      <div className="panel p-4 text-sm text-slate-400" role="status">
        Loading episode {curation.selectedEpisodeIndex}…
      </div>
    );
  }

  return (
    <TaskIndexCurationWorkspaceView
      curation={curation}
      objectSuggestions={normalizedSuggestions([
        ...objectSuggestions,
        ...curation.savedObjectSuggestions,
      ])}
    />
  );
}

export interface TaskIndexCurationWorkspaceViewProps {
  curation: CurationWorkspaceController;
  objectSuggestions?: string[];
}

export function TaskIndexCurationWorkspaceView({
  curation,
  objectSuggestions = [],
}: TaskIndexCurationWorkspaceViewProps) {
  const {
    episode,
    summary,
    grip,
    batch,
    audit,
    loading,
    saving,
    error,
    warning,
    conflict,
  } = curation;
  const reviewChoice = curation.reviewIntent.choice;
  const rejectionReason = curation.reviewIntent.rejectionReason;

  const proposalStatuses = useMemo(
    () =>
      episode?.activeProposal?.modelResponse.segments
        .slice(1)
        .map((segment) => segment.status),
    [episode?.activeProposal],
  );

  if (!curation.workspaceReady || loading || episode === null) {
    return (
      <div className="panel p-4 text-sm text-slate-400" role="status">
        {error ?? "Loading task-index curation…"}
      </div>
    );
  }

  const locked = episode.approvalLocked;
  const mutationBlocked = saving || locked || conflict !== null;
  const canApproveKeep =
    !saving && !locked && conflict === null && keepIsServerValid(episode);
  const canApproveReject =
    !saving &&
    !locked &&
    conflict === null &&
    (episode.decision.reviewState === "pending" ||
      episode.decision.reviewState === "draft");
  const staleApproval =
    locked &&
    episode.approvalRevision !== null &&
    episode.approvalRevision !== episode.revision;

  return (
    <div className="space-y-4" data-testid="task-index-curation-workspace">
      <section className="panel p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-[10px] uppercase tracking-wide text-cyan-300">
              task_index curation
            </p>
            <h2 className="mt-1 text-base font-medium text-slate-100">
              Episode {episode.sourceEpisodeIndex} ·{" "}
              {readableState(episode.decision.reviewState)}
            </h2>
            <p className="mt-1 text-xs text-slate-400">
              Server revision {episode.revision}
            </p>
          </div>
          {summary && (
            <div
              className="flex max-w-xl flex-wrap justify-end gap-2"
              aria-label="Review counts"
            >
              {Object.entries(summary.counts).map(([state, count]) => (
                <span
                  key={state}
                  className="rounded bg-slate-900/70 px-2 py-1 text-[11px] text-slate-300"
                >
                  {state}: {count}
                </span>
              ))}
            </div>
          )}
        </div>

        {conflict && (
          <div
            className="mt-3 rounded border border-red-500/50 bg-red-950/30 p-3 text-xs text-red-200"
            role="alert"
          >
            <p>{conflict.message}</p>
            <p className="mt-1 font-mono">
              Server revision {conflict.current.revision}
            </p>
            <button
              type="button"
              className="mt-2 rounded border border-red-400/50 px-2 py-1"
              onClick={curation.clearConflict}
            >
              Reconcile current revision
            </button>
          </div>
        )}
        {error && !conflict && (
          <p className="mt-3 text-xs text-red-300" role="alert">
            {error}
          </p>
        )}
        {warning && <p className="mt-3 text-xs text-amber-300">{warning}</p>}
        {staleApproval && (
          <p className="mt-3 text-xs text-red-300">
            Approval is stale at revision {episode.approvalRevision}; current
            revision is {episode.revision}.
          </p>
        )}
        {locked && (
          <p className="mt-3 rounded border border-emerald-600/40 bg-emerald-950/20 p-2 text-xs text-emerald-200">
            This approval is read-only. Reopen the episode before changing its
            decision.
          </p>
        )}
      </section>

      <section
        className="panel grid gap-4 p-4 lg:grid-cols-3"
        aria-label="Human decision fields"
      >
        <label className="text-xs text-slate-300">
          Object name
          <input
            type="text"
            list="curation-object-suggestions"
            className="mt-1 block w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            value={episode.decision.objectName ?? ""}
            disabled={mutationBlocked}
            onChange={(event) =>
              curation.updateDraft({ objectName: event.currentTarget.value })
            }
          />
        </label>
        <datalist id="curation-object-suggestions">
          {normalizedSuggestions(objectSuggestions).map((suggestion) => (
            <option key={suggestion} value={suggestion} />
          ))}
        </datalist>
        <label className="text-xs text-slate-300">
          Pickup hand
          <select
            className="mt-1 block w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            value={episode.decision.pickupHand ?? ""}
            disabled={mutationBlocked}
            onChange={(event) =>
              curation.updateDraft({
                pickupHand:
                  event.currentTarget.value === "left" ||
                  event.currentTarget.value === "right"
                    ? event.currentTarget.value
                    : null,
              })
            }
          >
            <option value="">Select hand</option>
            <option value="left">left</option>
            <option value="right">right</option>
          </select>
        </label>
        <label className="text-xs text-slate-300">
          Turn direction
          <select
            className="mt-1 block w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            value={episode.decision.turnDirection ?? ""}
            disabled={mutationBlocked}
            onChange={(event) =>
              curation.updateDraft({
                turnDirection:
                  event.currentTarget.value === "left" ||
                  event.currentTarget.value === "right"
                    ? event.currentTarget.value
                    : null,
              })
            }
          >
            <option value="">Select turn</option>
            <option value="left">left</option>
            <option value="right">right</option>
          </select>
        </label>
      </section>

      {episode.activeProposal && (
        <section
          className="panel p-4"
          aria-labelledby="cosmos-proposal-heading"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3
                id="cosmos-proposal-heading"
                className="text-sm font-medium text-slate-100"
              >
                Cosmos proposal
              </h3>
              <p className="mt-1 text-xs text-slate-400">
                Applying this proposal updates the draft only. Human approval
                remains separate.
              </p>
            </div>
            <button
              type="button"
              className="rounded border border-violet-500/60 px-3 py-1.5 text-xs text-violet-200 disabled:opacity-40"
              disabled={mutationBlocked}
              onClick={() => void curation.applyProposal()}
            >
              Apply Cosmos proposal to draft
            </button>
          </div>
          <ol className="mt-3 grid gap-1 sm:grid-cols-2 xl:grid-cols-4">
            {episode.activeProposal.modelResponse.segments.map((segment) => (
              <li
                key={segment.step}
                className="rounded bg-slate-900/50 px-2 py-1 text-[11px] text-slate-300"
              >
                Step {segment.step} ·{" "}
                {segment.status === "not_observed"
                  ? "NOT OBSERVED"
                  : segment.status}
              </li>
            ))}
          </ol>
        </section>
      )}

      <TaskIndexCurationTimeline
        timestamps={episode.timestamps}
        transitions={episode.decision.transitionFrames}
        proposalTransitions={episode.activeProposal?.transitionFrames}
        proposalStatuses={proposalStatuses}
        grip={grip}
        disabled={mutationBlocked}
        onChange={(transitionFrames) =>
          curation.updateDraft({ transitionFrames })
        }
      />

      <section className="panel p-4" aria-label="Grip diagnostic">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-slate-100">
              Grip cross-check
            </h3>
            {grip === null ? (
              <p className="mt-1 text-xs text-slate-400">Not loaded.</p>
            ) : grip.status === "available" ? (
              <p className="mt-1 text-xs text-slate-400">
                {grip.side} grasp frame {grip.grasp.frame}; release frame{" "}
                {grip.release.frame}
              </p>
            ) : (
              <p className="mt-1 text-xs text-amber-300">
                Unavailable: {grip.reason}
              </p>
            )}
          </div>
          <button
            type="button"
            className="rounded border border-amber-500/60 px-3 py-1.5 text-xs text-amber-200 disabled:opacity-40"
            disabled={saving || conflict !== null}
            onClick={() => void curation.refreshGrip()}
          >
            Refresh grip diagnostic
          </button>
        </div>
        {grip?.advisories.map((advisory) => (
          <p key={advisory} className="mt-2 text-xs text-amber-300">
            {advisory}
          </p>
        ))}
      </section>

      <section className="panel p-4" aria-label="Review decision">
        <fieldset disabled={saving || locked || conflict !== null}>
          <legend className="text-sm font-medium text-slate-100">
            Final review
          </legend>
          <div className="mt-2 flex gap-4 text-xs text-slate-300">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="curation-review-choice"
                checked={reviewChoice === "keep"}
                onChange={() => curation.updateReviewIntent({ choice: "keep" })}
              />
              Keep episode
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="curation-review-choice"
                checked={reviewChoice === "reject"}
                onChange={() =>
                  curation.updateReviewIntent({ choice: "reject" })
                }
              />
              Reject episode
            </label>
          </div>
          {reviewChoice === "reject" && (
            <label className="mt-3 block text-xs text-slate-300">
              Rejection reason
              <textarea
                className="mt-1 block min-h-20 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                value={rejectionReason}
                onChange={(event) =>
                  curation.updateReviewIntent({
                    rejectionReason: event.currentTarget.value,
                  })
                }
              />
            </label>
          )}
        </fieldset>

        {episode.warnings.length > 0 && (
          <div
            className="mt-3 rounded border border-amber-600/40 bg-amber-950/20 p-2"
            aria-label="Validation warnings"
          >
            {episode.warnings.map((item) => (
              <p key={item} className="text-xs text-amber-200">
                {item}
              </p>
            ))}
          </div>
        )}

        {episode.promptPreview && (
          <ol
            className="mt-3 list-decimal space-y-1 pl-5 text-xs text-slate-400"
            aria-label="Server prompt preview"
          >
            {episode.promptPreview.map((prompt, index) => (
              <li key={`${index}-${prompt}`}>{prompt}</li>
            ))}
          </ol>
        )}

        <div className="mt-4 flex flex-wrap gap-2">
          {locked ? (
            <button
              type="button"
              className="rounded border border-amber-500/60 px-3 py-1.5 text-xs text-amber-200 disabled:opacity-40"
              disabled={saving || conflict !== null}
              onClick={() => void curation.reopen()}
            >
              Reopen episode
            </button>
          ) : (
            <>
              <button
                type="button"
                className="rounded border border-slate-600 px-3 py-1.5 text-xs text-slate-200 disabled:opacity-40"
                disabled={saving || conflict !== null}
                onClick={() => void curation.saveDraft()}
              >
                Save draft
              </button>
              {reviewChoice === "keep" ? (
                <button
                  type="button"
                  className="rounded border border-emerald-500/60 px-3 py-1.5 text-xs text-emerald-200 disabled:opacity-40"
                  disabled={!canApproveKeep}
                  onClick={() => void curation.approveKeepAndNext()}
                >
                  Approve keep and next episode
                </button>
              ) : (
                <button
                  type="button"
                  className="rounded border border-red-500/60 px-3 py-1.5 text-xs text-red-200 disabled:opacity-40"
                  disabled={!canApproveReject}
                  onClick={() =>
                    void curation.approveRejectAndNext(
                      rejectionReason.trim() || null,
                    )
                  }
                >
                  Approve reject and next episode
                </button>
              )}
            </>
          )}
        </div>
      </section>

      <CurationBatchStatusView
        batch={batch}
        operationPending={curation.batchOperationPending}
        onStart={() => curation.startBatch()}
        onCancel={curation.cancelBatch}
        onRetry={() =>
          curation.retryBatch({
            failureStates: ["manual_only", "retryable", "cancelled"],
          })
        }
      />
      <CurationAuditView audit={audit} onRefresh={curation.refreshAudit} />
    </div>
  );
}
