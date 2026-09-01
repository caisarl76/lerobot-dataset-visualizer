"use client";

import type { BatchStatus } from "../types/curation.types";

const CANCELLABLE = new Set<BatchStatus["state"]>(["queued", "running"]);
const RETRYABLE = new Set<BatchStatus["state"]>([
  "completed_with_failures",
  "cancelled",
  "failed",
]);

export interface CurationBatchStatusViewProps {
  batch: BatchStatus | null;
  operationPending: boolean;
  onStart: () => void | Promise<void>;
  onCancel: () => void | Promise<void>;
  onRetry: () => void | Promise<void>;
}

export function CurationBatchStatusView({
  batch,
  operationPending,
  onStart,
  onCancel,
  onRetry,
}: CurationBatchStatusViewProps) {
  return (
    <section className="panel p-4" aria-labelledby="curation-batch-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3
            id="curation-batch-heading"
            className="text-sm font-medium text-slate-100"
          >
            Cosmos batch
          </h3>
          {batch === null ? (
            <p className="mt-1 text-xs text-slate-400">No batch selected.</p>
          ) : (
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-400">
              <span className="font-mono text-cyan-300">{batch.state}</span>
              <span>Job {batch.jobId}</span>
              <span>{batch.activeProposalCoverage} active proposals</span>
              {batch.currentEpisode !== null && (
                <span>Current episode {batch.currentEpisode}</span>
              )}
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          {batch === null && (
            <button
              type="button"
              className="rounded border border-cyan-500/60 px-3 py-1.5 text-xs text-cyan-200 disabled:opacity-40"
              disabled={operationPending}
              onClick={() => void onStart()}
            >
              Create Cosmos batch
            </button>
          )}
          {batch !== null && CANCELLABLE.has(batch.state) && (
            <button
              type="button"
              className="rounded border border-red-500/60 px-3 py-1.5 text-xs text-red-200 disabled:opacity-40"
              disabled={operationPending}
              onClick={() => void onCancel()}
            >
              Cancel Cosmos batch
            </button>
          )}
          {batch !== null && RETRYABLE.has(batch.state) && (
            <button
              type="button"
              className="rounded border border-amber-500/60 px-3 py-1.5 text-xs text-amber-200 disabled:opacity-40"
              disabled={operationPending}
              onClick={() => void onRetry()}
            >
              Retry failed episodes
            </button>
          )}
        </div>
      </div>

      {batch !== null && (
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(batch.counts).map(([state, count]) => (
            <span
              key={state}
              className="rounded bg-slate-900/70 px-2 py-1 text-[11px] text-slate-300"
            >
              {state}: {count}
            </span>
          ))}
          {batch.errors.map((error) => (
            <span key={error.attemptId} className="text-[11px] text-red-300">
              Episode {error.sourceEpisodeIndex}: {error.errorClass ?? "error"}
              {error.errorSummary ? ` — ${error.errorSummary}` : ""}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
