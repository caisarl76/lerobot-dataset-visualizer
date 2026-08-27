"use client";

import type { AuditDistribution, CurationAudit } from "../types/curation.types";

function seconds(value: number | null): string {
  return value === null ? "n/a" : `${value.toFixed(1)} s`;
}

function DistributionRows({
  title,
  values,
}: {
  title: string;
  values: Record<string, AuditDistribution>;
}) {
  const entries = Object.entries(values);
  return (
    <div>
      <h4 className="text-xs font-medium text-slate-200">{title}</h4>
      {entries.length === 0 ? (
        <p className="mt-1 text-[11px] text-slate-500">No observations.</p>
      ) : (
        <ul className="mt-1 space-y-1">
          {entries.map(([name, distribution]) => (
            <li key={name} className="text-[11px] text-slate-400">
              <span className="font-mono text-slate-300">{name}</span> · n=
              {distribution.count} · min {seconds(distribution.minSeconds)} ·
              median {seconds(distribution.medianSeconds)} · max{" "}
              {seconds(distribution.maxSeconds)} · p25{" "}
              {seconds(distribution.p25Seconds)} · p75{" "}
              {seconds(distribution.p75Seconds)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export interface CurationAuditViewProps {
  audit: CurationAudit | null;
  onRefresh: () => void | Promise<void>;
}

export function CurationAuditView({
  audit,
  onRefresh,
}: CurationAuditViewProps) {
  return (
    <section className="panel p-4" aria-labelledby="curation-audit-heading">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3
            id="curation-audit-heading"
            className="text-sm font-medium text-slate-100"
          >
            Dataset audit
          </h3>
          <p className="mt-1 text-xs text-slate-400">
            Review distributions, physical checks, and evidence integrity before
            export.
          </p>
        </div>
        <button
          type="button"
          className="rounded border border-slate-600 px-3 py-1.5 text-xs text-slate-200"
          onClick={() => void onRefresh()}
        >
          Refresh curation audit
        </button>
      </div>

      {audit === null ? (
        <p className="mt-3 text-xs text-slate-500">
          Run the audit to load dataset-wide checks.
        </p>
      ) : (
        <div className="mt-4 space-y-4">
          <div className="flex flex-wrap gap-2">
            {Object.entries(audit.reviewStateCounts).map(([state, count]) => (
              <span
                key={state}
                className="rounded bg-slate-900/70 px-2 py-1 text-[11px] text-slate-300"
              >
                {state}: {count}
              </span>
            ))}
            <span
              className={`rounded px-2 py-1 text-[11px] ${
                audit.sourceFingerprint.currentMatches
                  ? "bg-emerald-950/50 text-emerald-200"
                  : "bg-red-950/50 text-red-200"
              }`}
            >
              Source fingerprint{" "}
              {audit.sourceFingerprint.currentMatches ? "matches" : "changed"}
            </span>
          </div>

          <div className="space-y-2" aria-label="Cosmos audit counts">
            {(
              [
                ["jobs", audit.cosmos.jobStateCounts],
                ["attempts", audit.cosmos.attemptStateCounts],
                ["proposals", audit.cosmos.proposalStateCounts],
                ["results", audit.cosmos.proposalResultCounts],
                ["errors", audit.cosmos.attemptErrorClassCounts],
              ] as const
            ).map(([label, counts]) => (
              <div key={label} className="flex flex-wrap items-center gap-2">
                <span className="w-16 text-[10px] uppercase tracking-wide text-slate-500">
                  {label}
                </span>
                {Object.entries(counts).length === 0 ? (
                  <span className="text-[11px] text-slate-500">none</span>
                ) : (
                  Object.entries(counts).map(([state, count]) => (
                    <span
                      key={`${label}-${state}`}
                      className="rounded bg-slate-900/70 px-2 py-1 text-[11px] text-slate-300"
                    >
                      {state}: {count}
                    </span>
                  ))
                )}
              </div>
            ))}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <DistributionRows
              title="Transition times"
              values={audit.transitionTimeDistributions}
            />
            <DistributionRows
              title="Phase durations"
              values={audit.phaseDurationDistributions}
            />
          </div>

          <div className="space-y-1" aria-label="Audit warnings">
            {audit.boundaryErrors.map((item) => (
              <p
                key={`${item.sourceEpisodeIndex}-${item.code}`}
                className="text-[11px] text-red-300"
              >
                {item.code} · episode {item.sourceEpisodeIndex}
              </p>
            ))}
            {audit.gripDisagreements.flatMap((item) =>
              item.warnings.map((warning) => (
                <p
                  key={`${item.sourceEpisodeIndex}-${warning}`}
                  className="text-[11px] text-amber-300"
                >
                  {warning} · episode {item.sourceEpisodeIndex}
                </p>
              )),
            )}
            {audit.gripUnavailable.map((item) => (
              <p
                key={`${item.sourceEpisodeIndex}-${item.reason}`}
                className="text-[11px] text-slate-400"
              >
                grip unavailable: {item.reason} · episode{" "}
                {item.sourceEpisodeIndex}
              </p>
            ))}
            {audit.unreadableFiles.map((item, index) => (
              <p
                key={`${item.sourceEpisodeIndex}-${item.assetKind}-${index}`}
                className="text-[11px] text-red-300"
              >
                unreadable {item.assetKind}: {item.reason}
                {item.sourceEpisodeIndex === null
                  ? ""
                  : ` · episode ${item.sourceEpisodeIndex}`}
              </p>
            ))}
            {audit.contactSheetIssues.map((item) => (
              <p
                key={`${item.kind}-${item.sourceEpisodeIndex}-${item.reason}`}
                className="text-[11px] text-amber-300"
              >
                {item.kind} contact sheet {item.reason} · episode{" "}
                {item.sourceEpisodeIndex}
              </p>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
