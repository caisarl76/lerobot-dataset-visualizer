"use client";
import { useEffect, useRef, useState } from "react";
import {
  getAnnotationJob,
  prepareAnnotationDataset,
} from "../../utils/annotationsClient";
export default function AnnotatePage() {
  const [source, setSource] = useState("mncai/G1_Dex3_PickAndPlaceTrash");
  const [revision, setRevision] = useState("main");
  const [kind, setKind] = useState<"repo_id" | "local_path">("repo_id");
  const [status, setStatus] = useState("");
  const [repo, setRepo] = useState("");
  const [firstEpisode, setFirstEpisode] = useState(0);
  const [validationMessages, setValidationMessages] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const hosted =
    process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL?.startsWith("/") ?? false;
  const token = useRef(0);
  useEffect(
    () => () => {
      token.current += 1;
    },
    [],
  );
  const prepare = async () => {
    const mine = ++token.current;
    setBusy(true);
    setRepo("");
    setValidationMessages([]);
    setStatus("Preparing draft…");
    try {
      const job = await prepareAnnotationDataset({
        [kind]: source.trim(),
        revision: revision.trim() || "main",
      });
      if (mine !== token.current) return;
      let x = await getAnnotationJob(job.job_id);
      while (
        (x.status === "queued" || x.status === "running") &&
        mine === token.current
      ) {
        await new Promise((r) => setTimeout(r, 1000));
        if (mine !== token.current) return;
        x = await getAnnotationJob(job.job_id);
      }
      if (mine !== token.current) return;
      if (x.status === "completed" && x.result) {
        setRepo(x.result.repo_id);
        setFirstEpisode(x.result.first_episode_index ?? 0);
        setValidationMessages([
          ...x.result.validation.errors,
          ...x.result.validation.warnings,
        ]);
        setStatus("Ready to review");
      } else setStatus(x.error || "Preparation failed");
    } catch (e) {
      if (mine === token.current)
        setStatus(e instanceof Error ? e.message : String(e));
    } finally {
      if (mine === token.current) setBusy(false);
    }
  };
  return (
    <main className="annotations-skin annotation-prepare-page">
      <h1>Prepare annotation dataset</h1>
      <p>Prepare an independent draft; the source remains unchanged.</p>
      <div className="annotation-prepare-fields">
        <label className="annotation-field">
          Source type
          <select
            aria-label="Dataset source type"
            value={kind}
            onChange={(e) => setKind(e.target.value as typeof kind)}
            disabled={busy}
          >
            <option value="repo_id">Hugging Face repo ID</option>
            {!hosted && <option value="local_path">Local dataset path</option>}
          </select>
        </label>
        <label className="annotation-field">
          Dataset source
          <input
            type="text"
            aria-label="Dataset source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder={
              kind === "repo_id" ? "org/dataset" : "/path/to/dataset"
            }
            disabled={busy}
          />
        </label>
        {kind === "repo_id" && (
          <label className="annotation-field">
            Source revision
            <input
              type="text"
              aria-label="Source revision"
              value={revision}
              onInput={(e) => setRevision(e.currentTarget.value)}
              disabled={busy}
            />
          </label>
        )}
        <button
          className="annotation-prepare-primary"
          onClick={prepare}
          disabled={!source.trim() || busy}
        >
          {busy ? "Preparing…" : "Prepare draft"}
        </button>
      </div>
      {status && <p role="status">{status}</p>}
      {validationMessages.map((message) => (
        <p key={message}>{message}</p>
      ))}
      {repo && (
        <a
          className="annotation-prepare-output"
          href={`/${repo}/episode_${firstEpisode}?tab=annotations`}
        >
          Review prepared dataset
        </a>
      )}
    </main>
  );
}
