"use client";

import { useEffect, useRef, useState } from "react";
import { useAnnotations } from "../context/annotations-context";
import {
  deleteAnnotationEpisodes,
  getAnnotationJob,
} from "../utils/annotationsClient";

export function DeleteEpisodesControl() {
  const { ident, episodeId, dirty, save } = useAnnotations();
  const [episodes, setEpisodes] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [href, setHref] = useState<string | null>(null);
  const generation = useRef(0);

  useEffect(() => {
    setEpisodes(episodeId === null ? "" : String(episodeId));
    setBusy(false);
    setStatus("");
    setHref(null);
    generation.current += 1;
  }, [episodeId, ident.repoId, ident.localPath, ident.revision]);
  useEffect(
    () => () => {
      generation.current += 1;
    },
    [],
  );

  const run = async () => {
    const raw = episodes.trim();
    const entries = raw.split(",").map((value) => value.trim());
    const indices = entries.map(Number);
    if (
      entries.some((value) => !/^\d+$/.test(value)) ||
      indices.some((value) => !Number.isSafeInteger(value)) ||
      new Set(indices).size !== indices.length
    ) {
      setStatus(
        "Episode IDs must be distinct comma-separated nonnegative integers.",
      );
      return;
    }
    const token = ++generation.current;
    setBusy(true);
    setHref(null);
    setStatus("Starting cleaned draft…");
    try {
      if (dirty && !(await save()).ok) {
        if (token === generation.current)
          setStatus("Save the current episode before deleting episodes.");
        return;
      }
      if (token !== generation.current) return;
      const job = await deleteAnnotationEpisodes({
        ...ident,
        episode_indices: indices,
      });
      if (token !== generation.current) return;
      let current = await getAnnotationJob(job.job_id);
      while (
        (current.status === "queued" || current.status === "running") &&
        token === generation.current
      ) {
        setStatus("Creating cleaned draft…");
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (token !== generation.current) return;
        current = await getAnnotationJob(job.job_id);
      }
      if (token !== generation.current) return;
      if (current.status === "completed" && current.result) {
        const first = current.result.first_episode_index ?? 0;
        setHref(`/${current.result.repo_id}/episode_${first}`);
        setStatus(
          "Cleaned draft ready. The source dataset was kept unchanged.",
        );
      } else setStatus(current.error || "Could not create cleaned draft.");
    } catch (error) {
      if (token === generation.current)
        setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (token === generation.current) setBusy(false);
    }
  };

  return (
    <details className="annotation-composer">
      <summary>Delete episodes</summary>
      <div className="composer-copy">
        <p>
          Creates a cleaned copy, renumbers remaining episodes, and keeps the
          source dataset unchanged.
        </p>
      </div>
      <div className="quick-add">
        <label>
          Episode IDs to delete (comma-separated)
          <input
            value={episodes}
            onInput={(event) => setEpisodes(event.currentTarget.value)}
            placeholder={String(episodeId ?? 0)}
            disabled={busy}
          />
        </label>
        <button
          className="add-btn"
          onClick={run}
          disabled={busy || episodeId === null}
        >
          {busy ? "Creating…" : "Create cleaned draft"}
        </button>
      </div>
      {status && (
        <div className="save-status" role="status">
          {status}
        </div>
      )}
      {href && <a href={href}>Open cleaned draft</a>}
    </details>
  );
}
