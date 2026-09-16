"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import HfAuthButton from "@/components/hf-auth-button";
import styles from "./page.module.css";
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
  const [excludeTransitionPauses, setExcludeTransitionPauses] = useState(true);
  const hosted =
    process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL?.startsWith("/") ?? false;
  const localAllowed = Boolean(
    process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL && !hosted,
  );
  const token = useRef(0);
  useEffect(() => {
    if (!localAllowed) return;
    const localPath = new URLSearchParams(window.location.search).get(
      "local_path",
    );
    if (localPath) {
      setSource(localPath);
      setKind("local_path");
    }
  }, [localAllowed]);
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
        exclude_transition_pauses: excludeTransitionPauses,
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
    <main className={styles.page}>
      <div className={styles.background} aria-hidden="true">
        <video
          autoPlay
          muted
          loop
          playsInline
          tabIndex={-1}
          src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/level2.mp4"
        />
      </div>
      <header className={styles.navigation}>
        <Link href="/">← Dataset visualizer</Link>
        <HfAuthButton variant="ghost" />
      </header>
      <div className={styles.content}>
        <div className={styles.intro}>
          <p className={styles.eyebrow}>LeRobot · Annotation workspace</p>
          <h1>
            Prepare your <span>annotation dataset</span>
          </h1>
          <p>Turn robot episodes into reviewed, training-ready annotations.</p>
        </div>
        <ol className={styles.steps} aria-label="Annotation workflow">
          <li aria-current="step">
            <span>1</span> Prepare dataset
          </li>
          <li>
            <span>2</span> Annotate & review
          </li>
          <li>
            <span>3</span> Export dataset
          </li>
        </ol>
        <section className={styles.card} aria-labelledby="prepare-heading">
          <h2 id="prepare-heading">Prepare annotation dataset</h2>
          <p className={styles.description}>
            Choose a dataset to create an independent annotation draft.
          </p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (source.trim() && !busy) void prepare();
            }}
            aria-busy={busy}
            className={styles.form}
          >
            <div className={styles.options}>
              <label>
                Source type
                <select
                  aria-label="Dataset source type"
                  value={kind}
                  onChange={(e) => setKind(e.target.value as typeof kind)}
                  disabled={busy}
                >
                  <option value="repo_id">Hugging Face repo ID</option>
                  {localAllowed && (
                    <option value="local_path">Local dataset path</option>
                  )}
                </select>
              </label>
              {kind === "repo_id" && (
                <label>
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
            </div>
            <label>
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
                spellCheck={false}
                aria-describedby="source-hint"
              />
            </label>
            <p id="source-hint" className={styles.hint}>
              {kind === "repo_id"
                ? "Enter a Hugging Face dataset ID, such as org/dataset."
                : "Enter the full path to a LeRobot dataset on this machine. It will be prepared before you review annotations."}
            </p>
            <fieldset className={styles.filters} disabled={busy}>
              <legend>Import filters</legend>
              <label className={styles.filterToggle}>
                <input
                  type="checkbox"
                  checked={excludeTransitionPauses}
                  onChange={(event) =>
                    setExcludeTransitionPauses(event.target.checked)
                  }
                  aria-describedby="transition-filter-hint"
                />
                Exclude transition pauses
              </label>
              <p id="transition-filter-hint" className={styles.hint}>
                Automatically exclude low-motion pauses when switching between
                Pose and Planner modes. You can adjust or remove these intervals
                on the Exclude timeline. Datasets without the required robot
                fields are skipped.
              </p>
            </fieldset>
            <button
              type="submit"
              className={styles.primary}
              disabled={!source.trim() || busy}
            >
              {busy ? "Preparing draft…" : "Prepare draft"}
              <span aria-hidden="true">{busy ? "…" : "→"}</span>
            </button>
          </form>
          <p className={styles.note}>
            Your source dataset stays unchanged. Review edits before exporting
            or uploading.
          </p>
          {status && (
            <div
              className={`${styles.status} ${repo ? styles.ready : ""}`}
              role="status"
            >
              {status}
            </div>
          )}
          {validationMessages.length > 0 && (
            <div className={styles.validation}>
              <h3>Dataset checks</h3>
              <ul>
                {validationMessages.map((message, index) => (
                  <li key={`${index}-${message}`}>{message}</li>
                ))}
              </ul>
            </div>
          )}
          {repo && (
            <Link
              className={styles.review}
              href={`/${repo}/episode_${firstEpisode}?tab=annotations`}
            >
              Review prepared dataset <span aria-hidden="true">→</span>
            </Link>
          )}
        </section>
        <p className={styles.footer}>
          Generate prompts and timelines, inspect episodes, and export your
          reviewed dataset.
        </p>
      </div>
    </main>
  );
}
