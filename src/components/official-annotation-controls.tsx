"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useAnnotations } from "../context/annotations-context";
import {
  createAnnotationJob,
  fetchAnnotationConfig,
  fetchWorkflow,
  getAnnotationJob,
  validateAnnotation,
  type AnnotationValidation,
} from "../utils/annotationsClient";

export function OfficialAnnotationControls() {
  const { atoms, ident, episodeId, dirty, save } = useAnnotations();
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [json, setJson] = useState("{}");
  const [all, setAll] = useState(false);
  const [exampleEpisodes, setExampleEpisodes] = useState("");
  const [taskPrompt, setTaskPrompt] = useState("");
  const [subtaskPrompts, setSubtaskPrompts] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [report, setReport] = useState<AnnotationValidation | null>(null);
  const [resultHref, setResultHref] = useState<string | null>(null);
  const generation = useRef(0);
  const hosted =
    process.env.NEXT_PUBLIC_ANNOTATE_BACKEND_URL?.startsWith("/") ?? false;
  useEffect(() => {
    setExampleEpisodes("");
    setTaskPrompt("");
    setSubtaskPrompts("");
    let active = true;
    if (ident.repoId?.startsWith("local/"))
      fetchWorkflow(ident.repoId.slice(6))
        .then((run) => {
          if (active) {
            setTaskPrompt(run.task_prompt);
            setSubtaskPrompts(run.subtask_prompts.join("\n"));
          }
        })
        .catch(() => {});
    return () => {
      active = false;
    };
  }, [ident.repoId, ident.localPath, ident.revision]);
  const latestEdits = useRef({ dirty, atoms });
  useEffect(() => {
    latestEdits.current = { dirty, atoms };
  }, [dirty, atoms]);
  useEffect(() => {
    let active = true;
    fetchAnnotationConfig()
      .then((x) => {
        if (active) {
          const publicConfig = { ...x.config };
          if (hosted) delete publicConfig.vlm;
          setConfig(publicConfig);
          setJson(JSON.stringify(publicConfig, null, 2));
        }
      })
      .catch(
        () =>
          active &&
          setStatus(
            "Annotation backend unavailable. Set NEXT_PUBLIC_ANNOTATE_BACKEND_URL and start the backend.",
          ),
      );
    return () => {
      active = false;
    };
  }, [hosted]);
  useEffect(
    () => () => {
      generation.current += 1;
    },
    [episodeId, ident.repoId, ident.localPath, ident.revision],
  );
  useEffect(() => {
    setResultHref(null);
    setReport(null);
    setStatus("");
    setBusy(false);
    generation.current += 1;
  }, [episodeId, ident.repoId, ident.localPath, ident.revision]);
  const toggleModule = (key: string, value: boolean) => {
    let latest = config;
    try {
      latest = JSON.parse(json) as Record<string, unknown>;
    } catch {
      /* retain last valid config */
    }
    const current = latest[key];
    const next = {
      ...latest,
      [key]:
        typeof current === "object" && current !== null
          ? { ...(current as Record<string, unknown>), enabled: value }
          : { enabled: value },
    };
    setConfig(next);
    setJson(JSON.stringify(next, null, 2));
  };
  const validate = async () => {
    const token = ++generation.current;
    setStatus("Validating episode…");
    setBusy(true);
    try {
      const next = await validateAnnotation({
        ...ident,
        episode_index: episodeId ?? 0,
        atoms,
      });
      if (token !== generation.current) return;
      setReport(next);
      setStatus(next.ok ? "Validation passed." : "Validation found errors.");
    } catch (e) {
      if (token === generation.current)
        setStatus(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === generation.current) setBusy(false);
    }
  };
  const run = async () => {
    const entries = exampleEpisodes.trim()
      ? exampleEpisodes.split(",").map((value) => value.trim())
      : [];
    const examples = entries.map(Number);
    if (
      entries.some((value) => !/^\d+$/.test(value)) ||
      examples.some((value) => !Number.isSafeInteger(value)) ||
      new Set(examples).size !== examples.length ||
      examples.length > 5
    ) {
      setStatus(
        "Example episodes must be up to 5 distinct comma-separated nonnegative IDs.",
      );
      return;
    }
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(json) as Record<string, unknown>;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
        throw new Error("Expected object");
    } catch {
      setStatus("Advanced options must be valid JSON.");
      return;
    }
    const prompts = subtaskPrompts
      .split("\n")
      .map((x) => x.trim())
      .filter(Boolean);
    if (
      (hosted || taskPrompt.trim() || prompts.length) &&
      (!taskPrompt.trim() ||
        !prompts.length ||
        new Set(prompts).size !== prompts.length)
    ) {
      setStatus(
        "Enter a task description and distinct ordered subtask prompts.",
      );
      return;
    }
    if (hosted && parsed.vlm) {
      setStatus(
        "VLM connection settings are managed by the annotation server.",
      );
      return;
    }
    const token = ++generation.current;
    setBusy(true);
    setResultHref(null);
    setReport(null);
    setStatus("Starting generation…");
    const started = Date.now();
    try {
      if (dirty && !(await save()).ok) {
        if (token !== generation.current) return;
        setStatus("Save the current episode before generating.");
        return;
      }
      if (token !== generation.current) return;
      const job = await createAnnotationJob({
        ...ident,
        episode_indices: all ? undefined : [episodeId ?? 0],
        config: parsed,
        ...(examples.length ? { example_episode_indices: examples } : {}),
        ...(taskPrompt.trim() || subtaskPrompts.trim()
          ? {
              task_prompt: taskPrompt.trim(),
              subtask_prompts: prompts,
              assess_quality: true,
            }
          : {}),
      });
      if (token !== generation.current) return;
      setStatus("Generating draft…");
      let current = await getAnnotationJob(job.job_id);
      while (
        (current.status === "queued" || current.status === "running") &&
        token === generation.current
      ) {
        setStatus(
          `${current.status === "queued" ? "Queued" : "Generating"} ${all ? "all episodes" : `episode ${episodeId ?? 0}`}… ${Math.floor((Date.now() - started) / 1000)}s elapsed. The generated draft will open when ready.`,
        );
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (token !== generation.current) return;
        current = await getAnnotationJob(job.job_id);
      }
      if (token !== generation.current) return;
      if (current.status === "completed" && current.result) {
        setReport(current.result.validation);
        const reviewEpisode = examples.includes(episodeId ?? 0)
          ? (current.result.first_generated_episode_index ?? episodeId ?? 0)
          : (episodeId ?? 0);
        const href = `/${current.result.repo_id}/episode_${reviewEpisode}`;
        setResultHref(href);
        if (
          latestEdits.current.dirty &&
          JSON.stringify(latestEdits.current.atoms) !== JSON.stringify(atoms)
        ) {
          setStatus(
            "Draft generated. You have new unsaved edits; use the review link after saving them.",
          );
        } else {
          setStatus("Draft generated. Opening the generated episode…");
          window.location.assign(href);
        }
      } else setStatus(current.error || "Annotation failed.");
    } catch (e) {
      if (token === generation.current)
        setStatus(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === generation.current) setBusy(false);
    }
  };
  const ready = Object.keys(config).length > 0 && episodeId !== null;
  let displayed = config;
  try {
    const parsed = JSON.parse(json);
    if (parsed && typeof parsed === "object") displayed = parsed;
  } catch {
    /* keep last valid switches */
  }
  const enabled = (key: string) =>
    (displayed[key] as { enabled?: boolean } | undefined)?.enabled !== false;
  return (
    <section className="annotation-generation" aria-label="Official annotation">
      <div className="composer-copy">
        <h3>Generate annotations</h3>
        <Link href="/annotate">Prepare another dataset</Link>
      </div>
      <div className="generation-actions">
        <button
          type="button"
          className="add-btn"
          onClick={validate}
          disabled={busy || !ready}
        >
          Validate
        </button>
        <button
          type="button"
          className="add-btn"
          onClick={run}
          disabled={busy || !ready}
        >
          {busy ? "Generating…" : "Generate draft"}
        </button>
      </div>
      <form className="generation-fields" onSubmit={(e) => e.preventDefault()}>
        <label className="annotation-field">
          Task prompt
          <input
            type="text"
            value={taskPrompt}
            disabled={busy}
            onInput={(e) => setTaskPrompt(e.currentTarget.value)}
            placeholder={
              hosted ? "Describe the full task" : "Optional task description"
            }
          />
        </label>
        <label className="annotation-field">
          Ordered subtask prompts (one per line)
          <textarea
            value={subtaskPrompts}
            disabled={busy}
            onInput={(e) => setSubtaskPrompts(e.currentTarget.value)}
            rows={3}
          />
        </label>
        <label className="annotation-field">
          Example episodes (comma-separated IDs)
          <input
            type="text"
            value={exampleEpisodes}
            disabled={busy}
            onInput={(e) => setExampleEpisodes(e.currentTarget.value)}
            placeholder="e.g. 0, 3, 7"
            aria-label="Example episodes"
          />
        </label>
        <small>
          Save and review these episodes first. They are preserved and excluded
          from generation; all episodes means all remaining episodes.
        </small>
        <div className="generation-options">
          <label>
            <input
              type="checkbox"
              checked={all}
              disabled={busy}
              onChange={(e) => setAll(e.target.checked)}
            />{" "}
            All episodes
          </label>
          <label>
            <input
              type="checkbox"
              checked={enabled("plan")}
              disabled={!ready || busy}
              onChange={(e) => toggleModule("plan", e.target.checked)}
            />{" "}
            Subtasks, plans, memory and task phrasings
          </label>
          <label>
            <input
              type="checkbox"
              checked={enabled("interjections")}
              disabled={!ready || busy}
              onChange={(e) => toggleModule("interjections", e.target.checked)}
            />{" "}
            Interjections and speech
          </label>
          <label>
            <input
              type="checkbox"
              checked={enabled("vqa")}
              disabled={!ready || busy}
              onChange={(e) => toggleModule("vqa", e.target.checked)}
            />{" "}
            VQA
          </label>
        </div>
        <details>
          <summary>Advanced options</summary>
          <textarea
            aria-label="Official pipeline options"
            rows={7}
            value={json}
            disabled={busy}
            onChange={(e) => setJson(e.target.value)}
          />
        </details>
      </form>
      {status && (
        <div className="save-status" role="status">
          {status}
        </div>
      )}
      {report && (
        <div className="save-status">
          Errors: {report.errors.length}; warnings: {report.warnings.length};
          episodes checked: {report.episodes_checked}
        </div>
      )}
      {report?.errors.map((x) => (
        <div key={`e-${x}`} className="backend-offline">
          Error: {x}
        </div>
      ))}
      {report?.warnings.map((x) => (
        <div key={`w-${x}`} className="save-status">
          Warning: {x}
        </div>
      ))}
      {resultHref && <a href={resultHref}>Review generated dataset</a>}
    </section>
  );
}
