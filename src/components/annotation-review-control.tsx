"use client";

import React from "react";
import { useAnnotations } from "../context/annotations-context";
import {
  AnnotationRequestError,
  fetchEpisodeReview,
  setEpisodeReview,
  type EpisodeReview,
} from "../utils/annotationsClient";

export function AnnotationReviewControl() {
  const {
    episodeId,
    ident,
    atoms,
    dirty,
    saving,
    backendEnabled,
    annotationSha256,
  } = useAnnotations();
  const [review, setReview] = React.useState<EpisodeReview | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const requestId = React.useRef(0);
  const [clipRevision, setClipRevision] = React.useState(0);
  React.useEffect(() => {
    const changed = () => setClipRevision((value) => value + 1);
    window.addEventListener("annotation-clipping-changed", changed);
    return () =>
      window.removeEventListener("annotation-clipping-changed", changed);
  }, []);

  React.useEffect(() => {
    const id = ++requestId.current;
    if (!backendEnabled || episodeId == null || dirty || !annotationSha256) {
      setReview(null);
      setLoading(false);
      setError(null);
      return () => {
        requestId.current += 1;
      };
    }
    setLoading(true);
    setError(null);
    fetchEpisodeReview(episodeId, ident)
      .then((value) => {
        if (id !== requestId.current) return;
        if (value.annotation_sha256 !== annotationSha256) {
          setReview(null);
          setError(
            "Annotations changed in another tab. Reload the episode before reviewing.",
          );
        } else setReview(value);
      })
      .catch((cause) => {
        if (id !== requestId.current) return;
        setReview(null);
        setError(
          cause instanceof Error
            ? cause.message
            : "Unable to load review status.",
        );
      })
      .finally(() => {
        if (id === requestId.current) setLoading(false);
      });
    return () => {
      requestId.current += 1;
    };
  }, [
    backendEnabled,
    annotationSha256,
    episodeId,
    ident,
    atoms,
    dirty,
    clipRevision,
  ]);

  const markReviewed = async () => {
    if (
      !review ||
      episodeId == null ||
      dirty ||
      saving ||
      !annotationSha256 ||
      review.annotation_sha256 !== annotationSha256
    )
      return;
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const next = await setEpisodeReview(
        episodeId,
        ident,
        review.status !== "reviewed",
        annotationSha256,
        review.exclusions_sha256,
      );
      if (id === requestId.current) setReview(next);
    } catch (cause) {
      if (id !== requestId.current) return;
      setError(
        cause instanceof AnnotationRequestError && cause.status === 409
          ? "Review is out of date; reload the current episode before changing it."
          : cause instanceof Error
            ? cause.message
            : "Unable to update review status.",
      );
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  };

  if (!backendEnabled || episodeId == null) return null;
  return (
    <div className="flex items-center gap-2 text-xs" aria-live="polite">
      <span>
        {dirty
          ? "Save changes before reviewing"
          : loading
            ? "Review status loading…"
            : review?.status === "reviewed"
              ? "Reviewed"
              : "Unreviewed"}
      </span>
      {review && (
        <button
          type="button"
          disabled={loading || saving || dirty}
          onClick={markReviewed}
          className="h-7 px-3 rounded border border-violet-500/40 bg-violet-500/10 text-violet-200 hover:bg-violet-500/20 disabled:opacity-40"
        >
          {review.status === "reviewed" ? "Reopen review" : "Mark reviewed"}
        </button>
      )}
      {error && <span className="text-red-300">{error}</span>}
    </div>
  );
}
