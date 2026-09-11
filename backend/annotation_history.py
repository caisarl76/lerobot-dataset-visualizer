"""Dataset-local prediction snapshots and explicit, content-bound human reviews."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4


def annotation_hash(atoms: list[dict]) -> str:
    # Defaults and ordering can change during an editor save without changing labels.
    normalized = [
        {**dict.fromkeys(("content", "style", "camera", "tool_calls")), **a, "timestamp": float(a["timestamp"])}
        for a in atoms
    ]
    return sha256(
        json.dumps(sorted(normalized, key=lambda a: json.dumps(a, sort_keys=True)), sort_keys=True).encode()
    ).hexdigest()


def exclusions_hash(intervals: list[dict] | None = None) -> str:
    return sha256(json.dumps(intervals or [], sort_keys=True).encode()).hexdigest()


def read_reviews(root: Path) -> dict:
    path = root / "meta/annotation_reviews.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_reviews(root: Path, reviews: dict) -> None:
    path = root / "meta/annotation_reviews.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(reviews, indent=2))
    temporary.replace(path)


def review_status(root: Path, episode: int, atoms: list[dict], excluded_intervals=None) -> dict:
    digest = annotation_hash(atoms)
    review = read_reviews(root).get(str(episode), {})
    clip_digest = exclusions_hash(excluded_intervals)
    reviewed = (
        bool(review.get("reviewed_at"))
        and review.get("annotation_sha256") == digest
        and review.get("exclusions_sha256", exclusions_hash()) == clip_digest
    )
    available = any(
        str(episode) in json.loads(path.read_text())["episodes"]
        for path in (root / "meta/annotation_predictions").glob("*.json")
    )
    return {
        "status": "reviewed" if reviewed else "unreviewed",
        "annotation_sha256": digest,
        "exclusions_sha256": clip_digest,
        "reviewed_at": review.get("reviewed_at") if reviewed else None,
        "prediction_available": available,
    }


def save_review(
    root: Path, episode: int, atoms: list[dict], reviewed: bool, expected_hash: str, *, excluded_intervals=None
) -> dict:
    if annotation_hash(atoms) != expected_hash:
        raise ValueError("Annotations changed. Reload the episode before marking it reviewed.")
    reviews = read_reviews(root)
    reviews[str(episode)] = {
        "annotation_sha256": expected_hash,
        "exclusions_sha256": exclusions_hash(excluded_intervals),
        "reviewed_at": datetime.now(timezone.utc).isoformat() if reviewed else None,
    }
    write_reviews(root, reviews)
    return review_status(root, episode, atoms, excluded_intervals)


def snapshot_predictions(
    root: Path,
    source: Path,
    selected: set[int],
    examples: dict,
    config: dict,
    revision: str,
    *,
    task_prompt: str = "",
    subtask_prompts: list[str] | None = None,
    assess_quality: bool = False,
    episode_results: dict | None = None,
) -> Path:
    """Called once after generation, before a draft is exposed to the editor."""
    config = json.loads(json.dumps(config, default=str))
    for key in ("api_key", "serve_command"):
        config.get("vlm", {}).pop(key, None)
    sidecar = root / "meta/lerobot_annotations.json"
    episodes = json.loads(sidecar.read_text())["episodes"] if selected or sidecar.exists() else {}
    directory = root / "meta/annotation_predictions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x") as stream:
        json.dump(
            {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": str(source),
                "lerobot_revision": revision,
                "config": config,
                "task_prompt": task_prompt,
                "subtask_prompts": subtask_prompts or [],
                "assess_quality": assess_quality,
                "episode_results": episode_results or {},
                "original_episode_indices": {str(ep): ep for ep in sorted(selected)},
                "semantics": "Official pipeline output after postprocessing, not raw VLM replies",
                "example_episode_indices": sorted(examples),
                "examples": {
                    str(ep): {"atoms": atoms, "annotation_sha256": annotation_hash(atoms)}
                    for ep, atoms in examples.items()
                },
                "episodes": {str(ep): episodes[str(ep)] for ep in sorted(selected)},
            },
            stream,
            indent=2,
        )
    temporary.replace(path)
    reviews = read_reviews(root)
    for episode in selected:
        reviews.pop(str(episode), None)
    write_reviews(root, reviews)
    return path
