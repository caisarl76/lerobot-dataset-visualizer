"""Export causal rich-language instructions onto an original GR00T v2.1 dataset.

The v3 and v2.1 inputs must contain exactly the same episode/frame/timestamp
identities. Videos are hardlinked when possible; treat both datasets' media as
immutable. No model or trainer changes are required. Context must also be
available at inference; this adapter does not train language generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


def _json(path: Path):
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def _atoms(rows: list[dict]) -> list[dict]:
    """Deduplicate broadcast persistent lists; events use their own frame time."""
    unique = {}
    for row in rows:
        for column in ("language_persistent", "language_events"):
            for raw in row.get(column) or []:
                atom = dict(raw)
                if column == "language_events" and atom.get("timestamp") is None:
                    atom["timestamp"] = row["timestamp"]
                unique[json.dumps(atom, sort_keys=True)] = atom
    return list(unique.values())


def _prepare_atoms(atoms: list[dict]) -> list[dict]:
    usable = []
    for atom in atoms:
        style = atom.get("style")
        if style not in {"task_aug", "subtask", "plan", "memory", "interjection"}:
            continue
        if atom.get("tool_calls"):
            continue
        if style in {"task_aug", "interjection"} and atom.get("role") != "user":
            continue
        timestamp = atom.get("timestamp")
        if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp):
            raise ValueError(f"Invalid timestamp for {style} atom")
        if not isinstance(atom.get("content"), str) or not atom["content"].strip():
            raise ValueError(f"Missing text for {style} atom")
        usable.append(atom)
    return usable


def _active(atoms: list[dict], style: str, timestamp: float) -> str | None:
    # Same latest-timestamp <= t semantics as LeRobot's active_at resolver.
    matches = [a for a in atoms if a["style"] == style and a["timestamp"] <= timestamp]
    if not matches:
        return None
    latest = max(a["timestamp"] for a in matches)
    matches = [a for a in matches if a["timestamp"] == latest]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous {style} atoms at timestamp {latest}")
    return matches[0]["content"]


def _instruction(task: str, timestamp: float, atoms: list[dict], mode: str, task_variant: int | None) -> str:
    if task_variant is not None:
        variants = [a for a in atoms if a["style"] == "task_aug"]
        if task_variant >= len(variants):
            raise ValueError(f"task_variant {task_variant} is unavailable ({len(variants)} variants)")
        variant = variants[task_variant]
        if variant["timestamp"] <= timestamp:
            task = variant["content"]
    if mode == "task":
        return task
    subtask = _active(atoms, "subtask", timestamp)
    if mode == "subtask":
        return subtask or task
    parts = [f"Task: {task}"]
    for style in ("subtask", "plan", "memory", "interjection"):
        content = subtask if style == "subtask" else _active(atoms, style, timestamp)
        if content:
            parts.append(f"{style.capitalize()}: {content}")
    return "\n".join(parts)


def export_groot_dataset(
    annotated_root: Path,
    source_root: Path,
    output_root: Path,
    mode: str = "subtask",
    task_variant: int | None = None,
) -> dict:
    """Create a new v2.1 training view; saved per-episode sidecar edits win.

    ``task_variant`` is a zero-based index into episode task_aug atoms in stored
    order. Before that variant's timestamp, the original frame task is used.
    Missing active subtasks also fall back to the original frame task.
    """
    if mode not in {"task", "subtask", "context"}:
        raise ValueError("mode must be task, subtask, or context")
    if task_variant is not None and (type(task_variant) is not int or task_variant < 0):
        raise ValueError("task_variant must be a nonnegative integer")
    annotated_root, source_root, output_root = (
        Path(path).expanduser().resolve() for path in (annotated_root, source_root, output_root)
    )
    for source in (annotated_root, source_root):
        if output_root.is_relative_to(source) or source.is_relative_to(output_root):
            raise ValueError("Output must be separate from both source datasets")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    info = _json(source_root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("source_root must be the original GR00T-compatible v2.1 dataset")
    if not _json(annotated_root / "meta/info.json").get("codebase_version", "").startswith("v3"):
        raise ValueError("annotated_root must be a v3 dataset")
    modality = _json(source_root / "meta/modality.json")
    stats = _json(source_root / "meta/stats.json")
    episodes = _jsonl(source_root / "meta/episodes.jsonl")
    episode_ids = [row["episode_index"] for row in episodes]
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Missing or duplicate source episodes")
    source_tasks = _jsonl(source_root / "meta/tasks.jsonl")
    original_tasks = {row["task_index"]: row["task"] for row in source_tasks}
    if len(original_tasks) != len(source_tasks):
        raise ValueError("Duplicate source task indices")

    # ponytail: keep language/identity rows in memory; stream per episode if this
    # becomes too large. Images, actions and states from v3 are never loaded.
    annotated: dict[int, list[dict]] = {}
    for path in sorted((annotated_root / "data").rglob("*.parquet")):
        names = pq.read_schema(path).names
        columns = ["episode_index", "frame_index", "timestamp"]
        columns += [name for name in ("language_persistent", "language_events") if name in names]
        for row in pq.read_table(path, columns=columns).to_pylist():
            annotated.setdefault(row["episode_index"], []).append(row)
    metadata_ids = []
    for path in sorted((annotated_root / "meta/episodes").rglob("*.parquet")):
        metadata_ids.extend(pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist())
    if set(annotated) != set(episode_ids) or set(metadata_ids) != set(episode_ids):
        raise ValueError("Annotated/source episode sets do not match")
    if len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("Duplicate annotated episode metadata")
    sidecar_path = annotated_root / "meta/lerobot_annotations.json"
    sidecar = _json(sidecar_path).get("episodes", {}) if sidecar_path.exists() else {}
    if not set(sidecar).issubset({str(ep) for ep in episode_ids}):
        raise ValueError("Sidecar contains unknown episodes")

    tasks: dict[str, int] = {}
    replacements = {}
    frame_count = 0
    for episode in episodes:
        ep = episode["episode_index"]
        path = _within(
            source_root, info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        )
        table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", "task_index"])
        rows = table.to_pylist()
        rich_rows = sorted(annotated[ep], key=lambda row: row["frame_index"])
        expected_indices = list(range(episode["length"]))
        if (
            len(rows) != episode["length"]
            or [r["frame_index"] for r in rows] != expected_indices
            or [r["frame_index"] for r in rich_rows] != expected_indices
        ):
            raise ValueError(f"Missing, duplicate or out-of-order frames in episode {ep}")
        for original, rich in zip(rows, rich_rows, strict=True):
            if (
                original["episode_index"] != ep
                or not math.isfinite(original["timestamp"])
                or original["timestamp"] != rich["timestamp"]
            ):
                raise ValueError(f"Exact episode/frame/timestamp mismatch in episode {ep}")
        atoms = sidecar[str(ep)]["atoms"] if str(ep) in sidecar else _atoms(rich_rows)
        atoms = _prepare_atoms(atoms)
        indices, episode_tasks = [], []
        for row in rows:
            task = original_tasks.get(row["task_index"])
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"Missing source task for episode {ep}")
            text = _instruction(task, row["timestamp"], atoms, mode, task_variant)
            indices.append(tasks.setdefault(text, len(tasks)))
            if text not in episode_tasks:
                episode_tasks.append(text)
        episode["tasks"] = episode_tasks
        replacements[path.relative_to(source_root)] = indices
        frame_count += len(rows)
    source_data = {p.resolve() for p in (source_root / "data").rglob("*.parquet")}
    if source_data != {source_root / rel for rel in replacements}:
        raise ValueError("Source contains missing or unlisted episode parquet files")
    if info.get("total_frames", frame_count) != frame_count or info.get("total_episodes", len(episodes)) != len(
        episodes
    ):
        raise ValueError("Source metadata frame/episode counts do not match")

    files = sorted(source_root.rglob("*"))
    if any(path.is_symlink() for path in files):
        raise ValueError("Source symlinks are unsupported; use a materialized v2.1 dataset")
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        for path in files:
            relative = path.relative_to(source_root)
            target = output_root / relative
            if path.is_dir():
                target.mkdir(exist_ok=True)
            elif relative.parts[0] == "videos":
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
            else:
                shutil.copy2(path, target)
        for relative, indices in replacements.items():
            table = pq.read_table(source_root / relative)
            index = table.schema.get_field_index("task_index")
            table = table.set_column(
                index, table.schema.field(index), pa.array(indices, type=table.schema.field(index).type)
            )
            pq.write_table(table, output_root / relative)
        _write_jsonl(
            output_root / "meta/tasks.jsonl",
            [{"task_index": index, "task": text} for text, index in tasks.items()],
        )
        _write_jsonl(output_root / "meta/episodes.jsonl", episodes)
        annotation = modality.setdefault("annotation", {}).setdefault("human.task_description", {})
        annotation["original_key"] = "task_index"
        _write_json(output_root / "meta/modality.json", modality)
        info["total_tasks"] = len(tasks)
        _write_json(output_root / "meta/info.json", info)
        # GR00T normalizes state/action, never categorical task indices.
        if "task_index" in stats:
            stats.pop("task_index")
            _write_json(output_root / "meta/stats.json", stats)
        episode_stats_path = output_root / "meta/episodes_stats.jsonl"
        if episode_stats_path.exists():
            episode_stats = _jsonl(episode_stats_path)
            if any("task_index" in row.get("stats", {}) for row in episode_stats):
                for row in episode_stats:
                    row.get("stats", {}).pop("task_index", None)
                _write_jsonl(episode_stats_path, episode_stats)
        report = {
            "output_dir": str(output_root),
            "annotated_root": str(annotated_root),
            "source_root": str(source_root),
            "mode": mode,
            "task_variant": task_variant,
            "episodes": len(episodes),
            "frames": frame_count,
            "tasks": len(tasks),
        }
        _write_json(output_root / "meta/groot_instruction_export.json", report)
    except BaseException:
        shutil.rmtree(output_root)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("task", "subtask", "context"), default="subtask")
    parser.add_argument("--task-variant", type=int)
    args = parser.parse_args()
    print(json.dumps(export_groot_dataset(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
