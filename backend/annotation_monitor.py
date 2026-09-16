"""Read-only discovery and provenance for local LeRobot datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _dataset_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]


def _diag(code: str, message: str, severity: str = "warning") -> dict:
    return {"code": code, "message": message, "severity": severity}


def _episode_rows(root: Path) -> tuple[list[dict], list[dict]]:
    diagnostics: list[dict] = []
    legacy = root / "meta/episodes.jsonl"
    rows: list[dict] = []
    if legacy.exists():
        for number, line in enumerate(legacy.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("episode_index"), int) or value["episode_index"] < 0:
                    raise ValueError("missing integer episode_index")
                rows.append(value)
            except (json.JSONDecodeError, ValueError) as exc:
                diagnostics.append(_diag("invalid_episode_metadata", f"line {number}: {exc}"))
        ids = [r["episode_index"] for r in rows]
        if len(ids) != len(set(ids)):
            diagnostics.append(_diag("duplicate_episode_id", "duplicate episode_index values"))
        return rows, diagnostics
    try:
        import pyarrow.parquet as pq
        for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "length"])
            data = table.to_pylist()
            rows.extend(r for r in data if isinstance(r.get("episode_index"), int) and r["episode_index"] >= 0)
    except Exception as exc:
        diagnostics.append(_diag("invalid_episode_metadata", str(exc), "error"))
    return rows, diagnostics


def discover_datasets(root: Path, workspace: Path) -> list[dict]:
    root, workspace = Path(root), Path(workspace)
    if not root.exists():
        return [{"id": _dataset_id(root), "name": root.name, "path": str(root),
                 "canonical_path": str(root.resolve()), "state": "Updating", "collected": 0,
                 "reported_collected": None, "episode_lengths": {}, "diagnostics": [_diag("missing_root", str(root), "error")]}]
    candidates = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    rows: list[dict] = []
    seen: set[str] = set()
    for path in candidates:
        canonical = path.resolve()
        if str(canonical) in seen or not _inside(path, root) or _inside(path, workspace) or not (path / "meta").exists():
            continue
        seen.add(str(canonical))
        diagnostics: list[dict] = []
        try:
            info = json.loads((path / "meta/info.json").read_text()) if (path / "meta/info.json").exists() else {}
            episodes, diagnostics = _episode_rows(path)
            lengths = {str(r["episode_index"]): r.get("length") for r in episodes if isinstance(r.get("length"), int)}
            if len(lengths) != len(episodes):
                diagnostics.append(_diag("invalid_episode_metadata", "episode metadata contains invalid rows"))
            rows.append({"id": _dataset_id(path), "name": path.name, "path": str(path), "canonical_path": str(canonical),
                         "state": "Ready" if not diagnostics else "Updating", "collected": len(episodes),
                         "reported_collected": info.get("total_episodes"), "episode_lengths": lengths,
                         "parent_ids": [], "child_ids": [], "provenance": {"status": "unknown", "sources": []},
                         "diagnostics": diagnostics, "runs": [], "publications": [], "exports": []})
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            rows.append({"id": _dataset_id(path), "name": path.name, "path": str(path), "canonical_path": str(canonical),
                         "state": "Updating", "collected": 0, "reported_collected": None, "episode_lengths": {},
                         "parent_ids": [], "child_ids": [], "provenance": {"status": "unknown", "sources": []},
                         "diagnostics": [_diag("invalid_metadata", str(exc), "error")], "runs": [], "publications": [], "exports": []})
    return rows


def read_runs(workspace: Path) -> list[dict]:
    result = []
    for path in sorted((Path(workspace) / "runs").glob("*/run.json")):
        try:
            value = json.loads(path.read_text())
            if isinstance(value, dict) and value.get("run_id"):
                value["_path"] = str(path)
                result.append(value)
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
    return result


def attach_relationships(datasets: list[dict], runs: list[dict]) -> list[dict]:
    by_path = {d["canonical_path"]: d for d in datasets}
    for dataset in datasets:
        dataset.setdefault("runs", [])
        matching = [r for r in runs if str(Path(r.get("source_root", r.get("root", ""))).resolve()) == dataset["canonical_path"]]
        matching.sort(key=lambda r: (-(Path(r.get("_path", "")).stat().st_mtime_ns if Path(r.get("_path", "")).exists() else 0), r["run_id"]))
        dataset["runs"] = matching
        dataset["default_run_id"] = matching[0]["run_id"] if matching else None
        if matching:
            dataset["provenance"] = {"status": "confirmed", "sources": [dataset["canonical_path"]]}
        sources: list[str] = []
        for key in ("source_root", "root", "original_root", "source_root"):
            value = next((r.get(key) for r in matching if r.get(key)), None)
            if value:
                candidate = str(Path(value).resolve())
                if candidate != dataset["canonical_path"] and candidate in by_path:
                    sources.append(candidate)
        if sources:
            dataset["provenance"] = {"status": "confirmed", "sources": sorted(set(sources))}
    return datasets


def summarize_run(dataset: dict, run: dict, workspace: Path) -> dict:
    return {"run_id": run.get("run_id"), "updated_at": run.get("updated_at"), "current_repo_id": run.get("repo_id"),
            "first_retained_episode": None, "detail_signature": None, "metrics": {}, "job": run.get("job"),
            "freshness": {}, "diagnostics": [], "findings": {"unresolved": 0, "accepted_advisory": 0, "generation_failed": 0, "unreadable": 0},
            "exclusions": {"episodes": None, "frames": None}}


def prompt_distribution(run: dict) -> dict:
    return {"eligible_episodes": None, "evaluated_episodes": None, "unknown_episode_ids": [], "retained_frames": None,
            "unlabeled_frames": None, "ambiguous_frames": None, "complete": False, "rows": []}


def publication_records(run: dict, workspace: Path) -> list[dict]:
    return []


def check_publication(publication: dict, api) -> dict:
    return {"status": "unavailable", "checked_at": None, "current_commit": None, "message": "not checked"}
