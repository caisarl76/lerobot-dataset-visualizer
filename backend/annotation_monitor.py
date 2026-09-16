"""Read-only discovery and recorded lineage for local LeRobot datasets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_PROVENANCE_FILES = ("source_episode_mapping.json", "merge_manifest.json", "groot_instruction_export.json")
_INFRASTRUCTURE = {"cache", "exports", "staging", "tmp", "temp", "drafts", "runs", "jobs"}
_READ_ERRORS = (OSError, ValueError, TypeError, UnicodeError, RuntimeError)


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _dataset_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]


def _diag(code: str, message: str, severity: str = "warning") -> dict:
    return {"code": code, "message": message, "severity": severity}


def _nonnegative_int(value) -> bool:
    return type(value) is int and value >= 0


def _canonical(value) -> str | None:
    # Relative paths and sanitized Hub identities are not local source evidence.
    if not isinstance(value, str) or not value or "\x00" in value or not Path(value).is_absolute():
        return None
    try:
        return str(Path(value).resolve())
    except _READ_ERRORS:
        return None


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _signature(root: Path, paths, *, follow_symlinks: bool = True) -> tuple:
    """Directory entries and relevant file stats, without reading frame/video data."""
    entries = []
    for path in sorted(set(paths)):
        try:
            stat = path.stat(follow_symlinks=follow_symlinks)
            entries.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            entries.append((str(path.relative_to(root)), None, None))
    return tuple(entries)


def _metadata_signature(root: Path) -> tuple:
    meta = root / "meta"
    paths = [meta, meta / "info.json", meta / "episodes.jsonl", meta / "episodes"]
    paths.extend(meta / name for name in _PROVENANCE_FILES)
    if (meta / "episodes").is_dir():
        paths.extend((meta / "episodes").rglob("*"))
    return _signature(root, paths)


def _provenance_documents(root: Path, diagnostics: list) -> dict:
    documents = {}
    for name in _PROVENANCE_FILES:
        path = root / "meta" / name
        if path.exists():
            try:
                documents[name] = _json_object(path)
            except _READ_ERRORS as exc:
                diagnostics.append(_diag("invalid_provenance", f"{path}: {exc}"))
    return documents


def _episode_lengths(root: Path, info: dict, diagnostics: list) -> dict:
    records = []
    legacy = root / "meta/episodes.jsonl"
    version = info.get("codebase_version")
    if version == "v2.1" or (version is None and legacy.exists()):
        try:
            for number, line in enumerate(legacy.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append((f"{legacy}:{number}", json.loads(line)))
                except ValueError as exc:
                    diagnostics.append(_diag("invalid_episode_metadata", f"{legacy}:{number}: {exc}"))
        except _READ_ERRORS as exc:
            diagnostics.append(_diag("invalid_episode_metadata", f"{legacy}: {exc}"))
    else:
        import pyarrow.parquet as pq

        paths = sorted((root / "meta/episodes").rglob("*.parquet"))
        if not paths and info.get("total_episodes") != 0:
            diagnostics.append(_diag("missing_episode_metadata", f"{root}: no episode metadata files"))
        for path in paths:
            try:
                records.extend(
                    (str(path), row)
                    for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist()
                )
            except _READ_ERRORS as exc:
                diagnostics.append(_diag("invalid_episode_metadata", f"{path}: {exc}"))
    lengths = {}
    for location, row in records:
        if (
            not isinstance(row, dict)
            or not _nonnegative_int(row.get("episode_index"))
            or not _nonnegative_int(row.get("length"))
        ):
            diagnostics.append(
                _diag(
                    "invalid_episode_metadata",
                    f"{location}: expected nonnegative integer episode_index and length",
                )
            )
            continue
        key = str(row["episode_index"])
        if key in lengths:
            diagnostics.append(_diag("duplicate_episode_id", f"{location}: duplicate episode_index {key}"))
        else:
            lengths[key] = row["length"]
    return lengths


def _empty_dataset(path: Path) -> dict:
    return {
        "id": _dataset_id(path),
        "name": path.name,
        "path": str(path),
        "canonical_path": str(path.resolve()),
        "state": "Ready",
        "collected": 0,
        "reported_collected": None,
        "episode_lengths": {},
        "parent_ids": [],
        "child_ids": [],
        "provenance": {"status": "unknown", "sources": []},
        "default_run_id": None,
        "diagnostics": [],
        "runs": [],
        "publications": [],
        "exports": [],
    }


def _read_dataset(path: Path) -> dict:
    for attempt in range(2):
        row = _empty_dataset(path)
        diagnostics = row["diagnostics"]
        try:
            before = _metadata_signature(path)
            info = {}
            try:
                info = _json_object(path / "meta/info.json")
                if info.get("codebase_version") not in {"v2.1", "v3.0"}:
                    raise ValueError("missing or unsupported codebase_version")
                if not _nonnegative_int(info.get("total_episodes")):
                    raise ValueError("total_episodes must be a nonnegative integer")
                if not _nonnegative_int(info.get("total_frames")):
                    raise ValueError("total_frames must be a nonnegative integer")
                if type(info.get("fps")) not in (int, float) or not 0 < info["fps"] < float("inf"):
                    raise ValueError("fps must be a positive finite number")
                if not isinstance(info.get("features"), dict):
                    raise ValueError("features must be an object")
            except _READ_ERRORS as exc:
                diagnostics.append(_diag("invalid_metadata", f"{path}/meta/info.json: {exc}", "error"))
            row["reported_collected"] = (
                info.get("total_episodes") if _nonnegative_int(info.get("total_episodes")) else None
            )
            row["episode_lengths"] = _episode_lengths(path, info, diagnostics)
            row["collected"] = len(row["episode_lengths"])
            if row["reported_collected"] is not None and row["collected"] != row["reported_collected"]:
                diagnostics.append(
                    _diag(
                        "episode_count_mismatch",
                        f"{path}: metadata reports {row['reported_collected']} episodes; "
                        f"{row['collected']} valid unique records read",
                    )
                )
            row["_provenance_documents"] = _provenance_documents(path, diagnostics)
            after = _metadata_signature(path)
            row["_signature"] = after
            if before != after:
                if attempt == 0:
                    continue
                diagnostics.append(
                    _diag("metadata_changing", f"{path}: metadata changed during both read attempts")
                )
        except _READ_ERRORS as exc:
            diagnostics.append(_diag("invalid_metadata", f"{path}: {exc}", "error"))
        row["state"] = "Updating" if diagnostics else "Ready"
        return row


def discover_datasets(root: Path, workspace: Path) -> list[dict]:
    root, workspace = Path(root), Path(workspace)
    excluded = [
        workspace,
        Path(os.environ.get("LEROBOT_ANNOTATE_CACHE", "/tmp/lerobot_visualizer_annotate_cache")),
    ]
    if os.environ.get("LEROBOT_ANNOTATE_EXPORT"):
        excluded.append(Path(os.environ["LEROBOT_ANNOTATE_EXPORT"]))
    try:
        for attempt in range(2):
            children = sorted(root.iterdir())
            before = _signature(root, [root, *children], follow_symlinks=False)
            rows, seen = [], set()
            for path in children:
                try:
                    if (
                        path.name.startswith(".")
                        or path.name.lower() in _INFRASTRUCTURE
                        or path.name.endswith((".tmp", ".staging"))
                    ):
                        continue
                    canonical = path.resolve()
                    if not path.is_dir() or canonical in seen or not _inside(path, root):
                        continue
                    if (
                        canonical.name.startswith(".")
                        or canonical.name.lower() in _INFRASTRUCTURE
                        or canonical.name.endswith((".tmp", ".staging"))
                        or any(_inside(path, x) for x in excluded)
                    ):
                        continue
                    if not (path / "meta").exists() and not (path / "data").is_dir():
                        continue
                    seen.add(canonical)
                    rows.append(_read_dataset(path))
                except _READ_ERRORS as exc:
                    # A malformed candidate must not hide its healthy siblings.
                    if not path.is_symlink():
                        row = _empty_dataset(path)
                        row.update(
                            state="Updating", diagnostics=[_diag("invalid_metadata", f"{path}: {exc}", "error")]
                        )
                        rows.append(row)
            after = _signature(root, [root, *root.iterdir()], follow_symlinks=False)
            if before == after:
                return rows
            if attempt == 1:
                for row in rows:
                    row["state"] = "Updating"
                    row["diagnostics"].append(
                        _diag(
                            "collection_changing", f"{root}: directory listing changed during both read attempts"
                        )
                    )
                if rows:
                    return rows
                raise ValueError("collection changed during both read attempts")
    except _READ_ERRORS as exc:
        row = _empty_dataset(root)
        row.update(state="Updating", diagnostics=[_diag("root_unavailable", f"{root}: {exc}", "error")])
        return [row]


def _run_error(value: dict) -> str | None:
    if not isinstance(value, dict) or not isinstance(value.get("run_id"), str) or not value["run_id"].strip():
        return "missing string run_id"
    if not _canonical(value.get("source_root")) or not _canonical(value.get("root")):
        return "source_root and root must be absolute local paths"
    if not isinstance(value.get("episodes"), dict):
        return "episodes must be an episode-keyed object"
    return None


def _run_failure(path: Path, exc, source=None, code="invalid_run") -> dict:
    return {
        "_path": str(path),
        "_source_root": _canonical(source),
        "_diagnostic": _diag(code, f"{path}: {exc}", "error"),
    }


def read_runs(workspace: Path) -> list[dict]:
    root = Path(workspace) / "runs"
    for attempt in range(2):
        result = []
        try:
            paths = sorted(root.glob("*/run.json"))
            before = _signature(root, [root, *root.glob("*"), *paths])
            for path in paths:
                value = {}
                try:
                    if not _inside(path, root):
                        raise ValueError("run record escapes workspace")
                    value = _json_object(path)
                    error = _run_error(value)
                    if error:
                        raise ValueError(error)
                    value["_path"] = str(path)
                    value["_mtime_ns"] = path.stat().st_mtime_ns
                    result.append(value)
                except _READ_ERRORS as exc:
                    result.append(_run_failure(path, exc, value.get("source_root")))
            after_paths = sorted(root.glob("*/run.json"))
            after = _signature(root, [root, *root.glob("*"), *after_paths])
            if before == after:
                return result
            if attempt == 1:
                # Never attach a run whose identity may have changed mid-read.
                return [_run_failure(root, "run records changed during both read attempts", code="runs_changing")]
        except _READ_ERRORS as exc:
            return [_run_failure(root, exc)]


def attach_relationships(datasets: list[dict], runs: list[dict]) -> list[dict]:
    by_path = {d["canonical_path"]: d for d in datasets}
    valid_runs, failures = [], []
    for run in runs:
        if isinstance(run, dict) and run.get("_diagnostic"):
            failures.append(run)
        elif error := _run_error(run):
            value = run if isinstance(run, dict) else {}
            failures.append(_run_failure(Path(value.get("_path") or "run.json"), error, value.get("source_root")))
        else:
            valid_runs.append(run)
    runs_by_id = {}
    for run in valid_runs:
        runs_by_id.setdefault(run["run_id"], []).append(run)
    documents = {path: row.get("_provenance_documents", {}) for path, row in by_path.items()}
    external_diagnostics = {}

    def get_documents(path):
        if path not in documents:
            for attempt in range(2):
                diagnostics = []
                try:
                    before = _metadata_signature(Path(path))
                    docs = _provenance_documents(Path(path), diagnostics)
                    after = _metadata_signature(Path(path))
                    if before != after:
                        if attempt == 0:
                            continue
                        diagnostics.append(
                            _diag("metadata_changing", f"{path}: provenance changed during both read attempts")
                        )
                        docs = {}
                    documents[path] = docs
                except _READ_ERRORS as exc:
                    diagnostics.append(_diag("invalid_provenance", f"{path}: {exc}"))
                    documents[path] = {}
                external_diagnostics[path] = diagnostics
                break
        return documents[path]

    def evidence(path, diagnostics):
        groups = []
        docs = get_documents(path)
        diagnostics.extend(external_diagnostics.get(path, []))
        mapping = docs.get("source_episode_mapping.json", {})
        if "original_root" in mapping:
            groups.append([mapping["original_root"]])
        if "run_id" in mapping:
            run_id = mapping["run_id"]
            matches = runs_by_id.get(run_id, []) if isinstance(run_id, str) else []
            if matches:
                groups.extend([run["source_root"]] for run in matches)
            else:
                groups.append([None])
        manifest = docs.get("merge_manifest.json")
        if manifest is not None:
            sources = manifest.get("sources")
            if (
                not isinstance(sources, list)
                or not sources
                or any(not isinstance(s, dict) or "root" not in s for s in sources)
            ):
                diagnostics.append(_diag("invalid_provenance", f"{path}: merge sources must contain root objects"))
                groups.append([None])
            else:
                groups.append([s["root"] for s in sources])
        report = docs.get("groot_instruction_export.json", {})
        if "source_root" in report:
            groups.append([report["source_root"]])
        groups.extend(
            [run["source_root"]]
            for run in valid_runs
            if _canonical(run["root"]) == path and _canonical(run["source_root"]) != path
        )
        return groups

    def resolve(path, stack, diagnostics):
        if path in stack:
            diagnostics.append(_diag("provenance_cycle", f"Cycle in provenance at {path}"))
            return set(), True, False
        groups = evidence(path, diagnostics)
        resolved, conflict = [], False
        unresolved = any(d["code"] == "invalid_provenance" for d in diagnostics)
        for group in groups:
            sources = set()
            complete = True
            for value in group:
                candidate = _canonical(value)
                if candidate is None:
                    complete = False
                elif candidate in by_path:
                    sources.add(candidate)
                else:
                    upstream, bad, unknown = resolve(candidate, stack | {path}, diagnostics)
                    conflict |= bad
                    sources.update(upstream)
                    complete &= bool(upstream) and not bad and not unknown
            if complete and sources:
                resolved.append(sources)
            else:
                unresolved = True
                diagnostics.append(
                    _diag("unresolved_provenance", f"{path}: recorded source could not be confirmed")
                )
        if resolved and any(sources != resolved[0] for sources in resolved[1:]):
            conflict = True
            diagnostics.append(_diag("conflicting_provenance", f"{path}: recorded source identities disagree"))
        return set().union(*resolved), conflict, unresolved

    for path, row in by_path.items():
        row["parent_ids"], row["child_ids"] = [], []
        row["runs"] = sorted(
            (r for r in valid_runs if _canonical(r["source_root"]) == path),
            key=lambda r: (-r.get("_mtime_ns", 0), r["run_id"]),
        )
        row["default_run_id"] = row["runs"][0]["run_id"] if row["runs"] else None
        row["_run_diagnostics"] = [
            r["_diagnostic"] for r in failures if not r.get("_source_root") or r["_source_root"] not in by_path
        ]
        for failure in failures:
            if failure.get("_source_root") == path:
                row["diagnostics"].append(failure["_diagnostic"])
        sources, conflict, unresolved = resolve(path, set(), row["diagnostics"])
        row["provenance"] = {
            "status": "conflict" if conflict else "confirmed" if sources and not unresolved else "unknown",
            "sources": sorted(sources),
        }

    # Detect cycles across discovered datasets before allowing any nesting.
    graph = {p: set(r["provenance"]["sources"]) for p, r in by_path.items()}
    cyclic = set()

    def visit(path, stack, done):
        if path in stack:
            cyclic.update(stack[stack.index(path) :])
        elif path not in done:
            for parent in graph[path]:
                visit(parent, [*stack, path], done)
            done.add(path)

    done = set()
    for path in graph:
        visit(path, [], done)
    for path, row in by_path.items():
        if path in cyclic:
            row["provenance"]["status"] = "conflict"
            row["diagnostics"].append(_diag("provenance_cycle", f"{path}: cyclic recorded parentage"))
        if row["provenance"]["status"] == "confirmed":
            row["parent_ids"] = [by_path[p]["id"] for p in sorted(graph[path])]
            if len(row["parent_ids"]) == 1:
                by_path[next(iter(graph[path]))]["child_ids"].append(row["id"])
        if any(d["severity"] != "info" for d in row["diagnostics"]):
            row["state"] = "Updating"
    return datasets
