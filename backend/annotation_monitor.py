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


def _valid_episode_key(key) -> bool:
    return isinstance(key, str) and key.isascii() and key.isdigit() and str(int(key)) == key


def _checkpoint_signature(root: Path) -> tuple:
    paths = [root / "meta/lerobot_annotations.json", root / "meta/annotation_reviews.json", root / "data"]
    if (root / "data").is_dir():
        paths.extend((root / "data").rglob("*.parquet"))
    return _metadata_signature(root), _signature(root, paths)


def _saved_object(path: Path, diagnostics: list) -> dict | None:
    try:
        return _json_object(path) if path.exists() else {}
    except _READ_ERRORS as exc:
        diagnostics.append(_diag("invalid_review_evidence", f"{path}: {exc}"))
        return None


def _sidecar_atoms(payload: dict) -> list:
    """Match the editor's saved v2 atoms and legacy v1 conversion."""
    if not isinstance(payload, dict):
        raise ValueError("annotation episode must be an object")
    if payload.get("atoms") is not None:
        return payload["atoms"]
    if not any(key in payload for key in ("subtasks", "high_levels")):
        raise ValueError("annotation episode has no atoms or legacy segments")
    atoms = []
    for seg in payload.get("subtasks", []):
        if "label" in seg and "start" in seg:
            atoms.append(
                dict(
                    role="assistant",
                    content=str(seg["label"]),
                    style="subtask",
                    timestamp=float(seg["start"]),
                    tool_calls=None,
                )
            )
    for seg in payload.get("high_levels", []):
        timestamp = float(seg.get("start", 0.0))
        if seg.get("user_prompt"):
            atoms.append(
                dict(
                    role="user",
                    content=str(seg["user_prompt"]),
                    style="interjection",
                    timestamp=timestamp,
                    tool_calls=None,
                )
            )
        if seg.get("robot_utterance"):
            atoms.append(
                dict(
                    role="assistant",
                    content=None,
                    style=None,
                    timestamp=timestamp,
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {"name": "say", "arguments": {"text": str(seg["robot_utterance"])}},
                        }
                    ],
                )
            )
    return atoms


def _parquet_atoms(rows: list[dict]) -> list[dict]:
    """Editor fallback: first-row persistent atoms, all events, normalized/deduplicated."""
    atoms, seen = [], set()
    for index, row in enumerate(rows):
        groups = [(row.get("language_events") or [], row.get("timestamp"))]
        if index == 0:
            groups.insert(0, (row.get("language_persistent") or [], None))
        for values, fallback in groups:
            for raw in values:
                if not isinstance(raw, dict) or not raw.get("role"):
                    raise ValueError("invalid language atom")
                calls = raw.get("tool_calls")
                if calls is not None and not isinstance(calls, list):
                    calls = [calls]
                camera = raw.get("camera")
                timestamp = raw.get("timestamp")
                atom = dict(
                    role=str(raw["role"]),
                    content=None if raw.get("content") is None else str(raw["content"]),
                    style=raw.get("style"),
                    camera=camera if isinstance(camera, str) and camera else None,
                    tool_calls=calls or None,
                    timestamp=float(
                        timestamp if timestamp is not None else fallback if fallback is not None else 0
                    ),
                )
                key = json.dumps(atom, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    atoms.append(atom)
    return sorted(atoms, key=lambda atom: (atom["timestamp"], atom.get("style") or "", atom.get("role") or ""))


def _episode_tables(root: Path, info: dict, episode_ids, lengths: dict, diagnostics: list) -> dict:
    """Read only requested episodes; share one column-projected read per data shard.

    The returned rows include timestamps for prompt coverage. Episode identities,
    rather than global dataset offsets, delimit episodes in shared v3 shards.
    """
    import pyarrow.parquet as pq

    result, paths, metadata = {}, {}, {}
    if not episode_ids:
        return result
    if info.get("codebase_version") != "v2.1":
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            try:
                rows = pq.read_table(
                    path, columns=["episode_index", "data/chunk_index", "data/file_index"]
                ).to_pylist()
                for row in rows:
                    metadata[str(row["episode_index"])] = row
            except _READ_ERRORS as exc:
                diagnostics.append(_diag("invalid_episode_data", f"{path}: {exc}"))
    for ep in episode_ids:
        try:
            row = metadata.get(ep, {})
            legacy = info.get("codebase_version") == "v2.1"
            if not legacy and not row:
                raise ValueError("missing episode shard metadata")
            chunks = info.get("chunks_size", 1000)
            if type(chunks) is not int or chunks <= 0:
                raise ValueError("invalid chunks_size")
            rel = info["data_path"].format(
                episode_index=int(ep),
                episode_chunk=int(ep) // chunks,
                chunk_index=row.get("data/chunk_index", 0),
                file_index=row.get("data/file_index", 0),
            )
            path = (root / rel).resolve()
            if not _inside(path, root):
                raise ValueError("data path escapes checkpoint")
            paths.setdefault(path, []).append(ep)
        except (KeyError, IndexError, AttributeError, *_READ_ERRORS) as exc:
            diagnostics.append(_diag("invalid_episode_data", f"{root}: episode {ep}: {exc}"))
    for path, episodes in paths.items():
        try:
            names = pq.read_schema(path).names
            columns = [
                name
                for name in ("episode_index", "timestamp", "language_persistent", "language_events")
                if name in names
            ]
            if "episode_index" not in columns:
                raise ValueError("missing episode_index column")
            table = pq.read_table(path, columns=columns)
            grouped = {ep: [] for ep in episodes}
            for row in table.to_pylist():
                ep = str(row["episode_index"])
                if ep in grouped:
                    grouped[ep].append(row)
            for ep, rows in grouped.items():
                if len(rows) != lengths.get(ep) or not rows:
                    diagnostics.append(
                        _diag("invalid_episode_data", f"{path}: episode {ep} row count differs from metadata")
                    )
                else:
                    result[ep] = rows
        except _READ_ERRORS as exc:
            diagnostics.append(_diag("invalid_episode_data", f"{path}: {exc}"))
    return result


def _review_evidence(run: dict) -> dict:
    """Batch checkpoint evidence shared by summary and prompt calculations.

    Each retained episode has atoms, exclusions, length and reviewed. None means
    unreadable evidence; False means a known absent or stale explicit review.
    """
    import math

    try:
        from .annotation_clipping import normalize_exclusions
        from .annotation_history import annotation_hash, exclusions_hash
    except ImportError:
        from annotation_clipping import normalize_exclusions
        from annotation_history import annotation_hash, exclusions_hash

    root = Path(run["root"])
    checkpoint = _read_dataset(root)
    diagnostics = list(checkpoint["diagnostics"])
    try:
        info = _json_object(root / "meta/info.json")
    except _READ_ERRORS:
        info = {}
    fps = info.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        fps = None
    saved = _saved_object(root / "meta/lerobot_annotations.json", diagnostics)
    reviews = _saved_object(root / "meta/annotation_reviews.json", diagnostics)
    annotations = saved.get("episodes", {}) if saved is not None else None
    if annotations is not None and not isinstance(annotations, dict):
        diagnostics.append(_diag("invalid_review_evidence", f"{root}: annotation episodes must be an object"))
        annotations = None
    episodes, fallback = {}, []
    for ep, state in run["episodes"].items():
        if (
            not _valid_episode_key(ep)
            or not isinstance(state, dict)
            or state.get("decision") not in ("keep", "pending", "delete")
        ):
            diagnostics.append(
                _diag("invalid_episode_state", f"{root}: episode {ep}: invalid identity or decision")
            )
            continue
        if state["decision"] == "delete":
            continue
        length = checkpoint["episode_lengths"].get(ep)
        evidence = dict(state=state, length=length, atoms=None, exclusions=None, reviewed=None)
        episodes[ep] = evidence
        try:
            if length is None:
                raise ValueError("missing checkpoint frame count")
            evidence["exclusions"] = normalize_exclusions(state.get("excluded_intervals", []), length)
        except _READ_ERRORS as exc:
            diagnostics.append(_diag("invalid_exclusions", f"{root}: episode {ep}: {exc}"))
        if annotations is not None:
            if ep in annotations:
                try:
                    evidence["atoms"] = _sidecar_atoms(annotations[ep])
                except (KeyError, AttributeError, *_READ_ERRORS) as exc:
                    diagnostics.append(_diag("invalid_annotations", f"{root}: episode {ep}: {exc}"))
            elif length is not None:
                fallback.append(ep)
    tables = _episode_tables(root, info, fallback, checkpoint["episode_lengths"], diagnostics)
    for ep, evidence in episodes.items():
        try:
            if ep in tables:
                evidence["atoms"] = _parquet_atoms(tables[ep])
            atoms = evidence["atoms"]
            if not isinstance(atoms, list) or any(
                not isinstance(atom, dict)
                or not atom.get("role")
                or type(atom.get("timestamp")) not in (int, float)
                or not math.isfinite(atom["timestamp"])
                for atom in atoms
            ):
                raise ValueError("missing or invalid annotation atoms")
            evidence["annotation_sha256"] = annotation_hash(atoms)
            if evidence["exclusions"] is None or evidence["length"] is None or reviews is None:
                continue
            review = reviews.get(ep, {})
            if not isinstance(review, dict):
                raise ValueError("review record must be an object")
            if any(
                review.get(key) is not None and not isinstance(review[key], str)
                for key in ("reviewed_at", "annotation_sha256", "exclusions_sha256")
            ):
                raise ValueError("review timestamp and hashes must be strings or null")
            evidence["exclusions_sha256"] = exclusions_hash(evidence["exclusions"])
            evidence["reviewed"] = bool(review.get("reviewed_at")) and (
                review.get("annotation_sha256") == evidence["annotation_sha256"]
                and review.get("exclusions_sha256", exclusions_hash()) == evidence["exclusions_sha256"]
            )
        except (KeyError, AttributeError, *_READ_ERRORS) as exc:
            evidence["atoms"] = None
            diagnostics.append(_diag("invalid_review_evidence", f"{root}: episode {ep}: {exc}"))
    return dict(root=root, info=info, fps=fps, checkpoint=checkpoint, episodes=episodes, diagnostics=diagnostics)


def _current_alias(root: Path, workspace: Path, diagnostics: list) -> str | None:
    aliases = _saved_object(workspace / "local_datasets.json", diagnostics)
    return next(
        (
            alias
            for alias, path in sorted((aliases or {}).items())
            if alias.startswith("local/") and _canonical(path) == str(root.resolve())
        ),
        None,
    )


def _current_job(run: dict, workspace: Path, diagnostics: list) -> dict | None:
    job_id = run.get("current_job_id")
    if job_id is None:
        return None
    if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        diagnostics.append(_diag("invalid_job", "Current job ID is invalid"))
        return None
    path = workspace / "jobs" / (job_id + ".json")
    try:
        if not _inside(path, workspace):
            raise ValueError("job path escapes workspace")
        job = _json_object(path)
        return {key: job[key] for key in ("job_id", "status", "error") if key in job}
    except _READ_ERRORS as exc:
        diagnostics.append(_diag("invalid_job", f"{path}: {exc}"))
        return None


def summarize_run(dataset: dict, run: dict, workspace: Path) -> dict:
    """Summarize one immutable import identity against current checkpoint reviews."""
    from datetime import datetime, timezone

    workspace = Path(workspace)
    evidence = _review_evidence(run)
    root, episodes, diagnostics = evidence["root"], evidence["episodes"], evidence["diagnostics"]
    states = run["episodes"]
    decisions = {key: 0 for key in ("keep", "delete", "pending")}
    originals, identity_complete, decisions_complete = {}, True, True
    findings = dict(unresolved=0, accepted_advisory=0, generation_failed=0, unreadable=0)
    for ep, state in states.items():
        if not isinstance(state, dict):
            decisions_complete = identity_complete = False
            findings["unreadable"] += 1
            continue
        decision = state.get("decision")
        if isinstance(decision, str) and decision in decisions:
            decisions[decision] += 1
        else:
            decisions_complete = False
        if not _valid_episode_key(ep):
            decisions_complete = False
        original = state.get("original_episode_index")
        if not _nonnegative_int(original) or original in originals:
            identity_complete = False
            diagnostics.append(
                _diag(
                    "invalid_original_episode_id",
                    f"{root}: episode {ep}: missing, invalid or duplicate original ID",
                )
            )
        else:
            originals[original] = evidence["checkpoint"]["episode_lengths"].get(ep)
        if decision != "delete":
            issues = state.get("issues", [])
            if not isinstance(issues, list):
                diagnostics.append(_diag("invalid_episode_state", f"{root}: episode {ep}: issues must be a list"))
            elif issues:
                if decision == "keep":
                    findings["accepted_advisory"] += 1
                elif decision == "pending":
                    findings["unresolved"] += 1
            findings["generation_failed"] += state.get("generation_status") == "failed"
            findings["unreadable"] += (
                ep not in episodes
                or episodes[ep]["atoms"] is None
                or (
                    isinstance(issues, list)
                    and any(
                        isinstance(issue, dict) and issue.get("code") == "unreadable_episode" for issue in issues
                    )
                )
            )
    imported = len(states)
    retained = decisions["keep"] + decisions["pending"] if decisions_complete else None
    reviewed = (
        sum(ep["reviewed"] is True for ep in episodes.values())
        if decisions_complete and all(ep["reviewed"] is not None for ep in episodes.values())
        else None
    )
    exclusions_complete = decisions_complete and all(ep["exclusions"] is not None for ep in episodes.values())
    exclusions = {"episodes": None, "frames": None}
    if exclusions_complete:
        exclusions = dict(
            episodes=sum(bool(ep["exclusions"]) for ep in episodes.values()),
            frames=sum(
                span["end_frame"] - span["start_frame"] for ep in episodes.values() for span in ep["exclusions"]
            ),
        )
    accepted = [ep for ep in episodes.values() if ep["state"]["decision"] == "keep"]
    accepted_frames = None
    if decisions_complete and all(ep["length"] is not None and ep["exclusions"] is not None for ep in accepted):
        accepted_frames = sum(
            ep["length"] - sum(span["end_frame"] - span["start_frame"] for span in ep["exclusions"])
            for ep in accepted
        )
    accepted_seconds = (
        accepted_frames / evidence["fps"] if accepted_frames is not None and evidence["fps"] is not None else None
    )
    source_complete = dataset.get("state") == "Ready"
    live = {int(ep): length for ep, length in dataset["episode_lengths"].items()}
    new_ids = sorted(set(live) - set(originals)) if identity_complete and source_complete else None
    missing_ids = sorted(set(originals) - set(live)) if identity_complete and source_complete else None
    changed_ids = (
        sorted(ep for ep in set(originals) & set(live) if originals[ep] != live[ep])
        if identity_complete and source_complete and all(length is not None for length in originals.values())
        else None
    )
    if not source_complete:
        diagnostics.append(
            _diag("source_metadata_incomplete", "Source metadata is incomplete; growth comparison is unknown")
        )
    metadata_changed = any(bool(ids) for ids in (new_ids, missing_ids, changed_ids))
    source_changed = (
        True
        if metadata_changed
        else False
        if all(ids is not None for ids in (new_ids, missing_ids, changed_ids))
        else None
    )
    metrics = dict(
        imported=imported,
        accepted=decisions["keep"],
        rejected=decisions["delete"],
        pending=decisions["pending"],
        retained=retained,
        reviewed=reviewed,
        decision_rate=(decisions["keep"] + decisions["delete"]) / imported
        if imported and decisions_complete
        else None,
        review_rate=reviewed / retained if reviewed is not None and retained else None,
        new_episode_ids=new_ids,
        missing_episode_ids=missing_ids,
        changed_length_ids=changed_ids,
        accepted_frames=accepted_frames,
        accepted_seconds=accepted_seconds,
        counts_complete=not diagnostics and decisions_complete and identity_complete and source_complete,
    )
    try:
        signature = _checkpoint_signature(root)
    except _READ_ERRORS as exc:
        signature = None
        diagnostics.append(_diag("invalid_checkpoint_signature", f"{root}: {exc}"))
        metrics["counts_complete"] = False
    public_run = {key: value for key, value in run.items() if not key.startswith("_")}
    detail_signature = hashlib.sha256(
        json.dumps([public_run, signature, dataset.get("_signature")], sort_keys=True).encode()
    ).hexdigest()
    mtime = run.get("_mtime_ns")
    updated_at = datetime.fromtimestamp(mtime / 1e9, timezone.utc).isoformat() if mtime is not None else None
    return dict(
        run_id=run["run_id"],
        updated_at=updated_at,
        current_repo_id=_current_alias(root, workspace, diagnostics),
        first_retained_episode=min(map(int, episodes)) if episodes and decisions_complete else None,
        detail_signature=detail_signature,
        metrics=metrics,
        findings=findings,
        exclusions=exclusions,
        job=_current_job(run, workspace, diagnostics),
        freshness=dict(
            publication_state=run.get("publication_state"),
            metadata_only=True,
            source_changed=source_changed,
            local_changes=None,
            verifiable=False,
        ),
        diagnostics=diagnostics,
    )
