"""LeRobot visualizer: review drafts, generate with official modules, validate/export.

Run the isolated annotation runtime with:
    backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7861
See backend/README.md for pinned dependencies and VLM endpoint configuration.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

try:  # Supports both ``import backend.app`` and the legacy ``import app`` entrypoint.
    from . import annotation_history, annotation_runs
    from .annotation_access import AnnotationAccess, validate_dataset_paths
    from .curation.assets import LocalAssetService
    from .curation.config import CurationSettings, curation_is_configured, legacy_browser_origin
    from .curation.db import CurationDatabase
    from .curation.review import ReviewService
    from .curation.router import build_curation_router
    from .curation.security import CurationLoopbackGuard
    from .curation.source import SourceRegistry
    from .curation.worker import BatchService
    from .robot_motion import read_robot_motion
except ImportError:  # pragma: no cover - selected only by ``uvicorn app:app``.
    from annotation_access import AnnotationAccess, validate_dataset_paths
    import annotation_history
    import annotation_runs
    from curation.assets import LocalAssetService
    from curation.config import CurationSettings, curation_is_configured, legacy_browser_origin
    from curation.db import CurationDatabase
    from curation.review import ReviewService
    from curation.router import build_curation_router
    from curation.security import CurationLoopbackGuard
    from curation.source import SourceRegistry
    from curation.worker import BatchService
    from robot_motion import read_robot_motion

logger = logging.getLogger("lerobot-annotate")
logging.basicConfig(level=logging.INFO)

CACHE_ROOT = Path(os.environ.get("LEROBOT_ANNOTATE_CACHE", "/tmp/lerobot_visualizer_annotate_cache"))
EXPORT_ROOT = Path(os.environ.get("LEROBOT_ANNOTATE_EXPORT", "/tmp/lerobot_visualizer_annotate_exports"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets-cache"))

_alias_lock = Lock()


def _local_aliases() -> dict[str, str]:
    path = EXPORT_ROOT / "local_datasets.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _register_local_dataset(root: Path) -> str:
    root = root.resolve()
    alias = "local/annotation-" + sha256(str(root).encode()).hexdigest()[:16]
    with _alias_lock:
        aliases = _local_aliases()
        aliases[alias] = str(root)
        EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
        path = EXPORT_ROOT / "local_datasets.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(aliases, indent=2))
        temporary.replace(path)
    return alias


def _resolve_local_ref(req):
    if not req.local_path and req.repo_id and req.repo_id.startswith("local/"):
        root = _local_aliases().get(req.repo_id)
        if root is None:
            raise HTTPException(status_code=404, detail="Unknown local dataset; prepare it at /annotate")
        return req.model_copy(update={"repo_id": None, "local_path": root})
    return req


# --- Schema mirrors src/lerobot/datasets/language.py --------------------------

PERSISTENT_STYLES = {"task_aug", "subtask", "plan", "memory", "motion"}
EVENT_ONLY_STYLES = {"interjection", "vqa", "trace"}
KNOWN_STYLES = PERSISTENT_STYLES | EVENT_ONLY_STYLES
LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"


def column_for_style(style: str | None) -> str:
    if style is None:
        return LANGUAGE_EVENTS
    if style in PERSISTENT_STYLES:
        return LANGUAGE_PERSISTENT
    if style in EVENT_ONLY_STYLES:
        return LANGUAGE_EVENTS
    raise ValueError(f"Unknown language style: {style!r}")


# --- Pydantic models ----------------------------------------------------------


class DatasetRef(BaseModel):
    repo_id: str | None = None
    revision: str | None = None
    local_path: str | None = None


class LoadRequest(DatasetRef):
    pass


class LanguageAtom(BaseModel):
    role: str
    content: str | None = None
    style: str | None = None
    timestamp: float = Field(allow_inf_nan=False)
    # ``observation.images.*`` feature key for view-dependent atoms
    # (vqa / trace). ``None`` for camera-agnostic atoms. Mirrors the
    # row-level ``camera`` field added in lerobot PR 3467.
    camera: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class EpisodeAtomsPayload(DatasetRef):
    episode_index: int
    atoms: list[LanguageAtom] = []
    expected_annotation_sha256: str | None = None


class EpisodeReviewPayload(DatasetRef):
    episode_index: int
    reviewed: bool
    annotation_sha256: str
    expected_exclusions_sha256: str | None = None


class ExportRequest(DatasetRef):
    output_dir: str | None = None
    copy_videos: bool = False


class PushToHubRequest(DatasetRef):
    hf_token: str
    push_in_place: bool = True
    new_repo_id: str | None = None
    private: bool = False
    commit_message: str = "Add language annotations"


@dataclass
class EpisodeAnnotations:
    atoms: list[dict[str, Any]] = field(default_factory=list)


# --- Per-dataset state cache --------------------------------------------------


@dataclass
class DatasetState:
    repo_id: str | None
    local_path: str | None
    revision: str | None
    root: Path
    info: dict[str, Any]
    episodes_df: pd.DataFrame
    annotations: dict[int, EpisodeAnnotations] = field(default_factory=dict)
    frame_ts_cache: dict[int, list[float]] = field(default_factory=dict)

    @property
    def annotations_path(self) -> Path:
        return self.root / "meta" / "lerobot_annotations.json"


_states: dict[str, DatasetState] = {}
_annotation_edit_lock = annotation_runs.LOCK


def _state_key(req: DatasetRef) -> str:
    if req.local_path:
        return f"local::{Path(req.local_path).expanduser().resolve()}"
    if req.repo_id:
        return f"hf::{req.repo_id}@{req.revision or 'main'}"
    raise HTTPException(status_code=400, detail="need repo_id or local_path")


def _ensure_state(req: DatasetRef) -> DatasetState:
    with _annotation_edit_lock:
        if os.environ.get("ANNOTATION_BACKEND_TOKEN") and req.repo_id and not req.repo_id.startswith("local/"):
            raise HTTPException(403, "Prepare a pinned dataset and use its registered local alias")
        req = _resolve_local_ref(req)
        key = _state_key(req)
        if key in _states:
            return _states[key]
        return _load_state(req, key)


def _load_state(req: DatasetRef, key: str) -> DatasetState:
    if req.local_path:
        root = Path(req.local_path).expanduser().resolve()
        if not root.exists():
            raise HTTPException(status_code=404, detail=f"Dataset path not found: {root}")
    elif req.repo_id:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        slug = sha256(f"{req.repo_id}@{req.revision or 'main'}".encode()).hexdigest()
        root = CACHE_ROOT / slug
        root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            req.repo_id,
            repo_type="dataset",
            revision=req.revision,
            local_dir=root,
            allow_patterns=["meta/*"],
        )
    else:
        raise HTTPException(status_code=400, detail="need repo_id or local_path")

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing meta/info.json at {root}")
    info = json.loads(info_path.read_text())

    if info.get("codebase_version") == "v2.1":
        episodes_df = pd.DataFrame(_official_engine().source_episode_rows(root))
    else:
        episodes_root = root / "meta" / "episodes"
        files = sorted(episodes_root.rglob("*.parquet"))
        if not files:
            raise HTTPException(status_code=404, detail="No episodes parquet files found")
        episodes_df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    state = DatasetState(
        repo_id=req.repo_id,
        local_path=str(root) if req.local_path else None,
        revision=req.revision,
        root=root,
        info=info,
        episodes_df=episodes_df,
    )
    _load_existing_annotations(state)
    _states[key] = state
    return state


def _load_existing_annotations(state: DatasetState) -> None:
    path = state.annotations_path
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for ep_str, payload in data.get("episodes", {}).items():
        ep_idx = int(ep_str)
        atoms = payload.get("atoms")
        if atoms is None:
            # v1 format from older lerobot-annotate (legacy)
            atoms = []
            for seg in payload.get("subtasks", []):
                if "label" in seg and "start" in seg:
                    atoms.append(
                        {
                            "role": "assistant",
                            "content": str(seg["label"]),
                            "style": "subtask",
                            "timestamp": float(seg["start"]),
                            "tool_calls": None,
                        }
                    )
            for seg in payload.get("high_levels", []):
                ts = float(seg.get("start", 0.0))
                if seg.get("user_prompt"):
                    atoms.append(
                        {
                            "role": "user",
                            "content": str(seg["user_prompt"]),
                            "style": "interjection",
                            "timestamp": ts,
                            "tool_calls": None,
                        }
                    )
                if seg.get("robot_utterance"):
                    atoms.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "style": None,
                            "timestamp": ts,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "say",
                                        "arguments": {"text": str(seg["robot_utterance"])},
                                    },
                                }
                            ],
                        }
                    )
        state.annotations[ep_idx] = EpisodeAnnotations(atoms=[dict(a) for a in atoms])


def _save_annotations(state: DatasetState) -> None:
    path = state.annotations_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "schema": {
            "persistent_styles": sorted(PERSISTENT_STYLES),
            "event_styles": sorted(EVENT_ONLY_STYLES),
        },
        "episodes": {str(ep): {"atoms": ann.atoms} for ep, ann in state.annotations.items()},
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


# --- Frame-timestamp helpers --------------------------------------------------


def _episode_data_path(state: DatasetState, episode_index: int) -> Path | None:
    rows = state.episodes_df[state.episodes_df["episode_index"] == episode_index]
    if rows.empty:
        return None
    row = rows.iloc[0]
    chunk_col = "data/chunk_index"
    file_col = "data/file_index"
    legacy = state.info.get("codebase_version") == "v2.1"
    if not legacy and (chunk_col not in row or file_col not in row):
        return None
    rel = state.info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    rel = rel.format(
        chunk_index=int(row.get(chunk_col, 0)),
        file_index=int(row.get(file_col, 0)),
        episode_index=episode_index,
        episode_chunk=episode_index // int(state.info.get("chunks_size", 1000)),
    )
    full = (state.root / rel).resolve()
    if not full.is_relative_to(state.root.resolve()):
        raise HTTPException(422, "Dataset data path escapes its root")
    if full.exists():
        return full
    if state.repo_id:
        try:
            hf_hub_download(
                repo_id=state.repo_id,
                repo_type="dataset",
                filename=rel,
                revision=state.revision,
                local_dir=state.root,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("frame_ts download failed for ep %s: %s", episode_index, e)
            return None
    return full if full.exists() else None


def _frame_timestamps(state: DatasetState, episode_index: int) -> list[float]:
    if episode_index in state.frame_ts_cache:
        return state.frame_ts_cache[episode_index]
    path = _episode_data_path(state, episode_index)
    if path is None:
        return []
    try:
        df = pd.read_parquet(path, columns=["episode_index", "timestamp"])
    except Exception as e:  # noqa: BLE001
        logger.warning("frame_ts read failed for ep %s: %s", episode_index, e)
        return []
    ts = df.loc[df["episode_index"] == episode_index, "timestamp"].astype(float).tolist()
    ts.sort()
    state.frame_ts_cache[episode_index] = ts
    return ts


def _coerce_existing_atom(raw: Any, fallback_ts: float | None = None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except Exception:  # noqa: BLE001
            return None
    if not raw.get("role"):
        return None
    tool_calls = raw.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
    camera = raw.get("camera")
    if isinstance(camera, str) and not camera:
        camera = None
    raw_ts = raw.get("timestamp")
    if raw_ts is None:
        # v3.1 event rows don't carry a ``timestamp`` field in the struct —
        # the writer drops it because the parquet row's frame timestamp is
        # already the event's firing time. Use the caller-provided fallback
        # so dedup doesn't collapse every event atom into one (timestamp=0.0)
        # entry.
        timestamp = float(fallback_ts) if fallback_ts is not None else 0.0
    else:
        timestamp = float(raw_ts)
    return {
        "role": str(raw["role"]),
        "content": None if raw.get("content") is None else str(raw.get("content")),
        "style": raw.get("style"),
        "timestamp": timestamp,
        "camera": camera if isinstance(camera, str) else None,
        "tool_calls": tool_calls or None,
    }


def _extract_existing_atoms_from_table(table: pa.Table, episode_index: int) -> list[dict[str, Any]]:
    if "episode_index" not in table.column_names:
        return []

    episode_col = table.column("episode_index").to_pylist()
    persistent_col = (
        table.column(LANGUAGE_PERSISTENT).to_pylist() if LANGUAGE_PERSISTENT in table.column_names else None
    )
    events_col = table.column(LANGUAGE_EVENTS).to_pylist() if LANGUAGE_EVENTS in table.column_names else None
    # Event rows don't carry their own ``timestamp`` in the v3.1 struct;
    # the parquet row's frame timestamp IS the event's firing time. Read
    # the timestamp column so we can pass it as a fallback to
    # ``_coerce_existing_atom`` — without this, every event row defaults
    # to timestamp=0.0 and dedup collapses them all into one.
    ts_col = table.column("timestamp").to_pylist() if "timestamp" in table.column_names else None

    atoms: list[dict[str, Any]] = []
    seen: set[str] = set()
    persistent_loaded = False

    def add_many(raw_atoms: Any, fallback_ts: float | None = None) -> None:
        if not raw_atoms:
            return
        for raw in raw_atoms:
            atom = _coerce_existing_atom(raw, fallback_ts=fallback_ts)
            if atom is None:
                continue
            key = json.dumps(atom, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            atoms.append(atom)

    for row_idx, ep_value in enumerate(episode_col):
        if int(ep_value) != int(episode_index):
            continue
        if persistent_col is not None and not persistent_loaded:
            add_many(persistent_col[row_idx])
            persistent_loaded = True
        if events_col is not None:
            row_ts = float(ts_col[row_idx]) if ts_col is not None else None
            add_many(events_col[row_idx], fallback_ts=row_ts)

    atoms.sort(key=lambda a: (a["timestamp"], a.get("style") or "", a.get("role") or ""))
    return atoms


def _snap(ts: float, frame_ts: list[float]) -> float:
    if not frame_ts:
        return float(ts)
    return float(min(frame_ts, key=lambda f: abs(f - ts)))


VIEW_DEPENDENT_STYLES = {"vqa", "trace"}


def _validate_atom(atom: dict[str, Any]) -> None:
    style = atom.get("style")
    if style is not None and style not in KNOWN_STYLES:
        raise HTTPException(status_code=400, detail=f"Unknown language style: {style!r}")
    has_content = atom.get("content") is not None
    has_tools = bool(atom.get("tool_calls"))
    if not (has_content or has_tools):
        raise HTTPException(status_code=400, detail="atom must have content or tool_calls")
    if style is None and not has_tools:
        raise HTTPException(status_code=400, detail="style=None requires tool_calls (speech atom)")
    camera = atom.get("camera")
    if camera is not None and not isinstance(camera, str):
        raise HTTPException(status_code=400, detail="camera must be a string or null")
    # Mirror lerobot's row-level invariant: camera is set iff the style is
    # view-dependent. We don't enforce camera-required here because the
    # visualizer accepts in-progress edits where the user hasn't picked a
    # camera yet — the writer (or the next save round-trip) will surface
    # the missing tag. We DO reject camera-on-non-view-dependent so the
    # field can't drift onto task_aug/subtask/plan/memory rows.
    if camera is not None and style is not None and style not in VIEW_DEPENDENT_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"camera must be null for style={style!r} (only vqa/trace are view-dependent)",
        )


def _official_engine():
    try:
        if __package__:
            from . import official_annotations
        else:
            import official_annotations
        return official_annotations
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Official LeRobot runtime unavailable. Use Python >=3.12 and install "
            "backend/requirements-annotations.txt: " + str(exc),
        ) from exc


def _download_full_dataset(state: DatasetState) -> None:
    if state.repo_id:
        snapshot_download(
            state.repo_id,
            repo_type="dataset",
            revision=state.revision,
            local_dir=state.root,
            allow_patterns=["meta/**", "data/**", "videos/**"],
        )


def _do_export(state: DatasetState, output_dir: str | None, copy_videos: bool) -> dict[str, Any]:
    run = annotation_runs.RunStore(EXPORT_ROOT).for_root(state.root)
    if run and any(episode.get("excluded_intervals") for episode in run["episodes"].values()):
        raise HTTPException(
            status_code=409,
            detail="This dataset has excluded intervals. Use Export & publish → Preview export to apply clipping.",
        )
    engine = _official_engine()
    _download_full_dataset(state)
    out_root = Path(output_dir).expanduser().resolve() if output_dir else EXPORT_ROOT / f"annotated-{uuid4().hex}"
    try:
        result = engine.export_dataset(
            state.root,
            out_root,
            {ep: ann.atoms for ep, ann in state.annotations.items()},
            copy_videos=copy_videos,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {k: result[k] for k in ("output_dir", "persistent_rows", "event_rows")}


# --- FastAPI app --------------------------------------------------------------

# Curation is opt-in so an unchanged visualizer installation keeps serving its
# v3.1 annotation routes. Once any curation variable is provided, settings are
# deliberately all-or-nothing and source registration happens before serving.
_curation_settings: CurationSettings | None = None
_local_asset_service: LocalAssetService | None = None
_review_service: ReviewService | None = None
_batch_service: BatchService | None = None
if curation_is_configured():
    _curation_settings = CurationSettings.from_env()
    _source_registry = SourceRegistry.from_paths(
        _curation_settings.dataset_aliases, workspace=_curation_settings.workspace
    )
    _local_asset_service = LocalAssetService(_source_registry)
    _curation_database = CurationDatabase(_curation_settings.workspace / "curation.sqlite3")
    _curation_database.initialize()
    _review_service = ReviewService(database=_curation_database, source_registry=_source_registry)
    _batch_service = BatchService(
        database=_curation_database,
        source_registry=_source_registry,
        workspace=_curation_settings.workspace,
        cosmos_base_url=_curation_settings.cosmos_base_url,
        cosmos_model=_curation_settings.cosmos_model,
        cosmos_api_key_env=_curation_settings.cosmos_api_key_env,
        cosmos_endpoint_identity=_curation_settings.cosmos_endpoint_identity,
    )

app = FastAPI(title="LeRobot dataset visualizer — annotation backend")
app.add_middleware(AnnotationAccess)
app.add_middleware(
    CORSMiddleware,
    # Curation uses its explicit origin; legacy annotation keeps its documented
    # loopback Next.js origin rather than silently disabling browser access.
    allow_origins=[_curation_settings.browser_origin] if _curation_settings else [legacy_browser_origin()],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length", "ETag"],
)
if _curation_settings:
    app.add_middleware(CurationLoopbackGuard)
app.include_router(
    build_curation_router(
        _local_asset_service,
        review_service=_review_service,
        batch_service=_batch_service,
        bearer_token=_curation_settings.bearer_token if _curation_settings else None,
    )
)


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "lerobot-visualizer-annotate",
            "persistent_styles": sorted(PERSISTENT_STYLES),
            "event_styles": sorted(EVENT_ONLY_STYLES),
        }
    )


@app.post("/api/dataset/load")
def load_dataset(req: LoadRequest) -> JSONResponse:
    state = _ensure_state(req)
    return JSONResponse(
        {
            "repo_id": state.repo_id,
            "local_path": state.local_path,
            "revision": state.revision,
            "root": str(state.root),
            "fps": float(state.info.get("fps", 30)),
            "num_episodes": int(state.episodes_df["episode_index"].nunique()),
            "persistent_styles": sorted(PERSISTENT_STYLES),
            "event_styles": sorted(EVENT_ONLY_STYLES),
        }
    )


@app.get("/api/episodes/{episode_index}/atoms")
def get_episode_atoms(
    episode_index: int,
    repo_id: str | None = None,
    revision: str | None = None,
    local_path: str | None = None,
) -> JSONResponse:
    state = _ensure_state(DatasetRef(repo_id=repo_id, revision=revision, local_path=local_path))
    ann = state.annotations.get(episode_index)
    if ann is None:
        path = _episode_data_path(state, episode_index)
        atoms: list[dict[str, Any]] = []
        if path is not None:
            try:
                schema = pq.read_schema(path)
                columns = ["episode_index"]
                # Always pull the row timestamp — needed as a fallback for
                # event rows whose v3.1 struct intentionally omits it.
                if "timestamp" in schema.names:
                    columns.append("timestamp")
                if LANGUAGE_PERSISTENT in schema.names:
                    columns.append(LANGUAGE_PERSISTENT)
                if LANGUAGE_EVENTS in schema.names:
                    columns.append(LANGUAGE_EVENTS)
                if LANGUAGE_PERSISTENT in columns or LANGUAGE_EVENTS in columns:
                    atoms = _extract_existing_atoms_from_table(
                        pq.read_table(path, columns=columns),
                        episode_index,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("language column read failed for ep %s: %s", episode_index, e)
        ann = EpisodeAnnotations(atoms=atoms)
        if atoms:
            state.annotations[episode_index] = ann
    return JSONResponse(
        {
            "episode_index": episode_index,
            "atoms": ann.atoms,
            "annotation_sha256": annotation_history.annotation_hash(ann.atoms),
        }
    )


@app.post("/api/episodes/{episode_index}/atoms")
def set_episode_atoms(episode_index: int, payload: EpisodeAtomsPayload) -> JSONResponse:
    if episode_index != payload.episode_index:
        raise HTTPException(status_code=400, detail="episode index mismatch")
    state = _ensure_state(
        DatasetRef(repo_id=payload.repo_id, revision=payload.revision, local_path=payload.local_path)
    )
    if episode_index not in state.episodes_df["episode_index"].values:
        raise HTTPException(status_code=404, detail="Episode not found")
    atoms = [a.model_dump() for a in payload.atoms]
    for atom in atoms:
        _validate_atom(atom)
    # Snap event timestamps to exact frame timestamps (matches lerobot#3471).
    frame_ts = _frame_timestamps(state, episode_index)
    for atom in atoms:
        if column_for_style(atom.get("style")) == LANGUAGE_EVENTS and frame_ts:
            atom["timestamp"] = _snap(float(atom["timestamp"]), frame_ts)
    with _annotation_edit_lock:
        _require_editable(state.root)
        if (
            annotation_runs.RunStore(EXPORT_ROOT).for_root(state.root)
            and payload.expected_annotation_sha256 is None
        ):
            raise HTTPException(409, "Reload saved annotations before editing this workflow")
        if payload.expected_annotation_sha256 is not None:
            current = json.loads(
                get_episode_atoms(episode_index, payload.repo_id, payload.revision, payload.local_path).body
            )["atoms"]
            if annotation_history.annotation_hash(current) != payload.expected_annotation_sha256:
                raise HTTPException(409, "Annotations changed; reload before saving")
        reviews = annotation_history.read_reviews(state.root)
        review = reviews.get(str(episode_index))
        if review and review.get("annotation_sha256") != annotation_history.annotation_hash(atoms):
            reviews.pop(str(episode_index))
            annotation_history.write_reviews(state.root, reviews)
        state.annotations[episode_index] = EpisodeAnnotations(atoms=atoms)
        _save_annotations(state)
        annotation_runs.RunStore(EXPORT_ROOT).edited(state.root)
    result = {"ok": True, "saved": len(atoms), "path": str(state.annotations_path)}
    if payload.expected_annotation_sha256 is not None:
        result["annotation_sha256"] = annotation_history.annotation_hash(atoms)
    return JSONResponse(result)


def _review_atoms(episode_index: int, ref: DatasetRef) -> tuple[DatasetState, list[dict]]:
    state = _ensure_state(ref)
    if episode_index not in state.episodes_df["episode_index"].values:
        raise HTTPException(status_code=404, detail="Episode not found")
    response = get_episode_atoms(episode_index, ref.repo_id, ref.revision, ref.local_path)
    return state, json.loads(response.body)["atoms"]


@app.get("/api/episodes/{episode_index}/review")
def get_episode_review(
    episode_index: int,
    repo_id: str | None = None,
    revision: str | None = None,
    local_path: str | None = None,
) -> dict:
    with _annotation_edit_lock:
        state, atoms = _review_atoms(
            episode_index, DatasetRef(repo_id=repo_id, revision=revision, local_path=local_path)
        )
        run = annotation_runs.RunStore(EXPORT_ROOT).for_root(state.root)
        intervals = (run or {}).get("episodes", {}).get(str(episode_index), {}).get("excluded_intervals", [])
        return annotation_history.review_status(state.root, episode_index, atoms, intervals)


@app.post("/api/episodes/{episode_index}/review")
def set_episode_review(episode_index: int, payload: EpisodeReviewPayload) -> dict:
    if episode_index != payload.episode_index:
        raise HTTPException(status_code=400, detail="episode index mismatch")
    with _annotation_edit_lock:
        state, atoms = _review_atoms(episode_index, payload)
        try:
            _require_editable(state.root)
            run = annotation_runs.RunStore(EXPORT_ROOT).for_root(state.root)
            intervals = (run or {}).get("episodes", {}).get(str(episode_index), {}).get("excluded_intervals", [])
            clip_digest = annotation_history.exclusions_hash(intervals)
            if (
                payload.expected_exclusions_sha256 is not None or intervals
            ) and payload.expected_exclusions_sha256 != clip_digest:
                raise ValueError("Excluded intervals changed. Reload before reviewing.")
            result = annotation_history.save_review(
                state.root,
                episode_index,
                atoms,
                payload.reviewed,
                payload.annotation_sha256,
                excluded_intervals=intervals,
            )
            annotation_runs.RunStore(EXPORT_ROOT).edited(state.root)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/episodes/{episode_index}/frame_timestamps")
def episode_frame_timestamps(
    episode_index: int,
    repo_id: str | None = None,
    revision: str | None = None,
    local_path: str | None = None,
) -> JSONResponse:
    state = _ensure_state(DatasetRef(repo_id=repo_id, revision=revision, local_path=local_path))
    ts = _frame_timestamps(state, episode_index)
    return JSONResponse({"episode_index": episode_index, "timestamps": ts})


@app.get("/api/episodes/{episode_index}/robot-motion")
def episode_robot_motion(
    episode_index: int,
    repo_id: str | None = None,
    revision: str | None = None,
    local_path: str | None = None,
) -> JSONResponse:
    state = _ensure_state(DatasetRef(repo_id=repo_id, revision=revision, local_path=local_path))
    if state.episodes_df[state.episodes_df["episode_index"] == episode_index].empty:
        raise HTTPException(404, "Episode not found")
    path = _episode_data_path(state, episode_index)
    if path is None:
        raise HTTPException(422, "Episode motion data is unavailable")
    return JSONResponse(read_robot_motion(state, episode_index, path))


@app.post("/api/export")
def export_dataset(req: ExportRequest) -> JSONResponse:
    state = _ensure_state(req)
    return JSONResponse(_do_export(state, req.output_dir, req.copy_videos))


@app.post("/api/push_to_hub")
def push_to_hub(req: PushToHubRequest) -> JSONResponse:
    state = _ensure_state(req)
    if not state.repo_id and not req.new_repo_id:
        raise HTTPException(status_code=400, detail="repo_id or new_repo_id required")

    # Ensure data + videos are present locally before exporting.
    if state.repo_id:
        snapshot_download(
            state.repo_id,
            repo_type="dataset",
            revision=state.revision,
            local_dir=state.root,
            allow_patterns=["data/**/*.parquet", "videos/**/*.mp4"],
        )
    export_result = _do_export(state, output_dir=None, copy_videos=True)
    export_dir = Path(export_result["output_dir"])

    target_repo = state.repo_id if req.push_in_place else req.new_repo_id
    if not target_repo:
        raise HTTPException(status_code=400, detail="No target repo")

    api = HfApi(token=req.hf_token)
    if not req.push_in_place:
        api.create_repo(
            repo_id=target_repo,
            repo_type="dataset",
            private=req.private,
            exist_ok=True,
        )
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=target_repo,
        repo_type="dataset",
        commit_message=req.commit_message,
    )
    return JSONResponse(
        {
            "ok": True,
            "repo_id": target_repo,
            "url": f"https://huggingface.co/datasets/{target_repo}",
            "message": f"Pushed annotated dataset to {target_repo}",
        }
    )


# ponytail: one process owns this queue; use a durable worker if multi-process serving is needed.
# The official executor controls episode/VLM parallelism within each job.
_annotation_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lerobot-annotation")
_annotation_jobs: dict[str, dict] = {}


class GenerationRequest(DatasetRef):
    episode_indices: list[int] | None = None
    example_episode_indices: list[int] = Field(default_factory=list, max_length=5)
    config: dict[str, Any] = {}
    task_prompt: str = ""
    subtask_prompts: list[str] | None = None
    assess_quality: bool = False
    resume_unfinished: bool = False


class DeleteEpisodesRequest(DatasetRef):
    episode_indices: list[int] = Field(min_length=1)


def _start_annotation_job(operation, on_queued=None) -> dict:
    job_id = uuid4().hex
    job = {"job_id": job_id, "status": "queued"}
    jobs_root = EXPORT_ROOT / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    path = jobs_root / f"{job_id}.json"

    def persist() -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(path)

    def run() -> None:
        job["status"] = "running"
        persist()
        try:
            result = operation(EXPORT_ROOT / "drafts" / job_id)
            if "output_dir" in result:
                result["repo_id"] = _register_local_dataset(Path(result["output_dir"]))
            job.update(status="completed", result=result)
        except Exception as exc:
            logger.exception("Official annotation job %s failed", job_id)
            job.update(status="failed", error=str(exc))
        finally:
            persist()
            _annotation_jobs.pop(job_id, None)

    with _annotation_edit_lock:
        if on_queued:
            on_queued(job_id)
        persist()
        _annotation_jobs[job_id] = job
        _annotation_pool.submit(run)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/annotation/config")
def annotation_config() -> dict:
    engine = _official_engine()
    config = engine.default_config()
    if os.environ.get("ANNOTATION_BACKEND_TOKEN"):
        config.pop("vlm", None)
    return {"revision": engine.REVISION, "config": config}


@app.post("/api/annotation/prepare")
def prepare_annotation_dataset(req: DatasetRef) -> dict:
    engine = _official_engine()
    req = _resolve_local_ref(req)
    if not req.local_path and not req.repo_id:
        raise HTTPException(status_code=400, detail="Provide local_path or repo_id")

    def prepare(output: Path) -> dict:
        source_commit = None
        if req.local_path:
            source = Path(req.local_path).expanduser().resolve()
        else:
            source_commit = HfApi().dataset_info(req.repo_id, revision=req.revision or "main").sha
            source = CACHE_ROOT / sha256(f"{req.repo_id}@{source_commit}".encode()).hexdigest()
            snapshot_download(
                req.repo_id,
                repo_type="dataset",
                revision=source_commit,
                local_dir=source,
                allow_patterns=["meta/**", "data/**", "videos/**"],
            )
        validate_dataset_paths(source)
        source_hashes = annotation_runs.source_inventory(source)
        result = engine.prepare_dataset(source, output)
        indices = result.get("episode_indices")
        if indices is None:
            indices = [r.episode_index for r in engine.iter_episodes(output)]
        if not indices:
            raise ValueError("Dataset contains no episodes")
        if result.get("preparation_mode") == "source_review":
            result["validation"] = result["source_validation"]
        else:
            result["validation"] = engine.validate_atoms(output, list(engine.iter_episodes(output)), {})
        episode_results = result.get("episode_results", {})
        usable = [ep for ep in indices if episode_results.get(str(ep), {}).get("generation_status") != "failed"]
        result["first_episode_index"] = min(usable or indices)
        run = annotation_runs.RunStore(EXPORT_ROOT).create(
            output,
            source,
            req.repo_id,
            source_commit,
            json.loads((source / "meta/info.json").read_text())["codebase_version"],
            indices,
        )
        if run["source_file_hashes"] != source_hashes:
            raise ValueError("Source dataset changed during preparation; prepare it again")
        run["preparation_mode"] = result.get("preparation_mode", "ready")
        for ep, record in episode_results.items():
            run["episodes"][ep].update(record)
        run = annotation_runs.RunStore(EXPORT_ROOT).save(run, run["revision"])
        result["run_id"] = run["run_id"]
        return result

    return _start_annotation_job(prepare)


@app.post("/api/annotation/jobs")
def create_annotation_job(req: GenerationRequest) -> dict:
    engine = _official_engine()
    try:
        config = engine.parse_config(req.config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state = _ensure_state(req)
    if req.resume_unfinished:
        prior = annotation_runs.RunStore(EXPORT_ROOT).for_root(state.root)
        if prior:
            state = _ensure_state(DatasetRef(local_path=prior["root"]))
            req.task_prompt = prior.get("task_prompt", "")
            req.subtask_prompts = prior.get("subtask_prompts", [])
            req.example_episode_indices = prior.get("example_episode_indices", [])
            req.assess_quality = prior.get("assess_quality", False)
            if not req.config:
                req.config = prior.get("generation_config", {})
                config = engine.parse_config(req.config)
    if len(set(req.example_episode_indices)) != len(req.example_episode_indices) or set(
        req.example_episode_indices
    ) - set(state.episodes_df["episode_index"]):
        raise HTTPException(status_code=422, detail="Select up to five distinct existing example episodes")
    if req.episode_indices is not None and (
        not req.episode_indices or set(req.episode_indices) - set(state.episodes_df["episode_index"])
    ):
        raise HTTPException(status_code=422, detail="Select existing episode indices")
    with _annotation_edit_lock:
        _require_editable(state.root)
        store = annotation_runs.RunStore(EXPORT_ROOT)
        run = store.for_root(state.root)
        if run is None:
            run = store.create(
                state.root,
                state.root,
                None,
                None,
                state.info.get("codebase_version", "v3.1"),
                [int(ep) for ep in state.episodes_df["episode_index"]],
            )
        for ep in req.example_episode_indices:
            atoms = json.loads(get_episode_atoms(ep, local_path=str(state.root)).body)["atoms"]
            if annotation_history.review_status(state.root, ep, atoms)["status"] != "reviewed":
                raise HTTPException(422, "Few-shot examples must be explicitly marked reviewed")
        annotations = {ep: ann.atoms for ep, ann in state.annotations.items()}
        annotations = {int(ep): atoms for ep, atoms in json.loads(json.dumps(annotations)).items()}
        selected = set(req.episode_indices if req.episode_indices is not None else map(int, run["episodes"]))
        selected -= set(req.example_episode_indices)
        selected = {ep for ep in selected if run["episodes"][str(ep)].get("decision") != "delete"}
        if req.resume_unfinished:
            selected = {ep for ep in selected if run["episodes"][str(ep)]["generation_status"] != "generated"}
        if not selected:
            raise HTTPException(422, "No unfinished target episodes remain")
        selected = sorted(selected)

    def queued(job_id):
        with _annotation_edit_lock:
            _require_editable(state.root)
            current = store.read(run["run_id"])
            if current["revision"] != run["revision"]:
                raise HTTPException(409, "Annotations changed before generation was queued; retry")
            current.update(
                current_job_id=job_id,
                task_prompt=req.task_prompt,
                subtask_prompts=req.subtask_prompts or [],
                example_episode_indices=req.example_episode_indices,
                publication_state="draft",
                generation_config=req.config,
                assess_quality=req.assess_quality,
            )
            current.pop("export", None)
            for ep in selected:
                current["episodes"][str(ep)].update(
                    generation_status="pending", decision="pending", decision_reason=None
                )
            reviews = annotation_history.read_reviews(state.root)
            for ep in selected:
                reviews.pop(str(ep), None)
            annotation_history.write_reviews(state.root, reviews)
            store.save(current, run["revision"])

    def generate(output: Path) -> dict:
        # Each finished episode is a recoverable official-writer checkpoint.
        # ponytail: full shards are copied per episode; optimize checkpoint IO if it dominates VLM time.
        from dataclasses import asdict

        _download_full_dataset(state)
        validate_dataset_paths(state.root)
        source = state.root
        labels = annotations
        result = {
            "output_dir": str(source),
            "validation": {"ok": True, "errors": [], "warnings": []},
            "first_generated_episode_index": selected[0],
        }
        first_success = None
        for ep in selected:
            checkpoint = output / f"episode_{ep:06d}"
            with _annotation_edit_lock:
                current = store.read(run["run_id"])
                current["episodes"][str(ep)]["generation_status"] = "running"
                store.save(current, current["revision"])
            try:
                generated = engine.generate_dataset(
                    source,
                    checkpoint,
                    labels,
                    config,
                    [ep],
                    example_episode_indices=req.example_episode_indices,
                    task_prompt=req.task_prompt,
                    subtask_prompts=req.subtask_prompts,
                    assess_quality=req.assess_quality,
                )
                record = generated.get("episode_results", {}).get(
                    str(ep), {"generation_status": "generated", "issues": []}
                )
                source = Path(generated["output_dir"])
                labels = {
                    int(k): v["atoms"]
                    for k, v in json.loads((source / "meta/lerobot_annotations.json").read_text())[
                        "episodes"
                    ].items()
                }
                result = generated
                if record["generation_status"] == "generated" and first_success is None:
                    first_success = ep
            except Exception as exc:
                logger.exception("Generation failed for episode %s", ep)
                record = {
                    "generation_status": "failed",
                    "issues": [
                        {
                            "code": "generation_failed",
                            "source": "deterministic",
                            "severity": "error",
                            "message": str(exc),
                            "start": None,
                            "end": None,
                        }
                    ],
                }
                # Preserve failed attempts as failures; never freeze old labels as a prediction.
                annotation_history.snapshot_predictions(
                    source,
                    source,
                    set(),
                    {},
                    asdict(config),
                    engine.REVISION,
                    task_prompt=req.task_prompt,
                    subtask_prompts=req.subtask_prompts,
                    episode_results={str(ep): record},
                )
            with _annotation_edit_lock:
                current = store.read(run["run_id"])
                current["root"] = str(source.resolve())
                current["episodes"][str(ep)].update(record, decision="pending", decision_reason=None)
                store.attach(source, current)
                _register_local_dataset(source)
                store.save(current, current["revision"])
        result["output_dir"] = str(source)
        if first_success is None:
            result["validation"] = {
                "ok": False,
                "errors": ["No target episodes were generated; inspect the findings"],
                "warnings": [],
            }
        result["first_generated_episode_index"] = first_success if first_success is not None else selected[0]
        return result

    return _start_annotation_job(generate, queued)


@app.post("/api/annotation/delete-episodes")
def delete_annotation_episodes(req: DeleteEpisodesRequest) -> dict:
    engine = _official_engine()
    state = _ensure_state(req)
    indices = set(state.episodes_df["episode_index"])
    selected = set(req.episode_indices)
    if len(selected) != len(req.episode_indices) or selected - indices or selected == indices:
        raise HTTPException(
            status_code=422, detail="Select distinct existing episodes and keep at least one episode"
        )
    annotations = json.loads(json.dumps({str(ep): ann.atoms for ep, ann in state.annotations.items()}))
    annotations = {int(ep): atoms for ep, atoms in annotations.items()}

    def delete(output: Path) -> dict:
        _download_full_dataset(state)
        return engine.delete_dataset_episodes(state.root, output, annotations, req.episode_indices)

    return _start_annotation_job(delete)


@app.get("/api/annotation/jobs/{job_id}")
def get_annotation_job(job_id: str) -> dict:
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    if job_id in _annotation_jobs:
        return dict(_annotation_jobs[job_id])
    path = EXPORT_ROOT / "jobs" / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown job")
    job = json.loads(path.read_text())
    if job["status"] in {"running", "queued"}:
        job.update(status="interrupted", error="Worker restarted; explicitly resume unfinished episodes")
        annotation_runs.atomic_json(path, job)
    return job


@app.post("/api/annotation/validate")
def validate_annotation(payload: EpisodeAtomsPayload) -> dict:
    engine = _official_engine()
    state = _ensure_state(payload)
    if _episode_data_path(state, payload.episode_index) is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    records = list(engine.iter_episodes(state.root, only_episodes=(payload.episode_index,)))
    if not records:
        raise HTTPException(status_code=404, detail="Episode frames not found")
    return engine.validate_atoms(
        state.root, records, {payload.episode_index: [a.model_dump() for a in payload.atoms]}
    )


@app.api_route("/datasets/local/{dataset}/resolve/{revision}/{asset_path:path}", methods=["GET", "HEAD"])
def annotation_dataset_asset(dataset: str, revision: str, asset_path: str):
    root_text = _local_aliases().get(f"local/{dataset}")
    relative = Path(asset_path)
    if (
        not root_text
        or revision != "main"
        or not relative.parts
        or relative.parts[0] not in {"meta", "data", "videos"}
    ):
        raise HTTPException(status_code=404, detail="Unknown dataset asset")
    root = Path(root_text).resolve()
    path = (root / relative).resolve()
    if (
        root not in path.parents
        or path.suffix not in {".json", ".jsonl", ".parquet", ".mp4"}
        or not path.is_file()
    ):
        raise HTTPException(status_code=404, detail="Unknown dataset asset")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


def _require_editable(root: Path):
    manifest = root.parent / "manifest.json"
    if manifest.is_file():
        frozen = json.loads(manifest.read_text())
        if frozen.get("dataset_name") == root.name and frozen.get("format") in {"groot_v21", "rich"}:
            raise HTTPException(409, "Frozen export is read-only; edit the review draft and export again")
    run = annotation_runs.RunStore(EXPORT_ROOT).for_root(root)
    if run:
        if Path(run["root"]).resolve() != root.resolve():
            raise HTTPException(409, "This draft is superseded; open the generated dataset")
        job = _annotation_jobs.get(run.get("current_job_id"))
        if job and job["status"] in {"queued", "running"}:
            raise HTTPException(409, "Wait for the active annotation job before editing")


def _workflow(alias: str):
    root = _local_aliases().get("local/" + alias)
    if not root:
        raise HTTPException(404, "Unknown workflow dataset")
    store = annotation_runs.RunStore(EXPORT_ROOT)
    run = store.for_root(Path(root))
    if not run:
        raise HTTPException(404, "Prepare a dataset to start a workflow")
    return store, run


def _workflow_payload(run: dict) -> dict:
    from copy import deepcopy

    result = deepcopy(run)
    root = Path(result.pop("root"))
    result.pop("source_root", None)
    result.pop("source_file_hashes", None)
    result["current_repo_id"] = _register_local_dataset(root)
    sidecar = root / "meta/lerobot_annotations.json"
    episodes = json.loads(sidecar.read_text()).get("episodes", {}) if sidecar.exists() else {}
    snapshots = [json.loads(p.read_text()) for p in sorted((root / "meta/annotation_predictions").glob("*.json"))]
    for ep, data in result["episodes"].items():
        atoms = episodes.get(ep, {}).get("atoms")
        if atoms is None:
            try:
                atoms = json.loads(get_episode_atoms(int(ep), local_path=str(root)).body)["atoms"]
            except Exception as exc:
                atoms = []
                data.setdefault("issues", []).append(
                    {
                        "code": "unreadable_episode",
                        "source": "deterministic",
                        "severity": "error",
                        "message": str(exc),
                        "start": None,
                        "end": None,
                    }
                )
        data["atoms"] = atoms
        data["review"] = annotation_history.review_status(root, int(ep), atoms, data.get("excluded_intervals", []))
        data["predictions"] = [
            {"atoms": snap.get("episodes", {}).get(ep, {}).get("atoms"), "created_at": snap["created_at"]}
            for snap in snapshots
            if ep in snap.get("episodes", {}) or ep in snap.get("episode_results", {})
        ]
    try:
        from .annotation_metrics import summarize
    except ImportError:
        from annotation_metrics import summarize
    examples = set(result.get("example_episode_indices", []))
    for snap in snapshots:
        examples.update(snap.get("example_episode_indices", []))
    result["metrics"] = summarize(result["episodes"], list(examples))
    result["review_snapshot_sha256"] = _review_snapshot(result["episodes"])
    if result.get("export"):
        result["export"] = {
            k: v
            for k, v in result["export"].items()
            if k
            in {
                "manifest_sha256",
                "retained_episodes",
                "deleted_episodes",
                "old_to_new",
                "run_revision",
                "validation",
                "managed_changes",
                "main_files",
                "rich_files",
                "format",
                "instruction_mode",
                "dataset_name",
                "retained_frames",
                "clipped_episodes",
                "destination",
                "output_repo_id",
            }
        }
        if not os.environ.get("ANNOTATION_BACKEND_TOKEN") and run["export"].get("local_path"):
            result["export"]["local_path"] = run["export"]["local_path"]
    return result


def _review_snapshot(episodes):
    value = {
        ep: {
            "decision": data["decision"],
            "annotation_sha256": annotation_history.annotation_hash(data["atoms"]),
            "exclusions_sha256": annotation_history.exclusions_hash(data.get("excluded_intervals")),
        }
        for ep, data in episodes.items()
    }
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@app.get("/api/workflow/{alias}")
def get_workflow(alias: str):
    with _annotation_edit_lock:
        _, run = _workflow(alias)
        return _workflow_payload(run)


class WorkflowDecision(BaseModel):
    episode_index: int
    decision: str
    reason: str = ""
    expected_revision: int


@app.post("/api/workflow/{alias}/decision")
def set_workflow_decision(alias: str, payload: WorkflowDecision):
    with _annotation_edit_lock:
        store, run = _workflow(alias)
        _require_editable(Path(run["root"]))
        if (
            payload.decision not in {"pending", "keep", "delete"}
            or str(payload.episode_index) not in run["episodes"]
        ):
            raise HTTPException(422, "Select an existing episode and keep/delete/pending decision")
        if payload.decision == "delete" and not payload.reason.strip():
            raise HTTPException(422, "A deletion reason is required")
        run["episodes"][str(payload.episode_index)].update(
            decision=payload.decision, decision_reason=payload.reason
        )
        run["publication_state"] = "draft"
        run.pop("export", None)
        try:
            run = store.save(run, payload.expected_revision)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return _workflow_payload(run)


class WorkflowExclusions(BaseModel):
    episode_index: int
    expected_revision: int
    excluded_intervals: list[dict]


@app.post("/api/workflow/{alias}/exclusions")
def set_workflow_exclusions(alias: str, payload: WorkflowExclusions):
    try:
        from .annotation_clipping import normalize_exclusions
    except ImportError:
        from annotation_clipping import normalize_exclusions
    with _annotation_edit_lock:
        store, run = _workflow(alias)
        root = Path(run["root"])
        _require_editable(root)
        if run["revision"] != payload.expected_revision:
            raise HTTPException(409, "Workflow changed; reload before editing excluded intervals")
        ep = str(payload.episode_index)
        if ep not in run["episodes"]:
            raise HTTPException(422, "Select an existing episode")
        if run["episodes"][ep].get("decision") == "delete":
            raise HTTPException(422, "Undo episode deletion before editing excluded intervals")
        records = _official_engine().source_episode_rows(root)
        row = next((r for r in records if int(r["episode_index"]) == payload.episode_index), None)
        if row is None:
            raise HTTPException(422, "Episode metadata is missing")
        try:
            intervals = normalize_exclusions(payload.excluded_intervals, int(row["length"]))
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if intervals == run["episodes"][ep].get("excluded_intervals", []):
            return _workflow_payload(run)
        run["episodes"][ep]["excluded_intervals"] = intervals
        run["publication_state"] = "draft"
        run.pop("export", None)
        saved = store.save(run, payload.expected_revision)
        reviews = annotation_history.read_reviews(root)
        reviews.pop(ep, None)
        annotation_history.write_reviews(root, reviews)
        return _workflow_payload(saved)


class WorkflowRevision(BaseModel):
    expected_revision: int


class WorkflowBulkReview(WorkflowRevision):
    expected_review_sha256: str
    confirmed: Literal[True]


def _editable_workflow(alias, revision):
    store, run = _workflow(alias)
    _require_editable(Path(run["root"]))
    if run["revision"] != revision:
        raise HTTPException(409, "Workflow changed; reload before applying bulk actions")
    return store, run


@app.post("/api/workflow/{alias}/keep-remaining")
def keep_remaining(alias: str, payload: WorkflowRevision):
    with _annotation_edit_lock:
        store, run = _editable_workflow(alias, payload.expected_revision)
        changed = False
        for episode in run["episodes"].values():
            if episode.get("decision") == "pending":
                episode["decision"] = "keep"
                changed = True
        if changed:
            run["publication_state"] = "draft"
            run.pop("export", None)
            run = store.save(run, run["revision"])
        return _workflow_payload(run)


@app.post("/api/workflow/{alias}/review-retained")
def review_retained(alias: str, payload: WorkflowBulkReview):
    from datetime import datetime, timezone

    with _annotation_edit_lock:
        store, run = _editable_workflow(alias, payload.expected_revision)
        current = _workflow_payload(run)
        if current["review_snapshot_sha256"] != payload.expected_review_sha256:
            raise HTTPException(
                409, "Prompts, intervals or decisions changed; inspect the current review before confirming"
            )
        retained = {ep: row for ep, row in current["episodes"].items() if row["decision"] != "delete"}
        if not retained:
            raise HTTPException(422, "No retained episodes to review")
        # Validate every row first; a failure must not partially mark the batch.
        for ep, row in retained.items():
            if not row["atoms"]:
                raise HTTPException(422, f"Episode {ep} has no saved annotations to review")
            if any(issue.get("code") == "unreadable_episode" for issue in row.get("issues", [])):
                raise HTTPException(422, f"Episode {ep} cannot be read")
        root = Path(run["root"])
        reviews = annotation_history.read_reviews(root)
        timestamp = datetime.now(timezone.utc).isoformat()
        for ep, row in retained.items():
            reviews[ep] = {
                "annotation_sha256": row["review"]["annotation_sha256"],
                "exclusions_sha256": row["review"]["exclusions_sha256"],
                "reviewed_at": timestamp,
            }
        # Invalidate publication before the atomic review replacement. If the
        # review write fails, no stale frozen export can remain publishable.
        run["publication_state"] = "draft"
        run.pop("export", None)
        run = store.save(run, run["revision"])
        annotation_history.write_reviews(root, reviews)
        return _workflow_payload(run)


class WorkflowExport(WorkflowRevision):
    export_format: Literal["groot_v21", "rich"] | None = None
    instruction_mode: Literal["task", "subtask"] = "task"
    dataset_name: str = "retained-dataset"
    destination_repo_id: str | None = None
    destination_revision: str = "main"
    destination_private: bool = True


class WorkflowPublish(WorkflowExport):
    manifest_sha256: str


def _delivery_engine():
    try:
        from . import annotation_delivery
    except ImportError:
        import annotation_delivery
    return annotation_delivery


def _publication_engine():
    try:
        from . import annotation_publish
    except ImportError:
        import annotation_publish
    return annotation_publish


def _queue_workflow_operation(alias, expected_revision, operation):
    with _annotation_edit_lock:
        store, run = _workflow(alias)
        _require_editable(Path(run["root"]))
        if run["revision"] != expected_revision:
            raise HTTPException(409, "Workflow changed; refresh before exporting/publishing")

        def queued(job_id):
            run["current_job_id"] = job_id
            store.save(run, expected_revision)

        def execute(output):
            result = operation(run, output)
            with _annotation_edit_lock:
                current = store.read(run["run_id"])
                if "manifest_sha256" in result:
                    if result.get("local_path"):
                        result["output_repo_id"] = _register_local_dataset(Path(result["local_path"]))
                    current["export"] = result
                    current["publication_state"] = "exported"
                else:
                    current["publication"] = result
                    current["publication_state"] = "published"
                store.save(current, current["revision"])
            hidden = {"root", "rich_root", "export_root", "manifest_path"}
            if os.environ.get("ANNOTATION_BACKEND_TOKEN"):
                hidden.add("local_path")
            return {k: v for k, v in result.items() if k not in hidden}

        return _start_annotation_job(execute, queued)


@app.post("/api/workflow/{alias}/export")
def export_workflow(alias: str, payload: WorkflowExport):
    if payload.export_format:
        engine = _delivery_engine()
        options = payload.model_dump(exclude={"expected_revision"})
        try:
            engine.validate_options(options)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return _queue_workflow_operation(
            alias, payload.expected_revision, lambda run, output: engine.prepare_delivery(run, output, options)
        )
    return _queue_workflow_operation(
        alias, payload.expected_revision, lambda run, output: _publication_engine().prepare_export(run, output)
    )


@app.post("/api/workflow/{alias}/publish")
def publish_workflow(alias: str, payload: WorkflowPublish):
    with _annotation_edit_lock:
        _, run = _workflow(alias)
        if (
            run["publication_state"] != "exported"
            or run.get("export", {}).get("manifest_sha256") != payload.manifest_sha256
        ):
            raise HTTPException(409, "Prepare and review a current frozen export before publishing")
    return _queue_workflow_operation(
        alias,
        payload.expected_revision,
        lambda run, output: (
            _delivery_engine().publish_delivery(run, payload.manifest_sha256)
            if run.get("export", {}).get("format")
            else _publication_engine().publish_export(run, payload.manifest_sha256)
        ),
    )


@app.on_event("startup")
def recover_annotation_jobs():
    annotation_runs.recover_jobs(EXPORT_ROOT)
