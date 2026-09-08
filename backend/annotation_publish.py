"""Freeze reviewed datasets, then publish bounded commits with explicit parent SHAs.

Each revision is limited to 500 changed-file operations. Larger exports must use
a separately implemented staging transfer; this module never partially uploads main.
"""

from __future__ import annotations

from hashlib import sha1, sha256
import json
import math
import os
from pathlib import Path
import shutil
from urllib.parse import quote

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
import numpy as np
import pyarrow.parquet as pq

try:
    from . import official_annotations as engine
    from .annotation_history import annotation_hash, read_reviews
    from .annotation_quality import check_prompt_sequence
    from .annotation_runs import source_inventory
    from .annotation_source import align_source_v21 as _align_source_v21, v21_records
    from .groot_export import export_groot_dataset
    from .official_annotations import delete_dataset_episodes
except ImportError:
    from annotation_history import annotation_hash, read_reviews
    from annotation_quality import check_prompt_sequence
    from annotation_runs import source_inventory
    from annotation_source import align_source_v21 as _align_source_v21, v21_records
    from groot_export import export_groot_dataset
    import official_annotations as engine
    from official_annotations import delete_dataset_episodes

MAX_OPERATIONS = 500
RECEIPT = "meta/annotation_publication.json"


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _hash_file(path):
    size = path.stat().st_size
    digest, git_digest = sha256(), sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            git_digest.update(chunk)
    return {"sha256": digest.hexdigest(), "git_blob": git_digest.hexdigest(), "size": size}


def _inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.relative_to(root).parts[0] not in {"meta", "data", "videos"}:
            continue
        if path.is_symlink():
            raise ValueError("Frozen export must not contain symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _hash_file(path)
    return files


def _sanitize_metadata(value, source_ref):
    if isinstance(value, list):
        return [_sanitize_metadata(item, source_ref) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"api_key", "token", "serve_command", "api_base"}:
            continue
        if key in {
            "root",
            "source",
            "source_root",
            "annotated_root",
            "output_dir",
            "staging_dir",
            "original_root",
        }:
            if isinstance(item, str) and item.startswith("/"):
                result[key] = source_ref if "source" in key or key == "original_root" else "."
                continue
        result[key] = _sanitize_metadata(item, source_ref)
    return result


def _publication_metadata(root, run, mapping, reviews):
    for relative in ("meta/annotation_run.json", RECEIPT):
        pointer = root / relative
        if pointer.exists():
            pointer.unlink()
    for path in (root / "meta").rglob("*.json"):
        _write(path, _sanitize_metadata(_json(path), f"{run.get('repo_id')}@{run.get('source_commit')}"))
    _write(root / "meta/annotation_reviews.json", reviews)
    _write(
        root / "meta/source_episode_mapping.json",
        {
            "repo_id": run.get("repo_id"),
            "source_commit": run.get("source_commit"),
            "run_id": run["run_id"],
            "run_revision": run["revision"],
            "old_to_new": mapping,
            "prediction_identity_space": "source_episode_index",
            "task_prompt": run.get("task_prompt", ""),
            "subtask_prompts": run.get("subtask_prompts", []),
            "example_episode_indices": run.get("example_episode_indices", []),
            "episode_decisions": {
                str(state.get("original_episode_index", ep)): _sanitize_metadata(
                    state, f"{run.get('repo_id')}@{run.get('source_commit')}"
                )
                for ep, state in run["episodes"].items()
            },
            "decisions": {
                str(state.get("original_episode_index", ep)): state.get("decision")
                for ep, state in run["episodes"].items()
            },
        },
    )


def _review_digest(run):
    root = Path(run["root"])
    state = {
        key: run.get(key) for key in ("episodes", "task_prompt", "subtask_prompts", "example_episode_indices")
    }
    state.update(annotations=_json(root / "meta/lerobot_annotations.json"), reviews=read_reviews(root))
    return sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _source_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe source dataset file: {relative}")
    return path


def _validate_structure(root, records):
    """Check retained robot rows and every decoded camera timestamp, not just atoms.

    v3 video paths and offsets come from the official metadata implementation;
    PyAV checks actual media rather than a frame provider's fallback images.
    """
    import av
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    info = _json(root / "meta/info.json")
    fps = info.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid structural dataset FPS")
    features = info.get("features", {})
    cameras = {key: value for key, value in features.items() if value.get("dtype") == "video"}
    is_v21 = info.get("codebase_version") == "v2.1"
    metadata = None if is_v21 else LeRobotDatasetMetadata(repo_id="local", root=root)
    episode_metadata = (
        {row["episode_index"]: row for row in _jsonl(root / "meta/episodes.jsonl")}
        if is_v21
        else metadata.episodes
    )
    decoded = {}
    for record in records:
        ep = record.episode_index
        try:
            row = episode_metadata[ep]
            table = pq.read_table(_source_path(root, str(record.data_path.relative_to(root)))).slice(
                record.row_offset, record.row_count
            )
            count = table.num_rows
            if count <= 0 or count != row["length"] or count != record.row_count:
                raise ValueError("episode metadata/frame count mismatch")
            if table["episode_index"].to_pylist() != [ep] * count or table["frame_index"].to_pylist() != list(
                range(count)
            ):
                raise ValueError("episode/frame identity mismatch")
            times = np.asarray(table["timestamp"].to_pylist(), dtype=float)
            if not np.isfinite(times).all() or not np.allclose(
                times, np.arange(count) / fps, atol=1e-4, rtol=1e-5
            ):
                raise ValueError("invalid frame timestamps")
            indices = np.asarray(table["index"].to_pylist())
            if not np.array_equal(indices, np.arange(indices[0], indices[0] + count)):
                raise ValueError("invalid global frame indices")
            if not is_v21 and indices[0] != row["dataset_from_index"]:
                raise ValueError("episode metadata/global index mismatch")
            for key, feature in features.items():
                try:
                    dtype = np.dtype(feature["dtype"])
                except (TypeError, KeyError):
                    continue  # Official rich-language structs and media have their own validators.
                if dtype.kind not in "biuf":
                    continue
                column = table[key]
                values = np.asarray(column.to_pylist())
                shape = tuple(feature["shape"])
                if column.null_count or (
                    values.shape != (count, *shape) and not (shape == (1,) and values.shape == (count,))
                ):
                    raise ValueError(f"invalid numeric feature shape/nulls: {key}")
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite robot feature: {key}")
            for camera, feature in cameras.items():
                if is_v21:
                    relative = info["video_path"].format(
                        episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=camera
                    )
                    start, end = 0.0, count / fps
                else:
                    relative = metadata.get_video_file_path(ep, camera)
                    start = float(row[f"videos/{camera}/from_timestamp"])
                    end = float(row[f"videos/{camera}/to_timestamp"])
                video = _source_path(root, relative)
                if video not in decoded:
                    timestamps = []
                    with av.open(str(video)) as container:
                        if len(container.streams.video) != 1:
                            raise ValueError("expected a single video stream")
                        stream = container.streams.video[0]
                        if stream.average_rate is None or not math.isclose(
                            float(stream.average_rate), fps, rel_tol=1e-4
                        ):
                            raise ValueError("video FPS mismatch")
                        for frame in container.decode(stream):
                            if [frame.height, frame.width, 3] != list(feature["shape"]):
                                raise ValueError("video dimensions mismatch")
                            if frame.time is None:
                                raise ValueError("video frame has no timestamp")
                            timestamps.append(frame.time)
                    decoded[video] = np.asarray(timestamps)
                actual = decoded[video]
                if not len(actual) or not np.isfinite(actual).all() or (np.diff(actual) <= 0).any():
                    raise ValueError("invalid decoded video timestamps")
                expected = times + start
                closest = np.searchsorted(actual, expected)
                closest = np.clip(closest, 0, len(actual) - 1)
                previous = np.maximum(closest - 1, 0)
                distances = np.minimum(abs(actual[closest] - expected), abs(actual[previous] - expected))
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or end < expected[-1]
                    or end > actual[-1] + 1 / fps + 1e-3
                    or (distances > 1e-3).any()
                ):
                    raise ValueError("missing video frames or invalid episode video bounds")
                if is_v21 and len(actual) != count:
                    raise ValueError("video frame count mismatch")
        except Exception as error:
            raise ValueError(f"Episode {ep} failed structural data/video validation: {error}") from error


def _detach_export_files(root):
    """Replace any hardlinks inherited from upstream copying with independent bytes."""
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_nlink > 1:
            temporary = path.with_name(path.name + ".freeze-copy")
            try:
                shutil.copy2(path, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def _verify_references(api, repo, run, rich_commit, rich_revision):
    if rich_revision != f"annotations/{run['run_id']}":
        raise ValueError("Rich revision identity conflict")
    for revision, expected in ((f"raw/{run['run_id']}", run["source_commit"]), (rich_revision, rich_commit)):
        if api.repo_info(repo, repo_type="dataset", revision=revision).sha != expected:
            raise ValueError(f"Publication revision conflict: {revision}")


def prepare_export(run: dict, output: Path) -> dict:
    """Freeze a new independent export; caller serializes against run/editor changes."""
    root, source, output = Path(run["root"]).resolve(), Path(run["source_root"]).resolve(), Path(output).resolve()
    for input_root in (root, source):
        if output == input_root or output.is_relative_to(input_root) or input_root.is_relative_to(output):
            raise ValueError("Export must be outside the source and draft")
    if output.exists():
        raise ValueError("Export output already exists")
    expected_source = run.get("source_file_hashes")
    if not isinstance(expected_source, dict) or source_inventory(source) != expected_source:
        raise ValueError("Immutable source hashes are missing or source files changed; prepare a new pinned run")
    input_files = {path: _inventory(path) for path in {root, source}}
    states = run["episodes"]
    raw_v21 = (
        run.get("preparation_mode") == "source_review"
        and _json(root / "meta/info.json").get("codebase_version") == "v2.1"
    )
    if raw_v21:
        episode_rows = _jsonl(root / "meta/episodes.jsonl")
        ids = {str(row["episode_index"]) for row in episode_rows}
        if len(ids) != len(episode_rows):
            raise ValueError("Duplicate raw source episode identities")
        selected = [int(ep) for ep in ids if states.get(ep, {}).get("decision") != "delete"]
        records = v21_records(root, selected)
    else:
        try:
            records = list(engine.iter_episodes(root))
        except Exception as error:
            if run.get("preparation_mode") == "source_review":
                raise ValueError(
                    "Cannot safely recover corrupt/shared v3 parquet shards; prepare a verified repaired source"
                ) from error
            raise
        ids = {str(record.episode_index) for record in records}
    if not ids or set(states) != ids:
        raise ValueError("Run episode identities do not match the draft")
    deleted = sorted(int(ep) for ep in ids if states[ep].get("decision") == "delete")
    retained = [record for record in records if record.episode_index not in deleted]
    if not retained:
        raise ValueError("Cannot delete all episodes")
    _validate_structure(root, retained)
    saved = _json(root / "meta/lerobot_annotations.json")["episodes"]
    reviews = read_reviews(root)
    labels = {}
    for record in retained:
        ep = str(record.episode_index)
        if ep not in saved:
            raise ValueError(f"Episode {ep} has no saved human annotations")
        atoms = saved[ep]["atoms"]
        review = reviews.get(ep, {})
        if not review.get("reviewed_at") or review.get("annotation_sha256") != annotation_hash(atoms):
            raise ValueError(f"Episode {ep} needs a current explicit human review")
        if states[ep].get("issues") and states[ep].get("decision") != "keep":
            raise ValueError(f"Episode {ep} has unresolved quality findings")
        if check_prompt_sequence(atoms, run.get("subtask_prompts") or []):
            raise ValueError(f"Episode {ep} still has a subtask prompt mismatch")
        times = record.frame_timestamps
        if not times or len(times) != record.row_count or any(not math.isfinite(t) for t in times):
            raise ValueError(f"Episode {ep} has invalid source frames")
        for atom in atoms:
            timestamp = atom.get("timestamp")
            if (
                type(timestamp) not in (int, float)
                or not math.isfinite(timestamp)
                or not times[0] <= timestamp <= times[-1]
            ):
                raise ValueError(f"Episode {ep} has invalid annotation time bounds")
            engine._validate_atom_invariants(atom)
            engine._validate_speech_atom(atom)
            if engine.column_for_style(atom.get("style")) == engine.LANGUAGE_PERSISTENT:
                engine._normalize_persistent_row(atom)
            else:
                engine._normalize_event_row(atom)
        labels[record.episode_index] = atoms
    validation = engine.validate_atoms(root, retained, labels)
    if not validation["ok"]:
        raise ValueError("Retained annotations failed official validation: " + "\n".join(validation["errors"]))
    if run["source_format"] not in {"v2.1", "v3.0", "v3.1"}:
        raise ValueError("Unsupported source format")
    current_to_new = {
        record.episode_index: i for i, record in enumerate(sorted(retained, key=lambda r: r.episode_index))
    }
    mapping = {str(states[str(ep)].get("original_episode_index", ep)): new for ep, new in current_to_new.items()}
    if len(mapping) != len(current_to_new):
        raise ValueError("Original episode identities must be unique")
    if run["source_format"] == "v2.1":
        source_records = v21_records(source, {int(ep) for ep in mapping})
        if {str(record.episode_index) for record in source_records} != set(mapping):
            raise ValueError("Retained source episode identities are missing")
        _validate_structure(source, source_records)
    mapped_reviews = {str(new): reviews[str(ep)] for ep, new in current_to_new.items()}
    output.mkdir(parents=True)
    rich, main = output / "rich", output / "main"
    try:
        if raw_v21:
            # Pinned official delete_episodes requires a v3 LeRobotDataset and
            # cannot read raw v2 or corrupt excluded shards. In this recovery
            # exception only, materialize retained IDs into frozen v2 staging,
            # apply the single mapping there, then use official conversion and
            # writer. Review/source files and their identities remain untouched.
            aligned = output / "aligned_source"
            _align_source_v21(source, aligned, mapping)
            converted = output / "converted_source"
            engine.prepare_dataset(aligned, converted)
            if not _json(converted / "meta/info.json").get("codebase_version", "").startswith("v3"):
                raise ValueError("Retained raw source could not be converted safely")
            remapped_labels = {current_to_new[ep]: atoms for ep, atoms in labels.items()}
            engine.export_dataset(converted, rich, remapped_labels, copy_videos=True)
            history = root / "meta/annotation_predictions"
            if history.exists():
                shutil.copytree(history, rich / "meta/annotation_predictions", dirs_exist_ok=True)
        elif deleted:
            result = delete_dataset_episodes(root, rich, labels, deleted)
            if result["old_to_new"] != current_to_new:
                raise ValueError("Official deletion produced an unexpected identity mapping")
            history = root / "meta/annotation_predictions"
            if history.exists():
                shutil.copytree(history, rich / "meta/annotation_predictions", dirs_exist_ok=True)
        else:
            engine.export_dataset(root, rich, labels)
        _publication_metadata(rich, run, mapping, mapped_reviews)
        if run["source_format"] == "v2.1":
            aligned = output / "aligned_source"
            if not raw_v21:
                _align_source_v21(source, aligned, mapping)
            export_groot_dataset(rich, aligned, main, mode="subtask")
            for name in ("lerobot_annotations.json", "annotation_reviews.json", "source_episode_mapping.json"):
                shutil.copy2(rich / "meta" / name, main / "meta" / name)
            _publication_metadata(main, run, mapping, mapped_reviews)
        else:
            engine.copy_dataset(rich, main)
        for dataset in (rich, main):
            _detach_export_files(dataset)
        _validate_structure(rich, list(engine.iter_episodes(rich)))
        _validate_structure(main, list(engine.iter_episodes(main)))
        if any(_inventory(path) != files for path, files in input_files.items()):
            raise ValueError("Source or draft files changed during export preparation")
        manifest = {
            "version": 1,
            "run_id": run["run_id"],
            "run_revision": run["revision"],
            "repo_id": run.get("repo_id"),
            "source_commit": run.get("source_commit"),
            "source_format": run["source_format"],
            "old_to_new": mapping,
            "review_sha256": _review_digest(run),
            "source_files": input_files[source],
            "files": {"main": _inventory(main), "rich": _inventory(rich)},
        }
        _write(output / "manifest.json", manifest)
        return {
            "root": str(main),
            "rich_root": str(rich),
            "export_root": str(output),
            "manifest_sha256": _hash_file(output / "manifest.json")["sha256"],
            "run_revision": run["revision"],
            "retained_episodes": len(retained),
            "deleted_episodes": deleted,
            "old_to_new": mapping,
            "validation": validation,
            "managed_changes": {
                "added_or_updated": sorted(
                    name
                    for name, digest in manifest["files"]["main"].items()
                    if input_files[source].get(name) != digest
                ),
                "deleted": sorted(
                    name
                    for name in input_files[source]
                    if _managed(name) and name not in manifest["files"]["main"]
                ),
            },
            "main_files": len(manifest["files"]["main"]),
            "rich_files": len(manifest["files"]["rich"]),
        }
    except BaseException:
        shutil.rmtree(output)
        raise


def _managed(path):
    if path.startswith("data/") and path.endswith(".parquet"):
        return True
    if path.startswith("videos/") and path.endswith(".mp4"):
        return True
    return path.startswith(("meta/episodes/", "meta/annotation_predictions/")) or path in {
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/tasks.parquet",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/modality.json",
        "meta/lerobot_annotations.json",
        "meta/annotation_reviews.json",
        "meta/annotation_pipeline.json",
        "meta/source_episode_mapping.json",
        "meta/groot_instruction_export.json",
        "meta/annotation_run.json",
        RECEIPT,
    }


def _remote_files(api, repo, revision):
    paths = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    files = {}
    for offset in range(0, len(paths), 100):
        for item in api.get_paths_info(repo, paths[offset : offset + 100], repo_type="dataset", revision=revision):
            lfs = getattr(item, "lfs", None)
            digest = (
                (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            )
            files[item.path] = {"sha256": digest, "git_blob": item.blob_id, "size": item.size}
    return files


def _matches(local, remote):
    if not remote or local["size"] != remote["size"]:
        return False
    return local["sha256"] == remote["sha256"] if remote["sha256"] else local["git_blob"] == remote["git_blob"]


def _operations(root, files, remote):
    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(root / path))
        for path, value in files.items()
        if not _matches(value, remote.get(path))
    ]
    operations += [
        CommitOperationDelete(path_in_repo=path)
        for path in remote
        if _managed(path) and path not in files and path != RECEIPT
    ]
    # Leave one operation for the publication receipt.
    if len(operations) + 1 > MAX_OPERATIONS:
        raise ValueError(f"Export exceeds the {MAX_OPERATIONS}-operation single-commit publication cap")
    return operations


def _verify_remote(api, repo, revision, files):
    remote = _remote_files(api, repo, revision)
    if any(not _matches(value, remote.get(path)) for path, value in files.items()):
        raise ValueError("Published revision does not match the frozen manifest")
    if any(_managed(path) and path not in files and path != RECEIPT for path in remote):
        raise ValueError("Published revision still contains obsolete managed files")


def _receipt(api, repo, revision):
    if RECEIPT not in api.list_repo_files(repo, repo_type="dataset", revision=revision):
        return None
    return _json(api.hf_hub_download(repo, RECEIPT, repo_type="dataset", revision=revision))


def _result(repo, main_commit, rich_commit, rich_revision):
    raw_revision = rich_revision.replace("annotations/", "raw/", 1)
    return {
        "main_commit": main_commit,
        "rich_commit": rich_commit,
        "rich_revision": rich_revision,
        "urls": {
            "main": f"https://huggingface.co/datasets/{repo}/tree/{main_commit}",
            "rich": f"https://huggingface.co/datasets/{repo}/tree/{quote(rich_revision, safe='')}",
            "raw": f"https://huggingface.co/datasets/{repo}/tree/{quote(raw_revision, safe='')}",
        },
    }


def publish_export(run: dict, manifest_sha256: str, api=None) -> dict:
    """Publish a verified frozen export; retries recognize a completed main commit."""
    repo = run.get("repo_id")
    allowed = {item.strip() for item in os.environ.get("ANNOTATION_HUB_REPOS", "").split(",") if item.strip()}
    if not repo or repo not in allowed:
        raise ValueError("Target repository is not in the publication allowlist")
    frozen = run.get("export") or {}
    root = Path(frozen.get("export_root", ""))
    manifest_path = root / "manifest.json"
    if (
        not manifest_path.is_file()
        or _hash_file(manifest_path)["sha256"] != manifest_sha256
        or frozen.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Frozen export manifest changed or is missing")
    manifest = _json(manifest_path)
    if any(manifest.get(key) != run.get(key) for key in ("run_id", "repo_id", "source_commit", "source_format")):
        raise ValueError("Frozen export identity changed")
    if manifest.get("review_sha256") != _review_digest(run):
        raise ValueError("Frozen export is stale: annotations, reviews, or decisions changed")
    for name in ("main", "rich"):
        if _inventory(root / name) != manifest["files"][name]:
            raise ValueError("Frozen export files changed after preparation")
    api = api if api is not None else HfApi()
    main_sha = api.repo_info(repo, repo_type="dataset", revision="main").sha
    receipt = _receipt(api, repo, main_sha)
    if receipt and receipt.get("manifest_sha256") == manifest_sha256:
        _verify_references(api, repo, run, receipt["rich_commit"], receipt["rich_revision"])
        _verify_remote(api, repo, main_sha, manifest["files"]["main"])
        _verify_remote(api, repo, receipt["rich_commit"], manifest["files"]["rich"])
        return _result(repo, main_sha, receipt["rich_commit"], receipt["rich_revision"])
    source_sha = run.get("source_commit")
    if not source_sha or main_sha != source_sha:
        raise ValueError("Hub main conflict: source commit changed; preserve the draft and rebase explicitly")
    remote = _remote_files(api, repo, source_sha)
    main_ops = _operations(root / "main", manifest["files"]["main"], remote)
    rich_ops = _operations(root / "rich", manifest["files"]["rich"], remote)
    raw_revision, rich_revision = f"raw/{run['run_id']}", f"annotations/{run['run_id']}"
    api.create_branch(repo, repo_type="dataset", branch=raw_revision, revision=source_sha, exist_ok=True)
    if api.repo_info(repo, repo_type="dataset", revision=raw_revision).sha != source_sha:
        raise ValueError("Raw source revision conflict")
    api.create_branch(repo, repo_type="dataset", branch=rich_revision, revision=source_sha, exist_ok=True)
    rich_sha = api.repo_info(repo, repo_type="dataset", revision=rich_revision).sha
    rich_receipt = _receipt(api, repo, rich_sha)
    common = {
        "version": 1,
        "run_id": run["run_id"],
        "source_commit": source_sha,
        "manifest_sha256": manifest_sha256,
        "rich_revision": rich_revision,
    }
    if rich_sha != source_sha:
        if not rich_receipt or rich_receipt.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Rich annotation revision conflict")
    else:
        rich_ops.append(CommitOperationAdd(path_in_repo=RECEIPT, path_or_fileobj=json.dumps(common).encode()))
        rich_sha = api.create_commit(
            repo,
            repo_type="dataset",
            revision=rich_revision,
            parent_commit=source_sha,
            operations=rich_ops,
            commit_message="Freeze reviewed rich annotations",
        ).oid
    _verify_remote(api, repo, rich_sha, manifest["files"]["rich"])
    main_ops.append(
        CommitOperationAdd(
            path_in_repo=RECEIPT, path_or_fileobj=json.dumps({**common, "rich_commit": rich_sha}).encode()
        )
    )
    main_sha = api.create_commit(
        repo,
        repo_type="dataset",
        revision="main",
        parent_commit=source_sha,
        operations=main_ops,
        commit_message="Publish reviewed annotation dataset",
    ).oid
    _verify_remote(api, repo, main_sha, manifest["files"]["main"])
    _verify_references(api, repo, run, rich_sha, rich_revision)
    return _result(repo, main_sha, rich_sha, rich_revision)
