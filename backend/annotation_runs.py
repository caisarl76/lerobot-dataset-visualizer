"""Persistent annotation run identity; execution status belongs to job records."""

from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

LOCK = RLock()  # One annotation worker/process; shared by editor mutations.


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def source_inventory(root: Path) -> dict[str, str]:
    """Hash original dataset bytes, excluding mutable editor/run/history sidecars."""
    root = Path(root).resolve()

    def original_files():
        files = {}
        for directory in ("data", "videos", "meta"):
            base = root / directory
            if base.is_symlink():
                raise ValueError("Source dataset directories must not be symlinks")
            for path in base.rglob("*"):
                relative = path.relative_to(root)
                if directory == "meta" and (
                    relative.parts[1].startswith("annotation_") or relative.parts[1] == "lerobot_annotations.json"
                ):
                    continue
                if path.is_symlink():
                    raise ValueError(f"Source dataset file must not be a symlink: {relative}")
                if path.is_file():
                    stat = path.stat()
                    files[relative.as_posix()] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return files

    before = original_files()
    hashes = {}
    for relative in sorted(before):
        digest = sha256()
        with (root / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    if original_files() != before:
        raise ValueError("Source dataset changed while capturing source hashes")
    return hashes


class RunStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def path(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid run ID")
        return self.workspace / "runs" / run_id / "run.json"

    def read(self, run_id: str) -> dict:
        return json.loads(self.path(run_id).read_text())

    def for_root(self, root: Path) -> dict | None:
        pointer = root / "meta/annotation_run.json"
        if not pointer.exists():
            return None
        return self.read(json.loads(pointer.read_text())["run_id"])

    def create(
        self,
        root: Path,
        source: Path,
        repo_id: str | None,
        source_commit: str | None,
        source_format: str,
        episodes: list[int],
    ) -> dict:
        run = dict(
            version=1,
            run_id=uuid4().hex,
            root=str(root.resolve()),
            source_root=str(source.resolve()),
            repo_id=repo_id,
            source_commit=source_commit,
            source_format=source_format,
            source_file_hashes=source_inventory(source),
            revision=1,
            current_job_id=None,
            publication_state="draft",
            task_prompt="",
            subtask_prompts=[],
            example_episode_indices=[],
            episodes={
                str(ep): dict(
                    original_episode_index=ep,
                    generation_status="pending",
                    issues=[],
                    decision="pending",
                    decision_reason=None,
                )
                for ep in episodes
            },
        )
        with LOCK:
            atomic_json(self.path(run["run_id"]), run)
            self.attach(root, run)
        return run

    def attach(self, root: Path, run: dict):
        atomic_json(root / "meta/annotation_run.json", {"run_id": run["run_id"]})

    def save(self, run: dict, expected_revision: int) -> dict:
        with LOCK:
            current = self.read(run["run_id"])
            if current["revision"] != expected_revision:
                raise ValueError("Run revision conflict; reload the review workspace")
            for key in ("repo_id", "source_commit", "source_root", "source_format", "source_file_hashes"):
                if run.get(key) != current.get(key):
                    raise ValueError(f"Immutable source identity cannot change: {key}")

            def original_ids(value):
                return {
                    key: state.get("original_episode_index", int(key)) for key, state in value["episodes"].items()
                }

            if original_ids(run) != original_ids(current):
                raise ValueError("Immutable original episode identities cannot change")
            updated = {**run, "revision": expected_revision + 1}
            atomic_json(self.path(run["run_id"]), updated)
            return updated

    def edited(self, root: Path):
        with LOCK:
            run = self.for_root(root)
            if run:
                if Path(run["root"]).resolve() != root.resolve():
                    raise ValueError("This draft has been superseded; open the current run")
                run["publication_state"] = "draft"
                run.pop("export", None)
                return self.save(run, run["revision"])


def recover_jobs(workspace: Path):
    with LOCK:
        for path in (workspace / "jobs").glob("*.json"):
            job = json.loads(path.read_text())
            if job.get("status") in {"queued", "running"}:
                job.update(status="interrupted", error="Worker restarted; resume unfinished episodes explicitly")
                atomic_json(path, job)
