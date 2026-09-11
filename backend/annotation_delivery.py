"""One retained dataset, one explicit destination frozen at export preview."""

import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.utils import validate_repo_id

try:
    from . import annotation_publish as publication, official_annotations as engine
    from .annotation_source import v21_records
    from .groot_materialize import materialize_groot_v21
except ImportError:
    import annotation_publish as publication
    from annotation_source import v21_records
    from groot_materialize import materialize_groot_v21
    import official_annotations as engine


def validate_options(options):
    if options["export_format"] not in {"groot_v21", "rich"}:
        raise ValueError("Choose GR00T v2.1 or rich annotations")
    if options["instruction_mode"] not in {"task", "subtask"}:
        raise ValueError("Choose full task or subtask instructions")
    name = options["dataset_name"]
    if name in {"frozen", "manifest.json"}:
        raise ValueError("That dataset folder name is reserved for export metadata")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name) or ".." in name:
        raise ValueError("Dataset folder name must contain letters, digits, dots, dashes or underscores")
    repo = options.get("destination_repo_id")
    if not repo:
        return
    validate_repo_id(repo)
    if repo.count("/") != 1:
        raise ValueError("Enter the full Hugging Face namespace/dataset name")
    revision = options["destination_revision"]
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}", revision)
        or any(value in revision for value in ("..", "//", "@{"))
        or revision.endswith(("/", ".", ".lock"))
        or any(part.startswith(".") or part.endswith(".lock") for part in revision.split("/"))
        or re.fullmatch(r"[a-fA-F0-9]{40}", revision)
    ):
        raise ValueError("Use a branch name such as main or ver.260909 for the destination version")
    _allowed(repo)


def _allowed(repo):
    allowed = {v.strip() for v in os.environ.get("ANNOTATION_HUB_REPOS", "").split(",") if v.strip()}
    if (allowed or os.environ.get("ANNOTATION_BACKEND_TOKEN")) and repo not in allowed:
        raise ValueError("Target repository is not in the publication allowlist")


def _destination(options, api):
    repo = options.get("destination_repo_id")
    if not repo:
        return None
    revision = options["destination_revision"]
    result = {"repo_id": repo, "revision": revision, "private": options["destination_private"]}
    try:
        info = api.repo_info(repo, repo_type="dataset", revision="main")
    except RepositoryNotFoundError:
        return {**result, "exists": False, "revision_exists": False, "expected_commit": None}
    refs = api.list_repo_refs(repo, repo_type="dataset")
    if any(tag.name == revision for tag in refs.tags):
        raise ValueError("Destination version is a tag; choose a writable branch name")
    branches = {branch.name: branch.target_commit for branch in refs.branches}
    return {
        **result,
        "private": bool(info.private),
        "exists": True,
        "revision_exists": revision in branches,
        "expected_commit": branches.get(revision, info.sha),
    }


def _files(root):
    result = publication._inventory(root)
    card = root / "README.md"
    if card.exists():
        result["README.md"] = publication._hash_file(card)
    return result


def prepare_delivery(run, output, options, api=None):
    validate_options(options)
    # Destination inspection is read-only. Repository creation occurs only after
    # the user confirms the resulting preview and presses Publish.
    destination = _destination(options, api or HfApi())
    if output.exists():
        raise ValueError("Export output already exists")
    output.mkdir(parents=True)
    try:
        frozen = publication.prepare_export(run, output / "frozen")
        dataset = output / options["dataset_name"]
        rich = Path(frozen["rich_root"])
        if options["export_format"] == "groot_v21":
            materialize_groot_v21(rich, dataset, options["instruction_mode"])
            for name in ("lerobot_annotations.json", "annotation_reviews.json", "source_episode_mapping.json"):
                shutil.copy2(rich / "meta" / name, dataset / "meta" / name)
            records = v21_records(dataset, set(range(frozen["retained_episodes"])))
        else:
            shutil.copytree(rich, dataset)
            records = list(engine.iter_episodes(dataset))
        publication._detach_export_files(dataset)
        publication._validate_structure(dataset, records)
        frames = sum(record.row_count for record in records)
        description = (
            "GR00T LeRobot v2.1" if options["export_format"] == "groot_v21" else "LeRobot rich annotations"
        )
        # Generate a truthful card for the selected dataset instead of retaining
        # a destination card that could describe different episodes or formats.
        card = (
            "---\ntask_categories:\n  - robotics\ntags:\n  - lerobot\n---\n\n"
            f"# {options['dataset_name']}\n\n{description}. "
            f"{len(records)} retained episodes, {frames} frames.\n\n"
            f"Instruction mode: {options['instruction_mode']}. "
            f"Removed source episodes: {frozen['deleted_episodes']}.\n\n"
            "Episode identity and retained-frame mappings are recorded in "
            "`meta/source_episode_mapping.json`.\n"
        )
        (dataset / "README.md").write_text(card)
        if options["export_format"] == "groot_v21":
            report = dataset / "meta/groot_instruction_export.json"
            if report.exists():
                values = json.loads(report.read_text())
                for key in ("annotated_root", "source_root", "output_dir"):
                    values.pop(key, None)
                report.write_text(json.dumps(values, indent=2) + "\n")
        manifest = {
            "version": 1,
            "run_id": run["run_id"],
            "dataset_name": options["dataset_name"],
            "format": options["export_format"],
            "instruction_mode": options["instruction_mode"],
            "destination": destination,
            "review_sha256": publication._review_digest(run),
            "frozen_manifest_sha256": frozen["manifest_sha256"],
            "files": _files(dataset),
        }
        publication._write(output / "manifest.json", manifest)
        digest = publication._hash_file(output / "manifest.json")["sha256"]
        return {
            **frozen,
            "export_root": str(output),
            "manifest_sha256": digest,
            "root": str(dataset),
            "local_path": str(dataset),
            "format": options["export_format"],
            "instruction_mode": options["instruction_mode"],
            "dataset_name": options["dataset_name"],
            "retained_frames": frames,
            "destination": destination,
        }
    except BaseException:
        shutil.rmtree(output)
        raise


def _current(api, repo, revision):
    try:
        return api.repo_info(repo, repo_type="dataset", revision=revision).sha
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return None


def publish_delivery(run, manifest_sha256, api=None):
    frozen = run.get("export") or {}
    output = Path(frozen.get("export_root", ""))
    path = output / "manifest.json"
    if (
        not path.is_file()
        or publication._hash_file(path)["sha256"] != manifest_sha256
        or frozen.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Frozen export manifest changed or is missing")
    manifest = publication._json(path)
    if manifest["run_id"] != run["run_id"] or manifest["review_sha256"] != publication._review_digest(run):
        raise ValueError("Frozen export is stale; review and preview again")
    if any(
        manifest[key] != frozen.get(key) for key in ("destination", "dataset_name", "format", "instruction_mode")
    ):
        raise ValueError("Frozen export destination or format changed")
    dataset = output / manifest["dataset_name"]
    if _files(dataset) != manifest["files"]:
        raise ValueError("Frozen export files changed after preview")
    target = manifest.get("destination")
    if not target:
        raise ValueError("This preview is local only; enter a destination and preview again")
    repo, revision = target["repo_id"], target["revision"]
    _allowed(repo)
    api = api or HfApi()
    head = _current(api, repo, revision)
    if head:
        receipt = publication._receipt(api, repo, head)
        if receipt and receipt.get("manifest_sha256") == manifest_sha256:
            publication._verify_remote(api, repo, head, manifest["files"])
            return _result(repo, revision, head)
    if target["revision_exists"]:
        if head != target["expected_commit"]:
            raise ValueError("Destination branch changed since preview; create a new preview")
    elif head:
        raise ValueError("Destination branch was created since preview; create a new preview")
    elif target["exists"]:
        if _current(api, repo, "main") != target["expected_commit"]:
            raise ValueError("Destination main changed since preview; create a new preview")
    elif _current(api, repo, "main") is not None:
        raise ValueError("Destination repository appeared since preview; create a new preview")
    # Enforce the single-commit cap before any remote mutation, even repo creation.
    remote = publication._remote_files(api, repo, target["expected_commit"]) if target["exists"] else {}
    operations = publication._operations(dataset, manifest["files"], remote)
    if not target["exists"]:
        api.create_repo(repo, repo_type="dataset", private=target["private"], exist_ok=False)
        head = _current(api, repo, "main")
        if not head:
            raise ValueError("New repository has no initial commit; preview again")
    else:
        head = target["expected_commit"]
    if not target["revision_exists"] and revision != "main":
        api.create_branch(repo, repo_type="dataset", branch=revision, revision=head, exist_ok=False)
    receipt = {
        "version": 1,
        "run_id": run["run_id"],
        "manifest_sha256": manifest_sha256,
        "repo_id": repo,
        "revision": revision,
        "format": manifest["format"],
    }
    operations.append(
        CommitOperationAdd(path_in_repo=publication.RECEIPT, path_or_fileobj=json.dumps(receipt).encode())
    )
    commit = api.create_commit(
        repo,
        repo_type="dataset",
        revision=revision,
        parent_commit=head,
        operations=operations,
        commit_message=f"Publish {manifest['dataset_name']} ({manifest['format']})",
    ).oid
    publication._verify_remote(api, repo, commit, manifest["files"])
    return _result(repo, revision, commit)


def _result(repo, revision, commit):
    url = f"https://huggingface.co/datasets/{repo}/tree/{quote(revision, safe='')}"
    return {
        "repo_id": repo,
        "revision": revision,
        "main_commit": commit,
        "urls": {"main": url},
        "commit_url": f"https://huggingface.co/datasets/{repo}/commit/{commit}",
    }
