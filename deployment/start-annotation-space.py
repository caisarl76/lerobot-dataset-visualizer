"""Start the loopback annotation backend and public Next server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

MODEL = "qwen/qwen3.5-397b-a17b-fp8"


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _validate_api_base(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("ANNOTATION_VLM_API_BASE must be an HTTPS URL without credentials, query, or fragment")
    return value.rstrip("/")


def configure_environment() -> tuple[dict[str, str], Path]:
    workspace = Path(_required("ANNOTATION_WORKSPACE")).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    api_key = _required("GENON_API_KEY")
    api_base = _validate_api_base(_required("ANNOTATION_VLM_API_BASE"))
    hf_token = _required("HF_TOKEN")
    if _required("ANNOTATION_HOSTED_PRIVATE_SPACE") != "1":
        raise RuntimeError("ANNOTATION_HOSTED_PRIVATE_SPACE must be 1")
    origin = os.environ.get("ANNOTATION_BROWSER_ORIGIN", "").strip() or f"https://{_required('SPACE_HOST')}"
    config_path = workspace / "annotation-runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "vlm": {
                    "backend": "openai",
                    "api_base": api_base,
                    "model_id": MODEL,
                    "client_concurrency": 2,
                    "max_new_tokens": 2048,
                },
                "executor": {"episode_parallelism": 1},
                "video_backend": "pyav",
            },
            indent=2,
        )
        + "\n"
    )
    env = dict(os.environ)
    env.update(
        ANNOTATION_BACKEND_TOKEN=secrets.token_urlsafe(32),
        ANNOTATION_BACKEND_URL="http://127.0.0.1:7861",
        ANNOTATION_BROWSER_ORIGIN=origin,
        LEROBOT_ANNOTATE_CONFIG=str(config_path),
        LEROBOT_ANNOTATE_EXPORT=str(workspace / "annotations"),
        LEROBOT_ANNOTATE_CACHE=str(workspace / "cache"),
        LEROBOT_VLM_API_KEY=api_key,
        HF_TOKEN=hf_token,
        ANNOTATION_HUB_REPOS=os.environ.get("ANNOTATION_HUB_REPOS", "mncai/G1_Dex3_PickAndPlaceTrash"),
    )
    return env, config_path


def run() -> int:
    backend_env, _ = configure_environment()
    ui_env = dict(backend_env)
    for key in ("GENON_API_KEY", "LEROBOT_VLM_API_KEY", "HF_TOKEN"):
        ui_env.pop(key, None)
    ui_env["NODE_ENV"] = "production"
    children = []
    interrupted = False

    def cleanup() -> None:
        for child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(child.poll() is None for child in children):
            time.sleep(0.05)
        for child in children:
            if child.poll() is None:
                child.kill()

    def stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        cleanup()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        children.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "7861"],
                env=backend_env,
            )
        )
        children.append(
            subprocess.Popen(
                ["node", "node_modules/next/dist/bin/next", "start", "--hostname", "0.0.0.0", "--port", "7860"],
                env=ui_env,
            )
        )
        while True:
            statuses = [child.poll() for child in children]
            if any(status is not None for status in statuses):
                return 0 if interrupted else next(status for status in statuses if status is not None) or 1
            time.sleep(0.2)
    finally:
        cleanup()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(run())
