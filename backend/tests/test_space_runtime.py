import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "deployment" / "start-annotation-space.py"
spec = importlib.util.spec_from_file_location("space_runtime", SCRIPT)
space_runtime = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(space_runtime)


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("ANNOTATION_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GENON_API_KEY", "genon-placeholder")
    monkeypatch.setenv("ANNOTATION_VLM_API_BASE", "https://api.example.test/v1")
    monkeypatch.setenv("HF_TOKEN", "hf-placeholder")
    monkeypatch.setenv("SPACE_HOST", "space.example.test")
    monkeypatch.setenv("ANNOTATION_HOSTED_PRIVATE_SPACE", "1")


def test_runtime_config_keeps_secrets_out_of_config(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    backend_env, config_path = space_runtime.configure_environment()
    config = json.loads(config_path.read_text())
    assert config["vlm"] == {
        "backend": "openai",
        "api_base": "https://api.example.test/v1",
        "model_id": space_runtime.MODEL,
        "client_concurrency": 2,
        "max_new_tokens": 2048,
    }
    assert "genon-placeholder" not in config_path.read_text()
    assert backend_env["LEROBOT_VLM_API_KEY"] == "genon-placeholder"
    assert backend_env["ANNOTATION_BACKEND_URL"] == "http://127.0.0.1:7861"
    assert backend_env["ANNOTATION_BROWSER_ORIGIN"] == "https://space.example.test"
    assert backend_env["ANNOTATION_HOSTED_PRIVATE_SPACE"] == "1"


@pytest.mark.parametrize(
    "value", ["http://api.example.test", "https://user:pass@api.example.test", "https://api.example.test/v1?x=1"]
)
def test_api_base_must_be_https_without_url_secrets(monkeypatch, tmp_path, value):
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANNOTATION_VLM_API_BASE", value)
    with pytest.raises(RuntimeError):
        space_runtime.configure_environment()


def test_ui_environment_drops_all_runtime_secrets(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    processes = []

    class Child:
        def __init__(self, args, env):
            self.args, self.env, self.terminated = args, env, False
            self.returncode = None
            processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(space_runtime.subprocess, "Popen", Child)
    monkeypatch.setattr(space_runtime.time, "sleep", lambda _: processes[1].__setattr__("returncode", 0))
    assert space_runtime.run() == 1
    assert processes[0].env["LEROBOT_VLM_API_KEY"] == "genon-placeholder"
    assert all(key not in processes[1].env for key in ("GENON_API_KEY", "LEROBOT_VLM_API_KEY", "HF_TOKEN"))
    assert processes[1].env["ANNOTATION_BACKEND_TOKEN"] == processes[0].env["ANNOTATION_BACKEND_TOKEN"]
    assert processes[1].env["ANNOTATION_BACKEND_URL"] == "http://127.0.0.1:7861"
    assert processes[1].env["ANNOTATION_BROWSER_ORIGIN"] == "https://space.example.test"
    assert processes[0].terminated


def test_supervisor_terminates_sibling_when_backend_exits(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    processes = []

    class Child:
        def __init__(self, args, env):
            self.returncode = 3 if not processes else None
            self.terminated = False
            processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(space_runtime.subprocess, "Popen", Child)
    assert space_runtime.run() == 3
    assert processes[1].terminated


def test_config_is_accepted_by_official_pipeline(monkeypatch, tmp_path):
    from official_annotations import parse_config

    _env(monkeypatch, tmp_path)
    backend_env, config_path = space_runtime.configure_environment()
    monkeypatch.setenv("LEROBOT_ANNOTATE_CONFIG", str(config_path))
    monkeypatch.setenv("LEROBOT_VLM_API_KEY", backend_env["LEROBOT_VLM_API_KEY"])
    config = parse_config({})
    assert config.vlm.model_id == space_runtime.MODEL
    assert config.vlm.max_new_tokens == 2048
    assert not config.vlm.auto_serve


def test_ui_launch_failure_stops_backend_and_restores_signals(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    original = space_runtime.signal.getsignal(space_runtime.signal.SIGTERM)

    class Backend:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

    child = Backend()

    def launch(args, env):
        if args[0] == "node":
            raise FileNotFoundError("node unavailable")
        return child

    monkeypatch.setattr(space_runtime.subprocess, "Popen", launch)
    with pytest.raises(FileNotFoundError):
        space_runtime.run()
    assert child.returncode == 0
    assert space_runtime.signal.getsignal(space_runtime.signal.SIGTERM) == original
