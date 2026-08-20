from __future__ import annotations

import json
from pathlib import Path

from curation.config import CurationConfigurationError, CurationSettings
import pytest

REQUIRED = {
    "CURATION_DATASET_ALIASES_JSON",
    "CURATION_WORKSPACE",
    "CURATION_OUTPUT",
    "CURATION_BROWSER_ORIGIN",
    "CURATION_BEARER_TOKEN",
    "COSMOS_BASE_URL",
    "COSMOS_MODEL",
    "COSMOS_API_KEY_ENV",
    "COSMOS_ENDPOINT_IDENTITY",
    "ISAAC_GROOT_ROOT",
}


def _environment(tmp_path: Path) -> dict[str, str]:
    source = tmp_path / "source"
    source.mkdir()
    isaac = tmp_path / "isaac"
    isaac.mkdir()
    return {
        "CURATION_DATASET_ALIASES_JSON": json.dumps({"local/pnp_trash": str(source)}),
        "CURATION_WORKSPACE": str(tmp_path / "workspace"),
        "CURATION_OUTPUT": str(tmp_path / "output"),
        "CURATION_BROWSER_ORIGIN": "http://127.0.0.1:3000",
        "CURATION_BEARER_TOKEN": "test-token",
        "COSMOS_BASE_URL": "http://127.0.0.1:8001/v1",
        "COSMOS_MODEL": "cosmos3-nano",
        "COSMOS_API_KEY_ENV": "COSMOS_API_KEY",
        "COSMOS_ENDPOINT_IDENTITY": "h100-cosmos",
        "ISAAC_GROOT_ROOT": str(isaac),
    }


def test_settings_require_every_explicit_runtime_value(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    for name in REQUIRED:
        incomplete = dict(environment)
        incomplete.pop(name)
        with pytest.raises(CurationConfigurationError, match=name):
            CurationSettings.from_env(incomplete)


def test_settings_normalize_paths_and_freeze_default_limits(tmp_path: Path) -> None:
    settings = CurationSettings.from_env(_environment(tmp_path))

    assert settings.dataset_aliases["local/pnp_trash"] == (tmp_path / "source").resolve()
    with pytest.raises(TypeError):
        settings.dataset_aliases["local/other"] = tmp_path  # type: ignore[index]
    assert settings.workspace == (tmp_path / "workspace").resolve()
    assert settings.output == (tmp_path / "output").resolve()
    assert settings.browser_origin == "http://127.0.0.1:3000"
    assert settings.worker_concurrency == 1
    assert settings.http_timeout_seconds == 120
    assert settings.transport_attempts == 2
    assert settings.repair_attempts == 1
    assert settings.target_sampling_fps == 2
    assert settings.maximum_duration_seconds == 120
    assert settings.maximum_sampled_frames == 240
    assert settings.maximum_payload_bytes == 67_108_864


@pytest.mark.parametrize(
    "mutate",
    [
        lambda env: env.update({"CURATION_BROWSER_ORIGIN": "http://one.test,http://two.test"}),
        lambda env: env.update({"CURATION_BROWSER_ORIGIN": "http://one.test/path"}),
        lambda env: env.update({"CURATION_WORKSPACE": env["CURATION_OUTPUT"]}),
        lambda env: env.update({"CURATION_BACKEND_HOST": "0.0.0.0"}),
    ],
)
def test_settings_reject_unsafe_origins_paths_and_bindings(tmp_path: Path, mutate: object) -> None:
    environment = _environment(tmp_path)
    mutate(environment)  # type: ignore[operator]
    with pytest.raises(CurationConfigurationError):
        CurationSettings.from_env(environment)
