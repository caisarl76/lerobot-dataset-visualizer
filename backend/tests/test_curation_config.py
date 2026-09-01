from __future__ import annotations

import json
from pathlib import Path

from curation.config import (
    CURATION_REQUIRED_ENV_NAMES,
    CurationConfigurationError,
    CurationSettings,
    ExportSettings,
    WorkerSettings,
)
import pytest


def test_fastapi_required_environment_names_are_a_stable_public_contract() -> None:
    assert CURATION_REQUIRED_ENV_NAMES == (
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
    )


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


class _SecretReadGuard(dict[str, str]):
    def __init__(self, values: dict[str, str], forbidden: set[str]):
        super().__init__(values)
        self.forbidden = forbidden

    def get(self, key: str, default: str | None = None) -> str | None:
        if key in self.forbidden:
            raise AssertionError(f"forbidden environment read: {key}")
        return super().get(key, default)

    def __getitem__(self, key: str) -> str:
        if key in self.forbidden:
            raise AssertionError(f"forbidden environment read: {key}")
        return super().__getitem__(key)


def test_worker_settings_require_only_worker_authority_and_never_read_target_secret(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    for name in (
        "CURATION_OUTPUT",
        "CURATION_BROWSER_ORIGIN",
        "CURATION_BEARER_TOKEN",
        "ISAAC_GROOT_ROOT",
    ):
        environment.pop(name)
    environment["COSMOS_API_KEY"] = "must-not-be-read-during-settings-load"
    guarded = _SecretReadGuard(environment, {"COSMOS_API_KEY"})

    settings = WorkerSettings.from_env(guarded)

    assert settings.workspace == (tmp_path / "workspace").resolve()
    assert settings.dataset_aliases["local/pnp_trash"] == (tmp_path / "source").resolve()
    assert settings.cosmos_api_key_env == "COSMOS_API_KEY"
    assert settings.worker_concurrency == 1
    assert settings.target_sampling_fps == 2


def test_export_settings_require_only_export_authority_and_never_read_cosmos_secrets(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["COSMOS_API_KEY"] = "must-not-be-read-by-exporter"
    forbidden = {
        "CURATION_OUTPUT",
        "CURATION_BROWSER_ORIGIN",
        "CURATION_BEARER_TOKEN",
        "COSMOS_BASE_URL",
        "COSMOS_API_KEY_ENV",
        "COSMOS_API_KEY",
    }
    guarded = _SecretReadGuard(environment, forbidden)

    settings = ExportSettings.from_env(guarded)

    assert settings.workspace == (tmp_path / "workspace").resolve()
    assert settings.dataset_aliases["local/pnp_trash"] == (tmp_path / "source").resolve()
    assert settings.isaac_groot_root == (tmp_path / "isaac").resolve()
    assert settings.cosmos_model == "cosmos3-nano"
    assert settings.cosmos_endpoint_identity == "h100-cosmos"


def test_backend_settings_still_require_the_bearer_token(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment.pop("CURATION_BEARER_TOKEN")

    with pytest.raises(CurationConfigurationError, match="CURATION_BEARER_TOKEN"):
        CurationSettings.from_env(environment)


@pytest.mark.parametrize("settings_type", [CurationSettings, WorkerSettings])
@pytest.mark.parametrize(
    "reserved_name",
    [
        "PATH",
        "HOME",
        "PYTHONPATH",
        "LD_PRELOAD",
        "CURATION_BEARER_TOKEN",
        "CURATION_WORKSPACE",
        "COSMOS_MODEL",
        "COSMOS_BASE_URL",
        "COSMOS_API_KEY_ENV",
        "ISAAC_GROOT_ROOT",
        "NEXT_PUBLIC_DATASET_URL",
    ],
)
def test_cosmos_key_target_cannot_collide_with_process_or_configuration_names(
    tmp_path: Path,
    settings_type: type[CurationSettings] | type[WorkerSettings],
    reserved_name: str,
) -> None:
    environment = _environment(tmp_path)
    environment["COSMOS_API_KEY_ENV"] = reserved_name

    with pytest.raises(CurationConfigurationError, match="reserved") as raised:
        settings_type.from_env(environment)

    assert reserved_name not in str(raised.value)


@pytest.mark.parametrize("settings_type", [WorkerSettings, ExportSettings])
def test_process_settings_preserve_source_workspace_separation(
    tmp_path: Path,
    settings_type: type[WorkerSettings] | type[ExportSettings],
) -> None:
    environment = _environment(tmp_path)
    environment["CURATION_WORKSPACE"] = str(tmp_path / "source" / "nested-workspace")

    with pytest.raises(CurationConfigurationError, match="source and workspace paths must be separated"):
        settings_type.from_env(environment)


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
