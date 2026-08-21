from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


class CurationConfigurationError(ValueError):
    """Raised when the explicit curation runtime contract is invalid."""


_REQUIRED = (
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

_LEGACY_BROWSER_ORIGIN_ENV = "LEROBOT_ANNOTATE_BROWSER_ORIGIN"
_LEGACY_BROWSER_ORIGIN_DEFAULT = "http://localhost:3000"


def _canonical_absolute(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise CurationConfigurationError(f"{name} must be an absolute path")
    return path.resolve()


def _is_nested_or_equal(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _single_origin(value: str) -> str:
    if not value or "," in value:
        raise CurationConfigurationError("CURATION_BROWSER_ORIGIN must contain exactly one origin")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise CurationConfigurationError("CURATION_BROWSER_ORIGIN must be one origin without a path")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def legacy_browser_origin(environment: Mapping[str, str] | None = None) -> str:
    """Return the one non-public browser origin used by the legacy backend."""
    env = os.environ if environment is None else environment
    return _single_origin(env.get(_LEGACY_BROWSER_ORIGIN_ENV, _LEGACY_BROWSER_ORIGIN_DEFAULT))


def _validate_loopback(host: str) -> str:
    if host == "localhost":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise CurationConfigurationError("CURATION_BACKEND_HOST must be a loopback address")


@dataclass(frozen=True)
class CurationSettings:
    dataset_aliases: Mapping[str, Path]
    workspace: Path
    output: Path
    browser_origin: str
    bearer_token: str
    cosmos_base_url: str
    cosmos_model: str
    cosmos_api_key_env: str
    cosmos_endpoint_identity: str
    isaac_groot_root: Path
    backend_host: str
    worker_concurrency: int = 1
    http_timeout_seconds: int = 120
    transport_attempts: int = 2
    repair_attempts: int = 1
    target_sampling_fps: int = 2
    maximum_duration_seconds: int = 120
    maximum_sampled_frames: int = 240
    maximum_payload_bytes: int = 67_108_864

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "CurationSettings":
        env = os.environ if environment is None else environment
        missing = [name for name in _REQUIRED if not env.get(name)]
        if missing:
            raise CurationConfigurationError(f"missing required environment value(s): {', '.join(missing)}")
        try:
            raw_aliases = json.loads(env["CURATION_DATASET_ALIASES_JSON"])
        except json.JSONDecodeError as error:
            raise CurationConfigurationError("CURATION_DATASET_ALIASES_JSON must be JSON") from error
        if not isinstance(raw_aliases, dict) or not raw_aliases:
            raise CurationConfigurationError("CURATION_DATASET_ALIASES_JSON must be a nonempty object")
        aliases: dict[str, Path] = {}
        for alias, raw_path in raw_aliases.items():
            if not isinstance(alias, str) or alias.count("/") != 1 or not isinstance(raw_path, str):
                raise CurationConfigurationError("dataset aliases must be 'org/dataset' string paths")
            aliases[alias] = _canonical_absolute(raw_path, f"dataset alias {alias}")

        workspace = _canonical_absolute(env["CURATION_WORKSPACE"], "CURATION_WORKSPACE")
        output = _canonical_absolute(env["CURATION_OUTPUT"], "CURATION_OUTPUT")
        protected_paths = [*aliases.values(), workspace, output]
        for position, left in enumerate(protected_paths):
            for right in protected_paths[position + 1 :]:
                if _is_nested_or_equal(left, right):
                    raise CurationConfigurationError("source, workspace, and output paths must be separated")
        isaac = _canonical_absolute(env["ISAAC_GROOT_ROOT"], "ISAAC_GROOT_ROOT")
        return cls(
            dataset_aliases=MappingProxyType(aliases),
            workspace=workspace,
            output=output,
            browser_origin=_single_origin(env["CURATION_BROWSER_ORIGIN"]),
            bearer_token=env["CURATION_BEARER_TOKEN"],
            cosmos_base_url=env["COSMOS_BASE_URL"],
            cosmos_model=env["COSMOS_MODEL"],
            cosmos_api_key_env=env["COSMOS_API_KEY_ENV"],
            cosmos_endpoint_identity=env["COSMOS_ENDPOINT_IDENTITY"],
            isaac_groot_root=isaac,
            backend_host=_validate_loopback(env.get("CURATION_BACKEND_HOST", "127.0.0.1")),
        )


def curation_is_configured(environment: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environment is None else environment
    return any(name in env for name in _REQUIRED)
