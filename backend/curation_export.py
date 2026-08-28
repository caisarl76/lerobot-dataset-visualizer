"""Separate process entrypoint for deterministic cleaned-dataset exports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from uuid import UUID

from curation.config import CurationConfigurationError, CurationSettings
from curation.db import (
    CurationDatabase,
    IllegalStateTransition,
    IncompatibleCurationDatabase,
    RetryableDatabaseError,
    StateTransitionConflict,
    canonical_json,
)
from curation.exporter import ExportError, ValidatedDatasetExporter
from curation.source import SourceRegistry


def _absolute_workspace(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("workspace must be an absolute path")
    return path.resolve()


def _export_uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError):
        raise argparse.ArgumentTypeError("export ID must be a UUID") from None


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="curation_export.py")
    parser.add_argument("--workspace", required=True, type=_absolute_workspace)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "resume"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--export-id", required=True, type=_export_uuid)
    return parser


def _print(document: Mapping[str, Any]) -> None:
    print(canonical_json(dict(document)), flush=True)


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    database_path = arguments.workspace / "curation.sqlite3"
    if not database_path.is_file():
        _print({"error": "invalid_workspace", "export_id": arguments.export_id})
        return 2
    database = CurationDatabase(database_path)
    try:
        database.validate_worker_compatibility()
        settings = CurationSettings.from_env()
        if settings.workspace != arguments.workspace:
            raise CurationConfigurationError("CLI workspace does not match CURATION_WORKSPACE")
        export = database.get_export(export_id=arguments.export_id)
        if export is None:
            raise ExportError("export not found", {"error": "export_not_found"})
        registry = SourceRegistry.from_paths(settings.dataset_aliases, workspace=arguments.workspace)
        exporter = ValidatedDatasetExporter(
            database=database,
            source_registry=registry,
            workspace=arguments.workspace,
            isaac_root=settings.isaac_groot_root,
            visualizer_root=Path(__file__).parent.parent,
            cosmos_model=settings.cosmos_model,
            cosmos_endpoint_identity=settings.cosmos_endpoint_identity,
        )
        result = (
            exporter.run(arguments.export_id)
            if arguments.command == "run"
            else exporter.resume(arguments.export_id)
        )
    except RetryableDatabaseError:
        _print({"error": "database_busy", "export_id": arguments.export_id, "retryable": True})
        return 2
    except (IncompatibleCurationDatabase, sqlite3.DatabaseError):
        _print({"error": "invalid_database", "export_id": arguments.export_id})
        return 2
    except (StateTransitionConflict, IllegalStateTransition):
        _print({"error": "export_state_conflict", "export_id": arguments.export_id})
        return 1
    except ExportError as error:
        _print(dict(error.payload) | {"export_id": arguments.export_id})
        return 1
    except (CurationConfigurationError, OSError, ValueError):
        _print({"error": "invalid_export_configuration", "export_id": arguments.export_id})
        return 2
    _print({"event": "export_published", **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
