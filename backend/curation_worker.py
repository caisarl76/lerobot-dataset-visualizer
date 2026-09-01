"""Separate process entrypoint for persisted Cosmos curation jobs."""

from __future__ import annotations

from curation.worker import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
