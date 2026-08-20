from __future__ import annotations

from pathlib import Path
import sys

import pytest

BACKEND_ROOT = Path(__file__).parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

TESTS_ROOT = Path(__file__).parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from fixtures import legacy_v31_dataset  # noqa: E402, F401


@pytest.fixture(autouse=True)
def clear_dataset_state() -> None:
    """Keep the module-level app cache from crossing test fixture boundaries."""
    import app

    app._states.clear()
    yield
    app._states.clear()
