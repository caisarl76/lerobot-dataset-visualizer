"""The one canonical source for pnp-trash training and deployment prompts."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

PROMPT_TEMPLATE_VERSION = "pnp-trash-prompts-v1"

_PROMPT_TEMPLATES = (
    "approach the brown table",
    "pick up the {object} from the table with the {hand} hand",
    "turn {turn} to find the black trash bin",
    "approach the black trash bin while holding the {object}",
    "lean down to the black trash bin",
    "drop the {object} into the black trash bin",
    "go to a standing straight pose",
)

# The hash covers the version and every byte of every ordered template.  A
# trailing newline makes the serialization explicit and stable across callers.
PROMPT_TEMPLATE_BYTES = ("\n".join((PROMPT_TEMPLATE_VERSION, *_PROMPT_TEMPLATES)) + "\n").encode("utf-8")
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE_BYTES).hexdigest()

_WHITESPACE = re.compile(r"\s+")


def normalize_object_name(value: str) -> str:
    """Trim and collapse whitespace without changing any non-whitespace text."""
    if not isinstance(value, str):
        raise ValueError("object_name must be a string")
    normalized = _WHITESPACE.sub(" ", value.strip())
    if not normalized:
        raise ValueError("object_name must be nonempty after normalization")
    return normalized


def expand_prompts(
    *, object_name: str, hand: Literal["left", "right"], turn: Literal["left", "right"]
) -> list[str]:
    """Expand the exact ordered seven-prompt contract."""
    object_text = normalize_object_name(object_name)
    if hand not in {"left", "right"}:
        raise ValueError("hand must be 'left' or 'right'")
    if turn not in {"left", "right"}:
        raise ValueError("turn must be 'left' or 'right'")
    return [template.format(object=object_text, hand=hand, turn=turn) for template in _PROMPT_TEMPLATES]
