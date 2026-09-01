from __future__ import annotations

import hashlib

from curation.prompts import (
    PROMPT_TEMPLATE_BYTES,
    PROMPT_TEMPLATE_SHA256,
    PROMPT_TEMPLATE_VERSION,
    expand_prompts,
)
import pytest


def test_expands_the_exact_frozen_seven_prompt_contract() -> None:
    assert expand_prompts(object_name="crumpled can", hand="left", turn="right") == [
        "approach the brown table",
        "pick up the crumpled can from the table with the left hand",
        "turn right to find the black trash bin",
        "approach the black trash bin while holding the crumpled can",
        "lean down to the black trash bin",
        "drop the crumpled can into the black trash bin",
        "go to a standing straight pose",
    ]


def test_object_normalization_only_trims_and_collapses_whitespace() -> None:
    prompts = expand_prompts(object_name="  Crumpled\t CAN\nlabel  ", hand="right", turn="left")

    assert prompts[1] == "pick up the Crumpled CAN label from the table with the right hand"
    assert prompts[3] == "approach the black trash bin while holding the Crumpled CAN label"
    assert prompts[5] == "drop the Crumpled CAN label into the black trash bin"


@pytest.mark.parametrize("field,value", [("object_name", " \t\n "), ("hand", "Left"), ("turn", "RIGHT")])
def test_rejects_empty_objects_and_noncanonical_enums(field: str, value: str) -> None:
    arguments = {"object_name": "can", "hand": "left", "turn": "right"}
    arguments[field] = value

    with pytest.raises(ValueError):
        expand_prompts(**arguments)


def test_prompt_bundle_has_a_frozen_version_and_deterministic_sha256() -> None:
    assert PROMPT_TEMPLATE_VERSION == "pnp-trash-prompts-v1"
    assert hashlib.sha256(PROMPT_TEMPLATE_BYTES).hexdigest() == PROMPT_TEMPLATE_SHA256
    assert PROMPT_TEMPLATE_SHA256 == "290309cb8ec2ee155ce65d4233bb0d958437a05505c2b58c856c3caa966b253f"
