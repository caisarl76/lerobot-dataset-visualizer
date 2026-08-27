from __future__ import annotations

import json
import random

from curation.exporter import build_stable_tasks, serialize_tasks_jsonl


def test_stable_tasks_use_minimum_step_then_raw_utf8_prompt_bytes() -> None:
    prompt_sets = [
        ["same", "zeta", "step-3", "step-4", "step-5", "step-6", "step-7"],
        ["same", "alpha", "step-3-b", "step-4-b", "step-5-b", "step-6-b", "step-7-b"],
        ["한글", "same", "step-3-c", "step-4-c", "step-5-c", "step-6-c", "step-7-c"],
    ]

    shuffled = list(prompt_sets)
    random.Random(7).shuffle(shuffled)
    tasks = build_stable_tasks(shuffled)

    expected = sorted(
        {
            prompt: min(
                step for prompts in prompt_sets for step, value in enumerate(prompts, 1) if value == prompt
            )
            for prompts in prompt_sets
            for prompt in prompts
        }.items(),
        key=lambda item: (item[1], item[0].encode("utf-8")),
    )
    assert [(row.prompt, row.ordering_step) for row in tasks] == expected
    assert [row.task_index for row in tasks] == list(range(len(tasks)))


def test_tasks_jsonl_is_canonical_compact_utf8_and_newline_terminated() -> None:
    payload = serialize_tasks_jsonl(build_stable_tasks([["한글", "b", "c", "d", "e", "f", "g"]]))

    assert payload.endswith(b"\n")
    assert b" " not in payload
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    assert rows[0] == {"task": "한글", "task_index": 0}
    assert payload.splitlines()[0] == '{"task":"한글","task_index":0}'.encode()


def test_stable_task_bytes_are_independent_of_episode_completion_order() -> None:
    episodes = [
        ["approach", "pick can", "turn", "carry can", "lean", "drop can", "stand"],
        ["approach", "pick cup", "turn", "carry cup", "lean", "drop cup", "stand"],
    ]
    assert serialize_tasks_jsonl(build_stable_tasks(episodes)) == serialize_tasks_jsonl(
        build_stable_tasks(list(reversed(episodes)))
    )
