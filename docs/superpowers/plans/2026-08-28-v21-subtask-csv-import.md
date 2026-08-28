# LeRobot v2.1 Subtask CSV Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe bulk CSV workflow that previews timestamped subtask annotations and creates a new, fully annotated local LeRobot v2.1 dataset without modifying the source.

**Architecture:** Keep CSV parsing and staged validation pure in `backend/subtask_csv.py`; isolate v2.1 dataset inspection and transactional materialization in `backend/dataset_v21.py`; expose thin FastAPI preview/apply endpoints; and add a typed React import panel to the existing Annotations tab. Reuse one language-column codec for existing manual exports and the new v2.1 importer, then teach the v2.1 frontend loader to display the resulting atoms.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic 2, PyArrow, standard-library `csv`/`unittest`; Next.js 15, React 19, TypeScript, Bun tests, Tailwind plus the existing annotations CSS skin.

---

## Execution prerequisite

The current main worktree contains user-owned changes in `fetch-data.ts`, its tests, `episode-viewer.tsx`, and `versionUtils.ts`. Before Task 1, invoke the `using-git-worktrees` skill and create a clean feature worktree from commit `9a99316` on branch `feat/v21-subtask-csv-import`. Do not stash, reset, stage, or commit the current worktree's unrelated changes.

In the feature worktree, verify the baseline before editing:

```bash
git status --short
bun test
bun run type-check
python -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

Expected: clean status, frontend baseline commands exit `0`, and the isolated backend environment is ready. Dependency installation requires network approval if packages are not already cached. If a baseline command fails on the clean worktree, record the exact failure before continuing; do not silently mix a pre-existing failure into a feature commit.

## File map

Create:

- `backend/subtask_csv.py` — pure CSV contract parsing and structured issues.
- `backend/language_columns.py` — shared Arrow schema, atom extraction, and language-column materialization.
- `backend/dataset_v21.py` — v2.1 preview, snapping, safe sibling creation, and apply orchestration.
- `backend/tests/__init__.py` — test package marker.
- `backend/tests/fixtures.py` — real temporary v2.1 Parquet fixture builder.
- `backend/tests/test_subtask_csv.py` — parser and scalar-validation tests.
- `backend/tests/test_language_columns.py` — language codec characterization tests.
- `backend/tests/test_dataset_v21.py` — timeline and materialization integration tests.
- `backend/tests/test_subtask_import_api.py` — FastAPI endpoint tests.
- `backend/requirements-dev.txt` — backend test-only HTTP client dependency.
- `src/types/subtask-import.types.ts` — frontend API contract.
- `src/utils/subtaskImportClient.ts` — preview/apply HTTP client.
- `src/utils/subtaskImportModel.ts` — pure UI derivations and error grouping.
- `src/utils/__tests__/subtaskImportModel.test.ts` — frontend model tests.
- `src/components/subtask-csv-import.tsx` — bulk import workflow UI.
- `src/components/__tests__/subtask-csv-import.test.ts` — static component contract test.
- `examples/subtask_annotations.csv` — normalized CSV template.

Modify:

- `backend/app.py` — import the shared language codec and add thin preview/apply routes.
- `backend/README.md` — document v2.1 CSV endpoints and commands.
- `src/components/annotations-panel.tsx` — mount the bulk importer.
- `src/components/annotations-skin.css` — style the uploader, preview, and staged errors.
- `src/app/[org]/[dataset]/[episode]/fetch-data.ts` — load language columns for v2.1.
- `src/app/[org]/[dataset]/[episode]/__tests__/fetch-data.test.ts` — v2.1 language extraction regression tests.
- `.env.example` — document the local dataset path variable.
- `README.md` — document the end-to-end workflow.

### Task 1: Pure CSV parser and staged errors

**Files:**

- Create: `backend/subtask_csv.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_subtask_csv.py`

- [ ] **Step 1: Write failing parser tests**

Create `backend/tests/test_subtask_csv.py` with `unittest` cases that establish the public API and exact issue locations:

```python
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from subtask_csv import REQUIRED_HEADERS, ImportIssue, ParsedSubtask, parse_subtask_csv


class ParseSubtaskCsvTest(unittest.TestCase):
    def test_accepts_bom_quotes_multiline_prompts_and_arbitrary_row_order(self) -> None:
        text = (
            "\ufeffepisode_index,subtask_index,prompt,start_timestamp,end_timestamp\n"
            '1,1,"pick, then place",1.0,2.0\n'
            '0,0,"look\ncarefully",0,1\n'
            "1,0,approach,0,1\n"
        )
        result = parse_subtask_csv(text)
        self.assertEqual(result.issues, [])
        self.assertEqual(
            [(row.episode_index, row.subtask_index) for row in result.rows],
            [(0, 0), (1, 0), (1, 1)],
        )
        self.assertEqual(result.rows[0].prompt, "look\ncarefully")
        self.assertEqual(result.rows[0].csv_row, 3)

    def test_reports_missing_unknown_and_duplicate_headers_as_csv_schema(self) -> None:
        result = parse_subtask_csv(
            "episode_index,subtask_index,prompt,start_timestamp,prompt,extra\n"
            "0,0,a,0,a,x\n"
        )
        self.assertFalse(result.valid)
        self.assertEqual({issue.stage for issue in result.issues}, {"csv_schema"})
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"MISSING_HEADER", "DUPLICATE_HEADER", "UNKNOWN_HEADER"},
        )

    def test_reports_malformed_record_width_as_csv_parse(self) -> None:
        result = parse_subtask_csv(
            ",".join(REQUIRED_HEADERS) + "\n0,0,prompt,0\n"
        )
        issue = result.issues[0]
        self.assertEqual((issue.stage, issue.code, issue.row), ("csv_parse", "COLUMN_COUNT", 2))

    def test_reports_each_invalid_scalar_with_row_and_field(self) -> None:
        result = parse_subtask_csv(
            ",".join(REQUIRED_HEADERS)
            + "\n-1,x,,NaN,Infinity\n"
        )
        self.assertEqual(
            {(issue.code, issue.field) for issue in result.issues},
            {
                ("INVALID_INTEGER", "episode_index"),
                ("INVALID_INTEGER", "subtask_index"),
                ("EMPTY_PROMPT", "prompt"),
                ("INVALID_TIMESTAMP", "start_timestamp"),
                ("INVALID_TIMESTAMP", "end_timestamp"),
            },
        )
        self.assertTrue(all(issue.row == 2 for issue in result.issues))

    def test_reports_duplicate_keys_and_non_consecutive_indices(self) -> None:
        result = parse_subtask_csv(
            ",".join(REQUIRED_HEADERS)
            + "\n0,0,a,0,1\n0,0,b,1,2\n1,1,c,0,1\n"
        )
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"DUPLICATE_SUBTASK", "NON_CONSECUTIVE_SUBTASK_INDEX"},
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the parser test and verify RED**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_subtask_csv -v
```

Expected: import failure for `subtask_csv`; the feature module does not exist yet.

- [ ] **Step 3: Implement the parser contract**

Create `backend/subtask_csv.py` with these exact public types and entry point:

```python
from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

REQUIRED_HEADERS = (
    "episode_index",
    "subtask_index",
    "prompt",
    "start_timestamp",
    "end_timestamp",
)
_NON_NEGATIVE_INTEGER = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class ImportIssue:
    stage: str
    code: str
    message: str
    row: int | None = None
    field: str | None = None
    episode_index: int | None = None

    def to_dict(self) -> dict[str, object | None]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedSubtask:
    csv_row: int
    episode_index: int
    subtask_index: int
    prompt: str
    start_timestamp: float
    end_timestamp: float


@dataclass(frozen=True)
class CsvParseResult:
    rows: list[ParsedSubtask]
    issues: list[ImportIssue]

    @property
    def valid(self) -> bool:
        return not self.issues


def parse_subtask_csv(csv_text: str) -> CsvParseResult:
    """Parse normalized CSV without consulting a dataset or filesystem."""
```

Implementation rules:

- Remove only one leading `\ufeff` before constructing `csv.reader(io.StringIO(text, newline=""), strict=True)`.
- Treat an empty input as `csv_schema/EMPTY_FILE`.
- Compare the header as a multiset to detect every missing, duplicate, and unknown name; return no rows when the header is invalid.
- Enumerate logical CSV records starting at row `2`; ignore records whose cells are all blank.
- Emit `csv_parse/COLUMN_COUNT` when a record width differs from five and `csv_parse/MALFORMED_CSV` when `csv.Error` is raised.
- Parse integers only when the stripped value matches `_NON_NEGATIVE_INTEGER`.
- Parse timestamps with `float`, then require `math.isfinite(value)` and `value >= 0`.
- Preserve internal prompt content but store `prompt.strip()`.
- Add a row only when all five fields on that row are valid.
- After scalar parsing, emit `row_validation/DUPLICATE_SUBTASK` for every duplicate after the first and `row_validation/NON_CONSECUTIVE_SUBTASK_INDEX` when an episode's unique sorted indices differ from `range(len(indices))`.
- Return rows sorted by `(episode_index, subtask_index, csv_row)` and issues sorted by `(row or 0, field or "", code)`.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_subtask_csv -v
```

Expected: five tests pass.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add backend/subtask_csv.py backend/tests/__init__.py backend/tests/test_subtask_csv.py
git commit -m "feat: validate subtask CSV files"
```

Expected: validation exits `0`; commit contains only the parser and its tests.

### Task 2: Shared language-column codec

**Files:**

- Create: `backend/language_columns.py`
- Create: `backend/tests/test_language_columns.py`
- Modify: `backend/app.py:57-88, 354-585`

- [ ] **Step 1: Write failing Arrow codec tests**

Create tests that build a two-frame table, materialize persistent/event atoms, and read them back:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyarrow as pa

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from language_columns import (
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    extract_existing_atoms_from_table,
    materialize_language_columns,
)


class LanguageColumnsTest(unittest.TestCase):
    def test_materializes_explicit_list_struct_columns_and_round_trips(self) -> None:
        table = pa.table({"episode_index": [0, 0], "timestamp": [0.0, 1.0], "value": [1, 2]})
        atoms = [
            {"role": "assistant", "content": "approach", "style": "subtask", "timestamp": 0.0, "camera": None, "tool_calls": None},
            {"role": "user", "content": "stop", "style": "interjection", "timestamp": 1.0, "camera": None, "tool_calls": None},
        ]
        result, persistent_count, event_count = materialize_language_columns(table, {0: atoms})
        self.assertEqual((persistent_count, event_count), (1, 1))
        self.assertEqual(result.schema.field(LANGUAGE_PERSISTENT).type.value_type.field("timestamp").type, pa.float64())
        self.assertEqual(result.column(LANGUAGE_EVENTS).to_pylist()[0], [])
        restored = extract_existing_atoms_from_table(result, 0)
        self.assertEqual([(a["style"], a["timestamp"]) for a in restored], [("subtask", 0.0), ("interjection", 1.0)])

    def test_preserves_non_language_columns_and_drops_legacy_subtask_index(self) -> None:
        table = pa.table({"episode_index": [0], "timestamp": [0.0], "subtask_index": [4], "value": [9]})
        result, _, _ = materialize_language_columns(table, {0: []})
        self.assertEqual(result.column_names, ["episode_index", "timestamp", "value", LANGUAGE_PERSISTENT, LANGUAGE_EVENTS])
        self.assertEqual(result.column("value").to_pylist(), [9])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the codec test and verify RED**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_language_columns -v
```

Expected: import failure for `language_columns`.

- [ ] **Step 3: Extract and harden the language codec**

Create `backend/language_columns.py`. Move the following behavior out of `app.py` without changing its public results:

```python
LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"
PERSISTENT_STYLES = {"task_aug", "subtask", "plan", "memory"}
EVENT_ONLY_STYLES = {"interjection", "vqa"}
KNOWN_STYLES = PERSISTENT_STYLES | EVENT_ONLY_STYLES


def column_for_style(style: str | None) -> str:
    if style is None:
        return LANGUAGE_EVENTS
    if style in PERSISTENT_STYLES:
        return LANGUAGE_PERSISTENT
    if style in EVENT_ONLY_STYLES:
        return LANGUAGE_EVENTS
    raise ValueError(f"Unknown language style: {style!r}")


def snap_timestamp(timestamp: float, frame_timestamps: list[float]) -> float:
    if not frame_timestamps:
        return float(timestamp)
    index = bisect_left(frame_timestamps, timestamp)
    if index == 0:
        return float(frame_timestamps[0])
    if index == len(frame_timestamps):
        return float(frame_timestamps[-1])
    earlier = frame_timestamps[index - 1]
    later = frame_timestamps[index]
    return float(earlier if timestamp - earlier <= later - timestamp else later)


def validate_atom(atom: dict[str, object]) -> str | None:
    style = atom.get("style")
    if style is not None and style not in KNOWN_STYLES:
        return f"Unknown language style: {style!r}"
    if atom.get("content") is None and not atom.get("tool_calls"):
        return "atom must have content or tool_calls"
    if style is None and not atom.get("tool_calls"):
        return "style=None requires tool_calls"
    camera = atom.get("camera")
    if camera is not None and not isinstance(camera, str):
        return "camera must be a string or null"
    if camera is not None and style not in {"vqa", "trace"}:
        return f"camera must be null for style={style!r}"
    return None
```

The module additionally exports `extract_existing_atoms_from_table(table, episode_index)` and `materialize_language_columns(table, atoms_by_episode)`. Move the complete row coercion, persistent deduplication, event fallback-timestamp handling, sorting, and per-row broadcasting algorithms currently implemented by `_extract_existing_atoms_from_table` and `_materialize_table`; rename their callers in `app.py` to the public names.

Replace each ellipsis above with the corresponding complete implementation from `app.py`, with these deliberate changes:

- `validate_atom` returns an error string or `None`; the FastAPI layer converts it to `HTTPException`.
- `snap_timestamp` uses `bisect_left` and chooses the earlier frame on an equal-distance tie.
- Define explicit nested Arrow types named `TOOL_CALL_LIST_TYPE`, `PERSISTENT_ATOM_TYPE`, `EVENT_ATOM_TYPE`, `PERSISTENT_LIST_TYPE`, and `EVENT_LIST_TYPE`. Persistent structs contain `role`, `content`, `style`, `timestamp`, `camera`, and `tool_calls`; event structs omit `timestamp`. The supported tool-call struct contains `type`, a nested `function.name`, and a nested `function.arguments.text`, matching the existing `say` tool.
- Use `pa.array(per_row_persistent, type=PERSISTENT_LIST_TYPE)` and `pa.array(per_row_events, type=EVENT_LIST_TYPE)`, including when every list is empty.
- Preserve existing language atoms for an episode when `atoms_by_episode` has no key for it.
- Continue dropping `subtask_index`, existing language columns, and the stray per-row `tools` column before appending the canonical columns.

In `backend/app.py`, import the constants and functions from `language_columns`, delete the duplicated codec functions/constants, keep `SAY_TOOL_SCHEMA` in `app.py`, and adapt `_validate_atom` call sites to convert the returned error string into HTTP 400.

- [ ] **Step 4: Run codec and backend import checks**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_language_columns -v
backend/.venv/bin/python -m py_compile backend/app.py backend/language_columns.py
```

Expected: two tests pass and both modules compile.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add backend/app.py backend/language_columns.py backend/tests/test_language_columns.py
git commit -m "refactor: share language parquet codec"
```

Expected: validation exits `0`; the commit contains the characterized extraction only.

### Task 3: v2.1 dataset preview and timeline validation

**Files:**

- Create: `backend/dataset_v21.py`
- Create: `backend/tests/fixtures.py`
- Create: `backend/tests/test_dataset_v21.py`

- [ ] **Step 1: Add a real v2.1 fixture builder and failing preview tests**

Create `backend/tests/fixtures.py` with a `write_v21_dataset(root, episodes)` helper. It must write `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`, and one `data/chunk-000/episode_NNNNNN.parquet` per episode. Each Parquet file contains `episode_index`, `frame_index`, `timestamp`, and a numeric `observation.state` column.

Create preview tests with this public API:

```python
from dataset_v21 import preview_subtask_import, snap_to_nearest_frame


class DatasetV21PreviewTest(unittest.TestCase):
    def test_snapping_tie_chooses_earlier_frame(self) -> None:
        self.assertEqual(snap_to_nearest_frame(0.5, [0.0, 1.0]), 0.0)

    def test_valid_preview_snaps_and_groups_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = write_v21_dataset(Path(tmp) / "source", {0: [0.0, 1.0, 2.0]})
            output = Path(tmp) / "source_annotated"
            csv_text = csv_for(
                "0,0,approach,0.01,1.01",
                "0,1,grasp,1.01,2.01",
            )
            preview = preview_subtask_import(source, output, csv_text)
            self.assertTrue(preview.valid)
            self.assertEqual(preview.episode_count, 1)
            self.assertEqual(preview.subtask_count, 2)
            self.assertEqual(
                [(r.snapped_start, r.snapped_end) for r in preview.episodes[0].rows],
                [(0.0, 1.0), (1.0, 2.0)],
            )
            self.assertEqual(len(preview.warnings), 4)
            self.assertFalse(output.exists())

    def test_reports_dataset_and_timeline_stages_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = write_v21_dataset(Path(tmp) / "source", {0: [0.0, 1.0, 2.0]})
            preview = preview_subtask_import(
                source,
                Path(tmp) / "source_annotated",
                csv_for("0,0,a,0,1", "0,1,b,1.8,2", "9,0,c,0,1"),
            )
            self.assertFalse(preview.valid)
            self.assertIn("UNKNOWN_EPISODE", {e.code for e in preview.errors})
            self.assertIn("TIMELINE_GAP", {e.code for e in preview.errors})
```

Add separate tests for unsupported version, missing Parquet, reversed interval, zero duration after snapping, overlap, first-frame mismatch, final-frame mismatch, non-chronological subtask indices, output equal to/inside source, missing output parent, and existing output.

- [ ] **Step 2: Run preview tests and verify RED**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_dataset_v21.DatasetV21PreviewTest -v
```

Expected: import failure for `dataset_v21`.

- [ ] **Step 3: Implement v2.1 inspection and preview**

Create `backend/dataset_v21.py` with these public dataclasses/functions:

```python
@dataclass(frozen=True)
class PreviewRow:
    csv_row: int
    episode_index: int
    subtask_index: int
    prompt: str
    original_start: float
    original_end: float
    snapped_start: float
    snapped_end: float


@dataclass(frozen=True)
class EpisodePreview:
    episode_index: int
    first_timestamp: float
    final_timestamp: float
    rows: list[PreviewRow]


@dataclass(frozen=True)
class ImportPreview:
    valid: bool
    source_path: str
    output_path: str
    version: str | None
    episode_count: int
    subtask_count: int
    episodes: list[EpisodePreview]
    warnings: list[ImportIssue]
    errors: list[ImportIssue]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def snap_to_nearest_frame(timestamp: float, frames: list[float]) -> float:
    return snap_timestamp(timestamp, frames)
```

The module's orchestration entry point is `preview_subtask_import(local_path: str | Path, output_dir: str | Path, csv_text: str) -> ImportPreview`.

Implement the following exact validation flow:

1. Call `parse_subtask_csv` and retain all parser issues.
2. Resolve source/output with `Path.expanduser().resolve(strict=False)`.
3. Validate source directory, `meta/info.json`, exact version `v2.1`, integer `total_episodes`, `data_path`, and `chunks_size`.
4. Reject output equal to source, inside source, already existing, missing parent, or non-writable parent with `output_preflight` issues.
5. Resolve an episode path by formatting `data_path` with `episode_chunk=episode_index // chunks_size` and `episode_index`; reject indices outside `[0, total_episodes)` before reading.
6. Read only `episode_index` and `timestamp`, retain rows for the requested episode, require at least one finite timestamp, sort unique timestamps, and report unreadable/missing files as `dataset_validation`.
7. Snap every boundary. Emit `timeline_validation/TIMESTAMP_SNAPPED` as a warning whenever `abs(original - snapped) > 1e-12`.
8. Validate raw and snapped positive duration, chronological subtask order, first coverage, pairwise contiguity, and final coverage. Use distinct codes `INVALID_INTERVAL`, `ZERO_DURATION_AFTER_SNAP`, `NON_CHRONOLOGICAL_SUBTASKS`, `FIRST_FRAME_MISMATCH`, `TIMELINE_GAP`, `TIMELINE_OVERLAP`, and `FINAL_FRAME_MISMATCH`.
9. Sort episodes and errors deterministically. Set counts only from fully parsed rows, and set `valid` only when `errors` is empty.

- [ ] **Step 4: Run preview tests and verify GREEN**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_dataset_v21.DatasetV21PreviewTest -v
```

Expected: all preview and timeline cases pass.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add backend/dataset_v21.py backend/tests/fixtures.py backend/tests/test_dataset_v21.py
git commit -m "feat: preview v2.1 subtask imports"
```

Expected: validation exits `0`; commit contains the read-only preview path.

### Task 4: Transactional v2.1 dataset application

**Files:**

- Modify: `backend/dataset_v21.py`
- Modify: `backend/tests/test_dataset_v21.py`

- [ ] **Step 1: Write failing source-safety and materialization tests**

Extend the integration test fixture to optionally seed `language_persistent`, `language_events`, and `meta/lerobot_annotations.json`. Add tests asserting:

```python
from dataset_v21 import SubtaskImportFailure, apply_subtask_import


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class DatasetV21ApplyTest(unittest.TestCase):
    def test_apply_replaces_only_subtasks_and_never_changes_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = write_v21_dataset_with_language_atoms(Path(tmp) / "source")
            before = tree_hashes(source)
            output = Path(tmp) / "source_annotated"
            result = apply_subtask_import(
                source,
                output,
                csv_for("0,0,new approach,0,1", "0,1,new grasp,1,2"),
            )
            self.assertEqual(tree_hashes(source), before)
            self.assertEqual(result.rewritten_episodes, 1)
            atoms = atoms_from_episode(output, 0)
            self.assertEqual([a["content"] for a in atoms if a["style"] == "subtask"], ["new approach", "new grasp"])
            self.assertEqual([a["content"] for a in atoms if a["style"] == "plan"], ["existing plan"])
            self.assertEqual([a["content"] for a in atoms if a["style"] == "interjection"], ["existing event"])

    def test_apply_leaves_absent_episode_parquet_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = write_v21_dataset(Path(tmp) / "source", {0: [0.0, 1.0], 1: [0.0, 1.0]})
            output = Path(tmp) / "source_annotated"
            apply_subtask_import(source, output, csv_for("0,0,only,0,1"))
            rel = Path("data/chunk-000/episode_000001.parquet")
            self.assertEqual((source / rel).read_bytes(), (output / rel).read_bytes())

    def test_write_failure_removes_only_request_temp_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = write_v21_dataset(Path(tmp) / "source", {0: [0.0, 1.0]})
            output = Path(tmp) / "source_annotated"
            with mock.patch("dataset_v21.pq.write_table", side_effect=OSError("disk full")):
                with self.assertRaises(SubtaskImportFailure) as caught:
                    apply_subtask_import(source, output, csv_for("0,0,only,0,1"))
            self.assertEqual(caught.exception.issue.stage, "dataset_write")
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).glob(".source_annotated.tmp-*")), [])
```

Add tests for destination races, output-path lock contention, metadata feature updates, sidecar precedence, no symlinks, and explicit non-null Arrow list types when all events are empty.

- [ ] **Step 2: Run apply tests and verify RED**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_dataset_v21.DatasetV21ApplyTest -v
```

Expected: import failure for `apply_subtask_import` or missing application behavior.

- [ ] **Step 3: Implement safe application**

Add these public results and failure type:

```python
@dataclass(frozen=True)
class ApplyResult:
    ok: bool
    output_dir: str
    rewritten_episodes: int
    imported_subtasks: int
    materialized_files: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SubtaskImportFailure(RuntimeError):
    def __init__(self, issue: ImportIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue
```

Add `apply_subtask_import(local_path: str | Path, output_dir: str | Path, csv_text: str) -> ApplyResult` as the public write entry point.

Implementation sequence:

- Re-run `preview_subtask_import`; if invalid, raise `SubtaskImportFailure` with the first deterministic error and let the API return the complete preview errors separately.
- Acquire a non-blocking per-resolved-output `threading.Lock`; report `output_preflight/OUTPUT_BUSY` when held.
- Create the temporary directory with `tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)`.
- Recreate the complete source tree with `shutil.copytree(source, temp_root, dirs_exist_ok=True, copy_function=hardlink_then_copy, symlinks=False)`. `hardlink_then_copy` calls `os.link` and falls back to `shutil.copy2` on `OSError`.
- Build imported atoms from preview rows with role `assistant`, style `subtask`, snapped start timestamp, null camera, and null tool calls.
- For each imported episode, choose existing atoms from sidecar when present, otherwise from Parquet; remove only style `subtask`; append imported atoms; sort by timestamp/style/role; and call `materialize_language_columns` on the complete source table.
- Write changed Parquet and metadata JSON to uniquely named sibling temporary files inside the temporary output tree, then `os.replace` them onto their hard-linked destinations. Never open a hard-linked destination for writing.
- Add `language_persistent` and `language_events` feature descriptors to the output `meta/info.json`, preserve `codebase_version` and path templates, and update only imported episodes in the output sidecar.
- Immediately before publishing, fail with `output_preflight/OUTPUT_ALREADY_EXISTS` if another process created the final path.
- Publish with `os.rename(temp_root, output)` while holding the lock.
- On every exception before rename, `shutil.rmtree(temp_root)` only when it is the exact path returned by `mkdtemp` and remains inside `output.parent`; convert copy/write failures to `dataset_write` and rename failures to `finalize`.
- Release the lock in `finally` and remove its dictionary entry under a guard lock.

- [ ] **Step 4: Run the full v2.1 integration suite and verify GREEN**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_dataset_v21 -v
```

Expected: all preview and apply tests pass; temporary directories are absent after each test.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add backend/dataset_v21.py backend/tests/test_dataset_v21.py backend/tests/fixtures.py
git commit -m "feat: apply v2.1 subtask annotations safely"
```

Expected: validation exits `0`; source-safety tests are green.

### Task 5: FastAPI preview and apply endpoints

**Files:**

- Modify: `backend/app.py`
- Create: `backend/requirements-dev.txt`
- Create: `backend/tests/test_subtask_import_api.py`

- [ ] **Step 1: Add the backend test dependency and failing API tests**

Create `backend/requirements-dev.txt`:

```text
-r requirements.txt
httpx>=0.27,<1
```

Install it into the isolated backend environment:

```bash
backend/.venv/bin/pip install -r backend/requirements-dev.txt
```

Create endpoint tests with FastAPI `TestClient`:

```python
class SubtaskImportApiTest(unittest.TestCase):
    def test_preview_returns_all_staged_errors_with_http_200(self) -> None:
        response = self.client.post(
            "/api/subtasks/import/preview",
            json={"local_path": str(self.source), "output_dir": str(self.output), "csv_text": csv_for("9,0,x,0,1")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["valid"])
        self.assertEqual(body["errors"][0]["stage"], "dataset_validation")

    def test_apply_returns_400_for_invalid_preview(self) -> None:
        response = self.client.post(
            "/api/subtasks/import/apply",
            json={"local_path": str(self.source), "output_dir": str(self.output), "csv_text": "bad"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["ok"])
        self.assertEqual(response.json()["errors"][0]["stage"], "csv_schema")

    def test_apply_returns_counts_for_valid_import(self) -> None:
        response = self.client.post(
            "/api/subtasks/import/apply",
            json={"local_path": str(self.source), "output_dir": str(self.output), "csv_text": csv_for("0,0,only,0,1")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["imported_subtasks"], 1)
        self.assertTrue(self.output.exists())
```

- [ ] **Step 2: Run API tests and verify RED**

Run:

```bash
backend/.venv/bin/python -m unittest backend.tests.test_subtask_import_api -v
```

Expected: 404 responses because the routes do not exist.

- [ ] **Step 3: Add thin request models and routes**

Add this request model and routes to `backend/app.py`:

```python
class SubtaskImportRequest(BaseModel):
    local_path: str
    output_dir: str
    csv_text: str


@app.post("/api/subtasks/import/preview")
def preview_subtask_csv(req: SubtaskImportRequest) -> JSONResponse:
    preview = preview_subtask_import(req.local_path, req.output_dir, req.csv_text)
    return JSONResponse(preview.to_dict())


@app.post("/api/subtasks/import/apply")
def apply_subtask_csv(req: SubtaskImportRequest) -> JSONResponse:
    preview = preview_subtask_import(req.local_path, req.output_dir, req.csv_text)
    if not preview.valid:
        return JSONResponse(
            {"ok": False, "errors": [issue.to_dict() for issue in preview.errors]},
            status_code=400,
        )
    try:
        return JSONResponse(
            apply_subtask_import(req.local_path, req.output_dir, req.csv_text).to_dict()
        )
    except SubtaskImportFailure as exc:
        return JSONResponse(
            {"ok": False, "errors": [exc.issue.to_dict()]},
            status_code=409 if exc.issue.stage in {"output_preflight", "finalize"} else 500,
        )
```

Import `preview_subtask_import`, `apply_subtask_import`, and `SubtaskImportFailure` from `dataset_v21`. Do not route the request through `_ensure_state`, which remains v3-only.

- [ ] **Step 4: Run backend tests and verify GREEN**

Run:

```bash
backend/.venv/bin/python -m unittest discover -s backend/tests -v
```

Expected: parser, codec, dataset, and API tests all pass.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add backend/app.py backend/requirements-dev.txt backend/tests/test_subtask_import_api.py
git commit -m "feat: expose subtask CSV import API"
```

Expected: validation exits `0`; commit contains only API wiring and its test dependency.

### Task 6: Typed frontend client and pure UI model

**Files:**

- Create: `src/types/subtask-import.types.ts`
- Create: `src/utils/subtaskImportClient.ts`
- Create: `src/utils/subtaskImportModel.ts`
- Create: `src/utils/__tests__/subtaskImportModel.test.ts`

- [ ] **Step 1: Write failing frontend model tests**

Create tests for default paths, stable stage order, and Apply gating:

```typescript
import { describe, expect, test } from "bun:test";
import {
  canApplyPreview,
  deriveAnnotatedOutputPath,
  groupIssuesByStage,
} from "@/utils/subtaskImportModel";

describe("subtaskImportModel", () => {
  test("derives a sibling annotated path", () => {
    expect(deriveAnnotatedOutputPath("/datasets/pnp_trash")).toBe(
      "/datasets/pnp_trash_annotated",
    );
    expect(deriveAnnotatedOutputPath("/datasets/pnp_trash/")).toBe(
      "/datasets/pnp_trash_annotated",
    );
  });

  test("groups issues in workflow stage order", () => {
    const groups = groupIssuesByStage([
      {
        stage: "timeline_validation",
        code: "TIMELINE_GAP",
        message: "gap",
        row: 3,
        field: null,
        episode_index: 0,
      },
      {
        stage: "csv_schema",
        code: "MISSING_HEADER",
        message: "missing",
        row: 1,
        field: "prompt",
        episode_index: null,
      },
    ]);
    expect(groups.map((group) => group.stage)).toEqual([
      "csv_schema",
      "timeline_validation",
    ]);
  });

  test("allows apply only for the unchanged valid preview", () => {
    expect(
      canApplyPreview(
        { valid: true, sourcePath: "/a", outputPath: "/b", csvText: "x" },
        "/a",
        "/b",
        "x",
      ),
    ).toBe(true);
    expect(
      canApplyPreview(
        { valid: true, sourcePath: "/a", outputPath: "/b", csvText: "x" },
        "/a",
        "/c",
        "x",
      ),
    ).toBe(false);
  });
});
```

- [ ] **Step 2: Run the model test and verify RED**

Run:

```bash
bun test src/utils/__tests__/subtaskImportModel.test.ts
```

Expected: module-not-found failure for `subtaskImportModel`.

- [ ] **Step 3: Define the API contract and pure model**

In `src/types/subtask-import.types.ts`, define:

```typescript
export type ImportErrorStage =
  | "file_read"
  | "csv_parse"
  | "csv_schema"
  | "row_validation"
  | "dataset_validation"
  | "timeline_validation"
  | "output_preflight"
  | "dataset_write"
  | "finalize";

export interface SubtaskImportIssue {
  stage: ImportErrorStage;
  code: string;
  message: string;
  row: number | null;
  field: string | null;
  episode_index: number | null;
}

export interface SubtaskPreviewRow {
  csv_row: number;
  episode_index: number;
  subtask_index: number;
  prompt: string;
  original_start: number;
  original_end: number;
  snapped_start: number;
  snapped_end: number;
}

export interface SubtaskEpisodePreview {
  episode_index: number;
  first_timestamp: number;
  final_timestamp: number;
  rows: SubtaskPreviewRow[];
}

export interface SubtaskImportPreview {
  valid: boolean;
  source_path: string;
  output_path: string;
  version: string | null;
  episode_count: number;
  subtask_count: number;
  episodes: SubtaskEpisodePreview[];
  warnings: SubtaskImportIssue[];
  errors: SubtaskImportIssue[];
}

export interface SubtaskImportApplyResult {
  ok: boolean;
  output_dir: string;
  rewritten_episodes: number;
  imported_subtasks: number;
  materialized_files: number;
}
```

In `src/utils/subtaskImportModel.ts`, implement the exact stage-order constant, sibling path derivation for absolute POSIX paths, `groupIssuesByStage`, and `canApplyPreview`. The preview snapshot type used by `canApplyPreview` is:

```typescript
export interface PreviewSnapshot {
  valid: boolean;
  sourcePath: string;
  outputPath: string;
  csvText: string;
}
```

In `src/utils/subtaskImportClient.ts`, expose:

```typescript
export async function previewSubtaskImport(request: {
  local_path: string;
  output_dir: string;
  csv_text: string;
}): Promise<SubtaskImportPreview>;

export async function applySubtaskImport(request: {
  local_path: string;
  output_dir: string;
  csv_text: string;
}): Promise<SubtaskImportApplyResult>;
```

Both functions use `getAnnotateBackendUrl()`, JSON POST, and parse non-2xx bodies shaped as `{errors?: SubtaskImportIssue[]}`. Export `SubtaskImportApiError extends Error` with an `issues` property so the component can render backend stages without parsing strings.

- [ ] **Step 4: Run frontend model tests and type checks**

Run:

```bash
bun test src/utils/__tests__/subtaskImportModel.test.ts
bun run type-check
```

Expected: model tests pass and TypeScript exits `0`.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add src/types/subtask-import.types.ts src/utils/subtaskImportClient.ts src/utils/subtaskImportModel.ts src/utils/__tests__/subtaskImportModel.test.ts
git commit -m "feat: add typed subtask import client"
```

Expected: validation exits `0`; commit contains only pure frontend contracts and tests.

### Task 7: Bulk import component and Annotations-tab integration

**Files:**

- Create: `src/components/subtask-csv-import.tsx`
- Create: `src/components/__tests__/subtask-csv-import.test.ts`
- Modify: `src/components/annotations-panel.tsx:18-28, 620-690`
- Modify: `src/components/annotations-skin.css`

- [ ] **Step 1: Write the failing static component contract test**

Create `src/components/__tests__/subtask-csv-import.test.ts`:

```typescript
import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { SubtaskCsvImport } from "@/components/subtask-csv-import";

describe("SubtaskCsvImport", () => {
  test("renders the CSV contract and disables apply while offline", () => {
    const html = renderToStaticMarkup(
      createElement(SubtaskCsvImport, { backendEnabled: false }),
    );
    expect(html).toContain("Bulk subtask CSV");
    expect(html).toContain("episode_index");
    expect(html).toContain("annotation backend is unavailable");
    expect(html).toContain("Apply to new dataset");
    expect(html).toContain("disabled");
  });
});
```

Run:

```bash
bun test src/components/__tests__/subtask-csv-import.test.ts
```

Expected RED: module-not-found failure because `subtask-csv-import.tsx` does not exist.

- [ ] **Step 2: Implement the component state machine**

Create `src/components/subtask-csv-import.tsx` as a client component. It must contain these states:

```typescript
const [expanded, setExpanded] = useState(true);
const [sourcePath, setSourcePath] = useState(
  process.env.NEXT_PUBLIC_LOCAL_DATASET_PATH?.trim() ?? "",
);
const [outputPath, setOutputPath] = useState(() =>
  deriveAnnotatedOutputPath(
    process.env.NEXT_PUBLIC_LOCAL_DATASET_PATH?.trim() ?? "",
  ),
);
const [csvName, setCsvName] = useState("");
const [csvText, setCsvText] = useState("");
const [preview, setPreview] = useState<SubtaskImportPreview | null>(null);
const [snapshot, setSnapshot] = useState<PreviewSnapshot | null>(null);
const [issues, setIssues] = useState<SubtaskImportIssue[]>([]);
const [phase, setPhase] = useState<
  "idle" | "reading" | "previewing" | "ready" | "applying" | "done"
>("idle");
const [result, setResult] = useState<SubtaskImportApplyResult | null>(null);
```

Required handlers:

- `handleSourceChange` updates the source and refreshes the default output only while the old output still equals the old derived default.
- `handleFileChange` calls `await file.text()`; on failure, creates a local `file_read/FILE_READ_FAILED` issue with null locations.
- `handlePreview` clears prior result/issues, posts all three fields, stores the returned preview and an exact snapshot, and sets phase `ready` only for `valid: true`.
- `handleApply` first checks `canApplyPreview(snapshot, sourcePath, outputPath, csvText)`, posts the same three fields, and renders structured `SubtaskImportApiError.issues` on failure.
- Any source/output/file edit clears preview, snapshot, result, and moves phase back to `idle`.
- Buttons are disabled while reading/previewing/applying; Apply is disabled unless the backend is enabled and the unchanged preview is valid.

Render:

- a collapsible heading and short schema hint;
- source and output text fields;
- an `.csv` file input and Preview button;
- an offline configuration notice;
- summary cards for episode/subtask counts and version;
- errors grouped by stage with code, row, field, episode, and message;
- warnings in a separate group;
- a per-episode table with prompt and `original → snapped` boundaries;
- Apply button; and
- final output/count summary.

Use semantic `label`, `table`, `thead`, `tbody`, `aria-live="polite"`, and `aria-disabled` attributes. Do not add a new component framework.

- [ ] **Step 3: Wire the component into the panel and style it**

Import and render it immediately below the action bar/status and above the manual annotation composer:

```tsx
import { SubtaskCsvImport } from "./subtask-csv-import";

<SubtaskCsvImport backendEnabled={backendEnabled} />;
```

Add scoped classes under `.annotations-skin` for `.subtask-import`, `.subtask-import-grid`, `.subtask-import-summary`, `.subtask-import-errors`, `.subtask-import-table-wrap`, and `.subtask-import-result`. Reuse existing surface, border, cyan, green, red, and monospace timestamp tokens. At widths below `760px`, collapse the path grid and summary cards to one column and keep the table horizontally scrollable.

- [ ] **Step 4: Verify model tests, type checks, and lint**

Run:

```bash
bun test src/components/__tests__/subtask-csv-import.test.ts
bun test src/utils/__tests__/subtaskImportModel.test.ts
bun run type-check
bun run lint
```

Expected: all commands exit `0`; no React hook or accessibility lint errors.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add src/components/subtask-csv-import.tsx src/components/__tests__/subtask-csv-import.test.ts src/components/annotations-panel.tsx src/components/annotations-skin.css
git commit -m "feat: add bulk subtask CSV workflow"
```

Expected: validation exits `0`; the Annotations tab contains the complete preview/apply workflow.

### Task 8: Display imported language atoms in v2.1 episodes

**Files:**

- Modify: `src/app/[org]/[dataset]/[episode]/fetch-data.ts:587-597, 693-720, 1005-1084`
- Modify: `src/app/[org]/[dataset]/[episode]/__tests__/fetch-data.test.ts`

- [ ] **Step 1: Write failing v2.1 extraction tests**

Add tests that import the not-yet-exported `extractLanguageAtoms` and not-yet-created `languageColumnsForFeatures` from `fetch-data.ts`:

```typescript
test("selects v2.1 language columns only when metadata declares them", () => {
  expect(
    languageColumnsForFeatures({
      timestamp: { dtype: "float32", shape: [1], names: null },
      language_persistent: { dtype: "language", shape: [1], names: null },
      language_events: { dtype: "language", shape: [1], names: null },
    }),
  ).toEqual(["language_persistent", "language_events"]);
  expect(languageColumnsForFeatures({})).toEqual([]);
});

test("extracts broadcast subtasks and frame-timed events from v2.1 rows", () => {
  const rows = [
    {
      timestamp: 0,
      language_persistent: [
        {
          role: "assistant",
          content: "approach",
          style: "subtask",
          timestamp: 0,
          camera: null,
          tool_calls: null,
        },
      ],
      language_events: [],
    },
    {
      timestamp: 1,
      language_persistent: [
        {
          role: "assistant",
          content: "approach",
          style: "subtask",
          timestamp: 0,
          camera: null,
          tool_calls: null,
        },
      ],
      language_events: [
        {
          role: "user",
          content: "stop",
          style: "interjection",
          camera: null,
          tool_calls: null,
        },
      ],
    },
  ];
  expect(
    extractLanguageAtoms(rows).map(({ style, timestamp }) => ({
      style,
      timestamp,
    })),
  ).toEqual([
    { style: "subtask", timestamp: 0 },
    { style: "interjection", timestamp: 1 },
  ]);
});
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
bun test src/app/[org]/[dataset]/[episode]/__tests__/fetch-data.test.ts
```

Expected: missing export/helper failure.

- [ ] **Step 3: Load declared language columns in the v2.1 path**

Implement:

```typescript
export function languageColumnsForFeatures(
  features: DatasetMetadata["features"],
): string[] {
  return ["language_persistent", "language_events"].filter(
    (key) => key in features,
  );
}
```

Add these declared columns to `parquetColumns` in `getEpisodeDataV2` so old v2.1 files without them do not request missing columns. Export the existing `extractLanguageAtoms` function, and add to the v2.1 return value:

```typescript
languageAtoms: extractLanguageAtoms(allData),
frameTimestamps: allData
  .map((row) => Number(row.timestamp))
  .filter((timestamp) => Number.isFinite(timestamp)),
```

Keep frame timestamps unsampled so manual event snapping retains source-frame precision. Update comments that currently describe extraction as v3-only.

- [ ] **Step 4: Run focused and full frontend tests**

Run:

```bash
bun test src/app/[org]/[dataset]/[episode]/__tests__/fetch-data.test.ts
bun run type-check
```

Expected: focused tests and TypeScript pass.

- [ ] **Step 5: Run repository validation and commit**

Run:

```bash
bun run format && bun run validate
git add src/app/[org]/[dataset]/[episode]/fetch-data.ts src/app/[org]/[dataset]/[episode]/__tests__/fetch-data.test.ts
git commit -m "feat: read language annotations from v2.1"
```

Expected: validation exits `0`; existing v2.0/v2.1 datasets without language columns remain covered.

### Task 9: User documentation and CSV template

**Files:**

- Create: `examples/subtask_annotations.csv`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `backend/README.md`

- [ ] **Step 1: Add the exact CSV template**

Create:

```csv
episode_index,subtask_index,prompt,start_timestamp,end_timestamp
0,0,approach the object,0.0,2.5
0,1,pick up the object,2.5,5.0
```

- [ ] **Step 2: Document environment and launch commands**

Add to `.env.example`:

```dotenv
# Absolute local v2.1 dataset path used to prefill Bulk subtask CSV.
# NEXT_PUBLIC_LOCAL_DATASET_PATH=/absolute/path/to/pnp_trash
```

Update both READMEs with:

- the five exact CSV headers and interval rules;
- `python -m venv backend/.venv` and `backend/.venv/bin/pip install -r backend/requirements-dev.txt`;
- backend launch from `backend/` on port 7861;
- frontend launch with both `NEXT_PUBLIC_ANNOTATE_BACKEND_URL` and `NEXT_PUBLIC_LOCAL_DATASET_PATH`;
- preview versus apply behavior;
- staged error names;
- refusal to overwrite outputs; and
- the fact that the first release is local v2.1 only.

- [ ] **Step 3: Run formatting and link/path checks**

Run:

```bash
bun run format
test -f examples/subtask_annotations.csv
rg -n "Bulk subtask CSV|episode_index,subtask_index,prompt,start_timestamp,end_timestamp|NEXT_PUBLIC_LOCAL_DATASET_PATH" README.md backend/README.md .env.example examples/subtask_annotations.csv
```

Expected: formatter exits `0`; each required phrase appears in the intended file.

- [ ] **Step 4: Run repository validation and commit**

Run:

```bash
bun run validate
git add .env.example README.md backend/README.md examples/subtask_annotations.csv
git commit -m "docs: explain v2.1 subtask CSV import"
```

Expected: validation exits `0`; documentation matches the implemented fields and endpoints.

### Task 10: Full verification and real-dataset smoke test

**Files:**

- No production file changes expected.

- [ ] **Step 1: Run the complete backend suite**

Run:

```bash
backend/.venv/bin/python -m unittest discover -s backend/tests -v
```

Expected: all backend tests pass with zero failures/errors.

- [ ] **Step 2: Run the repository-required validation sequence**

Run exactly in this order:

```bash
bun run format
bun run type-check
bun run lint
bun test
```

Expected: all commands exit `0`; formatting produces no subsequent diff.

- [ ] **Step 3: Preview the actual `pnp_trash` dataset read-only**

Run a Python one-liner from the repository root that reads episode 0's first/final timestamps, builds a one-subtask CSV in memory, and calls `preview_subtask_import` with an unused `/tmp` destination:

```bash
PYTHONPATH=backend backend/.venv/bin/python -c 'from pathlib import Path; import tempfile; import pyarrow.parquet as pq; from dataset_v21 import preview_subtask_import; root=Path("/home/jihun/work/GR00T-WholeBodyControl/outputs/pnp_trash"); table=pq.read_table(root/"data/chunk-000/episode_000000.parquet", columns=["timestamp"]); ts=[float(x) for x in table.column("timestamp").to_pylist()]; csv_text=f"episode_index,subtask_index,prompt,start_timestamp,end_timestamp\n0,0,smoke test,{min(ts)},{max(ts)}\n"; temp=tempfile.TemporaryDirectory(); preview=preview_subtask_import(root, Path(temp.name)/"pnp_trash_annotated", csv_text); print(preview.to_dict()); temp.cleanup(); raise SystemExit(0 if preview.valid else 1)'
```

Expected: `valid` is `true`, version is `v2.1`, episode count is `1`, and no output directory is created.

- [ ] **Step 4: Audit source immutability and repository diff**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -10
```

Expected: no whitespace errors; only intentional feature changes/commits appear; the real source dataset was only read.

- [ ] **Step 5: Request code review before integration**

Invoke the `requesting-code-review` skill, address any correctness findings with test-first fixes, rerun Steps 1–4, then use `finishing-a-development-branch` to present integration options without modifying the user's dirty main worktree.

## Requirement coverage checklist

- Normalized one-row-per-subtask CSV: Task 1 and Task 9.
- Explicit wrong-format/error stages: Tasks 1, 3, 5, 6, and 7.
- Preview before mutation: Tasks 3, 5, and 7.
- Episode-relative seconds and deterministic frame snapping: Task 3.
- Contiguous full-episode intervals: Task 3.
- Replace existing subtasks while preserving other annotations: Tasks 2 and 4.
- New sibling dataset without overwrite/source mutation: Task 4.
- v2.1-first implementation: Tasks 3, 4, and 8.
- Imported annotations visible in the visualizer: Task 8.
- Backend-unavailable guidance and local path configuration: Tasks 7 and 9.
- Source-safety, atomicity, and real-dataset evidence: Tasks 4 and 10.
