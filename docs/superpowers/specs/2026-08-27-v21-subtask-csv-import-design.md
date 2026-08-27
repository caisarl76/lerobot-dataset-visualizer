# LeRobot v2.1 Subtask CSV Import Design

## Summary

Add a bulk CSV workflow to the visualizer's Annotations tab for applying timestamped subtask prompts to local LeRobot v2.1 datasets. The workflow validates and previews the complete CSV before it creates a new sibling dataset. It never mutates the source dataset and never partially applies an invalid import.

The first release supports local v2.1 datasets with one Parquet file per episode. Support for v2.0 and v3.x is explicitly out of scope.

## User workflow

1. Open an episode's Annotations tab and expand **Bulk subtask CSV**.
2. Enter the absolute source dataset path. When `NEXT_PUBLIC_LOCAL_DATASET_PATH` is configured, use it as the initial value. Confirm the output path, which defaults to `<source_dataset_path>_annotated`.
3. Select a UTF-8 CSV file and click **Preview**.
4. Review the affected episode/subtask counts, snapped timestamp boundaries, warnings, and errors grouped by validation stage.
5. Click **Apply to new dataset**. This action is available only after a valid preview.
6. Show the completed output path and counts of imported subtasks, rewritten episodes, and materialized files.

The bulk workflow is disabled when `NEXT_PUBLIC_ANNOTATE_BACKEND_URL` is unset or the annotation backend is unavailable. The UI explains how to configure the backend rather than silently falling back to session storage.

## CSV contract

The importer accepts a normalized, one-row-per-subtask CSV with exactly these headers:

```csv
episode_index,subtask_index,prompt,start_timestamp,end_timestamp
0,0,approach the trash,0.0,3.2
0,1,pick up the trash,3.2,6.8
1,0,locate the trash,0.0,2.5
```

Header order is not significant. A UTF-8 byte-order mark, standard CSV quoting, commas inside quoted prompts, and multiline quoted prompts are accepted. Missing, duplicate, or unknown headers are schema errors. Blank lines are ignored. CSV row numbers reported to users are one-based and include the header row, so the first data row is row 2.

Field rules:

- `episode_index` is a non-negative base-10 integer.
- `subtask_index` is a non-negative base-10 integer. Within each included episode, indices start at `0`, are unique, and are consecutive.
- `prompt` is non-empty after surrounding whitespace is trimmed. Internal whitespace and line breaks are preserved.
- `start_timestamp` and `end_timestamp` are finite, non-negative seconds relative to the start of the episode.
- `start_timestamp` is strictly less than `end_timestamp` before and after frame snapping.
- Rows may appear in any CSV order. Validation orders them by `episode_index` and `subtask_index`, then verifies that index order is also chronological.
- Each `(episode_index, subtask_index)` pair is unique.

The CSV may contain only a subset of dataset episodes. Episodes absent from the CSV remain unchanged. For every episode present in the CSV, its imported intervals replace all existing `subtask` atoms for that episode.

## Timeline semantics

CSV timestamps are episode-relative seconds. Both boundaries are snapped to the nearest source-frame timestamp before the preview is returned or data is written. When two frames are equally near, the earlier frame wins so snapping is deterministic.

An imported episode must be fully and contiguously covered:

- The first snapped start equals the episode's first frame timestamp.
- Each snapped end equals the next snapped start.
- No snapped intervals overlap or leave a gap.
- The final snapped end equals the episode's final frame timestamp.

A subtask is active from its snapped start through the frame before the next subtask starts. The final subtask remains active through the final frame. The canonical `language_persistent` representation stores each subtask's start timestamp; because coverage is contiguous, the next start reconstructs the preceding end. The CSV's explicit end values are used for validation and preview but are not added as a non-standard language-atom field.

## Architecture and data flow

### Frontend

A focused bulk-import component lives inside the existing Annotations tab. It owns:

- source path, output path, and selected-file state;
- reading the file as text without uploading multipart binary data;
- preview/apply request state;
- the preview summary and per-episode boundary table;
- errors grouped by stage, with row and field locations;
- invalidating the preview whenever the file, source path, or output path changes; and
- preventing duplicate apply requests.

A dedicated typed API client sends JSON to two backend endpoints:

- `POST /api/subtasks/import/preview`
- `POST /api/subtasks/import/apply`

Both requests carry `local_path`, `output_dir`, and `csv_text`. Apply always repeats validation using the submitted CSV before it touches the filesystem. It does not trust stale browser preview state.

The v2.1 episode loader is extended to request and extract `language_persistent` and `language_events` when those columns exist. This makes imported subtasks visible in the existing annotation list and timeline as soon as the annotated dataset is served by the visualizer.

### Backend

The backend is divided into three responsibilities:

- `backend/subtask_csv.py` parses CSV text, produces normalized rows, validates the CSV schema and row-level invariants, and creates staged errors.
- `backend/dataset_v21.py` validates a local v2.1 dataset, resolves per-episode Parquet paths from `meta/info.json`, reads source-frame timestamps, snaps and validates timelines, merges atoms, and materializes an annotated sibling dataset.
- `backend/app.py` defines request/response models and thin preview/apply routes.

The existing v3-oriented dataset-state loader is not reused for CSV import because it requires `meta/episodes/chunk-*/file-*.parquet`, while v2.1 uses `meta/episodes.jsonl` and per-episode data files. The new v2.1 adapter is isolated so later v3 support can be added without branching throughout the parser and UI.

## Preview behavior

Preview performs all checks needed to determine whether apply can begin:

- parse and validate the CSV;
- confirm the source exists and `meta/info.json` reports `codebase_version: "v2.1"`;
- confirm each episode exists and its Parquet file can be resolved;
- read the episode's `episode_index` and `timestamp` columns;
- snap boundaries and validate full contiguous coverage; and
- confirm the default or selected output path is distinct from the source and does not already exist.

A valid preview returns:

- source dataset path and detected version;
- affected episode and imported subtask counts;
- per-episode source duration and normalized rows;
- original and snapped start/end values for every subtask;
- timestamp-snap warnings; and
- `valid: true` with no errors.

An invalid preview returns `valid: false` and every independently discoverable error, rather than stopping at the first bad row. Dataset reads for rows that cannot be structurally parsed are skipped.

## Error stages

Every user-visible error has this shape:

```json
{
  "stage": "row_validation",
  "code": "INVALID_TIMESTAMP",
  "row": 7,
  "field": "start_timestamp",
  "episode_index": 3,
  "message": "start_timestamp must be a finite, non-negative number"
}
```

Location properties that do not apply are `null`. Stable stages are:

| Stage | Responsibility | Examples |
| --- | --- | --- |
| `file_read` | Browser file access | unreadable file or failed `File.text()` |
| `csv_parse` | CSV syntax and UTF-8 decoding | unterminated quote or inconsistent record width |
| `csv_schema` | Header contract | missing, duplicate, or unknown header |
| `row_validation` | Scalar row fields and uniqueness | empty prompt, invalid integer, non-finite timestamp, duplicate key |
| `dataset_validation` | Source dataset compatibility | missing dataset, unsupported version, unknown episode, unreadable Parquet |
| `timeline_validation` | Snapped interval rules | zero duration, chronological mismatch, gap, overlap, incomplete coverage |
| `output_preflight` | Destination safety | source equals output, output already exists, parent is not writable |
| `dataset_write` | Materialization and Parquet writing | copy, hard-link, or serialization failure |
| `finalize` | Publishing completed temporary output | atomic rename failure |

Preview errors use HTTP success with `valid: false` when the request itself is well-formed. Invalid JSON/request models use the normal FastAPI 422 response. Apply returns a structured error response with the same stage model and never reports success for partial output.

The UI groups errors by stage, shows row and field when available, and disables Apply while any error exists.

## Applying annotations

Apply re-runs the same parser, dataset checks, timestamp snapping, timeline validation, and output preflight used by preview. It then:

1. Creates a uniquely named temporary directory inside the output directory's parent, ensuring the final rename stays on one filesystem.
2. Recreates the source dataset tree using hard links where supported and byte copies as fallback.
3. For every imported episode, loads the complete source Parquet table and collects existing language atoms. If `meta/lerobot_annotations.json` contains an authoritative saved atom list for that episode, it takes precedence over atoms extracted from Parquet, matching the current annotation backend's export behavior.
4. Removes only atoms whose style is `subtask` and appends imported assistant subtask atoms with trimmed prompts, snapped start timestamps, `camera: null`, and `tool_calls: null`.
5. Preserves every non-subtask persistent atom and every event atom.
6. Broadcasts the complete persistent atom list across every frame of that episode and preserves event atoms on their source frames.
7. Writes each changed Parquet table to a new temporary file and uses `os.replace` inside the temporary output tree. It never truncates a hard-linked destination file, which would mutate the source inode.
8. Updates the output copy of `meta/info.json` with `language_persistent` and `language_events` feature entries while retaining `codebase_version: "v2.1"` and the existing v2.1 path templates.
9. Updates the output copy of `meta/lerobot_annotations.json` for imported episodes so later manual editing sees the same merged atom state. Metadata changes also use write-to-new-file plus `os.replace`, so no hard-linked source inode is modified.
10. Atomically renames the completed temporary tree to the requested output path.

If any step fails, the backend removes only the uniquely named temporary output it created. It never deletes or overwrites the requested final output path and never changes the source dataset. A pre-existing final output is an `output_preflight` error rather than an overwrite prompt.

## Output schema

Imported subtasks use the visualizer's existing language atom representation:

```json
{
  "role": "assistant",
  "content": "pick up the trash",
  "style": "subtask",
  "timestamp": 3.2,
  "camera": null,
  "tool_calls": null
}
```

`language_persistent` contains the full timestamp-ordered persistent atom list on every episode frame. `language_events` remains a per-frame list. Arrow struct field order and types match the language schema already written by the annotation backend. The importer uses explicit Arrow types so episodes with no event atoms do not produce ambiguous `list<null>` columns.

The output is a self-contained sibling dataset. Unchanged files may share storage with the source through hard links, but the output contains no symlinks and remains portable when copied normally.

## Concurrency and failure safety

- Preview is read-only and may run concurrently.
- Apply uses an in-process lock keyed by resolved output path so duplicate requests cannot build the same destination simultaneously.
- Output existence is checked before work starts and again immediately before final rename.
- Temporary paths include a random suffix and are scoped to the destination parent.
- Cleanup targets only the exact temporary path created by the request.
- Source and output paths are resolved before comparison; the source itself and any path inside it are rejected as outputs.
- The API does not accept a request to overwrite an existing destination.

## Testing strategy

Backend unit tests cover:

- UTF-8 BOM, quoted commas, multiline prompts, blank lines, and arbitrary row order;
- malformed CSV syntax and every header error;
- invalid integers, empty prompts, NaN/infinite/negative timestamps, duplicate keys, and non-consecutive indices;
- missing episodes and unsupported dataset versions;
- deterministic nearest-frame snapping, including equal-distance ties;
- valid contiguous coverage, gaps, overlaps, chronological mismatches, and incomplete first/final coverage; and
- stable error stage, code, row, field, and episode locations.

Backend integration tests construct small temporary v2.1 datasets with real PyArrow Parquet files and verify:

- preview is read-only;
- imported episodes replace only subtask atoms;
- unrelated persistent and event atoms survive;
- episodes absent from the CSV remain byte-identical;
- the source tree remains byte-identical after success and induced failures;
- output feature metadata and Arrow schemas are correct;
- an existing output is refused;
- write failures remove temporary output; and
- successful apply publishes a complete sibling dataset atomically.

Frontend tests cover API response normalization, error grouping and labels, preview invalidation when inputs change, disabled Apply states, and summary formatting. Existing fetch-data tests gain v2.1 language-column extraction cases.

Final verification runs the backend test suite followed by the repository-required commands:

```bash
python -m unittest discover -s backend/tests -v
bun run format
bun run type-check
bun run lint
bun test
```

## Out of scope

- LeRobot v2.0, v3.0, or v3.1 shared-shard import.
- Hugging Face repository import or push.
- Modifying a source dataset in place.
- Merging imported subtasks with existing subtasks.
- Non-contiguous or overlapping subtask timelines.
- Additional CSV layouts or spreadsheet formats.
- Editing CSV contents inside the preview UI.
