"""Real-file prompt coverage regressions; no review or parquet reader mocks."""

from collections import Counter
import json
from pathlib import Path

from annotation_clipping import normalize_exclusions
from annotation_history import annotation_hash, exclusions_hash
import annotation_monitor as monitor
from monitor_fixtures import write_json, write_run, write_v3, write_v21
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def atom(text, timestamp=0.0, **extra):
    return dict(role="assistant", style="subtask", content=text, timestamp=timestamp, **extra)


@pytest.fixture(params=["v2.1", "v3.0"])
def prompt_fixture(tmp_path, request):
    def make(*, lengths=(10,), labels=None, exclusions=None, fps=10, name="case"):
        root = (write_v21 if request.param == "v2.1" else write_v3)(tmp_path / name, list(lengths))
        info = json.loads((root / "meta/info.json").read_text())
        info["fps"] = fps
        write_json(root / "meta/info.json", info)
        run_path = write_run(tmp_path / (name + "-workspace"), root, root, list(range(len(lengths))))
        run = json.loads(run_path.read_text())
        labels = labels if labels is not None else [[atom("A"), atom("B", 0.3)] for _ in lengths]
        exclusions = (
            exclusions if exclusions is not None else [[{"start_frame": 2, "end_frame": 5}] for _ in lengths]
        )
        reviews, annotations = {}, {}
        for ep, (atoms, cuts) in enumerate(zip(labels, exclusions)):
            key = str(ep)
            annotations[key] = {"atoms": atoms}
            run["episodes"][key]["excluded_intervals"] = cuts
            reviews[key] = dict(
                annotation_sha256=annotation_hash(atoms),
                exclusions_sha256=exclusions_hash(normalize_exclusions(cuts, lengths[ep])),
                reviewed_at="2026-09-16T00:00:00Z",
            )
        write_json(root / "meta/lerobot_annotations.json", {"version": 2, "episodes": annotations})
        write_json(root / "meta/annotation_reviews.json", reviews)
        write_json(run_path, run)
        return run

    return make


def data_path(run, ep=0):
    root = Path(run["root"])
    info = json.loads((root / "meta/info.json").read_text())
    return root / info["data_path"].format(episode_chunk=0, episode_index=ep, chunk_index=0, file_index=ep)


def replace_column(run, name, values, ep=0):
    path = data_path(run, ep)
    table = pq.read_table(path)
    if name in table.column_names:
        table = table.drop([name])
    pq.write_table(table.append_column(name, pa.array(values)), path)


def counts(result):
    return {row["text"]: row["frames"] for row in result["rows"]}


def assert_partition(result):
    assert (
        sum(counts(result).values()) + result["unlabeled_frames"] + result["ambiguous_frames"]
        == result["retained_frames"]
    )
    if result["retained_frames"]:
        assert sum(row["ratio"] for row in result["rows"]) + (
            result["unlabeled_frames"] + result["ambiguous_frames"]
        ) / result["retained_frames"] == pytest.approx(1)
    json.dumps(result, allow_nan=False)


def test_prompt_inside_exclusion_remains_active_after_cut(prompt_fixture):
    result = monitor.prompt_distribution(prompt_fixture())
    assert counts(result) == {"A": 2, "B": 5}
    assert result["retained_frames"] == 7
    assert result["eligible_episodes"] == result["evaluated_episodes"] == 1
    assert result["complete"] is True
    assert {r["text"]: r["seconds"] for r in result["rows"]} == {"A": 0.2, "B": 0.5}
    assert_partition(result)


@pytest.mark.parametrize("change", ["exclusion", "annotation", "delete", "unreviewed"])
def test_stale_deleted_and_unreviewed_are_ineligible(prompt_fixture, change):
    run = prompt_fixture()
    root = Path(run["root"])
    if change == "exclusion":
        run["episodes"]["0"]["excluded_intervals"] = [{"start_frame": 1, "end_frame": 5}]
    elif change == "annotation":
        write_json(root / "meta/lerobot_annotations.json", {"episodes": {"0": {"atoms": [atom("changed")]}}})
    elif change == "delete":
        run["episodes"]["0"]["decision"] = "delete"
    else:
        write_json(root / "meta/annotation_reviews.json", {})
    result = monitor.prompt_distribution(run)
    assert result["eligible_episodes"] == result["evaluated_episodes"] == result["retained_frames"] == 0
    assert result["unknown_episode_ids"] == []
    assert result["rows"] == []
    assert result["complete"] is True


def test_actual_timestamps_and_exact_boundaries(prompt_fixture):
    run = prompt_fixture(labels=[[atom("first", 0.0), atom("next", 0.3), atom("last", 0.9)]], exclusions=[[]])
    replace_column(run, "timestamp", [0.0, 0.03, 0.07, 0.12, 0.2, 0.3, 0.45, 0.6, 0.8, 0.9])
    result = monitor.prompt_distribution(run)
    assert counts(result) == {"first": 5, "next": 4, "last": 1}
    assert_partition(result)


def test_unlabeled_no_task_aug_fallback_and_conflicting_latest_set(prompt_fixture):
    labels = [
        dict(role="user", style="task_aug", content="never fallback", timestamp=0.0),
        atom("A", 0.2),
        atom("A", 0.2),
        atom("B", 0.2),
        atom("C", 0.5),
        atom("C", 0.5),
    ]
    result = monitor.prompt_distribution(prompt_fixture(labels=[labels], exclusions=[[]]))
    assert counts(result) == {"C": 5}
    assert result["unlabeled_frames"] == 2
    assert result["ambiguous_frames"] == 3
    assert_partition(result)


def test_literals_named_like_buckets_and_whitespace_remain_distinct(prompt_fixture):
    result = monitor.prompt_distribution(
        prompt_fixture(
            labels=[[atom("Unlabeled", 0.1), atom("Ambiguous", 0.3), atom("A", 0.5), atom(" A ", 0.7)]],
            exclusions=[[]],
        )
    )
    assert counts(result) == {"Unlabeled": 2, "Ambiguous": 2, "A": 2, " A ": 3}
    assert result["unlabeled_frames"] == 1
    assert result["ambiguous_frames"] == 0
    assert_partition(result)


def test_counts_episodes_with_counted_frames_not_atom_occurrences(prompt_fixture):
    run = prompt_fixture(
        lengths=(10, 10), labels=[[atom("A"), atom("A", 0.3)], [atom("A"), atom("B", 0.5)]], exclusions=[[], []]
    )
    result = monitor.prompt_distribution(run)
    assert counts(result) == {"A": 15, "B": 5}
    assert {r["text"]: r["episodes"] for r in result["rows"]} == {"A": 2, "B": 1}
    assert result["evaluated_episodes"] == 2
    assert_partition(result)


def test_different_run_fps_uses_each_checkpoint_duration(prompt_fixture):
    slow = monitor.prompt_distribution(prompt_fixture(fps=10, name="slow"))
    fast = monitor.prompt_distribution(prompt_fixture(fps=20, name="fast"))
    assert {r["text"]: r["seconds"] for r in slow["rows"]} == {"A": 0.2, "B": 0.5}
    assert {r["text"]: r["seconds"] for r in fast["rows"]} == {"A": 0.1, "B": 0.25}
    assert counts(slow) == counts(fast)


@pytest.mark.parametrize(
    "bad", ["missing", "nan", "inf", "duplicate", "descending", "negative", "string", "boolean"]
)
def test_corrupt_timestamps_make_partial_coverage_explicit(prompt_fixture, bad):
    run = prompt_fixture(lengths=(10, 10))
    times = [i / 10 for i in range(10)]
    if bad == "missing":
        path = data_path(run, 1)
        pq.write_table(pq.read_table(path).drop(["timestamp"]), path)
    else:
        if bad == "nan":
            times[4] = float("nan")
        elif bad == "inf":
            times[4] = float("inf")
        elif bad == "duplicate":
            times[4] = times[3]
        elif bad == "descending":
            times[4] = 0.25
        elif bad == "negative":
            times[0] = -0.1
        elif bad == "string":
            times = list(map(str, times))
        else:
            times = [False] + [True] * 9
        replace_column(run, "timestamp", times, ep=1)
    result = monitor.prompt_distribution(run)
    assert result["eligible_episodes"] == 2
    assert result["evaluated_episodes"] == 1
    assert result["unknown_episode_ids"] == ["1"]
    assert result["complete"] is False
    assert result["retained_frames"] == 7
    assert counts(result) == {"A": 2, "B": 5}
    assert result["diagnostics"]
    assert_partition(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("timestamp", -0.1),
        ("timestamp", 1.0),
        ("timestamp", float("nan")),
        ("timestamp", float("inf")),
        ("timestamp", True),
        ("timestamp", "bad"),
        ("content", None),
        ("content", 42),
        ("content", ""),
        ("content", "  "),
    ],
)
def test_invalid_atoms_are_unknown_instead_of_unlabeled(prompt_fixture, field, value):
    run = prompt_fixture()
    root = Path(run["root"])
    labels = [atom("valid"), {**atom("invalid", 0.3), field: value}]
    write_json(root / "meta/lerobot_annotations.json", {"episodes": {"0": {"atoms": labels}}})
    reviews = json.loads((root / "meta/annotation_reviews.json").read_text())
    if not (field == "timestamp" and value == "bad"):
        reviews["0"]["annotation_sha256"] = annotation_hash(labels)
        write_json(root / "meta/annotation_reviews.json", reviews)
    result = monitor.prompt_distribution(run)
    assert result["unknown_episode_ids"] == ["0"]
    assert result["evaluated_episodes"] == result["retained_frames"] == 0
    assert result["complete"] is False
    assert result["rows"] == []


@pytest.mark.parametrize("damage", ["parquet", "sidecar", "review", "state", "exclusions", "fps"])
def test_unreadable_evidence_does_not_report_complete_zero(prompt_fixture, damage):
    run = prompt_fixture()
    root = Path(run["root"])
    if damage == "parquet":
        data_path(run).write_bytes(b"broken parquet")
    elif damage == "sidecar":
        (root / "meta/lerobot_annotations.json").write_text("{")
    elif damage == "review":
        (root / "meta/annotation_reviews.json").write_text("{")
    elif damage == "state":
        run["episodes"]["0"]["decision"] = "unknown"
    elif damage == "exclusions":
        run["episodes"]["0"]["excluded_intervals"] = [{"start_frame": 0, "end_frame": 99}]
    else:
        info = json.loads((root / "meta/info.json").read_text())
        info["fps"] = 0
        write_json(root / "meta/info.json", info)
    result = monitor.prompt_distribution(run)
    assert result["unknown_episode_ids"] == ["0"]
    assert result["complete"] is False
    assert result["diagnostics"]


def test_real_parquet_annotation_fallback(prompt_fixture):
    run = prompt_fixture()
    root = Path(run["root"])
    annotations = json.loads((root / "meta/lerobot_annotations.json").read_text())
    labels = annotations["episodes"]["0"]["atoms"]
    replace_column(run, "language_persistent", [labels] * 10)
    write_json(root / "meta/lerobot_annotations.json", {"episodes": {}})
    result = monitor.prompt_distribution(run)
    assert result["eligible_episodes"] == result["evaluated_episodes"] == 1
    assert counts(result) == {"A": 2, "B": 5}


def test_shared_v3_shards_respect_episode_identity_and_project_columns(tmp_path, monkeypatch):
    root = write_v3(tmp_path / "shared", [10, 10, 10])
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pq.read_table(metadata_path).to_pylist()
    for row in metadata:
        row["data/file_index"] = 0
        row["dataset_from_index"] += 1000
        row["dataset_to_index"] += 1000
    pq.write_table(pa.Table.from_pylist(metadata), metadata_path)
    rows = []
    for ep in (2, 0, 1):
        rows.extend(dict(episode_index=ep, timestamp=i / 10, pixels=b"never decode") for i in range(10))
    path = root / "data/chunk-000/file-000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    run = json.loads(write_run(tmp_path / "workspace", root, root, [0, 1, 2]).read_text())
    run["episodes"]["2"]["decision"] = "delete"
    annotations = {str(ep): {"atoms": [atom(str(ep))]} for ep in (0, 1, 2)}
    write_json(root / "meta/lerobot_annotations.json", {"episodes": annotations})
    write_json(
        root / "meta/annotation_reviews.json",
        {
            str(ep): dict(
                annotation_sha256=annotation_hash(annotations[str(ep)]["atoms"]),
                exclusions_sha256=exclusions_hash(),
                reviewed_at="now",
            )
            for ep in (0, 1, 2)
        },
    )
    read_table = pq.read_table
    reads = []

    def projected_read(file, *args, **kwargs):
        if Path(file) == path:
            reads.append(kwargs.get("columns"))
        return read_table(file, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", projected_read)
    result = monitor.prompt_distribution(run)
    assert counts(result) == {"0": 10, "1": 10}
    assert result["eligible_episodes"] == result["evaluated_episodes"] == 2
    assert reads == [["episode_index", "timestamp"]]


def test_active_prompt_parity_with_exporter_and_no_writes(prompt_fixture):
    from groot_export import _instruction, _prepare_atoms

    run = prompt_fixture()
    root = Path(run["root"])
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    labels = json.loads((root / "meta/lerobot_annotations.json").read_text())["episodes"]["0"]["atoms"]
    timestamps = pq.read_table(data_path(run), columns=["timestamp"])["timestamp"].to_pylist()
    expected = Counter(
        _instruction("task fallback", timestamps[i], _prepare_atoms(labels), "subtask", None)
        for i in (0, 1, 5, 6, 7, 8, 9)
    )
    result = monitor.prompt_distribution(run)
    assert counts(result) == expected == {"A": 2, "B": 5}
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "field,value", [("content", 42), ("content", None), ("timestamp", "0.3"), ("timestamp", True)]
)
def test_malformed_parquet_atoms_not_repaired_into_known_coverage(prompt_fixture, field, value):
    run = prompt_fixture(labels=[[atom("A", 0.3)]], exclusions=[[]])
    root = Path(run["root"])
    raw = {**atom("A", 0.3), field: value}
    replace_column(run, "language_persistent", [[raw]] * 10)
    write_json(root / "meta/lerobot_annotations.json", {"episodes": {}})
    # The editor's normalization can stringify content/coerce numeric timestamps.
    # Even a matching old review must not turn malformed evidence into known data.
    normalized = {
        **raw,
        "timestamp": float(raw["timestamp"]),
        "content": None if raw["content"] is None else str(raw["content"]),
    }
    write_json(
        root / "meta/annotation_reviews.json",
        {
            "0": dict(
                annotation_sha256=annotation_hash([normalized]),
                exclusions_sha256=exclusions_hash(),
                reviewed_at="now",
            )
        },
    )
    result = monitor.prompt_distribution(run)
    assert result["unknown_episode_ids"] == ["0"]
    assert result["complete"] is False
    assert result["retained_frames"] == 0


def test_sidecar_precedence_over_stale_parquet_annotations(prompt_fixture):
    run = prompt_fixture()
    replace_column(run, "language_persistent", [[atom(42)]] * 10)
    result = monitor.prompt_distribution(run)
    assert counts(result) == {"A": 2, "B": 5}
    assert result["complete"] is True


def test_parquet_event_timestamp_fallback_and_review_hash(prompt_fixture):
    run = prompt_fixture(labels=[[atom("A"), atom("B", 0.3)]], exclusions=[[]])
    root = Path(run["root"])
    replace_column(run, "language_persistent", [[atom("A")]] * 10)
    events = [[] for _ in range(10)]
    events[3] = [{key: value for key, value in atom("B").items() if key != "timestamp"}]
    replace_column(run, "language_events", events)
    write_json(root / "meta/lerobot_annotations.json", {"episodes": {}})
    result = monitor.prompt_distribution(run)
    assert counts(result) == {"A": 3, "B": 7}
    assert result["complete"] is True


def test_between_sample_label_starts_at_next_actual_frame(prompt_fixture):
    result = monitor.prompt_distribution(prompt_fixture(labels=[[atom("A", 0.15)]], exclusions=[[]]))
    assert counts(result) == {"A": 8}
    assert result["unlabeled_frames"] == 2
    assert_partition(result)


def test_normalized_overlapping_exclusions_keep_source_indices(prompt_fixture):
    result = monitor.prompt_distribution(
        prompt_fixture(exclusions=[[{"start_frame": 3, "end_frame": 5}, {"start_frame": 2, "end_frame": 4}]])
    )
    assert counts(result) == {"A": 2, "B": 5}
    assert result["complete"] is True


def test_only_task_aug_leaves_all_frames_unlabeled(prompt_fixture):
    run = prompt_fixture(
        labels=[[dict(role="user", style="task_aug", timestamp=0.0, content="task")]], exclusions=[[]]
    )
    result = monitor.prompt_distribution(run)
    assert result["retained_frames"] == result["unlabeled_frames"] == 10
    assert result["rows"] == []
    assert result["complete"] is True
    assert_partition(result)


def test_excluded_only_label_does_not_get_episode_or_zero_row(prompt_fixture):
    result = monitor.prompt_distribution(
        prompt_fixture(labels=[[atom("A"), atom("excluded", 0.2), atom("B", 0.5)]])
    )
    assert counts(result) == {"A": 2, "B": 5}
    assert all(row["episodes"] == 1 for row in result["rows"])


def test_missing_episode_rows_are_unknown(prompt_fixture):
    run = prompt_fixture()
    path = data_path(run)
    pq.write_table(pq.read_table(path).slice(0, 9), path)
    result = monitor.prompt_distribution(run)
    assert result["unknown_episode_ids"] == ["0"]
    assert result["retained_frames"] == 0
    assert result["complete"] is False


def test_no_video_or_application_imports_and_workspace_unchanged(prompt_fixture, monkeypatch):
    import builtins

    run = prompt_fixture()
    parent = Path(run["root"]).parent
    before = {p.relative_to(parent): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    actual_import = builtins.__import__

    def forbid_video(name, *args, **kwargs):
        if name.split(".")[0] in {"app", "cv2", "av", "decord", "official_annotations"}:
            raise AssertionError(f"Prompt monitor imported video/application module {name}")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_video)
    result = monitor.prompt_distribution(run)
    assert counts(result) == {"A": 2, "B": 5}
    assert before == {p.relative_to(parent): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
