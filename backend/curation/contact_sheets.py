"""Immutable six-cell visual evidence for proposal and final boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from uuid import UUID

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pyarrow as pa
import pyarrow.parquet as pq

from .cosmos_transport import ArtifactConflict, ArtifactSecurityError, AtomicArtifactStore
from .db import CurationDatabase
from .source import SourceRecord

FINAL_TRANSITION_NAMES = (
    "step 2: pick up object",
    "step 3: turn to find black trash bin",
    "step 4: approach black trash bin",
    "step 5: lean down to black trash bin",
    "step 6: drop object into black trash bin",
    "step 7: stand straight",
)

_CELL_WIDTH = 420
_CELL_HEIGHT = 270
_GRID_COLUMNS = 3
_GRID_ROWS = 2
_IMAGE_HEIGHT = 195
MAX_SOURCE_FRAME_WIDTH = 4096
MAX_SOURCE_FRAME_HEIGHT = 4096
MAX_SOURCE_FRAME_PIXELS = 9_000_000
MAX_SEQUENTIAL_DECODED_FRAMES = 10_000


class ContactSheetUnavailable(RuntimeError):
    """Required source evidence could not be rendered without fabrication."""


@dataclass(frozen=True)
class ContactSheetDatasetIdentity:
    dataset_id: int
    dataset_alias: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.dataset_id) is not int or self.dataset_id < 1:
            raise ValueError("dataset_id must be a positive integer")
        if not isinstance(self.dataset_alias, str) or not self.dataset_alias:
            raise ValueError("dataset_alias must be a nonempty string")
        if (
            not isinstance(self.source_manifest_sha256, str)
            or len(self.source_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_manifest_sha256)
        ):
            raise ValueError("source_manifest_sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class ContactSheetReconcileStatus:
    kind: str
    status: str
    source_episode_index: int
    relative_path: str
    proposal_id: str | None = None
    approval_revision: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ContactSheetCell:
    transition_name: str
    status: str
    frame: int | None
    timestamp_s: float | None
    proposal_delta_s: float | None
    image_decoded: bool
    placeholder: str | None
    display_lines: tuple[str, ...]


@dataclass(frozen=True)
class ContactSheetResult:
    relative_path: str
    sha256: str
    byte_size: int
    cells: tuple[ContactSheetCell, ...]


DecodeFrames = Callable[[tuple[int, ...]], Mapping[int, Image.Image]]


class ContactSheetService:
    """Render and durably install workspace-only contact sheets without DB rows."""

    def __init__(self, workspace: Path, *, identity: ContactSheetDatasetIdentity) -> None:
        self.store = AtomicArtifactStore(Path(workspace))
        self.identity = identity

    def write_proposal(
        self,
        *,
        proposal_id: str,
        transition_frames: Sequence[int | None],
        timestamps: Sequence[float],
        model_statuses: Sequence[str],
        decode_frames: DecodeFrames,
    ) -> ContactSheetResult:
        identifier = _canonical_uuid(proposal_id)
        frames = _nullable_frames(transition_frames, timestamps=timestamps)
        statuses = _statuses(model_statuses)
        requested = tuple(frame for frame in frames if frame is not None)
        decoded = _decode_exact(decode_frames, requested)
        cells: list[ContactSheetCell] = []
        images: list[Image.Image | None] = []
        timeline = tuple(float(value) for value in timestamps)
        for name, frame, status in zip(FINAL_TRANSITION_NAMES, frames, statuses, strict=True):
            if frame is None:
                lines = (name, f"status: {status}", "NOT OBSERVED")
                cells.append(
                    ContactSheetCell(
                        transition_name=name,
                        status=status,
                        frame=None,
                        timestamp_s=None,
                        proposal_delta_s=None,
                        image_decoded=False,
                        placeholder="NOT OBSERVED",
                        display_lines=lines,
                    )
                )
                images.append(None)
                continue
            timestamp = timeline[frame]
            lines = (name, f"status: {status}", f"frame: {frame}  time: {timestamp:.3f} s")
            cells.append(
                ContactSheetCell(
                    transition_name=name,
                    status=status,
                    frame=frame,
                    timestamp_s=timestamp,
                    proposal_delta_s=None,
                    image_decoded=True,
                    placeholder=None,
                    display_lines=lines,
                )
            )
            images.append(decoded[frame])
        return self._write(
            proposal_contact_sheet_path(self.identity, identifier),
            tuple(cells),
            tuple(images),
        )

    def write_final(
        self,
        *,
        review_state: str,
        source_episode_index: int,
        approval_revision: int,
        final_transition_frames: Sequence[int | None],
        proposal_transition_frames: Sequence[int | None],
        timestamps: Sequence[float],
        decode_frames: DecodeFrames,
    ) -> ContactSheetResult | None:
        if review_state == "approved_reject":
            return None
        if review_state != "approved_keep":
            raise ValueError("final contact sheets require an approved review state")
        if type(source_episode_index) is not int or source_episode_index < 0:
            raise ValueError("source_episode_index must be a nonnegative integer")
        if type(approval_revision) is not int or approval_revision < 1:
            raise ValueError("approval_revision must be a positive integer")
        finals = _nullable_frames(final_transition_frames, timestamps=timestamps)
        if any(frame is None for frame in finals):
            raise ContactSheetUnavailable("approved_keep final sheet requires six non-null human transitions")
        proposals = _nullable_frames(proposal_transition_frames, timestamps=timestamps)
        final_frames = tuple(int(frame) for frame in finals if frame is not None)
        decoded = _decode_exact(decode_frames, final_frames)
        timeline = tuple(float(value) for value in timestamps)
        cells: list[ContactSheetCell] = []
        images: list[Image.Image] = []
        for name, frame, proposal in zip(FINAL_TRANSITION_NAMES, final_frames, proposals, strict=True):
            timestamp = timeline[frame]
            if proposal is None:
                delta = None
                comparison = "proposal: NOT OBSERVED"
            else:
                delta = timestamp - timeline[proposal]
                comparison = f"proposal delta: {delta:+.3f} s"
            lines = (name, f"final frame: {frame}  time: {timestamp:.3f} s", comparison)
            cells.append(
                ContactSheetCell(
                    transition_name=name,
                    status="approved_keep",
                    frame=frame,
                    timestamp_s=timestamp,
                    proposal_delta_s=delta,
                    image_decoded=True,
                    placeholder=None,
                    display_lines=lines,
                )
            )
            images.append(decoded[frame])
        return self._write(
            final_contact_sheet_path(self.identity, source_episode_index, approval_revision),
            tuple(cells),
            tuple(images),
        )

    def _write(
        self,
        relative_path: str,
        cells: tuple[ContactSheetCell, ...],
        images: tuple[Image.Image | None, ...],
    ) -> ContactSheetResult:
        try:
            contents = _render(cells, images)
        finally:
            _close_images(image for image in images if image is not None)
        installed = self.store.write_bytes(relative_path, contents, media_type="image/png")
        return ContactSheetResult(
            relative_path=installed.record.relative_path,
            sha256=installed.record.sha256,
            byte_size=installed.record.byte_size,
            cells=cells,
        )


class ContactSheetCoordinator:
    """Reconcile DB lifecycle records to immutable workspace-only visual evidence."""

    def __init__(
        self,
        *,
        database: CurationDatabase,
        dataset_id: int,
        source_record: SourceRecord,
        workspace: Path,
        timestamp_loader: Callable[[int], Sequence[float]] | None = None,
        decoder_factory: Callable[[SourceRecord, str], DecodeFrames] | None = None,
    ) -> None:
        self.database = database
        self.dataset_id = dataset_id
        self.source_record = source_record
        self.workspace = Path(workspace)
        self.timestamp_loader = timestamp_loader or self._load_timestamps
        self.decoder_factory = decoder_factory or registered_frame_decoder
        dataset = database.get_dataset(alias=source_record.alias)
        if (
            dataset is None
            or dataset["id"] != dataset_id
            or dataset["source_path"] != str(source_record.root)
            or dataset["source_manifest_sha256"] != source_record.fingerprint
        ):
            raise ValueError("contact-sheet coordinator source authority does not match the workspace")
        self.identity = ContactSheetDatasetIdentity(
            dataset_id=dataset_id,
            dataset_alias=source_record.alias,
            source_manifest_sha256=source_record.fingerprint,
        )
        self.service = ContactSheetService(self.workspace, identity=self.identity)

    def ensure_proposal(self, proposal_id: str) -> ContactSheetResult:
        relative_path = proposal_contact_sheet_path(self.identity, proposal_id)
        with self.database._read() as connection:
            row = connection.execute(
                """
                SELECT proposal.*, attempt.source_episode_index
                FROM cosmos_proposals AS proposal
                JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE proposal.id=? AND job.dataset_id=?
                """,
                (proposal_id, self.dataset_id),
            ).fetchone()
        if row is None:
            raise ContactSheetUnavailable("proposal lifecycle record is unavailable")
        document = _proposal_document(row["model_response_json"])
        statuses = _proposal_statuses(document)
        transitions = tuple(row[f"step_{step}_start_frame"] for step in range(2, 8))
        episode_index = int(row["source_episode_index"])
        evidence = self._evidence_binding(
            kind="proposal",
            source_episode_index=episode_index,
            proposal_id=proposal_id,
            approval_revision=None,
            final_transition_frames=None,
            proposal_transition_frames=transitions,
        )
        existing = self._validated_existing(relative_path, evidence=evidence)
        if existing is not None:
            return existing
        timestamps = self.timestamp_loader(episode_index)
        video_path = self._video_path(episode_index)
        result = self.service.write_proposal(
            proposal_id=proposal_id,
            transition_frames=transitions,
            timestamps=timestamps,
            model_statuses=statuses,
            decode_frames=self.decoder_factory(self.source_record, video_path),
        )
        self._write_receipt(result, evidence=evidence)
        return result

    def ensure_final(self, source_episode_index: int, approval_revision: int) -> ContactSheetResult | None:
        if type(source_episode_index) is not int or source_episode_index < 0:
            raise ValueError("source_episode_index must be a nonnegative integer")
        if type(approval_revision) is not int or approval_revision < 1:
            raise ValueError("approval_revision must be a positive integer")
        relative_path = final_contact_sheet_path(self.identity, source_episode_index, approval_revision)
        with self.database._read() as connection:
            episode = connection.execute(
                """
                SELECT * FROM episodes
                WHERE dataset_id=? AND source_episode_index=?
                """,
                (self.dataset_id, source_episode_index),
            ).fetchone()
            if episode is None:
                raise ContactSheetUnavailable("approved episode is unavailable")
            events = connection.execute(
                """
                SELECT details_json FROM audit_events
                WHERE dataset_id=? AND episode_id=? AND operation='keep_approved' AND new_revision=?
                ORDER BY created_at, id
                """,
                (self.dataset_id, episode["id"], approval_revision),
            ).fetchall()
            if not events and episode["review_state"] == "approved_reject":
                return None
            if len(events) != 1:
                raise ContactSheetUnavailable("approval proposal snapshot is unavailable")
            try:
                details = json.loads(events[0]["details_json"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ContactSheetUnavailable("approval proposal snapshot is invalid") from error
            snapshot = approval_evidence_snapshot(
                details,
                identity=self.identity,
                source_episode_index=source_episode_index,
                approval_revision=approval_revision,
            )
            proposal_id = snapshot["proposal_id"]
            proposal = None
            if proposal_id is not None:
                proposal = connection.execute(
                    """
                    SELECT proposal.* FROM cosmos_proposals AS proposal
                    JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                    JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                    WHERE proposal.id=? AND job.dataset_id=?
                    """,
                    (proposal_id, self.dataset_id),
                ).fetchone()
                if proposal is None:
                    raise ContactSheetUnavailable("snapshotted proposal is unavailable")
            final_transitions = tuple(snapshot["final_transition_frames"])
            proposal_transitions = (
                (None,) * 6
                if proposal is None
                else tuple(proposal[f"step_{step}_start_frame"] for step in range(2, 8))
            )
            if list(proposal_transitions) != snapshot["proposal_transition_frames"]:
                raise ArtifactConflict("immutable proposal no longer matches approval evidence")
        evidence = self._evidence_binding(
            kind="final",
            source_episode_index=source_episode_index,
            proposal_id=proposal_id,
            approval_revision=approval_revision,
            final_transition_frames=final_transitions,
            proposal_transition_frames=proposal_transitions,
        )
        existing = self._validated_existing(relative_path, evidence=evidence)
        if existing is not None:
            return existing
        timestamps = self.timestamp_loader(source_episode_index)
        video_path = self._video_path(source_episode_index)
        result = self.service.write_final(
            review_state="approved_keep",
            source_episode_index=source_episode_index,
            approval_revision=approval_revision,
            final_transition_frames=final_transitions,
            proposal_transition_frames=proposal_transitions,
            timestamps=timestamps,
            decode_frames=self.decoder_factory(self.source_record, video_path),
        )
        if result is None:  # Defensive: the persisted row above was approved_keep.
            raise ContactSheetUnavailable("approved final sheet was not produced")
        self._write_receipt(result, evidence=evidence)
        return result

    def ensure_episode(self, source_episode_index: int) -> list[ContactSheetResult]:
        statuses = self.reconcile_episode(source_episode_index)
        if any(status.status == "conflict" for status in statuses):
            raise ArtifactConflict("one or more contact-sheet evidence records conflict")
        if any(status.status == "pending" for status in statuses):
            raise ContactSheetUnavailable("one or more contact-sheet evidence records are pending")
        return []

    def reconcile_episode(self, source_episode_index: int) -> list[ContactSheetReconcileStatus]:
        with self.database._read() as connection:
            proposals = connection.execute(
                """
                SELECT proposal.id FROM cosmos_proposals AS proposal
                JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE job.dataset_id=? AND attempt.source_episode_index=?
                ORDER BY proposal.created_at, proposal.id
                """,
                (self.dataset_id, source_episode_index),
            ).fetchall()
            finals = connection.execute(
                """
                SELECT event.new_revision AS approval_revision
                FROM audit_events AS event
                JOIN episodes AS episode ON episode.id=event.episode_id
                WHERE event.dataset_id=? AND event.operation='keep_approved'
                    AND episode.source_episode_index=?
                ORDER BY event.created_at, event.id
                """,
                (self.dataset_id, source_episode_index),
            ).fetchall()
        statuses = [
            self._reconcile_proposal(proposal["id"], source_episode_index=source_episode_index)
            for proposal in proposals
        ]
        statuses.extend(self._reconcile_final(source_episode_index, row["approval_revision"]) for row in finals)
        return statuses

    def reconcile_all(self) -> list[ContactSheetReconcileStatus]:
        with self.database._read() as connection:
            proposals = [
                row["id"]
                for row in connection.execute(
                    """
                    SELECT proposal.id FROM cosmos_proposals AS proposal
                    JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                    JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                    WHERE job.dataset_id=? ORDER BY proposal.created_at, proposal.id
                    """,
                    (self.dataset_id,),
                )
            ]
            finals = [
                (row["source_episode_index"], row["approval_revision"])
                for row in connection.execute(
                    """
                    SELECT episode.source_episode_index, event.new_revision AS approval_revision
                    FROM audit_events AS event
                    JOIN episodes AS episode ON episode.id=event.episode_id
                    WHERE event.dataset_id=? AND event.operation='keep_approved'
                    ORDER BY episode.source_episode_index, event.new_revision, event.created_at, event.id
                    """,
                    (self.dataset_id,),
                )
            ]
        proposal_episodes = self._proposal_episode_indices(proposals)
        statuses = [
            self._reconcile_proposal(proposal_id, source_episode_index=proposal_episodes[proposal_id])
            for proposal_id in proposals
        ]
        statuses.extend(self._reconcile_final(index, revision) for index, revision in finals)
        return statuses

    def _proposal_episode_indices(self, proposal_ids: Sequence[str]) -> dict[str, int]:
        if not proposal_ids:
            return {}
        with self.database._read() as connection:
            rows = connection.execute(
                """
                SELECT proposal.id, attempt.source_episode_index
                FROM cosmos_proposals AS proposal
                JOIN cosmos_attempts AS attempt ON attempt.id=proposal.attempt_id
                JOIN cosmos_jobs AS job ON job.id=attempt.job_id
                WHERE job.dataset_id=?
                """,
                (self.dataset_id,),
            ).fetchall()
        return {row["id"]: row["source_episode_index"] for row in rows}

    def _reconcile_proposal(self, proposal_id: str, *, source_episode_index: int) -> ContactSheetReconcileStatus:
        try:
            relative_path = proposal_contact_sheet_path(self.identity, proposal_id)
            result = self.ensure_proposal(proposal_id)
        except (ArtifactConflict, ArtifactSecurityError, TypeError, ValueError):
            return ContactSheetReconcileStatus(
                kind="proposal",
                status="conflict",
                source_episode_index=source_episode_index,
                proposal_id=proposal_id,
                relative_path=locals().get("relative_path", ""),
                reason="immutable_evidence_conflict",
            )
        except (ContactSheetUnavailable, OSError):
            return ContactSheetReconcileStatus(
                kind="proposal",
                status="pending",
                source_episode_index=source_episode_index,
                proposal_id=proposal_id,
                relative_path=relative_path,
                reason="evidence_unavailable",
            )
        return ContactSheetReconcileStatus(
            kind="proposal",
            status="available" if not result.cells else "created",
            source_episode_index=source_episode_index,
            proposal_id=proposal_id,
            relative_path=result.relative_path,
        )

    def _reconcile_final(self, source_episode_index: int, approval_revision: int) -> ContactSheetReconcileStatus:
        relative_path = final_contact_sheet_path(self.identity, source_episode_index, approval_revision)
        try:
            result = self.ensure_final(source_episode_index, approval_revision)
        except (ArtifactConflict, ArtifactSecurityError, TypeError, ValueError):
            return ContactSheetReconcileStatus(
                kind="final",
                status="conflict",
                source_episode_index=source_episode_index,
                approval_revision=approval_revision,
                relative_path=relative_path,
                reason="immutable_evidence_conflict",
            )
        except (ContactSheetUnavailable, OSError):
            return ContactSheetReconcileStatus(
                kind="final",
                status="pending",
                source_episode_index=source_episode_index,
                approval_revision=approval_revision,
                relative_path=relative_path,
                reason="evidence_unavailable",
            )
        if result is None:
            raise RuntimeError("historical keep approval unexpectedly produced no final sheet")
        return ContactSheetReconcileStatus(
            kind="final",
            status="available" if not result.cells else "created",
            source_episode_index=source_episode_index,
            approval_revision=approval_revision,
            relative_path=result.relative_path,
        )

    def _validated_existing(
        self, relative_path: str, *, evidence: Mapping[str, object]
    ) -> ContactSheetResult | None:
        receipt_relative_path = receipt_path(relative_path)
        try:
            receipt = self.service.store.read_existing(
                receipt_relative_path,
                media_type="application/json",
            )
        except ArtifactConflict as error:
            if str(error) in {"artifact does not exist", "artifact parent directory does not exist"}:
                return None
            raise
        receipt_document = _parse_receipt(receipt.contents)
        expected_binding = dict(evidence)
        for key, value in expected_binding.items():
            if receipt_document.get(key) != value:
                raise ArtifactConflict("contact-sheet receipt evidence binding does not match")
        expected_sha256 = receipt_document["sha256"]
        expected_size = receipt_document["byte_size"]
        if receipt_document["relative_path"] != relative_path:
            raise ArtifactConflict("contact-sheet receipt identifies a different artifact")
        record = self.service.store.inspect_existing(
            relative_path,
            media_type="image/png",
            expected_sha256=expected_sha256,
            expected_byte_size=expected_size,
        )
        return ContactSheetResult(
            relative_path=record.relative_path,
            sha256=record.sha256,
            byte_size=record.byte_size,
            cells=(),
        )

    def _write_receipt(self, result: ContactSheetResult, *, evidence: Mapping[str, object]) -> None:
        document = {
            **evidence,
            "relative_path": result.relative_path,
            "sha256": result.sha256,
            "byte_size": result.byte_size,
        }
        self.service.store.write_json(
            receipt_path(result.relative_path),
            document,
        )

    def _evidence_binding(
        self,
        *,
        kind: str,
        source_episode_index: int,
        proposal_id: str | None,
        approval_revision: int | None,
        final_transition_frames: Sequence[int] | None,
        proposal_transition_frames: Sequence[int | None],
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": kind,
            "dataset_id": self.identity.dataset_id,
            "dataset_alias": self.identity.dataset_alias,
            "source_manifest_sha256": self.identity.source_manifest_sha256,
            "source_episode_index": source_episode_index,
            "proposal_id": proposal_id,
            "approval_revision": approval_revision,
            "final_transition_frames": (
                None if final_transition_frames is None else list(final_transition_frames)
            ),
            "proposal_transition_frames": list(proposal_transition_frames),
        }

    def _video_path(self, source_episode_index: int) -> str:
        name = f"episode_{source_episode_index:06d}.mp4"
        paths = sorted(
            path
            for path in self.source_record.file_hashes
            if path.startswith("videos/") and "/observation.images.ego_view/" in path and path.endswith(f"/{name}")
        )
        if len(paths) != 1:
            raise ContactSheetUnavailable("registered episode video is unavailable")
        return paths[0]

    def _load_timestamps(self, source_episode_index: int) -> Sequence[float]:
        with self.database._read() as connection:
            episode = connection.execute(
                "SELECT source_length FROM episodes WHERE dataset_id=? AND source_episode_index=?",
                (self.dataset_id, source_episode_index),
            ).fetchone()
        if episode is None:
            raise ContactSheetUnavailable("episode timeline record is unavailable")
        name = f"episode_{source_episode_index:06d}.parquet"
        paths = sorted(
            path
            for path in self.source_record.file_hashes
            if path.startswith("data/") and path.endswith(f"/{name}")
        )
        if len(paths) != 1:
            raise ContactSheetUnavailable("registered episode parquet is unavailable")
        asset = self.source_record.open_asset(paths[0])
        if asset is None:
            raise ContactSheetUnavailable("registered episode parquet is unavailable")
        try:
            with os.fdopen(os.dup(asset.fd), "rb") as handle:
                table = pq.read_table(handle, columns=["episode_index", "frame_index", "timestamp"])
            digest = _sha256_fd(asset.fd)
            if not self.source_record.verify_pinned_asset(asset, sha256=digest):
                raise ContactSheetUnavailable("registered episode parquet changed during decoding")
        except ContactSheetUnavailable:
            raise
        except (OSError, pa.ArrowException, TypeError, ValueError) as error:
            raise ContactSheetUnavailable("registered episode parquet could not be read") from error
        finally:
            asset.close()
        rows = sorted(
            (
                (int(frame), float(timestamp))
                for source, frame, timestamp in zip(
                    table.column("episode_index").to_pylist(),
                    table.column("frame_index").to_pylist(),
                    table.column("timestamp").to_pylist(),
                    strict=True,
                )
                if int(source) == source_episode_index
            ),
            key=lambda item: item[0],
        )
        if len(rows) != episode["source_length"] or [frame for frame, _ in rows] != list(range(len(rows))):
            raise ContactSheetUnavailable("registered episode timeline does not match its source length")
        return tuple(timestamp for _, timestamp in rows)


def registered_frame_decoder(
    record: SourceRecord,
    video_asset_path: str,
) -> DecodeFrames:
    """Build a decoder that keeps one immutable registered video descriptor pinned."""

    def decode(indices: tuple[int, ...]) -> Mapping[int, Image.Image]:
        return _decode_registered_frames(record, video_asset_path, indices)

    return decode


def _dataset_evidence_namespace(identity: ContactSheetDatasetIdentity) -> str:
    return f"contact_sheets/datasets/dataset_{identity.dataset_id}_{identity.source_manifest_sha256}"


def proposal_contact_sheet_path(identity: ContactSheetDatasetIdentity, proposal_id: str) -> str:
    namespace = _dataset_evidence_namespace(identity)
    return f"{namespace}/proposals/proposal_{_canonical_uuid(proposal_id)}.png"


def final_contact_sheet_path(
    identity: ContactSheetDatasetIdentity,
    source_episode_index: int,
    approval_revision: int,
) -> str:
    if type(source_episode_index) is not int or source_episode_index < 0:
        raise ValueError("source_episode_index must be a nonnegative integer")
    if type(approval_revision) is not int or approval_revision < 1:
        raise ValueError("approval_revision must be a positive integer")
    namespace = _dataset_evidence_namespace(identity)
    return f"{namespace}/finals/episode_{source_episode_index}_revision_{approval_revision}.png"


def receipt_path(relative_path: str) -> str:
    path = Path(relative_path)
    parts = path.parts
    if (
        len(parts) != 5
        or parts[0:2] != ("contact_sheets", "datasets")
        or parts[3] not in {"proposals", "finals"}
        or path.suffix != ".png"
    ):
        raise ValueError("contact-sheet path is outside its dataset evidence namespace")
    return str(Path(*parts[:3]) / "receipts" / parts[3] / f"{parts[4]}.receipt.json")


class ContactSheetEvidenceInspector:
    """Descriptor-safe evidence checks with no cleanup or filesystem writes."""

    def __init__(self, workspace: Path) -> None:
        self.store = AtomicArtifactStore(Path(workspace), cleanup_on_start=False)

    def status(
        self,
        relative_path: str,
        *,
        expected_binding: Mapping[str, object] | None = None,
    ) -> str:
        return _contact_sheet_evidence_status(
            self.store,
            relative_path,
            expected_binding=expected_binding,
        )


def contact_sheet_evidence_status(
    workspace: Path,
    relative_path: str,
    *,
    expected_binding: Mapping[str, object] | None = None,
) -> str:
    """Return available, missing, incomplete, or conflict without decoding source video."""

    return ContactSheetEvidenceInspector(Path(workspace)).status(
        relative_path,
        expected_binding=expected_binding,
    )


def _contact_sheet_evidence_status(
    store: AtomicArtifactStore,
    relative_path: str,
    *,
    expected_binding: Mapping[str, object] | None,
) -> str:
    receipt_relative_path = receipt_path(relative_path)
    try:
        receipt = store.read_existing(
            receipt_relative_path,
            media_type="application/json",
        )
    except ArtifactConflict as error:
        if str(error) not in {"artifact does not exist", "artifact parent directory does not exist"}:
            return "conflict"
        try:
            store.inspect_existing(relative_path, media_type="image/png")
        except ArtifactConflict as image_error:
            if str(image_error) in {
                "artifact does not exist",
                "artifact parent directory does not exist",
            }:
                return "missing"
            return "conflict"
        except ArtifactSecurityError:
            return "conflict"
        return "incomplete"
    except ArtifactSecurityError:
        return "conflict"
    try:
        document = _parse_receipt(receipt.contents)
        if expected_binding is not None and any(
            document.get(key) != value for key, value in expected_binding.items()
        ):
            return "conflict"
        expected_sha256 = document["sha256"]
        expected_size = document["byte_size"]
        if document["relative_path"] != relative_path:
            return "conflict"
        store.inspect_existing(
            relative_path,
            media_type="image/png",
            expected_sha256=expected_sha256,
            expected_byte_size=expected_size,
        )
    except (ArtifactConflict, ArtifactSecurityError):
        return "conflict"
    return "available"


def _parse_receipt(contents: bytes) -> dict[str, object]:
    try:
        document = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactConflict("contact-sheet receipt is not canonical JSON") from error
    required = {
        "schema_version",
        "kind",
        "dataset_id",
        "dataset_alias",
        "source_manifest_sha256",
        "source_episode_index",
        "proposal_id",
        "approval_revision",
        "final_transition_frames",
        "proposal_transition_frames",
        "relative_path",
        "sha256",
        "byte_size",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ArtifactConflict("contact-sheet receipt has an invalid schema")
    sha256 = document["sha256"]
    byte_size = document["byte_size"]
    relative_path = document["relative_path"]
    if (
        document["schema_version"] != 1
        or document["kind"] not in {"proposal", "final"}
        or type(document["dataset_id"]) is not int
        or document["dataset_id"] < 1
        or not isinstance(document["dataset_alias"], str)
        or not document["dataset_alias"]
        or not isinstance(document["source_manifest_sha256"], str)
        or len(document["source_manifest_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in document["source_manifest_sha256"])
        or type(document["source_episode_index"]) is not int
        or document["source_episode_index"] < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or type(byte_size) is not int
        or byte_size < 0
        or not isinstance(relative_path, str)
        or not relative_path
    ):
        raise ArtifactConflict("contact-sheet receipt has invalid evidence fields")
    return document


def approval_evidence_snapshot(
    details: object,
    *,
    identity: ContactSheetDatasetIdentity,
    source_episode_index: int,
    approval_revision: int,
) -> dict[str, object]:
    if not isinstance(details, Mapping):
        raise ContactSheetUnavailable("approval proposal snapshot is invalid")
    snapshot = details.get("contact_sheet_evidence")
    required = {
        "schema_version",
        "dataset_id",
        "dataset_alias",
        "source_manifest_sha256",
        "source_episode_index",
        "approval_revision",
        "final_transition_frames",
        "proposal_id",
        "proposal_transition_frames",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ContactSheetUnavailable("approval proposal snapshot is invalid")
    proposal_id = snapshot["proposal_id"]
    final_frames = snapshot["final_transition_frames"]
    proposal_frames = snapshot["proposal_transition_frames"]
    if (
        snapshot["schema_version"] != 1
        or snapshot["dataset_id"] != identity.dataset_id
        or snapshot["dataset_alias"] != identity.dataset_alias
        or snapshot["source_manifest_sha256"] != identity.source_manifest_sha256
        or snapshot["source_episode_index"] != source_episode_index
        or snapshot["approval_revision"] != approval_revision
        or not isinstance(final_frames, list)
        or len(final_frames) != 6
        or any(type(frame) is not int or frame < 0 for frame in final_frames)
        or not isinstance(proposal_frames, list)
        or len(proposal_frames) != 6
        or any(frame is not None and (type(frame) is not int or frame < 0) for frame in proposal_frames)
        or (proposal_id is not None and not isinstance(proposal_id, str))
    ):
        raise ContactSheetUnavailable("approval proposal snapshot is invalid")
    if proposal_id is not None:
        _canonical_uuid(proposal_id)
    return snapshot


def _proposal_document(raw: str) -> Mapping[str, object]:
    try:
        document = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ContactSheetUnavailable("proposal model response is invalid") from error
    if not isinstance(document, Mapping):
        raise ContactSheetUnavailable("proposal model response is invalid")
    return document


def _proposal_statuses(document: Mapping[str, object]) -> tuple[str, ...]:
    segments = document.get("segments")
    if not isinstance(segments, list) or len(segments) != 7:
        raise ContactSheetUnavailable("proposal segment statuses are unavailable")
    by_step: dict[int, str] = {}
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ContactSheetUnavailable("proposal segment statuses are unavailable")
        step = segment.get("step")
        status = segment.get("status")
        if (
            type(step) is not int
            or step in by_step
            or status
            not in {
                "completed",
                "partial",
                "not_observed",
            }
        ):
            raise ContactSheetUnavailable("proposal segment statuses are unavailable")
        by_step[step] = str(status)
    if set(by_step) != set(range(1, 8)):
        raise ContactSheetUnavailable("proposal segment statuses are unavailable")
    return tuple(by_step[step] for step in range(2, 8))


def _decode_registered_frames(
    record: SourceRecord,
    video_asset_path: str,
    indices: tuple[int, ...],
) -> dict[int, Image.Image]:
    if not indices:
        return {}
    if any(type(index) is not int or index < 0 for index in indices):
        raise ContactSheetUnavailable("source video frame indices are invalid")
    if max(indices) >= MAX_SEQUENTIAL_DECODED_FRAMES:
        raise ContactSheetUnavailable("source video frame exceeds the sequential decode work limit")
    asset = record.open_asset(video_asset_path)
    if asset is None:
        raise ContactSheetUnavailable("registered source video is unavailable")
    selected = set(indices)
    decoded: dict[int, Image.Image] = {}
    try:
        with os.fdopen(os.dup(asset.fd), "rb") as handle:
            with av.open(handle, mode="r") as container:
                streams = container.streams.video
                if len(streams) != 1:
                    raise ContactSheetUnavailable("source video must contain exactly one stream")
                _validate_image_dimensions(streams[0].width, streams[0].height)
                for index, frame in enumerate(container.decode(streams[0])):
                    if index >= MAX_SEQUENTIAL_DECODED_FRAMES:
                        raise ContactSheetUnavailable("source video exceeded the sequential decode work limit")
                    _validate_image_dimensions(frame.width, frame.height)
                    if index in selected:
                        decoded[index] = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
                    if len(decoded) == len(selected):
                        break
        digest = _sha256_fd(asset.fd)
        if not record.verify_pinned_asset(asset, sha256=digest):
            raise ContactSheetUnavailable("registered source video changed during decoding")
    except ContactSheetUnavailable:
        _close_images(decoded.values())
        raise
    except (OSError, av.error.FFmpegError, ValueError) as error:
        _close_images(decoded.values())
        raise ContactSheetUnavailable("registered source video could not be decoded") from error
    finally:
        asset.close()
    missing = selected - set(decoded)
    if missing:
        _close_images(decoded.values())
        raise ContactSheetUnavailable(f"source video frames were unavailable: {sorted(missing)}")
    return {index: decoded[index] for index in indices}


def _canonical_uuid(value: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("proposal_id must be a canonical UUID") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError("proposal_id must be a canonical UUID")
    return canonical


def _timeline(values: Sequence[float]) -> tuple[float, ...]:
    try:
        timeline = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("timestamps must be finite and strictly increasing") from error
    if (
        not timeline
        or any(not math.isfinite(value) for value in timeline)
        or any(left >= right for left, right in zip(timeline, timeline[1:]))
    ):
        raise ValueError("timestamps must be finite and strictly increasing")
    return timeline


def _nullable_frames(
    values: Sequence[int | None],
    *,
    timestamps: Sequence[float],
) -> tuple[int | None, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 6:
        raise ValueError("transition frames must contain exactly six entries")
    timeline = _timeline(timestamps)
    result: list[int | None] = []
    for value in values:
        if value is None:
            result.append(None)
        elif type(value) is int and 0 <= value < len(timeline):
            result.append(value)
        else:
            raise ValueError("transition frames must be in-range integers or null")
    return tuple(result)


def _statuses(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 6:
        raise ValueError("model statuses must contain exactly six entries")
    result = tuple(values)
    if any(value not in {"completed", "partial", "not_observed"} for value in result):
        raise ValueError("model statuses must use the Cosmos completion vocabulary")
    return result


def _decode_exact(decode_frames: DecodeFrames, frames: tuple[int, ...]) -> dict[int, Image.Image]:
    if not callable(decode_frames):
        raise TypeError("decode_frames must be callable")
    decoded = dict(decode_frames(frames)) if frames else {}
    result: dict[int, Image.Image] = {}
    try:
        for image in decoded.values():
            if not isinstance(image, Image.Image):
                raise ContactSheetUnavailable("decoder returned a non-image frame")
            _validate_image_dimensions(image.width, image.height)
        for frame in frames:
            image = decoded.get(frame)
            if not isinstance(image, Image.Image):
                raise ContactSheetUnavailable(f"decoded frame {frame} is unavailable")
            result[frame] = image.convert("RGB")
        return result
    except Exception:
        _close_images(result.values())
        raise
    finally:
        _close_images(decoded.values())


def _validate_image_dimensions(width: object, height: object) -> None:
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or width > MAX_SOURCE_FRAME_WIDTH
        or height > MAX_SOURCE_FRAME_HEIGHT
        or width * height > MAX_SOURCE_FRAME_PIXELS
    ):
        raise ContactSheetUnavailable("source frame dimensions exceed the contact-sheet limits")


def _close_images(images: Iterable[Image.Image]) -> None:
    seen: set[int] = set()
    for image in images:
        identifier = id(image)
        if identifier in seen:
            continue
        seen.add(identifier)
        image.close()


def _render(
    cells: tuple[ContactSheetCell, ...],
    images: tuple[Image.Image | None, ...],
) -> bytes:
    if len(cells) != 6 or len(images) != 6:
        raise ValueError("contact sheets require exactly six cells")
    canvas = Image.new("RGB", (_CELL_WIDTH * _GRID_COLUMNS, _CELL_HEIGHT * _GRID_ROWS), "white")
    try:
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for position, (cell, source_image) in enumerate(zip(cells, images, strict=True)):
            column = position % _GRID_COLUMNS
            row = position // _GRID_COLUMNS
            left = column * _CELL_WIDTH
            top = row * _CELL_HEIGHT
            draw.rectangle(
                (left, top, left + _CELL_WIDTH - 1, top + _CELL_HEIGHT - 1),
                outline=(64, 64, 64),
                width=1,
            )
            image_box = (left + 1, top + 1, left + _CELL_WIDTH - 1, top + _IMAGE_HEIGHT)
            if source_image is None:
                draw.rectangle(image_box, fill=(40, 40, 40))
                marker = cell.placeholder or "NOT OBSERVED"
                marker_box = draw.textbbox((0, 0), marker, font=font)
                marker_width = marker_box[2] - marker_box[0]
                draw.text(
                    (left + (_CELL_WIDTH - marker_width) // 2, top + _IMAGE_HEIGHT // 2),
                    marker,
                    fill="white",
                    font=font,
                )
            else:
                fitted = ImageOps.contain(
                    source_image,
                    (_CELL_WIDTH - 2, _IMAGE_HEIGHT - 2),
                    method=Image.Resampling.LANCZOS,
                )
                try:
                    image_left = left + (_CELL_WIDTH - fitted.width) // 2
                    image_top = top + 1 + (_IMAGE_HEIGHT - 2 - fitted.height) // 2
                    canvas.paste(fitted, (image_left, image_top))
                finally:
                    fitted.close()
            text_top = top + _IMAGE_HEIGHT + 4
            for line in cell.display_lines:
                draw.text((left + 6, text_top), line, fill="black", font=font)
                text_top += 18
        with BytesIO() as buffer:
            canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
            return buffer.getvalue()
    finally:
        canvas.close()


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()
