"""Stable curation-state vocabulary shared by persistence and API layers."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum


class ReviewState(str, Enum):
    PENDING = "pending"
    DRAFT = "draft"
    APPROVED_KEEP = "approved_keep"
    APPROVED_REJECT = "approved_reject"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AttemptState(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    REQUESTING = "requesting"
    SUCCEEDED = "succeeded"
    MANUAL_ONLY = "manual_only"
    RETRYABLE = "retryable"
    CANCELLED = "cancelled"


class ExportState(str, Enum):
    QUEUED = "queued"
    BUILDING = "building"
    CORE_STRUCTURAL_VALIDATED = "core_structural_validated"
    GROOT_STATS_VALIDATED = "gr00t_stats_validated"
    GROOT_LOADER_VALIDATED = "gr00t_loader_validated"
    PROVENANCE_WRITTEN = "provenance_written"
    FINAL_CONSISTENCY_VALIDATED = "final_consistency_validated"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


class ProposalState(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class CompletionState(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NOT_OBSERVED = "not_observed"


JOB_STATE_TRANSITIONS: Mapping[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELLED}),
    JobState.RUNNING: frozenset(
        {
            JobState.CANCEL_REQUESTED,
            JobState.COMPLETED,
            JobState.COMPLETED_WITH_FAILURES,
            JobState.FAILED,
        }
    ),
    JobState.CANCEL_REQUESTED: frozenset({JobState.CANCELLED}),
    JobState.COMPLETED: frozenset(),
    JobState.COMPLETED_WITH_FAILURES: frozenset(),
    JobState.CANCELLED: frozenset(),
    JobState.FAILED: frozenset(),
}


EXPORT_STATE_TRANSITIONS: Mapping[ExportState, frozenset[ExportState]] = {
    ExportState.QUEUED: frozenset({ExportState.BUILDING, ExportState.FAILED}),
    ExportState.BUILDING: frozenset({ExportState.CORE_STRUCTURAL_VALIDATED, ExportState.FAILED}),
    ExportState.CORE_STRUCTURAL_VALIDATED: frozenset({ExportState.GROOT_STATS_VALIDATED, ExportState.FAILED}),
    ExportState.GROOT_STATS_VALIDATED: frozenset({ExportState.GROOT_LOADER_VALIDATED, ExportState.FAILED}),
    ExportState.GROOT_LOADER_VALIDATED: frozenset({ExportState.PROVENANCE_WRITTEN, ExportState.FAILED}),
    ExportState.PROVENANCE_WRITTEN: frozenset({ExportState.FINAL_CONSISTENCY_VALIDATED, ExportState.FAILED}),
    ExportState.FINAL_CONSISTENCY_VALIDATED: frozenset({ExportState.PUBLISHING, ExportState.FAILED}),
    ExportState.PUBLISHING: frozenset(
        {ExportState.PUBLISHED, ExportState.FINAL_CONSISTENCY_VALIDATED, ExportState.FAILED}
    ),
    ExportState.PUBLISHED: frozenset(),
    ExportState.FAILED: frozenset(),
}
