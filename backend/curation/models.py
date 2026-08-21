"""Stable curation-state vocabulary shared by persistence and API layers."""

from __future__ import annotations

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
