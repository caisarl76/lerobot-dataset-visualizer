from __future__ import annotations

import secrets
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from .assets import LocalAssetService
from .db import RetryableDatabaseError
from .review import ReviewError, ReviewService
from .worker import BatchError, BatchService


class _StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OpenWorkspaceBody(_StrictBody):
    dataset_alias: str
    actor: str


class ReviewMutationBody(_StrictBody):
    dataset_alias: str
    expected_revision: int
    actor: str


class DraftBody(ReviewMutationBody):
    object_name: str | None = None
    pickup_hand: Literal["left", "right"] | None = None
    turn_direction: Literal["left", "right"] | None = None
    transition_frames: list[int | None] | None = None


class ApprovalBody(ReviewMutationBody):
    reviewer: str


class RejectBody(ApprovalBody):
    reason: str | None = None


class StartBatchBody(_StrictBody):
    dataset_alias: str
    episode_indices: list[int] | None = None


class RetryBatchBody(_StrictBody):
    episode_indices: list[int] | None = None
    failure_states: list[Literal["manual_only", "retryable", "cancelled"]] | None = None


def _review_response(operation: Any) -> JSONResponse:
    try:
        return JSONResponse(operation())
    except (ReviewError, RetryableDatabaseError) as error:
        return JSONResponse(error.payload, status_code=error.status_code)


def build_curation_router(
    asset_service: LocalAssetService | None = None,
    review_service: ReviewService | None = None,
    batch_service: BatchService | None = None,
    bearer_token: str | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.api_route(
        "/api/local-datasets/{org}/{dataset}/resolve/{revision}/{asset_path:path}",
        methods=["GET", "HEAD"],
    )
    async def local_dataset_asset(
        org: str, dataset: str, revision: str, asset_path: str, request: Request
    ) -> Response:
        if asset_service is None:
            return Response(status_code=503)
        return asset_service.serve(
            org,
            dataset,
            revision,
            asset_path,
            method=request.method,
            headers=request.headers,
        )

    if review_service is None and batch_service is None:
        return router
    if not bearer_token:
        raise ValueError("bearer_token is required when review_service is configured")

    def require_curation_bearer(authorization: str | None = Header(default=None)) -> None:
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
        if not supplied or not secrets.compare_digest(supplied, bearer_token):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=401,
                detail={"error": "unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    curation = APIRouter(dependencies=[Depends(require_curation_bearer)])

    if batch_service is not None:

        def batch_response(operation: Any, *, success_status: int = 200) -> JSONResponse:
            try:
                return JSONResponse(operation(), status_code=success_status)
            except (BatchError, RetryableDatabaseError) as error:
                return JSONResponse(error.payload, status_code=error.status_code)

        @curation.post("/api/curation/batches")
        def start_batch(body: StartBatchBody) -> JSONResponse:
            return batch_response(
                lambda: batch_service.start(body.dataset_alias, body.episode_indices), success_status=201
            )

        @curation.get("/api/curation/batches/{job_id}")
        def batch_status(job_id: str) -> JSONResponse:
            return batch_response(lambda: batch_service.status(job_id))

        @curation.post("/api/curation/batches/{job_id}/retry")
        def retry_batch(job_id: str, body: RetryBatchBody) -> JSONResponse:
            return batch_response(
                lambda: batch_service.retry(
                    job_id,
                    episode_indices=body.episode_indices,
                    failure_states=body.failure_states,
                ),
                success_status=201,
            )

        @curation.post("/api/curation/batches/{job_id}/cancel")
        def cancel_batch(job_id: str) -> JSONResponse:
            try:
                status_code, payload = batch_service.cancel(job_id)
                return JSONResponse(payload, status_code=status_code)
            except RetryableDatabaseError as error:
                return JSONResponse(error.payload, status_code=error.status_code)

    if review_service is None:
        router.include_router(curation)
        return router

    @curation.post("/api/curation/workspaces/open")
    def open_workspace(body: OpenWorkspaceBody) -> JSONResponse:
        return _review_response(lambda: review_service.open_workspace(body.dataset_alias, actor=body.actor))

    @curation.get("/api/curation/summary")
    def review_summary(dataset_alias: str) -> JSONResponse:
        return _review_response(lambda: review_service.summary(dataset_alias))

    @curation.get("/api/curation/episodes/{source_episode_index}")
    def get_review_episode(source_episode_index: int, dataset_alias: str) -> JSONResponse:
        return _review_response(lambda: review_service.get_episode(dataset_alias, source_episode_index))

    @curation.get("/api/curation/episodes/{source_episode_index}/grip")
    def get_grip_diagnostic(source_episode_index: int, dataset_alias: str) -> JSONResponse:
        return _review_response(lambda: review_service.grip_diagnostic(dataset_alias, source_episode_index))

    @curation.get("/api/curation/audit")
    def get_curation_audit(dataset_alias: str) -> JSONResponse:
        return _review_response(lambda: review_service.audit(dataset_alias))

    @curation.patch("/api/curation/episodes/{source_episode_index}/draft")
    def save_review_draft(source_episode_index: int, body: DraftBody) -> JSONResponse:
        optional: dict[str, Any] = {}
        for field_name in ("object_name", "pickup_hand", "turn_direction", "transition_frames"):
            if field_name in body.model_fields_set:
                optional[field_name] = getattr(body, field_name)
        return _review_response(
            lambda: review_service.save_draft(
                dataset_alias=body.dataset_alias,
                source_episode_index=source_episode_index,
                expected_revision=body.expected_revision,
                actor=body.actor,
                **optional,
            )
        )

    @curation.post("/api/curation/episodes/{source_episode_index}/apply-proposal")
    def apply_review_proposal(source_episode_index: int, body: ReviewMutationBody) -> JSONResponse:
        return _review_response(
            lambda: review_service.apply_proposal(
                dataset_alias=body.dataset_alias,
                source_episode_index=source_episode_index,
                expected_revision=body.expected_revision,
                actor=body.actor,
            )
        )

    @curation.post("/api/curation/episodes/{source_episode_index}/approve-keep")
    def approve_review_keep(source_episode_index: int, body: ApprovalBody) -> JSONResponse:
        return _review_response(
            lambda: review_service.approve_keep(
                dataset_alias=body.dataset_alias,
                source_episode_index=source_episode_index,
                expected_revision=body.expected_revision,
                actor=body.actor,
                reviewer=body.reviewer,
            )
        )

    @curation.post("/api/curation/episodes/{source_episode_index}/approve-reject")
    def approve_review_reject(source_episode_index: int, body: RejectBody) -> JSONResponse:
        return _review_response(
            lambda: review_service.approve_reject(
                dataset_alias=body.dataset_alias,
                source_episode_index=source_episode_index,
                expected_revision=body.expected_revision,
                actor=body.actor,
                reviewer=body.reviewer,
                reason=body.reason,
            )
        )

    @curation.post("/api/curation/episodes/{source_episode_index}/reopen")
    def reopen_review_episode(source_episode_index: int, body: ReviewMutationBody) -> JSONResponse:
        return _review_response(
            lambda: review_service.reopen(
                dataset_alias=body.dataset_alias,
                source_episode_index=source_episode_index,
                expected_revision=body.expected_revision,
                actor=body.actor,
            )
        )

    router.include_router(curation)

    return router
