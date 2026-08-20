from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .assets import LocalAssetService


def build_curation_router(asset_service: LocalAssetService | None = None) -> APIRouter:
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

    return router
