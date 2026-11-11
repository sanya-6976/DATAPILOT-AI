from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile, status

from backend.app.schemas.dataset import (
    DatasetListResponse,
    DatasetMetadataResponse,
    DatasetPreviewResponse,
    DatasetUnderstandingResponse,
    DatasetUploadResponse,
)
from backend.app.services.dataset_service import (
    MAX_DATASET_LIST_PAGE_SIZE,
    MAX_PREVIEW_ROWS,
    get_dataset_metadata,
    ingest_dataset,
    list_datasets,
    preview_dataset,
)
from backend.app.services.understanding_service import (
    analyze_dataset_understanding,
    get_understanding_result,
)

router = APIRouter()


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: Annotated[UploadFile, File()],
):
    return await ingest_dataset(file)


@router.get(
    "",
    response_model=DatasetListResponse,
)
def get_datasets(
    limit: Annotated[int, Query(ge=1, le=MAX_DATASET_LIST_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return list_datasets(limit=limit, offset=offset)


@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
)
def get_dataset_preview(
    dataset_id: str,
    limit: Annotated[int, Query(ge=1, le=MAX_PREVIEW_ROWS)] = min(
        20,
        MAX_PREVIEW_ROWS,
    ),
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return preview_dataset(dataset_id, limit=limit, offset=offset)


@router.post(
    "/{dataset_id}/understanding",
    response_model=DatasetUnderstandingResponse,
)
def post_dataset_understanding(dataset_id: str):
    return analyze_dataset_understanding(dataset_id)


@router.get(
    "/{dataset_id}/understanding",
    response_model=DatasetUnderstandingResponse,
)
def get_dataset_understanding(dataset_id: str):
    return get_understanding_result(dataset_id)


@router.get(
    "/{dataset_id}",
    response_model=DatasetMetadataResponse,
)
def get_dataset(dataset_id: str):
    return get_dataset_metadata(dataset_id)
