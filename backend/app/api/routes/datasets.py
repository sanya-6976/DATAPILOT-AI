from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from backend.app.schemas.dataset import DatasetUploadResponse
from backend.app.services.dataset_service import ingest_dataset

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
