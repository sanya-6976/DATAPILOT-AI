
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from engine.understanding.models import DatasetUnderstandingResult


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int
    source_name: str | None = None
    index: int | None = None


class QualityIssue(BaseModel):
    issue: str
    severity: str
    description: str
    column: str | None = None
    count: int | None = None
    percentage: float | None = None


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    file_format: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    missing_values: dict[str, int]
    preview: list[dict[str, Any]]
    sha256: str
    profile: dict[str, Any]
    quality_issues: list[QualityIssue] = Field(default_factory=list)


class DatasetMetadataResponse(BaseModel):
    dataset_id: str
    filename: str
    file_extension: str
    file_format: str
    created_at: datetime
    size_bytes: int
    row_count: int
    column_count: int
    sha256: str
    storage_status: str


class DatasetListResponse(BaseModel):
    datasets: list[DatasetMetadataResponse]
    total: int
    offset: int
    limit: int
    unavailable_count: int


class DatasetPreviewColumn(BaseModel):
    name: str
    index: int


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: list[DatasetPreviewColumn]
    rows: list[list[Any]]
    offset: int
    limit: int
    total_rows: int


class DatasetUnderstandingResponse(BaseModel):
    dataset_id: str
    dataset_fingerprint: str
    stale: bool = False
    result: DatasetUnderstandingResult
