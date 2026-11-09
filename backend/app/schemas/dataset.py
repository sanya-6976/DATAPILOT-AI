from typing import Any

from pydantic import BaseModel


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int


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
