import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from fastapi import HTTPException
from openpyxl import load_workbook


def check_processing_time(started_at: float, limit_seconds: float) -> None:
    if time.monotonic() - started_at > limit_seconds:
        raise HTTPException(
            status_code=408,
            detail=(
                "Dataset processing exceeded the configured time limit."
            ),
        )


def inspect_dataset_resources(
    file_path: Path,
    file_format: str,
    *,
    max_rows: int,
    max_columns: int,
    max_cells: int,
    max_xlsx_uncompressed_bytes: int,
    max_parquet_uncompressed_bytes: int,
    max_archive_compression_ratio: float,
    max_json_nesting_depth: int,
    started_at: float,
    max_processing_seconds: float,
) -> None:
    """Reject oversized dataset structures before DataFrame materialization."""

    check_processing_time(started_at, max_processing_seconds)

    if file_format == "csv":
        _inspect_csv(
            file_path,
            max_rows,
            max_columns,
            max_cells,
            started_at,
            max_processing_seconds,
        )
    elif file_format == "xlsx":
        _inspect_xlsx(
            file_path,
            max_rows,
            max_columns,
            max_cells,
            max_xlsx_uncompressed_bytes,
            max_archive_compression_ratio,
            started_at,
            max_processing_seconds,
        )
    elif file_format == "json":
        _inspect_json_depth(
            file_path,
            max_json_nesting_depth,
            max_rows,
            max_columns,
            started_at,
            max_processing_seconds,
        )
    elif file_format == "parquet":
        _inspect_parquet(
            file_path,
            max_rows,
            max_columns,
            max_cells,
            max_parquet_uncompressed_bytes,
            started_at,
            max_processing_seconds,
        )

    check_processing_time(started_at, max_processing_seconds)


def validate_dataframe_resources(
    dataframe_rows: int,
    dataframe_columns: int,
    dataframe_memory_bytes: int,
    *,
    max_rows: int,
    max_columns: int,
    max_cells: int,
    max_dataframe_memory_bytes: int,
) -> None:
    _check_dimensions(
        dataframe_rows,
        dataframe_columns,
        max_rows,
        max_columns,
        max_cells,
    )
    if dataframe_memory_bytes > max_dataframe_memory_bytes:
        raise HTTPException(
            status_code=413,
            detail="Parsed dataset exceeds the configured memory limit.",
        )


def _check_dimensions(
    rows: int,
    columns: int,
    max_rows: int,
    max_columns: int,
    max_cells: int,
) -> None:
    if rows > max_rows:
        raise HTTPException(
            status_code=413,
            detail="Dataset exceeds the configured row limit.",
        )
    if columns > max_columns:
        raise HTTPException(
            status_code=413,
            detail="Dataset exceeds the configured column limit.",
        )
    if rows * columns > max_cells:
        raise HTTPException(
            status_code=413,
            detail="Dataset exceeds the configured cell limit.",
        )


def _inspect_csv(
    file_path: Path,
    max_rows: int,
    max_columns: int,
    max_cells: int,
    started_at: float,
    max_processing_seconds: float,
) -> None:
    header = pd.read_csv(file_path, nrows=0)
    column_count = len(header.columns)
    _check_dimensions(0, column_count, max_rows, max_columns, max_cells)

    row_count = 0
    for chunk in pd.read_csv(file_path, chunksize=10_000):
        check_processing_time(started_at, max_processing_seconds)
        row_count += len(chunk)
        _check_dimensions(
            row_count,
            column_count,
            max_rows,
            max_columns,
            max_cells,
        )


def _inspect_xlsx(
    file_path: Path,
    max_rows: int,
    max_columns: int,
    max_cells: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: float,
    started_at: float,
    max_processing_seconds: float,
) -> None:
    with zipfile.ZipFile(file_path) as archive:
        total_uncompressed_bytes = 0
        for member in archive.infolist():
            check_processing_time(started_at, max_processing_seconds)
            total_uncompressed_bytes += member.file_size
            ratio = (
                member.file_size / member.compress_size
                if member.compress_size
                else float("inf") if member.file_size else 1.0
            )
            if ratio > max_compression_ratio:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Spreadsheet exceeds the configured "
                        "compression-ratio limit."
                    ),
                )
            if total_uncompressed_bytes > max_uncompressed_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Spreadsheet exceeds the configured expanded-size "
                        "limit."
                    ),
                )

    workbook = load_workbook(file_path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        rows = max((worksheet.max_row or 1) - 1, 0)
        columns = worksheet.max_column or 0
        _check_dimensions(rows, columns, max_rows, max_columns, max_cells)
    finally:
        workbook.close()


def _inspect_json_depth(
    file_path: Path,
    max_nesting_depth: int,
    max_rows: int,
    max_columns: int,
    started_at: float,
    max_processing_seconds: float,
) -> None:
    depth = 0
    in_string = False
    escaped = False
    root_type: int | None = None
    root_container_items = 0
    root_container_has_item = False
    with file_path.open("rb") as source:
        while chunk := source.read(64 * 1024):
            check_processing_time(started_at, max_processing_seconds)
            for byte in chunk:
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                    continue
                if byte == ord('"'):
                    if depth == 1:
                        root_container_has_item = True
                    in_string = True
                elif byte in (ord("{"), ord("[")):
                    if depth == 0:
                        root_type = byte
                    elif depth == 1:
                        root_container_has_item = True
                    depth += 1
                    if depth > max_nesting_depth:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "JSON exceeds the configured nesting-depth "
                                "limit."
                            ),
                        )
                elif depth == 1 and byte == ord(","):
                    root_container_items += 1
                elif byte in (ord("}"), ord("]")):
                    depth -= 1
                elif depth == 1 and not chr(byte).isspace():
                    root_container_has_item = True

    root_item_count = root_container_items + int(root_container_has_item)
    if root_type == ord("[") and root_item_count > max_rows:
        raise HTTPException(
            status_code=413,
            detail="Dataset exceeds the configured row limit.",
        )
    if root_type == ord("{") and root_item_count > max_columns:
        raise HTTPException(
            status_code=413,
            detail="Dataset exceeds the configured column limit.",
        )


def _inspect_parquet(
    file_path: Path,
    max_rows: int,
    max_columns: int,
    max_cells: int,
    max_uncompressed_bytes: int,
    started_at: float,
    max_processing_seconds: float,
) -> None:
    check_processing_time(started_at, max_processing_seconds)
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    if metadata is None:
        raise ValueError("Parquet metadata is missing")

    rows = metadata.num_rows
    columns = metadata.num_columns
    _check_dimensions(rows, columns, max_rows, max_columns, max_cells)

    uncompressed_bytes = metadata.serialized_size + sum(
        metadata.row_group(index).total_byte_size
        for index in range(metadata.num_row_groups)
    )
    if uncompressed_bytes > max_uncompressed_bytes:
        raise HTTPException(
            status_code=413,
            detail="Parquet dataset exceeds the configured expanded-size limit.",
        )
