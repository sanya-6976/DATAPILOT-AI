import contextlib
import hashlib
import json
import logging
import os
import stat
import tempfile
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

from backend.app.core.config import get_settings
from backend.app.services.dataset_limits import (
    check_processing_time,
    inspect_dataset_resources,
    validate_dataframe_resources,
)
from backend.app.services.dataset_storage import (
    DatasetIntegrityError,
    DatasetMetadata,
    DatasetStorageManager,
    InvalidDatasetIdError,
    InvalidMetadataError,
    StorageIntegrityError,
    StorageQuotaExceeded,
)
from engine.profiling.profiler import profile_dataframe
from engine.quality.data_quality import detect_data_quality_issues

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_STORAGE_DIR = PROJECT_ROOT / "storage" / "datasets"

_settings = get_settings()

MAX_UPLOAD_SIZE = _settings.dataset_max_upload_size_bytes
MAX_STORAGE_BYTES = _settings.dataset_storage_max_bytes
MAX_PREVIEW_ROWS = _settings.dataset_preview_max_rows
MAX_DATASET_ROWS = _settings.dataset_max_rows
MAX_DATASET_COLUMNS = _settings.dataset_max_columns
MAX_DATASET_CELLS = _settings.dataset_max_cells
MAX_DATAFRAME_MEMORY_BYTES = (
    _settings.dataset_max_dataframe_memory_bytes
)
MAX_PROCESSING_SECONDS = _settings.dataset_max_processing_seconds
MAX_XLSX_UNCOMPRESSED_BYTES = (
    _settings.dataset_max_xlsx_uncompressed_bytes
)
MAX_PARQUET_UNCOMPRESSED_BYTES = (
    _settings.dataset_max_parquet_uncompressed_bytes
)
MAX_ARCHIVE_COMPRESSION_RATIO = (
    _settings.dataset_max_archive_compression_ratio
)
MAX_JSON_NESTING_DEPTH = _settings.dataset_max_json_nesting_depth
CHUNK_SIZE = 1024 * 1024

SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".json": "json",
    ".parquet": "parquet",
}
MAX_DATASET_LIST_PAGE_SIZE = 100

logger = logging.getLogger(__name__)


def load_dataframe(file_path: Path, file_format: str) -> pd.DataFrame:
    """Read a supported dataset into a Pandas DataFrame."""

    if file_format == "csv":
        return pd.read_csv(file_path)

    if file_format == "xlsx":
        return pd.read_excel(file_path, engine="openpyxl")

    if file_format == "json":
        return pd.read_json(file_path, dtype=False)

    if file_format == "parquet":
        return pd.read_parquet(file_path, engine="pyarrow")

    raise ValueError("Unsupported dataset format")


def _read_source_column_names(
    file_path: Path,
    file_format: str,
    dataframe: pd.DataFrame,
) -> list[str]:
    """Read source headers without pandas' duplicate-name mangling."""

    if file_format == "csv":
        header = pd.read_csv(file_path, header=None, nrows=1)
    elif file_format == "xlsx":
        header = pd.read_excel(
            file_path,
            header=None,
            nrows=1,
            engine="openpyxl",
        )
    else:
        return [str(column) for column in dataframe.columns]

    if header.empty:
        return [str(column) for column in dataframe.columns]

    source_names = header.iloc[0].tolist()
    if len(source_names) != len(dataframe.columns):
        raise ValueError("Dataset header and parsed columns do not match")

    return [
        (
            str(source_name)
            if not pd.isna(source_name) and str(source_name)
            else str(dataframe.columns[index])
        )
        for index, source_name in enumerate(source_names)
    ]


def _is_json_content(file_path: Path) -> bool:
    """Identify unambiguous JSON documents uploaded with a CSV extension."""

    try:
        with file_path.open("r", encoding="utf-8-sig") as source:
            first_character = source.read(1)
            while first_character and first_character.isspace():
                first_character = source.read(1)
            if first_character not in {"{", "["}:
                return False
            source.seek(0)
            payload = json.load(source)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False

    return isinstance(payload, dict) or (
        isinstance(payload, list)
        and bool(payload)
        and all(isinstance(row, dict) for row in payload)
    )


def _make_column_names_unique(
    columns: pd.Index,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Create unique string labels and report duplicated source labels."""

    source_names = [str(column) for column in columns]
    reserved_names = set(source_names)
    used_names = set()
    occurrence_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    unique_names = []

    for name in source_names:
        occurrence = occurrence_counts.get(name, 0) + 1
        occurrence_counts[name] = occurrence

        if occurrence == 1 and name not in used_names:
            unique_name = name
        else:
            duplicate_counts[name] = duplicate_counts.get(name, 0) + 1
            suffix = 2
            unique_name = f"{name}__{suffix}"

            while unique_name in used_names or unique_name in reserved_names:
                suffix += 1
                unique_name = f"{name}__{suffix}"

        used_names.add(unique_name)
        unique_names.append(unique_name)

    return unique_names, list(duplicate_counts.items())


async def ingest_dataset(upload: UploadFile) -> dict:
    """Validate, store and profile an uploaded dataset."""

    raw_filename = upload.filename or "dataset"
    filename = raw_filename.replace("\\", "/").split("/")[-1]

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported format. Use CSV, XLSX, JSON or Parquet.",
        )

    file_format = SUPPORTED_FORMATS[extension]
    dataset_id = uuid.uuid4().hex
    storage_manager = DatasetStorageManager(
        DATASET_STORAGE_DIR,
        max_bytes=MAX_STORAGE_BYTES,
    )
    stored_path = DATASET_STORAGE_DIR / f"{dataset_id}{extension}"
    reservation = None
    promoted = False
    file_size = 0
    file_hash = hashlib.sha256()
    started_at = time.monotonic()

    try:
        DATASET_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        with storage_manager.upload_staging_directory() as staging_directory:
            staged_path = Path(staging_directory) / f"upload{extension}"
            with staged_path.open("xb") as destination:
                while True:
                    check_processing_time(
                        started_at,
                        MAX_PROCESSING_SECONDS,
                    )
                    chunk = await upload.read(CHUNK_SIZE)

                    if not chunk:
                        break

                    file_size += len(chunk)
                    if file_size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Uploaded file exceeds the configured "
                                "file-size limit."
                            ),
                        )

                    file_hash.update(chunk)
                    destination.write(chunk)

                destination.flush()
                os.fsync(destination.fileno())

            check_processing_time(started_at, MAX_PROCESSING_SECONDS)
            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty.",
                )

            try:
                reservation = storage_manager.reserve_storage(file_size)
            except StorageQuotaExceeded as exc:
                raise HTTPException(
                    status_code=507,
                    detail="Dataset storage capacity would be exceeded.",
                ) from exc

            try:
                inspect_dataset_resources(
                    staged_path,
                    file_format,
                    max_rows=MAX_DATASET_ROWS,
                    max_columns=MAX_DATASET_COLUMNS,
                    max_cells=MAX_DATASET_CELLS,
                    max_xlsx_uncompressed_bytes=(
                        MAX_XLSX_UNCOMPRESSED_BYTES
                    ),
                    max_parquet_uncompressed_bytes=(
                        MAX_PARQUET_UNCOMPRESSED_BYTES
                    ),
                    max_archive_compression_ratio=(
                        MAX_ARCHIVE_COMPRESSION_RATIO
                    ),
                    max_json_nesting_depth=MAX_JSON_NESTING_DEPTH,
                    started_at=started_at,
                    max_processing_seconds=MAX_PROCESSING_SECONDS,
                )

                if file_format == "csv" and _is_json_content(staged_path):
                    raise ValueError(
                        "The uploaded contents are JSON, not a CSV table."
                    )

                dataframe = load_dataframe(staged_path, file_format)
                validate_dataframe_resources(
                    len(dataframe),
                    len(dataframe.columns),
                    int(dataframe.memory_usage(deep=True).sum()),
                    max_rows=MAX_DATASET_ROWS,
                    max_columns=MAX_DATASET_COLUMNS,
                    max_cells=MAX_DATASET_CELLS,
                    max_dataframe_memory_bytes=(
                        MAX_DATAFRAME_MEMORY_BYTES
                    ),
                )
                check_processing_time(started_at, MAX_PROCESSING_SECONDS)

                source_column_names = _read_source_column_names(
                    staged_path,
                    file_format,
                    dataframe,
                )
                dataframe.columns = source_column_names
                column_names, duplicate_columns = _make_column_names_unique(
                    dataframe.columns
                )
                dataframe.columns = column_names
                profile = profile_dataframe(dataframe)
                check_processing_time(started_at, MAX_PROCESSING_SECONDS)
                for index, column_profile in enumerate(profile["columns"]):
                    column_profile["source_column_name"] = (
                        source_column_names[index]
                    )
                    column_profile["column_index"] = index
                quality_issues = detect_data_quality_issues(dataframe)
                check_processing_time(started_at, MAX_PROCESSING_SECONDS)
                quality_issues.extend(
                    {
                        "issue": "duplicate_column_names",
                        "column": column,
                        "severity": "low",
                        "count": duplicate_count,
                        "description": (
                            f"Column label '{column}' occurs more than once; "
                            "duplicate labels were renamed for analysis."
                        ),
                    }
                    for column, duplicate_count in duplicate_columns
                )

                columns = []
                for index, column in enumerate(dataframe.columns):
                    series = dataframe.iloc[:, index]
                    columns.append({
                        "name": str(column),
                        "source_name": source_column_names[index],
                        "index": index,
                        "dtype": str(series.dtype),
                        "missing_count": int(series.isna().sum()),
                    })

                missing_values = {
                    column["name"]: column["missing_count"]
                    for column in columns
                }
                preview = json.loads(
                    dataframe.head(10).to_json(
                        orient="records",
                        date_format="iso",
                    )
                )
                check_processing_time(started_at, MAX_PROCESSING_SECONDS)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Unable to parse uploaded dataset. Check that the "
                        "file contents match its extension and contain "
                        "tabular data."
                    ),
                ) from exc

            metadata = DatasetMetadata(
                dataset_id=dataset_id,
                original_filename=filename,
                file_extension=extension,
                format=file_format,
                created_at=datetime.now(UTC),
                file_size_bytes=file_size,
                sha256=file_hash.hexdigest(),
                row_count=len(dataframe),
                column_count=len(dataframe.columns),
                storage_status="ready",
            )

            try:
                with storage_manager.storage_lock():
                    metadata_path = storage_manager.resolve_metadata_path(
                        dataset_id
                    )
                    if metadata_path.exists() or metadata_path.is_symlink():
                        raise FileExistsError(
                            "Dataset metadata destination already exists."
                        )
                    os.link(staged_path, stored_path, follow_symlinks=False)
                    promoted = True
                    _fsync_directory(DATASET_STORAGE_DIR)
            except Exception as exc:
                detail = (
                    "Dataset file was committed, but storage synchronization "
                    "failed; reconciliation is required."
                    if promoted
                    else "Unable to commit uploaded dataset."
                )
                raise HTTPException(
                    status_code=500,
                    detail=detail,
                ) from exc

            try:
                storage_manager.write_metadata_atomic(metadata)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Dataset file was committed, but metadata "
                        "persistence failed; reconciliation is required."
                    ),
                ) from exc

            try:
                reservation.release()
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Dataset was committed, but its storage reservation "
                        "could not be released."
                    ),
                ) from exc
            reservation = None

            return {
                "dataset_id": dataset_id,
                "filename": filename,
                "file_format": file_format,
                "size_bytes": file_size,
                "row_count": len(dataframe),
                "column_count": len(dataframe.columns),
                "columns": columns,
                "missing_values": missing_values,
                "preview": preview,
                "sha256": file_hash.hexdigest(),
                "profile": profile,
                "quality_issues": quality_issues,
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Dataset storage operation failed.",
        ) from exc

    finally:
        if reservation is not None and not promoted:
            try:
                reservation.release()
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Dataset storage reservation could not be released.",
                ) from exc
        await upload.close()


def list_datasets(*, limit: int, offset: int) -> dict:
    """List valid, committed datasets in stable UUID order."""

    storage_manager = DatasetStorageManager(DATASET_STORAGE_DIR)
    datasets = []
    unavailable_count = 0

    for dataset_id in storage_manager.list_dataset_ids():
        try:
            metadata = storage_manager.read_metadata(dataset_id)
            stored_path = storage_manager.find_dataset_file(dataset_id)
            expected_path = storage_manager.resolve_dataset_file_path(
                dataset_id,
                metadata.file_extension,
            )
            if (
                stored_path != expected_path
                or stored_path.stat(follow_symlinks=False).st_size
                != metadata.file_size_bytes
            ):
                raise DatasetIntegrityError(
                    "Stored dataset does not match its metadata."
                )
            storage_manager.verify_dataset_integrity(dataset_id, metadata)
        except (
            InvalidMetadataError,
            FileNotFoundError,
            StorageIntegrityError,
        ):
            unavailable_count += 1
            logger.warning("Omitting unavailable dataset from listing: %s", dataset_id)
            continue
        datasets.append(_metadata_response(metadata))

    return {
        "datasets": datasets[offset : offset + limit],
        "total": len(datasets),
        "offset": offset,
        "limit": limit,
        "unavailable_count": unavailable_count,
    }


def get_dataset_metadata(dataset_id: str) -> dict:
    """Return validated metadata only after the original passes integrity checks."""

    storage_manager = DatasetStorageManager(DATASET_STORAGE_DIR)
    try:
        metadata = storage_manager.read_metadata(dataset_id)
        storage_manager.find_dataset_file(dataset_id)
        storage_manager.verify_dataset_integrity(dataset_id, metadata)
    except InvalidDatasetIdError as exc:
        raise HTTPException(
            status_code=422,
            detail="Dataset ID must be a valid UUID.",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Dataset was not found.",
        ) from exc
    except InvalidMetadataError as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset metadata is invalid.",
        ) from exc
    except StorageIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset failed integrity verification.",
        ) from exc

    return _metadata_response(metadata)


def preview_dataset(
    dataset_id: str,
    *,
    limit: int,
    offset: int,
) -> dict:
    """Return bounded rows after integrity and full-dataset resource checks."""

    storage_manager = DatasetStorageManager(DATASET_STORAGE_DIR)
    try:
        metadata = storage_manager.read_metadata(dataset_id)
        if metadata.file_size_bytes > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Dataset exceeds the configured file-size limit.",
            )
        file_path = storage_manager.find_dataset_file(dataset_id)
        storage_manager.verify_dataset_integrity(dataset_id, metadata)
    except HTTPException:
        raise
    except InvalidDatasetIdError as exc:
        raise HTTPException(
            status_code=422,
            detail="Dataset ID must be a valid UUID.",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Dataset was not found.",
        ) from exc
    except InvalidMetadataError as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset metadata is invalid.",
        ) from exc
    except StorageIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset failed integrity verification.",
        ) from exc

    started_at = time.monotonic()
    try:
        with _verified_preview_snapshot(file_path, metadata) as snapshot_path:
            inspect_dataset_resources(
                snapshot_path,
                metadata.format,
                max_rows=MAX_DATASET_ROWS,
                max_columns=MAX_DATASET_COLUMNS,
                max_cells=MAX_DATASET_CELLS,
                max_xlsx_uncompressed_bytes=MAX_XLSX_UNCOMPRESSED_BYTES,
                max_parquet_uncompressed_bytes=MAX_PARQUET_UNCOMPRESSED_BYTES,
                max_archive_compression_ratio=MAX_ARCHIVE_COMPRESSION_RATIO,
                max_json_nesting_depth=MAX_JSON_NESTING_DEPTH,
                started_at=started_at,
                max_processing_seconds=MAX_PROCESSING_SECONDS,
            )
            dataframe = load_dataframe(snapshot_path, metadata.format)
            validate_dataframe_resources(
                len(dataframe),
                len(dataframe.columns),
                int(dataframe.memory_usage(deep=True).sum()),
                max_rows=MAX_DATASET_ROWS,
                max_columns=MAX_DATASET_COLUMNS,
                max_cells=MAX_DATASET_CELLS,
                max_dataframe_memory_bytes=MAX_DATAFRAME_MEMORY_BYTES,
            )
            source_column_names = _read_source_column_names(
                snapshot_path,
                metadata.format,
                dataframe,
            )
            check_processing_time(started_at, MAX_PROCESSING_SECONDS)
    except HTTPException:
        raise
    except StorageIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="Dataset failed integrity verification.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Unable to preview dataset contents.",
        ) from exc

    if (
        len(dataframe) != metadata.row_count
        or len(dataframe.columns) != metadata.column_count
        or len(source_column_names) != len(dataframe.columns)
    ):
        raise HTTPException(
            status_code=409,
            detail="Dataset dimensions do not match validated metadata.",
        )

    rows = json.loads(
        dataframe.iloc[offset : offset + limit].to_json(
            orient="values",
            date_format="iso",
        )
    )
    return {
        "dataset_id": metadata.dataset_id,
        "columns": [
            {"name": name, "index": index}
            for index, name in enumerate(source_column_names)
        ],
        "rows": rows,
        "offset": offset,
        "limit": limit,
        "total_rows": len(dataframe),
    }


@contextlib.contextmanager
def _verified_preview_snapshot(
    source_path: Path,
    metadata: DatasetMetadata,
) -> Iterator[Path]:
    """Parse an immutable, hash-checked copy of the committed dataset."""

    with tempfile.TemporaryDirectory(prefix="datapilot-preview-") as directory:
        snapshot_path = Path(directory) / f"dataset{metadata.file_extension}"
        try:
            descriptor = os.open(
                source_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise DatasetIntegrityError(
                "Dataset file changed during preview."
            ) from exc

        digest = hashlib.sha256()
        copied_size = 0
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise DatasetIntegrityError(
                    "Dataset path is not a regular file."
                )
            with snapshot_path.open("xb") as snapshot:
                while chunk := source.read(CHUNK_SIZE):
                    copied_size += len(chunk)
                    digest.update(chunk)
                    snapshot.write(chunk)
                snapshot.flush()

        if (
            copied_size != metadata.file_size_bytes
            or digest.hexdigest() != metadata.sha256
        ):
            raise DatasetIntegrityError(
                "Dataset changed during preview."
            )
        yield snapshot_path


def _metadata_response(metadata: DatasetMetadata) -> dict:
    return {
        "dataset_id": metadata.dataset_id,
        "filename": metadata.original_filename,
        "file_extension": metadata.file_extension,
        "file_format": metadata.format,
        "created_at": metadata.created_at,
        "size_bytes": metadata.file_size_bytes,
        "row_count": metadata.row_count,
        "column_count": metadata.column_count,
        "sha256": metadata.sha256,
        "storage_status": metadata.storage_status,
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
