"""Safe backend access and persistence for dataset understanding results."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend.app.services import dataset_service
from backend.app.services.dataset_storage import (
    DatasetMetadata,
    DatasetStorageManager,
    InvalidDatasetIdError,
    InvalidMetadataError,
    StorageIntegrityError,
)
from engine.understanding.models import DatasetUnderstandingResult
from engine.understanding.pipeline import run_understanding_pipeline

_RESULT_SCHEMA_VERSION = 1


def _result_directory() -> Path:
    return dataset_service.DATASET_STORAGE_DIR.parent / "understanding"


def _result_path(dataset_id: str) -> Path:
    return _result_directory() / f"{dataset_id}.json"


def _load_persisted_result(
    path: Path,
    metadata: DatasetMetadata,
) -> DatasetUnderstandingResult | None:
    if path.is_symlink():
        raise HTTPException(
            status_code=409,
            detail="Persisted understanding result is not a regular file.",
        )
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as source:
            envelope: dict[str, Any] = json.load(source)
        if (
            set(envelope) != {
                "schema_version",
                "dataset_id",
                "dataset_fingerprint",
                "result",
            }
            or envelope["schema_version"] != _RESULT_SCHEMA_VERSION
            or envelope["dataset_id"] != metadata.dataset_id
            or envelope["dataset_fingerprint"] != metadata.sha256
        ):
            return None
        result = DatasetUnderstandingResult.model_validate(envelope["result"])
        if (
            result.dataset_id != metadata.dataset_id
            or result.dataset_fingerprint != metadata.sha256
        ):
            return None
        return result
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=409,
            detail="Persisted understanding result is invalid.",
        ) from exc


def _persist_result(
    path: Path,
    result: DatasetUnderstandingResult,
) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise HTTPException(
            status_code=409,
            detail="Understanding result path must not be a symbolic link.",
        )
    envelope = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "dataset_id": result.dataset_id,
        "dataset_fingerprint": result.dataset_fingerprint,
        "result": result.model_dump(mode="json"),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result.dataset_id}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(
                envelope,
                target,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def get_understanding_result(dataset_id: str) -> dict[str, Any]:
    storage = DatasetStorageManager(dataset_service.DATASET_STORAGE_DIR)
    try:
        metadata = storage.read_metadata(dataset_id)
        file_path = storage.find_dataset_file(dataset_id)
        storage.verify_dataset_integrity(metadata.dataset_id, metadata)
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

    del file_path
    path = _result_path(metadata.dataset_id)
    with storage.storage_lock():
        result = _load_persisted_result(path, metadata)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No current understanding result exists.",
        )
    return {
        "dataset_id": metadata.dataset_id,
        "dataset_fingerprint": metadata.sha256,
        "stale": False,
        "result": result.model_dump(mode="json"),
    }


def analyze_dataset_understanding(dataset_id: str) -> dict[str, Any]:
    storage = DatasetStorageManager(dataset_service.DATASET_STORAGE_DIR)
    try:
        metadata = storage.read_metadata(dataset_id)
        file_path = storage.find_dataset_file(metadata.dataset_id)
        storage.verify_dataset_integrity(metadata.dataset_id, metadata)
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

    result_path = _result_path(metadata.dataset_id)
    with storage.storage_lock():
        existing = _load_persisted_result(result_path, metadata)
        if existing is not None:
            return {
                "dataset_id": metadata.dataset_id,
                "dataset_fingerprint": metadata.sha256,
                "stale": False,
                "result": existing.model_dump(mode="json"),
            }
        try:
            started_at = time.monotonic()
            with dataset_service._verified_preview_snapshot(
                file_path,
                metadata,
            ) as snapshot_path:
                dataset_service.inspect_dataset_resources(
                    snapshot_path,
                    metadata.format,
                    max_rows=dataset_service.MAX_DATASET_ROWS,
                    max_columns=dataset_service.MAX_DATASET_COLUMNS,
                    max_cells=dataset_service.MAX_DATASET_CELLS,
                    max_xlsx_uncompressed_bytes=(
                        dataset_service.MAX_XLSX_UNCOMPRESSED_BYTES
                    ),
                    max_parquet_uncompressed_bytes=(
                        dataset_service.MAX_PARQUET_UNCOMPRESSED_BYTES
                    ),
                    max_archive_compression_ratio=(
                        dataset_service.MAX_ARCHIVE_COMPRESSION_RATIO
                    ),
                    max_json_nesting_depth=(
                        dataset_service.MAX_JSON_NESTING_DEPTH
                    ),
                    started_at=started_at,
                    max_processing_seconds=(
                        dataset_service.MAX_PROCESSING_SECONDS
                    ),
                )
                dataframe = dataset_service.load_dataframe(
                    snapshot_path,
                    metadata.format,
                )
                dataset_service.validate_dataframe_resources(
                    len(dataframe),
                    len(dataframe.columns),
                    int(dataframe.memory_usage(deep=True).sum()),
                    max_rows=dataset_service.MAX_DATASET_ROWS,
                    max_columns=dataset_service.MAX_DATASET_COLUMNS,
                    max_cells=dataset_service.MAX_DATASET_CELLS,
                    max_dataframe_memory_bytes=(
                        dataset_service.MAX_DATAFRAME_MEMORY_BYTES
                    ),
                )
                dataframe.columns = dataset_service._read_source_column_names(
                    snapshot_path,
                    metadata.format,
                    dataframe,
                )
                result = run_understanding_pipeline(
                    dataframe,
                    dataset_id=metadata.dataset_id,
                    dataset_fingerprint=metadata.sha256,
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="Dataset understanding could not be completed.",
            ) from exc
        _persist_result(result_path, result)

    return {
        "dataset_id": metadata.dataset_id,
        "dataset_fingerprint": metadata.sha256,
        "stale": False,
        "result": result.model_dump(mode="json"),
    }
