import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_STORAGE_DIR = PROJECT_ROOT / "storage" / "datasets"

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".json": "json",
    ".parquet": "parquet",
}


def load_dataframe(file_path: Path, file_format: str) -> pd.DataFrame:
    """Read a supported dataset into a Pandas DataFrame."""

    if file_format == "csv":
        return pd.read_csv(file_path)

    if file_format == "xlsx":
        return pd.read_excel(file_path, engine="openpyxl")

    if file_format == "json":
        return pd.read_json(file_path)

    if file_format == "parquet":
        return pd.read_parquet(file_path, engine="pyarrow")

    raise ValueError("Unsupported dataset format")


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

    DATASET_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    stored_path = DATASET_STORAGE_DIR / f"{dataset_id}{extension}"

    file_size = 0
    file_hash = hashlib.sha256()

    try:
        with stored_path.open("wb") as destination:
            while True:
                chunk = await upload.read(CHUNK_SIZE)

                if not chunk:
                    break

                file_size += len(chunk)

                if file_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Maximum upload size is 50 MB.",
                    )

                file_hash.update(chunk)
                destination.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        dataframe = load_dataframe(stored_path, file_format)

        columns = []

        for index, column in enumerate(dataframe.columns):
            series = dataframe.iloc[:, index]

            columns.append({
                "name": str(column),
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
        }

    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        stored_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail="Unable to parse the uploaded dataset.",
        ) from exc

    finally:
        await upload.close()
