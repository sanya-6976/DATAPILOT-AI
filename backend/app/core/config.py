from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DataPilot AI"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"
    dataset_max_upload_size_bytes: int = 50 * 1024 * 1024
    dataset_storage_max_bytes: int = 1_073_741_824
    dataset_preview_max_rows: int = Field(default=100, ge=1, le=1000)
    dataset_max_rows: int = 1_000_000
    dataset_max_columns: int = 2_000
    dataset_max_cells: int = 10_000_000
    dataset_max_dataframe_memory_bytes: int = 512 * 1024 * 1024
    dataset_max_processing_seconds: float = 30.0
    dataset_max_xlsx_uncompressed_bytes: int = 256 * 1024 * 1024
    dataset_max_parquet_uncompressed_bytes: int = 512 * 1024 * 1024
    dataset_max_archive_compression_ratio: float = 100.0
    dataset_max_json_nesting_depth: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
