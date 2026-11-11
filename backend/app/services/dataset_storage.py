"""Local dataset storage primitives.

Uploads stage to a temporary directory, reserve their measured size, and retain
the reservation while validation and profiling run. Promotion uses a create-only
hard link, metadata is persisted atomically, and the reservation is released
only after both files are committed. If metadata persistence fails after
promotion, the original remains quota-accounted and
``report_files_without_metadata`` identifies it for explicit reconciliation;
never roll back by deleting a possibly pre-existing file or unrelated upload.
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from backend.app.core.config import get_settings

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".json": "json",
    ".parquet": "parquet",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UUID_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_UUID_DASHED_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_RESERVATION_FILENAME_PATTERN = re.compile(
    r"^\.reservation-([0-9a-f]{32})\.json$"
)
_METADATA_FILENAME_PATTERN = re.compile(
    r"^([0-9a-f]{32})\.metadata\.json$"
)
_METADATA_TEMP_FILENAME_PATTERN = re.compile(
    r"^\.([0-9a-f]{32})\.metadata-([a-z0-9_]{8})\.tmp$"
)
_STAGING_DIRECTORY_PATTERN = re.compile(
    r"^\.datapilot-staging-([0-9a-f]{32})$"
)
_LEGACY_STAGING_DIRECTORY_PATTERN = re.compile(r"^\.staging-[a-z0-9_]{8}$")
_STAGING_LOCK_FILENAME = ".active.lock"
_METADATA_FIELDS = {
    "dataset_id",
    "original_filename",
    "file_extension",
    "format",
    "created_at",
    "file_size_bytes",
    "sha256",
    "row_count",
    "column_count",
    "storage_status",
}


class DatasetStorageError(Exception):
    """Base error for dataset storage operations."""


class InvalidDatasetIdError(DatasetStorageError, ValueError):
    """Raised when a dataset identifier is not a supported UUID string."""


class InvalidMetadataError(DatasetStorageError, ValueError):
    """Raised when persisted dataset metadata is invalid or corrupted."""


class StorageQuotaExceeded(DatasetStorageError):
    """Raised when committed and reserved bytes exceed storage capacity."""


class StorageIntegrityError(DatasetStorageError):
    """Raised when stored files conflict with a unique dataset identity."""


class DatasetIntegrityError(StorageIntegrityError):
    """Raised when the stored file does not match validated metadata."""


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    original_filename: str
    file_extension: str
    format: str
    created_at: datetime
    file_size_bytes: int
    sha256: str
    row_count: int
    column_count: int
    storage_status: str

    def __post_init__(self) -> None:
        normalized_id = validate_dataset_id(self.dataset_id)
        object.__setattr__(self, "dataset_id", normalized_id)

        if (
            not isinstance(self.original_filename, str)
            or not self.original_filename
            or self.original_filename in {".", ".."}
            or "/" in self.original_filename
            or "\\" in self.original_filename
            or "\x00" in self.original_filename
        ):
            raise InvalidMetadataError("Invalid original filename.")

        try:
            extension = _normalize_extension(self.file_extension)
        except (TypeError, ValueError) as exc:
            raise InvalidMetadataError(
                "Unsupported dataset extension."
            ) from exc
        if extension not in SUPPORTED_EXTENSIONS:
            raise InvalidMetadataError("Unsupported dataset extension.")
        if self.format != SUPPORTED_EXTENSIONS[extension]:
            raise InvalidMetadataError(
                "Dataset format does not match its extension."
            )
        object.__setattr__(self, "file_extension", extension)

        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() != UTC.utcoffset(None)
        ):
            raise InvalidMetadataError(
                "created_at must be a timezone-aware UTC datetime."
            )

        for field_name in ("file_size_bytes", "row_count", "column_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise InvalidMetadataError(
                    f"{field_name} must be a non-negative integer."
                )

        if (
            not isinstance(self.sha256, str)
            or _SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise InvalidMetadataError("sha256 must be 64 lowercase hex digits.")

        if (
            not isinstance(self.storage_status, str)
            or self.storage_status not in {"ready", "deletion_pending"}
        ):
            raise InvalidMetadataError("Unsupported storage status.")

    @classmethod
    def from_dict(cls, data: Any) -> "DatasetMetadata":
        if not isinstance(data, dict) or set(data) != _METADATA_FIELDS:
            raise InvalidMetadataError("Metadata fields do not match schema.")

        created_at = data["created_at"]
        if not isinstance(created_at, str):
            raise InvalidMetadataError("created_at must be an ISO timestamp.")
        try:
            parsed_created_at = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise InvalidMetadataError("Invalid created_at timestamp.") from exc

        try:
            return cls(
                dataset_id=data["dataset_id"],
                original_filename=data["original_filename"],
                file_extension=data["file_extension"],
                format=data["format"],
                created_at=parsed_created_at,
                file_size_bytes=data["file_size_bytes"],
                sha256=data["sha256"],
                row_count=data["row_count"],
                column_count=data["column_count"],
                storage_status=data["storage_status"],
            )
        except InvalidMetadataError:
            raise
        except (TypeError, ValueError) as exc:
            raise InvalidMetadataError(
                "Metadata contains invalid values."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "original_filename": self.original_filename,
            "file_extension": self.file_extension,
            "format": self.format,
            "created_at": (
                self.created_at.isoformat().replace("+00:00", "Z")
            ),
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "storage_status": self.storage_status,
        }


@dataclass(frozen=True)
class DatasetIntegrityResult:
    dataset_id: str
    file_extension: str
    file_size_bytes: int
    sha256: str


@dataclass(frozen=True)
class MetadataTemporaryArtifact:
    filename: str
    dataset_id: str


@dataclass(frozen=True)
class DatasetFileWithoutMetadata:
    dataset_id: str
    file_extension: str
    file_size_bytes: int


@dataclass(frozen=True)
class StagingDirectoryArtifact:
    directory_name: str
    staging_id: str
    file_size_bytes: int
    active: bool | None


class ActiveStagingDirectoryError(DatasetStorageError):
    """Raised when staging recovery targets an active upload."""


def validate_dataset_id(dataset_id: str) -> str:
    """Validate a UUID and return the lowercase 32-hex storage form."""

    if not isinstance(dataset_id, str) or not (
        _UUID_HEX_PATTERN.fullmatch(dataset_id)
        or _UUID_DASHED_PATTERN.fullmatch(dataset_id)
    ):
        raise InvalidDatasetIdError("Invalid dataset UUID.")

    try:
        return uuid.UUID(dataset_id).hex
    except ValueError as exc:
        raise InvalidDatasetIdError("Invalid dataset UUID.") from exc


def _normalize_extension(extension: str) -> str:
    if not isinstance(extension, str):
        raise TypeError("Dataset extension must be a string.")
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if normalized not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported dataset extension.")
    return normalized


class StorageReservation:
    """A cross-process reservation released by closing or leaving its context."""

    def __init__(
        self,
        manager: "DatasetStorageManager",
        reservation_path: Path,
        reservation_file,
        size_bytes: int,
    ) -> None:
        self._manager = manager
        self._reservation_path = reservation_path
        self._reservation_file = reservation_file
        self.size_bytes = size_bytes
        self._released = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return

        with self._manager.storage_lock():
            self._reservation_path.unlink(missing_ok=True)
        self._reservation_file.close()
        self._released = True


class DatasetStorageManager:
    """Manage dataset file identities, metadata, quota and reservations."""

    def __init__(
        self,
        storage_dir: Path,
        max_bytes: int | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        configured_limit = (
            get_settings().dataset_storage_max_bytes
            if max_bytes is None
            else max_bytes
        )
        if type(configured_limit) is not int or configured_limit < 0:
            raise ValueError("Storage capacity must be a non-negative integer.")
        self.max_bytes = configured_limit

    def resolve_dataset_file_path(
        self,
        dataset_id: str,
        file_extension: str,
    ) -> Path:
        normalized_id = validate_dataset_id(dataset_id)
        extension = _normalize_extension(file_extension)
        return self.storage_dir / f"{normalized_id}{extension}"

    def resolve_metadata_path(self, dataset_id: str) -> Path:
        normalized_id = validate_dataset_id(dataset_id)
        return self.storage_dir / f"{normalized_id}.metadata.json"

    def list_dataset_ids(self) -> tuple[str, ...]:
        """Return sorted UUIDs with exact manager-named metadata sidecars."""

        if not self.storage_dir.exists():
            return ()
        with self.storage_lock():
            dataset_ids = {
                match.group(1)
                for path in self.storage_dir.iterdir()
                if (
                    (match := _METADATA_FILENAME_PATTERN.fullmatch(path.name))
                    is not None
                )
            }
        return tuple(sorted(dataset_ids))

    def find_dataset_file(self, dataset_id: str) -> Path:
        normalized_id = validate_dataset_id(dataset_id)
        matches = []
        for extension in SUPPORTED_EXTENSIONS:
            candidate = self.storage_dir / f"{normalized_id}{extension}"
            if candidate.is_symlink():
                raise StorageIntegrityError(
                    "Dataset file must not be a symbolic link."
                )
            if candidate.exists() and not _is_regular_file(candidate):
                raise StorageIntegrityError(
                    "Dataset path is not a regular file."
                )
            if _is_regular_file(candidate):
                matches.append(candidate)
        if len(matches) > 1:
            raise StorageIntegrityError(
                "Multiple files use the same dataset UUID."
            )
        if not matches:
            raise FileNotFoundError("Dataset file does not exist.")
        return matches[0]

    def read_metadata(self, dataset_id: str) -> DatasetMetadata:
        normalized_id = validate_dataset_id(dataset_id)
        metadata_path = self.resolve_metadata_path(normalized_id)
        if not _is_regular_file(metadata_path):
            if metadata_path.exists() or metadata_path.is_symlink():
                raise InvalidMetadataError(
                    "Metadata sidecar is not a regular file."
                )
            raise FileNotFoundError("Dataset metadata does not exist.")

        try:
            with metadata_path.open("r", encoding="utf-8") as source:
                metadata = DatasetMetadata.from_dict(json.load(source))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidMetadataError(
                "Dataset metadata is malformed or corrupted."
            ) from exc

        if metadata.dataset_id != normalized_id:
            raise InvalidMetadataError(
                "Metadata dataset UUID does not match its filename."
            )
        return metadata

    def verify_dataset_integrity(
        self,
        dataset_id: str,
        metadata: DatasetMetadata | None = None,
    ) -> DatasetIntegrityResult:
        """Verify the original file against metadata without repairing either."""

        normalized_id = validate_dataset_id(dataset_id)
        with self.storage_lock():
            if metadata is None:
                validated_metadata = self.read_metadata(normalized_id)
            elif isinstance(metadata, DatasetMetadata):
                validated_metadata = DatasetMetadata.from_dict(
                    metadata.to_dict()
                )
            else:
                raise InvalidMetadataError(
                    "Integrity verification requires validated metadata."
                )
            if validated_metadata.dataset_id != normalized_id:
                raise DatasetIntegrityError(
                    "Metadata UUID does not match the requested dataset."
                )

            expected_path = self.resolve_dataset_file_path(
                normalized_id,
                validated_metadata.file_extension,
            )
            matching_files = []
            for extension in SUPPORTED_EXTENSIONS:
                candidate = self.resolve_dataset_file_path(
                    normalized_id,
                    extension,
                )
                if candidate.is_symlink():
                    raise DatasetIntegrityError(
                        "Dataset file must not be a symbolic link."
                    )
                if _is_regular_file(candidate):
                    matching_files.append(candidate)

            if len(matching_files) > 1:
                raise DatasetIntegrityError(
                    "Multiple files use the same dataset UUID."
                )
            if not matching_files:
                raise DatasetIntegrityError("Dataset file does not exist.")
            if matching_files[0] != expected_path:
                raise DatasetIntegrityError(
                    "Dataset file extension does not match metadata."
                )

            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(expected_path, flags)
            except OSError as exc:
                raise DatasetIntegrityError(
                    "Dataset file cannot be opened safely."
                ) from exc

            with os.fdopen(descriptor, "rb") as source:
                file_stat = os.fstat(source.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    raise DatasetIntegrityError(
                        "Dataset file is not a regular file."
                    )
                if file_stat.st_size != validated_metadata.file_size_bytes:
                    raise DatasetIntegrityError(
                        "Dataset file size does not match metadata."
                    )

                digest = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)

            actual_sha256 = digest.hexdigest()
            if actual_sha256 != validated_metadata.sha256:
                raise DatasetIntegrityError(
                    "Dataset file SHA-256 does not match metadata."
                )

            return DatasetIntegrityResult(
                dataset_id=normalized_id,
                file_extension=validated_metadata.file_extension,
                file_size_bytes=file_stat.st_size,
                sha256=actual_sha256,
            )

    def report_metadata_temporary_files(
        self,
    ) -> tuple[MetadataTemporaryArtifact, ...]:
        """Report exact manager-generated temp files without removing them."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        with self.storage_lock():
            for path in sorted(self.storage_dir.iterdir()):
                match = _METADATA_TEMP_FILENAME_PATTERN.fullmatch(path.name)
                if match is None or not _is_regular_file(path):
                    continue
                artifacts.append(
                    MetadataTemporaryArtifact(
                        filename=path.name,
                        dataset_id=match.group(1),
                    )
                )
        return tuple(artifacts)

    def report_files_without_metadata(
        self,
    ) -> tuple[DatasetFileWithoutMetadata, ...]:
        """Report UUID-named dataset files lacking sidecars for reconciliation.

        This intentionally includes legacy uploads and never creates metadata.
        """

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        with self.storage_lock():
            for path in sorted(self.storage_dir.iterdir()):
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    dataset_id = validate_dataset_id(path.stem)
                except InvalidDatasetIdError:
                    continue
                if not _is_regular_file(path):
                    continue
                metadata_path = self.resolve_metadata_path(dataset_id)
                if metadata_path.exists() or metadata_path.is_symlink():
                    continue
                artifacts.append(
                    DatasetFileWithoutMetadata(
                        dataset_id=dataset_id,
                        file_extension=path.suffix.lower(),
                        file_size_bytes=path.stat(
                            follow_symlinks=False
                        ).st_size,
                    )
                )
        return tuple(artifacts)

    @contextmanager
    def upload_staging_directory(self) -> Iterator[Path]:
        """Create an upload staging directory protected by a process lock."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        staging_id = uuid.uuid4().hex
        directory = self.storage_dir / f".datapilot-staging-{staging_id}"
        with self.storage_lock():
            directory.mkdir()
            lock_path = directory / _STAGING_LOCK_FILENAME
            lock_file = lock_path.open("x+b")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            yield directory
        finally:
            try:
                with self.storage_lock():
                    shutil.rmtree(directory)
                    _fsync_directory(self.storage_dir)
            finally:
                lock_file.close()

    def report_staging_directories(
        self,
    ) -> tuple[StagingDirectoryArtifact, ...]:
        """Report exact managed staging directories and their disk usage."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        with self.storage_lock():
            for directory in sorted(self.storage_dir.iterdir()):
                match = _STAGING_DIRECTORY_PATTERN.fullmatch(directory.name)
                is_legacy = (
                    match is None
                    and _LEGACY_STAGING_DIRECTORY_PATTERN.fullmatch(
                        directory.name
                    )
                    is not None
                )
                if (
                    (match is None and not is_legacy)
                    or not _is_real_directory(directory)
                ):
                    continue
                staging_id = (
                    match.group(1)
                    if match is not None
                    else directory.name.removeprefix(".staging-")
                )
                active = (
                    self._staging_directory_is_active(
                        directory / _STAGING_LOCK_FILENAME
                    )
                    if match is not None
                    else None
                )
                artifacts.append(
                    StagingDirectoryArtifact(
                        directory_name=directory.name,
                        staging_id=staging_id,
                        file_size_bytes=_staging_directory_size(directory),
                        active=active,
                    )
                )
        return tuple(artifacts)

    def get_staging_usage_bytes(self) -> int:
        """Return staged regular-file bytes separately from committed usage."""

        return sum(
            artifact.file_size_bytes
            for artifact in self.report_staging_directories()
        )

    def remove_staging_directory(self, directory_name: str) -> None:
        """Explicitly remove one inactive, exact manager staging directory."""

        match = _STAGING_DIRECTORY_PATTERN.fullmatch(directory_name)
        if match is None:
            raise ValueError("Invalid managed staging directory name.")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        directory = self.storage_dir / directory_name
        with self.storage_lock():
            if directory.is_symlink():
                raise DatasetStorageError(
                    "Managed staging path must not be a symbolic link."
                )
            if not directory.exists():
                return
            if not _is_real_directory(directory):
                raise FileNotFoundError(
                    "Managed staging directory does not exist."
                )

            lock_path = directory / _STAGING_LOCK_FILENAME
            lock_file = self._open_staging_lock(lock_path)
            try:
                try:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError as exc:
                    raise ActiveStagingDirectoryError(
                        "Cannot remove an active upload staging directory."
                    ) from exc

                shutil.rmtree(directory)
                _fsync_directory(self.storage_dir)
            finally:
                lock_file.close()

    @staticmethod
    def _open_staging_lock(lock_path: Path):
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags)
        except FileNotFoundError:
            descriptor = os.open(
                lock_path,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise DatasetStorageError(
                "Staging lock is not a regular file."
            )
        return os.fdopen(descriptor, "r+b")

    @classmethod
    def _staging_directory_is_active(cls, lock_path: Path) -> bool:
        lock_file = cls._open_staging_lock(lock_path)
        try:
            try:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                return True
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return False
        finally:
            lock_file.close()

    def remove_metadata_temporary_file(self, filename: str) -> None:
        """Explicitly remove one exact manager temp file after reconciliation."""

        match = _METADATA_TEMP_FILENAME_PATTERN.fullmatch(filename)
        if match is None:
            raise ValueError("Invalid metadata temporary filename.")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        path = self.storage_dir / filename
        with self.storage_lock():
            if path.is_symlink() or not _is_regular_file(path):
                raise FileNotFoundError(
                    "Metadata temporary file does not exist as a regular file."
                )
            path.unlink()
            _fsync_directory(self.storage_dir)

    def write_metadata_atomic(
        self,
        metadata: DatasetMetadata,
    ) -> Path:
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError("metadata must be a DatasetMetadata instance.")

        validated = DatasetMetadata.from_dict(metadata.to_dict())
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.resolve_metadata_path(validated.dataset_id)
        temporary_path: Path | None = None

        with self.storage_lock():
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{validated.dataset_id}.metadata-",
                    suffix=".tmp",
                    dir=self.storage_dir,
                )
                temporary_path = Path(temporary_name)
                with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                    json.dump(
                        validated.to_dict(),
                        target,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    target.write("\n")
                    target.flush()
                    os.fsync(target.fileno())

                os.replace(temporary_path, target_path)
                _fsync_directory(self.storage_dir)
            except Exception:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise

        return target_path

    def get_storage_usage_bytes(self) -> int:
        """Return committed original-file bytes, excluding metadata/locks."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with self.storage_lock():
            self._reap_abandoned_reservations()
            return self._calculate_committed_usage()

    def get_reserved_bytes(self) -> int:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with self.storage_lock():
            return self._calculate_reserved_bytes()

    def check_capacity_available(self, additional_bytes: int) -> int:
        """Return remaining capacity or raise if a reservation would exceed it."""

        _validate_size(additional_bytes)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with self.storage_lock():
            self._reap_abandoned_reservations()
            used = self._calculate_committed_usage()
            reserved = self._calculate_reserved_bytes()
            remaining = self.max_bytes - used - reserved
            if additional_bytes > remaining:
                raise StorageQuotaExceeded(
                    "Dataset storage capacity would be exceeded."
                )
            return remaining - additional_bytes

    def reserve_storage(self, size_bytes: int) -> StorageReservation:
        """Reserve quota atomically across processes until released."""

        _validate_size(size_bytes)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        with self.storage_lock():
            self._reap_abandoned_reservations()
            used = self._calculate_committed_usage()
            reserved = self._calculate_reserved_bytes()
            if used + reserved + size_bytes > self.max_bytes:
                raise StorageQuotaExceeded(
                    "Dataset storage capacity would be exceeded."
                )

            reservation_id = uuid.uuid4().hex
            reservation_path = (
                self.storage_dir / f".reservation-{reservation_id}.json"
            )
            reservation_file = reservation_path.open("x+b")
            try:
                fcntl.flock(
                    reservation_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                reservation_file.write(
                    json.dumps({"size_bytes": size_bytes}).encode("ascii")
                )
                reservation_file.flush()
                os.fsync(reservation_file.fileno())
                _fsync_directory(self.storage_dir)
            except Exception:
                reservation_file.close()
                reservation_path.unlink(missing_ok=True)
                raise

        return StorageReservation(
            self,
            reservation_path,
            reservation_file,
            size_bytes,
        )

    @contextmanager
    def storage_lock(self) -> Iterator[None]:
        """Acquire the local-filesystem process lock for storage mutations."""

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.storage_dir / ".storage.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _calculate_committed_usage(self) -> int:
        sizes_by_uuid: dict[str, int] = {}
        total_bytes = 0

        for path in self.storage_dir.iterdir():
            if _is_internal_storage_file(path.name):
                continue
            if not _is_regular_file(path):
                continue

            file_size = path.stat(follow_symlinks=False).st_size
            suffix = path.suffix.lower()
            if suffix in SUPPORTED_EXTENSIONS:
                try:
                    normalized_id = validate_dataset_id(path.stem)
                except InvalidDatasetIdError:
                    normalized_id = ""
                if normalized_id:
                    if normalized_id in sizes_by_uuid:
                        raise StorageIntegrityError(
                            "Multiple files use the same dataset UUID."
                        )
                    sizes_by_uuid[normalized_id] = file_size
                    continue
            total_bytes += file_size

        return total_bytes + sum(sizes_by_uuid.values())

    def _calculate_reserved_bytes(self) -> int:
        self._reap_abandoned_reservations()
        return sum(
            _read_reservation_size(path)
            for path in self._reservation_paths()
        )

    def _reap_abandoned_reservations(self) -> None:
        for reservation_path in self._reservation_paths():
            try:
                with reservation_path.open("rb") as reservation_file:
                    try:
                        fcntl.flock(
                            reservation_file.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except BlockingIOError:
                        _read_reservation_size(reservation_path)
                    else:
                        reservation_path.unlink(missing_ok=True)
            except FileNotFoundError:
                continue

    def _reservation_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.storage_dir.glob(".reservation-*.json")
            if _RESERVATION_FILENAME_PATTERN.fullmatch(path.name)
        )


def _read_reservation_size(path: Path) -> int:
    try:
        with path.open("r", encoding="ascii") as source:
            value = json.load(source)["size_bytes"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise DatasetStorageError(
            "Reservation record is malformed or inaccessible."
        ) from exc
    _validate_size(value)
    return value


def _validate_size(size_bytes: int) -> None:
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("Storage size must be a non-negative integer.")


def _is_internal_storage_file(filename: str) -> bool:
    return (
        filename == ".storage.lock"
        or _RESERVATION_FILENAME_PATTERN.fullmatch(filename) is not None
        or _METADATA_FILENAME_PATTERN.fullmatch(filename) is not None
        or _METADATA_TEMP_FILENAME_PATTERN.fullmatch(filename) is not None
    )


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _staging_directory_size(directory: Path) -> int:
    total_bytes = 0
    for current_root, subdirectories, filenames in os.walk(
        directory,
        followlinks=False,
    ):
        root = Path(current_root)
        subdirectories[:] = [
            name
            for name in subdirectories
            if not (root / name).is_symlink()
        ]
        for filename in filenames:
            path = root / filename
            if (
                root == directory
                and filename == _STAGING_LOCK_FILENAME
            ):
                continue
            try:
                file_stat = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(file_stat.st_mode):
                total_bytes += file_stat.st_size
    return total_bytes


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
