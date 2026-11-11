import hashlib
import json
import multiprocessing
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.services.dataset_storage import (
    ActiveStagingDirectoryError,
    DatasetIntegrityError,
    DatasetMetadata,
    DatasetStorageManager,
    InvalidDatasetIdError,
    InvalidMetadataError,
    StorageIntegrityError,
    StorageQuotaExceeded,
    validate_dataset_id,
)


def _metadata(dataset_id: str | None = None) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id=dataset_id or "12345678123456781234567812345678",
        original_filename="source.csv",
        file_extension=".csv",
        format="csv",
        created_at=datetime(2026, 11, 10, tzinfo=UTC),
        file_size_bytes=12,
        sha256="a" * 64,
        row_count=3,
        column_count=2,
        storage_status="ready",
    )


def _reserve_in_process(
    storage_dir: str,
    max_bytes: int,
    reservation_bytes: int,
    barrier,
    results,
) -> None:
    manager = DatasetStorageManager(storage_dir, max_bytes=max_bytes)
    barrier.wait(timeout=10)
    try:
        reservation = manager.reserve_storage(reservation_bytes)
    except StorageQuotaExceeded:
        results.put("rejected")
    else:
        results.put("reserved")
        time.sleep(0.3)
        reservation.release()


def _abandon_reservation_in_process(storage_dir: str) -> None:
    manager = DatasetStorageManager(storage_dir, max_bytes=10)
    manager.reserve_storage(7)
    os._exit(0)


def _probe_storage_lock(storage_dir: str, results) -> None:
    manager = DatasetStorageManager(storage_dir, max_bytes=10)
    with manager.storage_lock():
        results.put("acquired")


def _hold_staging_directory(storage_dir: str, ready, finish) -> None:
    manager = DatasetStorageManager(storage_dir, max_bytes=10)
    with manager.upload_staging_directory() as directory:
        (directory / "partial.csv").write_bytes(b"live upload")
        ready.put(directory.name)
        if not finish.wait(timeout=10):
            raise TimeoutError("Staging test process was not released.")


def test_dataset_uuid_validation_accepts_uuid_forms_and_rejects_traversal():
    dataset_id = "12345678123456781234567812345678"

    assert validate_dataset_id(dataset_id) == dataset_id
    assert validate_dataset_id(
        "12345678-1234-5678-1234-567812345678"
    ) == dataset_id

    for invalid_id in ("../dataset", "x" * 32, "", "{" + dataset_id + "}"):
        with pytest.raises(InvalidDatasetIdError):
            validate_dataset_id(invalid_id)


def test_file_and_metadata_paths_are_resolved_from_validated_uuid(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = "12345678123456781234567812345678"

    assert manager.resolve_dataset_file_path(dataset_id, ".CSV") == (
        tmp_path / f"{dataset_id}.csv"
    )
    assert manager.resolve_metadata_path(dataset_id) == (
        tmp_path / f"{dataset_id}.metadata.json"
    )

    with pytest.raises(InvalidDatasetIdError):
        manager.resolve_metadata_path("../outside")
    with pytest.raises(ValueError):
        manager.resolve_dataset_file_path(dataset_id, "../outside")


def test_metadata_round_trip_and_safe_file_resolution(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    metadata = _metadata()
    file_path = manager.resolve_dataset_file_path(
        metadata.dataset_id,
        metadata.file_extension,
    )
    file_path.write_bytes(b"original upload")

    metadata_path = manager.write_metadata_atomic(metadata)

    assert metadata_path == tmp_path / f"{metadata.dataset_id}.metadata.json"
    assert manager.read_metadata(metadata.dataset_id) == metadata
    assert manager.find_dataset_file(metadata.dataset_id) == file_path
    assert str(tmp_path) not in json.dumps(metadata.to_dict())
    with pytest.raises(DatasetIntegrityError, match="size"):
        manager.verify_dataset_integrity(metadata.dataset_id)


def test_integrity_verification_checks_file_size_hash_uuid_and_extension(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    content = b"immutable original dataset"
    metadata = replace(
        _metadata(),
        file_size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    manager.resolve_dataset_file_path(
        dataset_id,
        metadata.file_extension,
    ).write_bytes(content)
    manager.write_metadata_atomic(metadata)

    result = manager.verify_dataset_integrity(dataset_id)

    assert result.dataset_id == dataset_id
    assert result.file_extension == ".csv"
    assert result.file_size_bytes == len(content)
    assert result.sha256 == metadata.sha256


@pytest.mark.parametrize(
    ("file_action", "message"),
    [
        ("missing", "Dataset file does not exist."),
        ("wrong_size", "Dataset file size does not match metadata."),
        ("wrong_hash", "Dataset file SHA-256 does not match metadata."),
        ("symlink", "Dataset file must not be a symbolic link."),
        ("wrong_extension", "Dataset file extension does not match metadata."),
        ("duplicate", "Multiple files use the same dataset UUID."),
    ],
)
def test_integrity_verification_rejects_file_inconsistency(
    tmp_path,
    file_action,
    message,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    content = b"verified file"
    metadata = replace(
        _metadata(),
        file_size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    expected_path = manager.resolve_dataset_file_path(dataset_id, ".csv")
    if file_action == "wrong_size":
        expected_path.write_bytes(content + b"x")
    elif file_action == "wrong_hash":
        expected_path.write_bytes(b"other content")
    elif file_action == "symlink":
        other_file = tmp_path / "outside.csv"
        other_file.write_bytes(content)
        expected_path.symlink_to(other_file)
    elif file_action == "wrong_extension":
        manager.resolve_dataset_file_path(dataset_id, ".json").write_bytes(
            content
        )
    elif file_action == "duplicate":
        expected_path.write_bytes(content)
        manager.resolve_dataset_file_path(dataset_id, ".json").write_bytes(
            content
        )
    elif file_action != "missing":
        raise AssertionError(f"Unknown action {file_action}")

    manager.write_metadata_atomic(metadata)

    with pytest.raises(DatasetIntegrityError, match=message):
        manager.verify_dataset_integrity(dataset_id)

    if file_action == "missing":
        assert not expected_path.exists()


def test_integrity_verification_rejects_metadata_uuid_mismatch(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    first_id = _metadata().dataset_id
    second_id = "87654321876543218765432187654321"
    content = b"same file bytes"
    metadata = replace(
        _metadata(second_id),
        file_size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    manager.resolve_dataset_file_path(first_id, ".csv").write_bytes(content)

    with pytest.raises(DatasetIntegrityError, match="UUID"):
        manager.verify_dataset_integrity(first_id, metadata)


def test_metadata_replacement_is_atomic_in_the_same_directory(
    tmp_path,
    monkeypatch,
):
    import backend.app.services.dataset_storage as storage_module

    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    metadata = _metadata()
    target_path = manager.write_metadata_atomic(metadata)
    old_contents = target_path.read_bytes()
    replacement = DatasetMetadata(
        **{
            **metadata.__dict__,
            "row_count": 8,
        }
    )
    real_replace = storage_module.os.replace

    def checked_replace(source, destination):
        source_path = storage_module.Path(source)
        destination_path = storage_module.Path(destination)
        assert source_path.parent == tmp_path
        assert source_path != destination_path
        assert destination_path.read_bytes() == old_contents
        return real_replace(source, destination)

    monkeypatch.setattr(storage_module.os, "replace", checked_replace)

    manager.write_metadata_atomic(replacement)

    assert manager.read_metadata(metadata.dataset_id) == replacement
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        b"{not-json",
        json.dumps({"dataset_id": "different"}).encode(),
        json.dumps({
            **_metadata().to_dict(),
            "file_extension": "../outside",
        }).encode(),
        json.dumps({
            **_metadata().to_dict(),
            "dataset_id": "../outside",
        }).encode(),
        json.dumps({
            **_metadata().to_dict(),
            "file_extension": True,
        }).encode(),
        json.dumps({
            **_metadata().to_dict(),
            "original_filename": "../outside.csv",
        }).encode(),
        json.dumps({
            **_metadata().to_dict(),
            "sha256": "not-a-digest",
        }).encode(),
        json.dumps({
            **_metadata().to_dict(),
            "row_count": True,
        }).encode(),
    ],
)
def test_invalid_or_corrupted_metadata_is_rejected(
    tmp_path,
    invalid_metadata,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    manager.resolve_metadata_path(dataset_id).write_bytes(invalid_metadata)

    with pytest.raises(InvalidMetadataError):
        manager.read_metadata(dataset_id)


def test_metadata_schema_rejects_invalid_values():
    for changes in (
        {"created_at": datetime(2026, 11, 10)},
        {"file_size_bytes": -1},
        {"row_count": True},
        {"storage_status": "unknown"},
        {"format": "json"},
    ):
        values = {**_metadata().__dict__, **changes}
        with pytest.raises(InvalidMetadataError):
            DatasetMetadata(**values)


def test_storage_usage_counts_legacy_files_but_not_metadata_or_locks(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    (tmp_path / f"{dataset_id}.csv").write_bytes(b"legacy")
    manager.write_metadata_atomic(_metadata())
    orphan_id = "87654321876543218765432187654321"
    (tmp_path / f"{orphan_id}.metadata.json").write_bytes(b"orphan metadata")
    (tmp_path / "report.metadata.json").write_bytes(b"other data")
    (tmp_path / ".unrelated.metadata-123.tmp").write_bytes(b"temporary")
    (tmp_path / f".{dataset_id}.metadata-abcdefgh.tmp").write_bytes(
        b"manager temp"
    )
    (tmp_path / ".reservation-not-a-uuid.json").write_bytes(b"other data")
    (tmp_path / ".storage.lock").write_bytes(b"lock bytes")

    expected = (
        len(b"legacy")
        + len(b"other data")
        + len(b"temporary")
        + len(b"other data")
    )
    assert manager.get_storage_usage_bytes() == expected
    assert (tmp_path / ".reservation-not-a-uuid.json").exists()


def test_metadata_temp_recovery_reports_exact_files_and_removes_only_explicit(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    original_path = manager.resolve_dataset_file_path(dataset_id, ".csv")
    original_path.write_bytes(b"keep this dataset")
    generated_temp = tmp_path / f".{dataset_id}.metadata-abcdefgh.tmp"
    generated_temp.write_bytes(b"incomplete sidecar")
    similar_temp = tmp_path / ".outside.metadata-not-generated!.tmp"
    similar_temp.write_bytes(b"unrelated file")

    artifacts = manager.report_metadata_temporary_files()

    assert [(item.filename, item.dataset_id) for item in artifacts] == [
        (generated_temp.name, dataset_id),
    ]
    assert generated_temp.exists()
    assert original_path.read_bytes() == b"keep this dataset"

    manager.remove_metadata_temporary_file(generated_temp.name)

    assert not generated_temp.exists()
    assert similar_temp.exists()
    assert original_path.read_bytes() == b"keep this dataset"
    with pytest.raises(ValueError):
        manager.remove_metadata_temporary_file("../outside")


def test_staging_report_ignores_missing_and_malformed_directories(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    (tmp_path / ".datapilot-staging-not-a-uuid").mkdir()
    (tmp_path / ".staging-malformed").mkdir()

    assert manager.report_staging_directories() == ()
    assert manager.get_staging_usage_bytes() == 0


def test_legacy_staging_directory_is_reported_as_unknown_not_cleanable(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    directory = tmp_path / ".staging-a1b2c3d4"
    directory.mkdir()
    (directory / "upload.csv").write_bytes(b"legacy staging")

    artifact, = manager.report_staging_directories()

    assert artifact.directory_name == directory.name
    assert artifact.active is None
    assert artifact.file_size_bytes == len(b"legacy staging")
    assert manager.get_staging_usage_bytes() == len(b"legacy staging")
    with pytest.raises(ValueError):
        manager.remove_staging_directory(directory.name)
    assert directory.exists()


def test_active_staging_directory_is_reported_but_not_removed(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    entered = threading.Event()
    finish = threading.Event()

    def active_upload():
        with manager.upload_staging_directory() as directory:
            (directory / "upload.csv").write_bytes(b"active upload")
            entered.set()
            assert finish.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        upload = executor.submit(active_upload)
        assert entered.wait(timeout=5)
        artifact, = manager.report_staging_directories()
        assert artifact.active
        assert artifact.file_size_bytes == len(b"active upload")
        with pytest.raises(ActiveStagingDirectoryError):
            manager.remove_staging_directory(artifact.directory_name)
        assert Path(manager.storage_dir / artifact.directory_name).exists()
        finish.set()
        upload.result(timeout=5)

    assert manager.report_staging_directories() == ()


def test_recovery_does_not_remove_staging_held_by_another_process(tmp_path):
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    finish = context.Event()
    process = context.Process(
        target=_hold_staging_directory,
        args=(str(tmp_path), ready, finish),
    )
    process.start()
    directory_name = ready.get(timeout=5)
    manager = DatasetStorageManager(tmp_path, max_bytes=10)
    try:
        artifact, = manager.report_staging_directories()
        assert artifact.directory_name == directory_name
        assert artifact.active
        with pytest.raises(ActiveStagingDirectoryError):
            manager.remove_staging_directory(directory_name)
    finally:
        finish.set()
        process.join(timeout=10)
    assert process.exitcode == 0
    assert manager.report_staging_directories() == ()


def test_abandoned_staging_directory_is_reported_and_explicitly_removed(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    committed_path = manager.resolve_dataset_file_path(dataset_id, ".csv")
    committed_path.write_bytes(b"committed original")
    staging_id = "12345678123456781234567812345678"
    directory = tmp_path / f".datapilot-staging-{staging_id}"
    directory.mkdir()
    (directory / ".active.lock").write_bytes(b"")
    (directory / "nested").mkdir()
    (directory / "nested" / "partial-upload.csv").write_bytes(b"staged data")

    artifact, = manager.report_staging_directories()
    assert artifact.directory_name == directory.name
    assert artifact.staging_id == staging_id
    assert not artifact.active
    assert artifact.file_size_bytes == len(b"staged data")
    assert manager.get_staging_usage_bytes() == len(b"staged data")

    manager.remove_staging_directory(artifact.directory_name)
    manager.remove_staging_directory(artifact.directory_name)
    assert not directory.exists()
    assert committed_path.read_bytes() == b"committed original"


def test_staging_cleanup_failure_is_visible_and_retryable(
    tmp_path,
    monkeypatch,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    staging_id = "12345678123456781234567812345678"
    directory = tmp_path / f".datapilot-staging-{staging_id}"
    directory.mkdir()
    (directory / ".active.lock").write_bytes(b"")
    (directory / "partial.csv").write_bytes(b"preserve until retry")
    real_rmtree = shutil.rmtree

    def fail_cleanup(path, *args, **kwargs):
        if Path(path) == directory:
            raise PermissionError("simulated staging cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("backend.app.services.dataset_storage.shutil.rmtree", fail_cleanup)
    with pytest.raises(PermissionError, match="cleanup failure"):
        manager.remove_staging_directory(directory.name)
    assert directory.joinpath("partial.csv").read_bytes() == (
        b"preserve until retry"
    )
    assert manager.report_staging_directories()[0].file_size_bytes == len(
        b"preserve until retry"
    )

    monkeypatch.setattr("backend.app.services.dataset_storage.shutil.rmtree", real_rmtree)
    manager.remove_staging_directory(directory.name)
    assert not directory.exists()


def test_staging_bytes_are_separate_from_committed_usage_and_reservations(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    original = b"committed dataset"
    manager.resolve_dataset_file_path(dataset_id, ".csv").write_bytes(original)
    staging_id = "87654321876543218765432187654321"
    directory = tmp_path / f".datapilot-staging-{staging_id}"
    directory.mkdir()
    (directory / ".active.lock").write_bytes(b"lock")
    (directory / "upload.csv").write_bytes(b"temporary bytes")
    reservation = manager.reserve_storage(5)
    try:
        assert manager.get_storage_usage_bytes() == len(original)
        assert manager.get_reserved_bytes() == 5
        assert manager.get_staging_usage_bytes() == len(b"temporary bytes")
        assert manager.report_staging_directories()[0].file_size_bytes == (
            len(b"temporary bytes")
        )
    finally:
        reservation.release()


def test_stale_reservation_recovery_preserves_staging_artifacts(tmp_path):
    process = multiprocessing.get_context("fork").Process(
        target=_abandon_reservation_in_process,
        args=(str(tmp_path),),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0

    manager = DatasetStorageManager(tmp_path, max_bytes=10)
    staging_id = "12345678123456781234567812345678"
    directory = tmp_path / f".datapilot-staging-{staging_id}"
    directory.mkdir()
    (directory / ".active.lock").write_bytes(b"")
    (directory / "partial.csv").write_bytes(b"recover separately")

    assert manager.get_reserved_bytes() == 0
    artifact, = manager.report_staging_directories()
    assert artifact.directory_name == directory.name
    assert artifact.file_size_bytes == len(b"recover separately")
    assert directory.joinpath("partial.csv").exists()


def test_missing_metadata_report_identifies_files_without_migrating_them(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    content = b"legacy original"
    original_path = manager.resolve_dataset_file_path(dataset_id, ".csv")
    original_path.write_bytes(content)

    artifacts = manager.report_files_without_metadata()

    assert len(artifacts) == 1
    assert artifacts[0].dataset_id == dataset_id
    assert artifacts[0].file_extension == ".csv"
    assert artifacts[0].file_size_bytes == len(content)
    assert original_path.read_bytes() == content
    assert not manager.resolve_metadata_path(dataset_id).exists()


def test_duplicate_files_for_one_uuid_are_not_double_counted(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=100)
    dataset_id = _metadata().dataset_id
    (tmp_path / f"{dataset_id}.csv").write_bytes(b"one")
    (tmp_path / f"{dataset_id}.json").write_bytes(b"two")

    with pytest.raises(StorageIntegrityError):
        manager.get_storage_usage_bytes()


def test_quota_boundary_includes_legacy_files_and_reserved_bytes(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=10)
    (tmp_path / "legacy.csv").write_bytes(b"1234")

    assert manager.check_capacity_available(6) == 0
    with pytest.raises(StorageQuotaExceeded):
        manager.check_capacity_available(7)

    with manager.reserve_storage(5):
        assert manager.get_reserved_bytes() == 5
        with pytest.raises(StorageQuotaExceeded):
            manager.reserve_storage(2)

    assert manager.get_reserved_bytes() == 0
    assert manager.get_storage_usage_bytes() == 4


def test_failed_reservation_is_released_without_touching_existing_files(
    tmp_path,
):
    manager = DatasetStorageManager(tmp_path, max_bytes=8)
    original_path = tmp_path / "legacy.csv"
    original_path.write_bytes(b"legacy")

    with pytest.raises(StorageQuotaExceeded):
        manager.reserve_storage(4)

    with pytest.raises(RuntimeError), manager.reserve_storage(2):
        raise RuntimeError("simulated upload failure")

    assert manager.get_reserved_bytes() == 0
    assert original_path.read_bytes() == b"legacy"
    assert not list(tmp_path.glob(".reservation-*.json"))


def test_abandoned_process_reservation_is_reclaimed_safely(tmp_path):
    process = multiprocessing.get_context("fork").Process(
        target=_abandon_reservation_in_process,
        args=(str(tmp_path),),
    )
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    manager = DatasetStorageManager(tmp_path, max_bytes=10)

    assert manager.get_reserved_bytes() == 0
    assert not list(tmp_path.glob(".reservation-*.json"))
    assert manager.get_reserved_bytes() == 0
    assert not list(tmp_path.glob(".reservation-*.json"))


def test_concurrent_processes_cannot_reserve_the_same_capacity(tmp_path):
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_reserve_in_process,
            args=(str(tmp_path), 10, 7, barrier, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results.get(timeout=2) for _ in processes) == [
        "rejected",
        "reserved",
    ]


def test_storage_lock_is_released_when_locked_operation_raises(tmp_path):
    manager = DatasetStorageManager(tmp_path, max_bytes=10)
    with pytest.raises(RuntimeError), manager.storage_lock():
        raise RuntimeError("simulated failure")

    context = multiprocessing.get_context("fork")
    results = context.Queue()
    process = context.Process(
        target=_probe_storage_lock,
        args=(str(tmp_path), results),
    )
    process.start()
    process.join(timeout=5)

    assert process.exitcode == 0
    assert results.get(timeout=2) == "acquired"


def test_reservation_release_unlink_failure_preserves_active_reservation(
    tmp_path,
    monkeypatch,
):
    from pathlib import Path

    manager = DatasetStorageManager(tmp_path, max_bytes=10)
    reservation = manager.reserve_storage(7)
    reservation_path = reservation._reservation_path
    original_unlink = Path.unlink

    def fail_reservation_unlink(path, *args, **kwargs):
        if path == reservation_path:
            raise PermissionError("simulated unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_reservation_unlink)

    with pytest.raises(PermissionError):
        reservation.release()

    assert reservation_path.exists()
    assert manager.get_reserved_bytes() == 7

    monkeypatch.setattr(Path, "unlink", original_unlink)
    reservation.release()

    assert manager.get_reserved_bytes() == 0
    assert not reservation_path.exists()


def test_stale_reservation_recovery_failure_keeps_accounting_record(
    tmp_path,
    monkeypatch,
):
    from pathlib import Path

    process = multiprocessing.get_context("fork").Process(
        target=_abandon_reservation_in_process,
        args=(str(tmp_path),),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0

    reservation_path = next(tmp_path.glob(".reservation-*.json"))
    original_unlink = Path.unlink

    def fail_reservation_unlink(path, *args, **kwargs):
        if path == reservation_path:
            raise PermissionError("simulated recovery unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_reservation_unlink)
    manager = DatasetStorageManager(tmp_path, max_bytes=10)

    with pytest.raises(PermissionError):
        manager.get_reserved_bytes()
    assert reservation_path.exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert manager.get_reserved_bytes() == 0
    assert not reservation_path.exists()


@pytest.mark.parametrize(
    ("committed_size", "requested_bytes", "allowed"),
    [
        (1_073_741_823, 0, True),
        (1_073_741_823, 1, True),
        (1_073_741_824, 0, True),
        (1_073_741_824, 1, False),
        (1_073_741_825, 0, False),
    ],
    ids=[
        "one-below",
        "one-below-reserve-to-exact",
        "exact",
        "one-over-exact",
        "already-over",
    ],
)
def test_default_one_gibibyte_quota_boundaries(
    tmp_path,
    committed_size,
    requested_bytes,
    allowed,
    monkeypatch,
):
    monkeypatch.delenv("DATASET_STORAGE_MAX_BYTES", raising=False)
    limit = Settings(_env_file=None).dataset_storage_max_bytes
    assert limit == 1_073_741_824
    manager = DatasetStorageManager(tmp_path, max_bytes=limit)
    with (tmp_path / "legacy.csv").open("wb") as dataset_file:
        dataset_file.truncate(committed_size)

    if allowed:
        assert manager.check_capacity_available(requested_bytes) == (
            limit - committed_size - requested_bytes
        )
        with manager.reserve_storage(requested_bytes):
            assert manager.get_reserved_bytes() == requested_bytes
    else:
        with pytest.raises(StorageQuotaExceeded):
            manager.reserve_storage(requested_bytes)


def test_default_quota_counts_legacy_files_and_active_reservations(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("DATASET_STORAGE_MAX_BYTES", raising=False)
    limit = Settings(_env_file=None).dataset_storage_max_bytes
    manager = DatasetStorageManager(tmp_path, max_bytes=limit)
    with (tmp_path / "legacy.csv").open("wb") as dataset_file:
        dataset_file.truncate(limit - 10)

    with manager.reserve_storage(5):
        assert manager.get_reserved_bytes() == 5
        with pytest.raises(StorageQuotaExceeded):
            manager.reserve_storage(6)
        with manager.reserve_storage(5):
            assert manager.get_reserved_bytes() == 10


def test_default_quota_concurrent_processes_compete_for_remaining_bytes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("DATASET_STORAGE_MAX_BYTES", raising=False)
    limit = Settings(_env_file=None).dataset_storage_max_bytes
    with (tmp_path / "legacy.csv").open("wb") as dataset_file:
        dataset_file.truncate(limit - 7)
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_reserve_in_process,
            args=(str(tmp_path), limit, 7, barrier, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results.get(timeout=2) for _ in processes) == [
        "rejected",
        "reserved",
    ]


def test_storage_capacity_setting_has_one_gibibyte_default(monkeypatch):
    monkeypatch.delenv("DATASET_STORAGE_MAX_BYTES", raising=False)

    settings = Settings(_env_file=None)

    assert settings.dataset_storage_max_bytes == 1_073_741_824


def test_storage_capacity_setting_reads_environment_override(monkeypatch):
    monkeypatch.setenv("DATASET_STORAGE_MAX_BYTES", "2048")

    settings = Settings(_env_file=None)

    assert settings.dataset_storage_max_bytes == 2048
