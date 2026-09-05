"""Atomic metadata and hashing helpers for persistent indexes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .format import INDEX_FORMAT, INDEX_VERSION, STATE_BUILDING, STATE_COMPLETE


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fsync_path(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers/fsync.
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())


def atomic_write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        if os.name != "nt":
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def load_index_metadata(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid index metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("index metadata must be a JSON object")
    if value.get("index_format") != INDEX_FORMAT:
        raise ValueError("unsupported index format")
    if value.get("index_version") != INDEX_VERSION:
        raise ValueError("unsupported index version")
    if value.get("state") not in {STATE_BUILDING, STATE_COMPLETE}:
        raise ValueError("invalid index state")
    return value


def hash_array_rows(array: Any, start: int, end: int) -> str:
    import numpy as np

    view = np.ascontiguousarray(array[start:end])
    return hashlib.sha256(view.tobytes(order="C")).hexdigest()


def hash_state_rows(array: Any, start: int, end: int) -> str:
    return hashlib.sha256(bytes(array[start:end])).hexdigest()


def file_seal(path: Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "file_id": int(stat.st_ino),
    }


def seal_matches(path: Path, seal: dict[str, Any]) -> bool:
    return file_seal(path) == {
        "size": int(seal.get("size", -1)),
        "mtime_ns": int(seal.get("mtime_ns", -1)),
        "ctime_ns": int(seal.get("ctime_ns", -1)),
        "file_id": int(seal.get("file_id", -1)),
    }


def require_keys(value: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")
