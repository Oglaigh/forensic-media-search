"""Durable, non-chained build commit journal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .metadata import hash_array_rows, hash_state_rows, require_keys


def truncate_incomplete_tail(path: Path) -> None:
    """Discard bytes after the last durable newline before resuming appends."""
    journal = Path(path)
    if not journal.exists():
        return
    data = journal.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    last_newline = data.rfind(b"\n")
    length = last_newline + 1 if last_newline >= 0 else 0
    with journal.open("r+b") as stream:
        stream.truncate(length)
        stream.flush()
        os.fsync(stream.fileno())


def append_commit(path: Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_commits(path: Path) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    if not Path(path).exists():
        return commits
    data = Path(path).read_bytes()
    lines = data.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        if not raw.endswith((b"\n", b"\r")):
            # A crash-truncated final record is not a commit.
            if index == len(lines) - 1:
                break
            raise ValueError("truncated build journal record")
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid build journal record {index + 1}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"build journal record {index + 1} is not an object")
        require_keys(
            value,
            ("sequence", "model_id", "start", "end", "valid_count", "error_count", "embeddings_sha256", "rows_sha256", "errors_offset"),
            f"build journal record {index + 1}",
        )
        commits.append(value)
    return commits


def validate_model_commits(
    commits: list[dict[str, Any]], model_id: str, embeddings: Any, rows: Any
) -> int:
    expected_start = 0
    expected_sequence = 0
    for record in commits:
        if record["model_id"] != model_id:
            continue
        if int(record["sequence"]) != expected_sequence:
            raise ValueError(f"non-contiguous journal sequence for {model_id}")
        start, end = int(record["start"]), int(record["end"])
        if start != expected_start or end <= start or end > len(rows):
            raise ValueError(f"invalid or non-contiguous journal range for {model_id}")
        if hash_array_rows(embeddings, start, end) != record["embeddings_sha256"]:
            raise ValueError(f"committed embedding range hash mismatch for {model_id}")
        if hash_state_rows(rows, start, end) != record["rows_sha256"]:
            raise ValueError(f"committed row-state range hash mismatch for {model_id}")
        expected_start = end
        expected_sequence += 1
    return expected_start
