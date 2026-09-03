"""Forensic-safe journals, CSV reporting, and console summaries."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, TextIO

from .queries import QuerySpec
from .ranking import FusedCandidate
from .scanner import ManifestEntry


CSV_FIELDS = (
    "FilePath",
    "OriginalQuery",
    "MatchedQuery",
    "SigLIP2Score",
    "CLIPCos",
    "SigLIP2Rank",
    "CLIPRank",
    "ModelsMatched",
    "FinalRank",
    "FusionScore",
    "SigLIP2Model",
    "SigLIP2Revision",
    "CLIPModel",
    "CLIPRevision",
    "ScanTimestamp",
)


class ErrorJournal:
    """Incremental JSONL journal stored outside evidence."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stream: TextIO | None = None

    def __enter__(self) -> "ErrorJournal":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8", newline="\n")
        return self

    def record(self, *, phase: str, model_id: str | None, error: Any) -> None:
        if self._stream is None:
            raise RuntimeError("error journal is not open")
        payload = asdict(error) if is_dataclass(error) else {"message": str(error)}
        payload.update({"phase": phase, "model_id": model_id})
        self._stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        self._stream.write("\n")
        self._stream.flush()

    def __exit__(self, *_: object) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def display_path(entry: ManifestEntry, display_root: str | None) -> str:
    if not display_root:
        return entry.source_path
    return str(PureWindowsPath(display_root).joinpath(*Path(entry.relative_path).parts))


def build_rows(
    candidates: Iterable[FusedCandidate],
    entries: Mapping[int, ManifestEntry],
    queries: Mapping[str, QuerySpec],
    *,
    display_root: str | None,
    model_metadata: Mapping[str, Mapping[str, str | None]],
    scan_timestamp: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        entry = entries[candidate.file_id]
        query = queries[candidate.query_id]
        siglip = model_metadata["siglip2"]
        clip = model_metadata["clip"]
        rows.append(
            {
                "FilePath": display_path(entry, display_root),
                "OriginalQuery": query.original_query,
                "MatchedQuery": query.matched_query,
                "SigLIP2Score": candidate.siglip2_score,
                "CLIPCos": candidate.clip_cos,
                "SigLIP2Rank": candidate.siglip2_rank,
                "CLIPRank": candidate.clip_rank,
                "ModelsMatched": candidate.models_matched,
                "FinalRank": candidate.final_rank,
                "FusionScore": candidate.fusion_score,
                "SigLIP2Model": siglip["name"],
                "SigLIP2Revision": siglip["revision"],
                "CLIPModel": clip["name"],
                "CLIPRevision": clip["revision"],
                "ScanTimestamp": scan_timestamp,
            }
        )
    return rows


def _format_row(row: Mapping[str, Any]) -> dict[str, Any]:
    formatted = dict(row)
    for field in ("SigLIP2Score", "CLIPCos"):
        value = formatted[field]
        formatted[field] = "" if value is None else f"{value:.6f}"
    formatted["FusionScore"] = f'{formatted["FusionScore"]:.12f}'
    for field in ("SigLIP2Rank", "CLIPRank"):
        if formatted[field] is None:
            formatted[field] = ""
    return formatted


def write_csv(rows: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    """Atomically write the final report outside evidence."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(_format_row(row))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

