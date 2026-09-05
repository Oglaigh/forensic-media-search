"""Fail-closed structural and full integrity validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .format import (
    MODEL_IDS, ROW_DECODE_ERROR, ROW_PENDING, ROW_VALID, STATE_COMPLETE, IndexPaths,
)
from .metadata import load_index_metadata, require_keys, seal_matches, sha256_file
from ..scanner import read_manifest
from .journal import read_commits, validate_model_commits


@dataclass(frozen=True, slots=True)
class IndexVerificationResult:
    index_path: Path
    state: str
    image_count: int
    full: bool
    models: dict[str, dict[str, int]]


def _validate_sealed_file(
    path: Path, *, size: int, digest: str, seal: dict[str, Any], full: bool, label: str
) -> None:
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"{label} is missing or has the wrong size")
    # Routine SEARCH avoids multi-GB hashing. A changed fast identity triggers
    # a full hash once; unchanged identities use the COMPLETE seal.
    if full or not seal_matches(path, seal):
        if sha256_file(path) != digest:
            raise ValueError(f"{label} hash mismatch")


def _open_model_files(
    paths: IndexPaths, metadata: dict[str, Any], model_id: str
) -> tuple[Any, Any, dict[str, Any]]:
    models = metadata.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(model_id), dict):
        raise ValueError(f"missing metadata for model {model_id}")
    model_meta = models[model_id]
    require_keys(
        model_meta,
        ("dimension", "dtype", "layout", "committed_prefix", "valid_count", "error_count"),
        f"model metadata {model_id}",
    )
    if model_meta["dtype"] != "float32" or model_meta["layout"] != "C":
        raise ValueError(f"unsupported embedding storage contract for {model_id}")
    if not isinstance(model_meta.get("identity"), dict):
        raise ValueError(f"missing model identity for {model_id}")
    embeddings_path = paths.embeddings(model_id)
    rows_path = paths.rows(model_id)
    if not embeddings_path.is_file() or not rows_path.is_file():
        raise ValueError(f"missing index data for {model_id}")
    try:
        embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid embeddings.npy for {model_id}: {error}") from error
    count = int(metadata["image_count"])
    dimension = int(model_meta["dimension"])
    if embeddings.shape != (count, dimension):
        raise ValueError(f"embedding shape mismatch for {model_id}")
    if embeddings.dtype != np.dtype("<f4") or not embeddings.flags.c_contiguous:
        raise ValueError(f"embedding dtype/layout mismatch for {model_id}")
    if rows_path.stat().st_size != count:
        raise ValueError(f"row-state size mismatch for {model_id}")
    rows = np.memmap(rows_path, dtype=np.uint8, mode="r", shape=(count,))
    states = set(int(value) for value in np.unique(rows))
    if not states.issubset({ROW_PENDING, ROW_VALID, ROW_DECODE_ERROR}):
        raise ValueError(f"unknown row state for {model_id}")
    return embeddings, rows, model_meta


def verify_index(index_path: Path, *, full: bool = False, require_complete: bool = False) -> IndexVerificationResult:
    paths = IndexPaths(Path(index_path).resolve(strict=True))
    metadata = load_index_metadata(paths.metadata)
    require_keys(
        metadata,
        ("image_count", "manifest_sha256", "manifest_size", "row_mapping", "models"),
        "index metadata",
    )
    if require_complete and metadata["state"] != STATE_COMPLETE:
        raise ValueError("SEARCH requires a COMPLETE index")
    if metadata["row_mapping"] != "row_equals_manifest_ordinal":
        raise ValueError("unsupported index row mapping")
    if not paths.manifest.is_file():
        raise ValueError("index manifest is missing")
    if paths.manifest.stat().st_size != int(metadata["manifest_size"]):
        raise ValueError("index manifest size mismatch")
    if sha256_file(paths.manifest) != metadata["manifest_sha256"]:
        raise ValueError("index manifest hash mismatch")
    if sum(1 for _ in read_manifest(paths.manifest)) != int(metadata["image_count"]):
        raise ValueError("index manifest row count mismatch")

    summaries: dict[str, dict[str, int]] = {}
    commits = read_commits(paths.journal)
    unknown_models = sorted({str(item["model_id"]) for item in commits}.difference(MODEL_IDS))
    if unknown_models:
        raise ValueError(f"build journal contains unknown models: {unknown_models}")
    for model_id in MODEL_IDS:
        model_files = metadata["models"].get(model_id, {}).get("files", {})
        for kind in ("embeddings", "rows", "errors"):
            _validate_sealed_file(
                getattr(paths, kind)(model_id),
                size=int(model_files.get(f"{kind}_size", -1)),
                digest=str(model_files.get(f"{kind}_sha256", "")),
                seal=model_files.get(f"{kind}_seal", {}),
                full=full,
                label=f"{model_id} {kind}",
            )
        embeddings, rows, model_meta = _open_model_files(paths, metadata, model_id)
        valid_count = int(np.count_nonzero(rows == ROW_VALID))
        error_count = int(np.count_nonzero(rows == ROW_DECODE_ERROR))
        pending_count = int(np.count_nonzero(rows == ROW_PENDING))
        if int(model_meta["valid_count"]) != valid_count:
            raise ValueError(f"valid row count mismatch for {model_id}")
        if int(model_meta["error_count"]) != error_count:
            raise ValueError(f"error row count mismatch for {model_id}")
        with paths.errors(model_id).open("r", encoding="utf-8") as stream:
            if sum(1 for line in stream if line.strip()) != error_count:
                raise ValueError(f"decode error journal count mismatch for {model_id}")
        if metadata["state"] == STATE_COMPLETE:
            if pending_count or int(model_meta["committed_prefix"]) != len(rows):
                raise ValueError(f"COMPLETE index has uncommitted rows for {model_id}")
        if full:
            committed_prefix = validate_model_commits(commits, model_id, embeddings, rows)
            if committed_prefix != int(model_meta["committed_prefix"]):
                raise ValueError(f"journal/metadata committed prefix mismatch for {model_id}")
            model_commits = [item for item in commits if item["model_id"] == model_id]
            previous_offset = 0
            journal_valid = 0
            journal_errors = 0
            for item in model_commits:
                offset = int(item["errors_offset"])
                if offset < previous_offset:
                    raise ValueError(f"non-monotonic error offset for {model_id}")
                previous_offset = offset
                journal_valid += int(item.get("valid_count", -1))
                journal_errors += int(item.get("error_count", -1))
            if journal_valid != valid_count or journal_errors != error_count:
                raise ValueError(f"journal row counts mismatch for {model_id}")
            if previous_offset > paths.errors(model_id).stat().st_size:
                raise ValueError(f"error journal offset exceeds file size for {model_id}")
            if metadata["state"] == STATE_COMPLETE and previous_offset != paths.errors(model_id).stat().st_size:
                raise ValueError(f"COMPLETE error journal has uncommitted bytes for {model_id}")
            for start in range(0, len(rows), 65536):
                end = min(start + 65536, len(rows))
                mask = np.asarray(rows[start:end]) == ROW_VALID
                if not np.any(mask):
                    continue
                chunk = np.asarray(embeddings[start:end])[mask]
                if not np.isfinite(chunk).all():
                    raise ValueError(f"non-finite embedding row for {model_id}")
                norms = np.linalg.norm(chunk, axis=1)
                if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"embedding normalization mismatch for {model_id}")
        summaries[model_id] = {
            "valid": valid_count, "decode_errors": error_count, "pending": pending_count,
        }
    discovery = metadata.get("discovery_errors", {})
    _validate_sealed_file(
        paths.discovery_errors,
        size=int(discovery.get("size", -1)), digest=str(discovery.get("sha256", "")),
        seal=discovery.get("seal", {}), full=full, label="discovery error journal",
    )
    with paths.discovery_errors.open("r", encoding="utf-8") as stream:
        if sum(1 for line in stream if line.strip()) != int(discovery.get("count", -1)):
            raise ValueError("discovery error journal count mismatch")
    files = metadata.get("files", {})
    _validate_sealed_file(
        paths.journal,
        size=int(files.get("journal_size", -1)), digest=str(files.get("journal_sha256", "")),
        seal=files.get("journal_seal", {}), full=full, label="build journal",
    )
    return IndexVerificationResult(
        index_path=paths.root,
        state=str(metadata["state"]),
        image_count=int(metadata["image_count"]),
        full=full,
        models=summaries,
    )
