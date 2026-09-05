"""Crash-resumable two-pass persistent embedding index builder."""

from __future__ import annotations

import json
import os
import platform
import importlib.metadata
import hashlib
import io
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageOps

from ..models import VisionLanguageModel
from ..scanner import (
    DecodeError, DecodedBatch, DecodedImage, ManifestEntry, batched,
    discover_images, validate_artifact_path,
)
from .format import (
    INDEX_FORMAT, INDEX_VERSION, MODEL_IDS, ROW_DECODE_ERROR, ROW_PENDING, ROW_VALID,
    STATE_BUILDING, STATE_COMPLETE, IndexPaths,
)
from .identity import compatible_identity, model_identity
from .integrity import verify_index
from .journal import append_commit, read_commits, truncate_incomplete_tail, validate_model_commits
from .metadata import (
    atomic_write_json, file_seal, fsync_path, hash_array_rows, hash_state_rows,
    load_index_metadata, sha256_file,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class IndexBuildConfig:
    evidence_root: Path
    index_path: Path
    batch_size: int = 64
    checkpoint_rows: int = 4096
    max_images: int | None = None
    resume: bool = False


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    index_path: Path
    image_count: int
    resumed: bool
    models: dict[str, dict[str, int]]
    model_seconds: dict[str, float]
    total_seconds: float
    throughput_images_per_second: float
    index_size_bytes: int


def _write_index_manifest(root: Path, paths: IndexPaths, max_images: int | None) -> tuple[int, int]:
    errors = paths.discovery_errors.open("w", encoding="utf-8", newline="\n")
    temporary_name: str | None = None
    error_count = 0

    def record_error(error: Any) -> None:
        nonlocal error_count
        error_count += 1
        errors.write(json.dumps(asdict(error), ensure_ascii=False, separators=(",", ":")) + "\n")
        errors.flush()

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".manifest.", suffix=".tmp",
            dir=paths.root, delete=False,
        ) as stream:
            temporary_name = stream.name
            count = 0
            for entry in discover_images(root, on_error=record_error, max_images=max_images):
                source = Path(entry.source_path)
                stat = source.stat()
                payload = {
                    "ordinal": entry.ordinal,
                    "relative_path": entry.relative_path,
                    "source_path": entry.source_path,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(source),
                }
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, paths.manifest)
        temporary_name = None
        errors.flush()
        os.fsync(errors.fileno())
        return count, error_count
    finally:
        errors.close()
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _new_metadata(
    root: Path, paths: IndexPaths, count: int, discovery_error_count: int,
    config: IndexBuildConfig,
) -> dict[str, Any]:
    import PIL

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    try:
        app_version = importlib.metadata.version("forensic-media-search")
    except importlib.metadata.PackageNotFoundError:
        app_version = "source-tree"
    cuda_environment: dict[str, Any] = {
        "runtime": None, "cudnn": None, "matmul_allow_tf32": None,
        "cudnn_allow_tf32": None, "deterministic_algorithms": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import torch
        cuda_environment.update({
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        })
    except ImportError:
        pass

    now = _utc()
    return {
        "index_format": INDEX_FORMAT,
        "index_version": INDEX_VERSION,
        "build_id": os.urandom(16).hex(),
        "state": STATE_BUILDING,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "evidence_root": str(root),
        "discovery": {
            "ordering": "casefold_then_original_files_before_directories",
            "symlinks": "skip",
            "max_images": config.max_images,
            "content_identity": "sha256",
        },
        "image_count": count,
        "manifest_size": paths.manifest.stat().st_size,
        "manifest_sha256": sha256_file(paths.manifest),
        "discovery_errors": {
            "count": discovery_error_count,
            "size": paths.discovery_errors.stat().st_size,
            "sha256": sha256_file(paths.discovery_errors),
            "seal": file_seal(paths.discovery_errors),
        },
        "row_mapping": "row_equals_manifest_ordinal",
        "storage_dtype": "float32",
        "normalization": "l2_fp32_adapter_output_then_l2_fp32_search",
        "environment": {
            "application": app_version,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch_version,
            "cuda": cuda_environment,
        },
        "models": {
            model_id: {
                "dimension": None, "dtype": "float32", "layout": "C",
                "byte_order": "little", "committed_prefix": 0,
                "valid_count": 0, "error_count": 0, "identity": None, "files": {},
            }
            for model_id in MODEL_IDS
        },
    }


def _index_manifest_rows(manifest: Path) -> Iterator[dict[str, Any]]:
    expected = 0
    with Path(manifest).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                ordinal = int(value["ordinal"])
                if ordinal != expected:
                    raise ValueError(f"expected ordinal {expected}, got {ordinal}")
                for key in ("relative_path", "source_path", "size", "mtime_ns", "sha256"):
                    if key not in value:
                        raise ValueError(f"missing {key}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid index manifest row {line_number}: {error}") from error
            expected += 1
            yield value


def _verify_source(value: dict[str, Any], *, full_hash: bool = False) -> None:
    source = Path(value["source_path"])
    stat = source.stat()
    if stat.st_size != int(value["size"]) or stat.st_mtime_ns != int(value["mtime_ns"]):
        raise RuntimeError(f"evidence metadata changed after discovery: {source}")
    if full_hash and sha256_file(source) != value["sha256"]:
        raise RuntimeError(f"evidence content changed after discovery: {source}")


def _entries_from(manifest: Path, start: int) -> Iterator[ManifestEntry]:
    for value in _index_manifest_rows(manifest):
        if int(value["ordinal"]) < start:
            continue
        _verify_source(value)
        yield ManifestEntry(
            ordinal=int(value["ordinal"]),
            relative_path=str(value["relative_path"]),
            source_path=str(value["source_path"]),
        )


def _decode_authenticated(value: dict[str, Any]) -> DecodedImage | DecodeError:
    """Hash and decode the exact same bytes obtained from one read-only handle."""
    entry = ManifestEntry(
        ordinal=int(value["ordinal"]),
        relative_path=str(value["relative_path"]),
        source_path=str(value["source_path"]),
    )
    source_path = Path(entry.source_path)
    try:
        with source_path.open("rb") as source:
            stat = os.fstat(source.fileno())
            if stat.st_size != int(value["size"]) or stat.st_mtime_ns != int(value["mtime_ns"]):
                raise RuntimeError(f"evidence metadata changed after discovery: {source_path}")
            encoded_bytes = source.read()
        if hashlib.sha256(encoded_bytes).hexdigest() != value["sha256"]:
            raise RuntimeError(f"evidence content changed after discovery: {source_path}")
        with Image.open(io.BytesIO(encoded_bytes)) as encoded:
            oriented = ImageOps.exif_transpose(encoded)
            try:
                rgb = oriented.convert("RGB")
                rgb.load()
                detached = rgb.copy()
            finally:
                if "rgb" in locals():
                    rgb.close()
                if oriented is not encoded:
                    oriented.close()
        return DecodedImage(entry=entry, image=detached)
    except RuntimeError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        return DecodeError(entry=entry, error_type=type(error).__name__, message=str(error))


def _iter_authenticated_batches(
    manifest: Path, start: int, batch_size: int
) -> Iterator[DecodedBatch]:
    rows = (value for value in _index_manifest_rows(manifest) if int(value["ordinal"]) >= start)
    for row_batch in batched(rows, batch_size):
        decoded = [_decode_authenticated(value) for value in row_batch]
        yield DecodedBatch(
            images=tuple(item for item in decoded if isinstance(item, DecodedImage)),
            errors=tuple(item for item in decoded if isinstance(item, DecodeError)),
        )


def _verify_all_evidence(manifest: Path) -> None:
    for value in _index_manifest_rows(manifest):
        _verify_source(value, full_hash=True)


def _validate_resume_manifest(paths: IndexPaths, metadata: dict[str, Any]) -> None:
    if not paths.manifest.is_file():
        raise ValueError("resume index manifest is missing")
    if paths.manifest.stat().st_size != int(metadata["manifest_size"]):
        raise ValueError("resume index manifest size mismatch")
    if sha256_file(paths.manifest) != metadata["manifest_sha256"]:
        raise ValueError("resume index manifest hash mismatch")
    count = sum(1 for _ in _index_manifest_rows(paths.manifest))
    if count != int(metadata["image_count"]):
        raise ValueError("resume index manifest row count mismatch")


def _rollback_error_log(path: Path, offset: int) -> None:
    if not path.exists():
        if offset:
            raise ValueError(f"missing committed error journal: {path}")
        path.touch()
    if path.stat().st_size < offset:
        raise ValueError(f"committed error journal is shorter than its checkpoint: {path}")
    with path.open("r+b") as stream:
        stream.truncate(offset)
        stream.flush()
        os.fsync(stream.fileno())


def _to_numpy(embeddings: Any) -> np.ndarray:
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().to("cpu").numpy()
    value = np.asarray(embeddings)
    if value.ndim != 2:
        raise ValueError("image encoder must return a two-dimensional embedding matrix")
    if value.dtype != np.float32:
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("image encoder returned non-finite embeddings")
    return np.ascontiguousarray(value, dtype="<f4")


def _sync(array: Any, path: Path) -> None:
    array.flush()
    fsync_path(path)


def _build_model(
    model: VisionLanguageModel,
    config: IndexBuildConfig,
    paths: IndexPaths,
    metadata: dict[str, Any],
    progress: Callable[[str], None],
) -> None:
    model_id = model.model_id
    count = int(metadata["image_count"])
    model_meta = metadata["models"][model_id]
    model_dir = paths.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows_path = paths.rows(model_id)
    if rows_path.exists():
        if rows_path.stat().st_size != count:
            raise ValueError(f"row-state size mismatch for {model_id}")
        rows = np.memmap(rows_path, dtype=np.uint8, mode="r+", shape=(count,))
    else:
        rows = np.memmap(rows_path, dtype=np.uint8, mode="w+", shape=(count,))
        rows[:] = ROW_PENDING
        _sync(rows, rows_path)

    model.load()
    try:
        identity = model_identity(model)
        if model_meta["identity"] is not None:
            compatible_identity(model_meta["identity"], identity)
        else:
            model_meta["identity"] = identity
            model_meta["inference_device"] = model.info.device
            model_meta["inference_dtype"] = model.info.dtype

        embeddings: Any | None = None
        if model_meta["dimension"] is not None:
            embeddings = np.load(paths.embeddings(model_id), mmap_mode="r+", allow_pickle=False)
            commits = read_commits(paths.journal)
            start = validate_model_commits(commits, model_id, embeddings, rows)
            if int(model_meta["committed_prefix"]) > start:
                raise ValueError(f"metadata is ahead of durable journal for {model_id}")
            model_meta["committed_prefix"] = start
        else:
            commits = read_commits(paths.journal)
            start = 0

        model_commits = [item for item in commits if item["model_id"] == model_id]
        durable_error_offset = int(model_commits[-1]["errors_offset"]) if model_commits else 0
        _rollback_error_log(paths.errors(model_id), durable_error_offset)
        with paths.errors(model_id).open("ab") as error_stream:
            commit_start = start
            sequence = len(model_commits)
            examined = start
            for batch in _iter_authenticated_batches(paths.manifest, start, config.batch_size):
                ordinals = [item.entry.ordinal for item in batch.images]
                error_ordinals = [error.entry.ordinal for error in batch.errors]
                for error in batch.errors:
                    rows[error.entry.ordinal] = ROW_DECODE_ERROR
                    error_stream.write(
                        (json.dumps(asdict(error), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    )
                if batch.images:
                    try:
                        values = _to_numpy(model.encode_images([item.image for item in batch.images]))
                    finally:
                        for item in batch.images:
                            item.image.close()
                    if values.shape[0] != len(batch.images):
                        raise ValueError(f"embedding row count mismatch for {model_id}")
                    if embeddings is None:
                        dimension = int(values.shape[1])
                        if dimension <= 0:
                            raise ValueError(f"invalid embedding dimension for {model_id}")
                        model_meta["dimension"] = dimension
                        embeddings = np.lib.format.open_memmap(
                            paths.embeddings(model_id), mode="w+", dtype="<f4", shape=(count, dimension),
                        )
                        metadata["updated_at"] = _utc()
                        atomic_write_json(paths.metadata, metadata)
                    elif values.shape[1] != int(model_meta["dimension"]):
                        raise ValueError(f"embedding dimension changed during {model_id} build")
                    embeddings[ordinals] = values
                    rows[ordinals] = ROW_VALID
                if ordinals or error_ordinals:
                    examined = max(ordinals + error_ordinals) + 1
                progress(f"@@PROGRESS\tindex-{model_id}\t{examined}\t{count}")
                if embeddings is not None and (examined - commit_start >= config.checkpoint_rows or examined == count):
                    _sync(embeddings, paths.embeddings(model_id))
                    _sync(rows, rows_path)
                    error_stream.flush()
                    os.fsync(error_stream.fileno())
                    errors_offset = error_stream.tell()
                    append_commit(paths.journal, {
                        "sequence": sequence, "model_id": model_id,
                        "start": commit_start, "end": examined,
                        "valid_count": int(np.count_nonzero(rows[commit_start:examined] == ROW_VALID)),
                        "error_count": int(np.count_nonzero(rows[commit_start:examined] == ROW_DECODE_ERROR)),
                        "embeddings_sha256": hash_array_rows(embeddings, commit_start, examined),
                        "rows_sha256": hash_state_rows(rows, commit_start, examined),
                        "errors_offset": errors_offset,
                        "committed_at": _utc(),
                    })
                    commit_start = examined
                    sequence += 1
                    model_meta["committed_prefix"] = examined
                    model_meta["valid_count"] = int(np.count_nonzero(rows[:examined] == ROW_VALID))
                    model_meta["error_count"] = int(np.count_nonzero(rows[:examined] == ROW_DECODE_ERROR))
                    metadata["updated_at"] = _utc()
                    atomic_write_json(paths.metadata, metadata)
            if embeddings is None:
                raise RuntimeError(f"{model_id} index has no decodable image from which to determine dimension")
            if commit_start != count:
                raise RuntimeError(f"{model_id} build stopped before all manifest rows were committed")
            if np.any(rows == ROW_PENDING):
                raise RuntimeError(f"{model_id} has pending rows after build")
            model_meta["files"] = {
                "embeddings": str(paths.embeddings(model_id).relative_to(paths.root)),
                "embeddings_size": paths.embeddings(model_id).stat().st_size,
                "embeddings_sha256": sha256_file(paths.embeddings(model_id)),
                "embeddings_seal": file_seal(paths.embeddings(model_id)),
                "rows": str(rows_path.relative_to(paths.root)),
                "rows_size": rows_path.stat().st_size,
                "rows_sha256": sha256_file(rows_path),
                "rows_seal": file_seal(rows_path),
                "errors": str(paths.errors(model_id).relative_to(paths.root)),
                "errors_size": paths.errors(model_id).stat().st_size,
                "errors_sha256": sha256_file(paths.errors(model_id)),
                "errors_seal": file_seal(paths.errors(model_id)),
            }
            metadata["updated_at"] = _utc()
            atomic_write_json(paths.metadata, metadata)
    finally:
        model.unload()


def build_index(
    config: IndexBuildConfig,
    models: Sequence[VisionLanguageModel],
    *,
    progress: Callable[[str], None] = print,
) -> IndexBuildResult:
    started = perf_counter()
    if [model.model_id for model in models] != list(MODEL_IDS):
        raise ValueError("models must be ordered as SigLIP2 followed by CLIP")
    if config.batch_size <= 0 or config.checkpoint_rows <= 0:
        raise ValueError("batch_size and checkpoint_rows must be greater than zero")
    root = Path(config.evidence_root).resolve(strict=True)
    index_root = Path(config.index_path).resolve(strict=False)
    validate_artifact_path(root, index_root)
    for model in models:
        for attribute in ("cache_dir", "download_root"):
            value = getattr(model, attribute, None)
            if value:
                cache = Path(value).resolve(strict=False)
                validate_artifact_path(root, cache)
                if cache == index_root or index_root in cache.parents:
                    raise ValueError("model caches must be outside the index directory")
    paths = IndexPaths(index_root)
    if config.resume:
        if not index_root.is_dir():
            raise ValueError("resume requires an existing index directory")
        metadata = load_index_metadata(paths.metadata)
        if metadata["state"] != STATE_BUILDING:
            raise ValueError("only a BUILDING index can be resumed")
        if os.path.normcase(str(Path(metadata["evidence_root"]).resolve(strict=False))) != os.path.normcase(str(root)):
            raise ValueError("resume evidence root does not match the index")
        if config.max_images != metadata["discovery"]["max_images"]:
            raise ValueError("resume max_images does not match the index")
        _validate_resume_manifest(paths, metadata)
        truncate_incomplete_tail(paths.journal)
    else:
        if index_root.exists() and any(index_root.iterdir()):
            raise FileExistsError(f"index directory is not empty: {index_root}")
        index_root.mkdir(parents=True, exist_ok=True)
        count, discovery_error_count = _write_index_manifest(root, paths, config.max_images)
        if count == 0:
            raise ValueError("no supported images were discovered")
        paths.journal.touch(exist_ok=False)
        metadata = _new_metadata(root, paths, count, discovery_error_count, config)
        atomic_write_json(paths.metadata, metadata)

    model_seconds: dict[str, float] = {}
    for model in models:
        model_started = perf_counter()
        _build_model(model, config, paths, metadata, progress)
        model_seconds[model.model_id] = perf_counter() - model_started
    _verify_all_evidence(paths.manifest)
    metadata["files"] = {
        "journal_size": paths.journal.stat().st_size,
        "journal_sha256": sha256_file(paths.journal),
        "journal_seal": file_seal(paths.journal),
    }
    atomic_write_json(paths.metadata, metadata)
    verify_index(index_root, full=True)
    metadata["state"] = STATE_COMPLETE
    metadata["completed_at"] = _utc()
    metadata["updated_at"] = metadata["completed_at"]
    atomic_write_json(paths.metadata, metadata)
    verify_index(index_root, require_complete=True)
    total_seconds = perf_counter() - started
    index_size_bytes = sum(
        path.stat().st_size for path in index_root.rglob("*") if path.is_file()
    )
    return IndexBuildResult(
        index_path=index_root,
        image_count=int(metadata["image_count"]),
        resumed=config.resume,
        models={
            model_id: {
                "valid": int(metadata["models"][model_id]["valid_count"]),
                "decode_errors": int(metadata["models"][model_id]["error_count"]),
            }
            for model_id in MODEL_IDS
        },
        model_seconds=model_seconds,
        total_seconds=total_seconds,
        throughput_images_per_second=(
            int(metadata["image_count"]) / total_seconds if total_seconds else 0.0
        ),
        index_size_bytes=index_size_bytes,
    )
