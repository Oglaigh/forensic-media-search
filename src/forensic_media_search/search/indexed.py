"""Exact, chunked indexed search without evidence image access."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from ..indexing.format import MODEL_IDS, ROW_DECODE_ERROR, ROW_VALID, IndexPaths
from ..indexing.identity import compatible_identity, model_identity
from ..indexing.integrity import verify_index
from ..indexing.metadata import atomic_write_json, file_seal, load_index_metadata, sha256_file
from ..models import VisionLanguageModel
from ..pipeline import SearchResult
from ..queries import QuerySpec
from ..ranking import EvaluatedCandidate, PerQueryTopK, evaluate_files, fuse_rankings
from ..reporting import build_final_rows, build_rows, utc_timestamp, write_csv, write_final_csv
from ..scanner import ManifestEntry, ScanMetrics, read_manifest, validate_artifact_path


def _cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


@dataclass(frozen=True, slots=True)
class IndexedSearchConfig:
    index_path: Path
    output_path: Path
    queries: tuple[QuerySpec, ...]
    top_k: int
    chunk_rows: int
    rrf_constant: int
    display_root: str | None = None
    final_output_path: Path | None = None
    evaluator_top_k: int | None = None
    evaluator_rrf_constant: int | None = None


def _entries_for_ids(manifest: Path, wanted: set[int]) -> dict[int, ManifestEntry]:
    found: dict[int, ManifestEntry] = {}
    for entry in read_manifest(manifest):
        if entry.ordinal in wanted:
            found[entry.ordinal] = entry
            if len(found) == len(wanted):
                break
    if len(found) != len(wanted):
        missing = sorted(wanted.difference(found))
        raise RuntimeError(f"candidate ordinals missing from index manifest: {missing[:10]}")
    return found


def _atomic_empty(path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _rank_model(
    model: VisionLanguageModel,
    model_meta: dict,
    paths: IndexPaths,
    queries: Sequence[QuerySpec],
    top_k: int,
    chunk_rows: int,
    metrics: ScanMetrics,
    progress: Callable[[str], None],
) -> tuple[Mapping[str, tuple], float, float, dict[str, str | None]]:
    model_id = model.model_id
    started = perf_counter()
    model.load()
    try:
        compatible_identity(model_meta["identity"], model_identity(model))
        _cuda_sync(model.info.device)
        text_started = perf_counter()
        query_embeddings = model.encode_queries(queries)
        _cuda_sync(model.info.device)
        text_seconds = perf_counter() - text_started
        if query_embeddings.ndim != 2 or int(query_embeddings.shape[1]) != int(model_meta["dimension"]):
            raise ValueError(f"query embedding dimension is incompatible with {model_id} index")
        preparation = perf_counter() - started
        embeddings = np.load(paths.embeddings(model_id), mmap_mode="r", allow_pickle=False)
        rows = np.memmap(paths.rows(model_id), dtype=np.uint8, mode="r", shape=(len(embeddings),))
        ranking = PerQueryTopK(model_id, queries, k=top_k)
        _cuda_sync(model.info.device)
        vector_started = perf_counter()
        metrics.begin_pass(model_id)
        try:
            for start in range(0, len(embeddings), chunk_rows):
                end = min(start + chunk_rows, len(embeddings))
                states = np.asarray(rows[start:end])
                valid_offsets = np.flatnonzero(states == ROW_VALID)
                error_offsets = np.flatnonzero(states == ROW_DECODE_ERROR)
                for offset in error_offsets:
                    metrics.record_decode_error(model_id, start + int(offset))
                if valid_offsets.size:
                    host = np.array(embeddings[start:end][valid_offsets], dtype=np.float32, copy=True, order="C")
                    if hasattr(query_embeddings, "detach"):
                        import torch
                        image_embeddings = torch.from_numpy(host).to(model.info.device)
                    else:
                        image_embeddings = host
                    try:
                        scores = model.similarity(image_embeddings, query_embeddings)
                    except RuntimeError as error:
                        if "out of memory" in str(error).casefold():
                            raise RuntimeError(
                                f"{model_id} CUDA out of memory during indexed search; reduce --chunk-rows"
                            ) from error
                        raise
                    file_ids = [start + int(offset) for offset in valid_offsets]
                    ranking.update_batch(file_ids, scores)
                    for file_id in file_ids:
                        metrics.record_processed(model_id, file_id)
                progress(f"@@PROGRESS\tsearch-{model_id}\t{end}\t{len(embeddings)}")
        finally:
            _cuda_sync(model.info.device)
            metrics.finish_pass(model_id)
        vector_seconds = perf_counter() - vector_started
        return ranking.ranked(), text_seconds, vector_seconds, {
            "name": model.info.name,
            "revision": model.info.revision,
            "dtype": model.info.dtype,
            "device": model.info.device,
        }
    finally:
        model.unload()


def run_indexed_search(
    config: IndexedSearchConfig,
    models: Sequence[VisionLanguageModel],
    *,
    progress: Callable[[str], None] = print,
) -> SearchResult:
    if [model.model_id for model in models] != list(MODEL_IDS):
        raise ValueError("models must be ordered as SigLIP2 followed by CLIP")
    if not config.queries or config.top_k <= 0 or config.chunk_rows <= 0:
        raise ValueError("queries, top_k and chunk_rows must be valid")
    if config.rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")
    if (config.final_output_path is None) != (config.evaluator_top_k is None):
        raise ValueError("final_output_path and evaluator_top_k must be provided together")
    if config.evaluator_top_k is not None and config.evaluator_top_k <= 0:
        raise ValueError("evaluator_top_k must be greater than zero")
    if config.evaluator_rrf_constant is not None and config.evaluator_rrf_constant < 0:
        raise ValueError("evaluator_rrf_constant must be non-negative")

    started = perf_counter()
    index_open_started = perf_counter()
    index_root = Path(config.index_path).resolve(strict=True)
    paths = IndexPaths(index_root)
    verify_index(index_root, require_complete=True)
    metadata = load_index_metadata(paths.metadata)
    evidence_root = Path(metadata["evidence_root"]).resolve(strict=False)
    output = Path(config.output_path).resolve(strict=False)
    errors = output.with_suffix(output.suffix + ".errors.jsonl")
    run_metadata = output.with_suffix(output.suffix + ".run.json")
    for artifact in (output, errors, run_metadata):
        validate_artifact_path(evidence_root, artifact)
        validate_artifact_path(index_root, artifact)
    final_output = Path(config.final_output_path).resolve(strict=False) if config.final_output_path else None
    if final_output is not None:
        validate_artifact_path(evidence_root, final_output)
        validate_artifact_path(index_root, final_output)
        if final_output in {output, errors, run_metadata}:
            raise ValueError("final output must differ from audit and error outputs")
    index_open_seconds = perf_counter() - index_open_started

    run_record = {
        "run_format": "forensic-media-search-indexed-search",
        "run_version": 1,
        "state": "BUILDING",
        "started_at": utc_timestamp(),
        "completed_at": None,
        "index": {
            "path": str(index_root),
            "build_id": metadata["build_id"],
            "index_version": metadata["index_version"],
            "manifest_sha256": metadata["manifest_sha256"],
        },
        "queries": [asdict(query) for query in config.queries],
        "search": {
            "top_k": config.top_k,
            "chunk_rows": config.chunk_rows,
            "rrf_constant": config.rrf_constant,
            "evaluator_top_k": config.evaluator_top_k,
            "evaluator_rrf_constant": config.evaluator_rrf_constant,
            "display_root": config.display_root,
            "audit_report": str(output),
            "final_report": str(final_output) if final_output is not None else None,
        },
        "models": metadata["models"],
        "reports": {},
        "result": None,
    }
    atomic_write_json(run_metadata, run_record)
    count = int(metadata["image_count"])
    metrics = ScanMetrics(images_discovered=count)
    rankings: dict[str, Mapping[str, tuple]] = {}
    preparation: dict[str, float] = {}
    vector_seconds: dict[str, float] = {}
    model_metadata: dict[str, dict[str, str | None]] = {}
    for model in models:
        for attribute in ("cache_dir", "download_root"):
            value = getattr(model, attribute, None)
            if value:
                cache = Path(value).resolve(strict=False)
                validate_artifact_path(evidence_root, cache)
                validate_artifact_path(index_root, cache)
        ranked, text_time, vector_time, trace = _rank_model(
            model, metadata["models"][model.model_id], paths, config.queries,
            config.top_k, config.chunk_rows, metrics, progress,
        )
        rankings[model.model_id] = ranked
        preparation[model.model_id] = text_time
        vector_seconds[model.model_id] = vector_time
        model_metadata[model.model_id] = trace
        progress(f"@@MODEL_COMPLETE\t{model.model_id}")

    rrf_started = perf_counter()
    candidates = fuse_rankings(
        rankings, rrf_constant=config.rrf_constant,
        query_order=[query.query_id for query in config.queries],
    )
    rrf_seconds = perf_counter() - rrf_started
    entries = _entries_for_ids(paths.manifest, {candidate.file_id for candidate in candidates})
    query_map = {query.query_id: query for query in config.queries}
    stamp = utc_timestamp()
    write_csv(build_rows(
        candidates, entries, query_map, display_root=config.display_root,
        model_metadata=model_metadata, scan_timestamp=stamp,
    ), output)
    _atomic_empty(errors)

    evaluated: list[EvaluatedCandidate] = []
    evaluator_seconds = 0.0
    if final_output is not None:
        evaluator_constant = 60 if config.evaluator_rrf_constant is None else config.evaluator_rrf_constant
        evaluator_started = perf_counter()
        evaluated = evaluate_files(
            candidates, evaluator_top_k=config.evaluator_top_k,
            rrf_constant=evaluator_constant,
            query_order=[query.query_id for query in config.queries],
        )
        evaluator_seconds = perf_counter() - evaluator_started
        final_entries = _entries_for_ids(paths.manifest, {item.file_id for item in evaluated})
        write_final_csv(build_final_rows(
            evaluated, final_entries, query_map, display_root=config.display_root,
            scan_timestamp=stamp, evaluator_top_k=config.evaluator_top_k,
            evaluator_rrf_constant=evaluator_constant, audit_report=output,
        ), final_output)

    report_paths = {"audit": output, "errors": errors}
    if final_output is not None:
        report_paths["final"] = final_output
    total_seconds = perf_counter() - started
    timings = {
        "index_open_seconds": index_open_seconds,
        "text_encoding_siglip2_seconds": preparation["siglip2"],
        "text_encoding_clip_seconds": preparation["clip"],
        "vector_search_siglip2_seconds": vector_seconds["siglip2"],
        "vector_search_clip_seconds": vector_seconds["clip"],
        "rrf_seconds": rrf_seconds,
        "evaluator_seconds": evaluator_seconds,
        "total_seconds": total_seconds,
    }
    run_record.update({
        "state": "COMPLETE",
        "scan_timestamp": stamp,
        "completed_at": utc_timestamp(),
        "reports": {
            label: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "seal": file_seal(path),
            }
            for label, path in report_paths.items()
        },
        "result": {"audit_rows": len(candidates), "final_rows": len(evaluated)},
        "timings": timings,
    })
    atomic_write_json(run_metadata, run_record)

    return SearchResult(
        candidates=candidates,
        metrics=metrics,
        preparation_seconds=preparation,
        total_seconds=total_seconds,
        manifest_path=paths.manifest,
        error_log_path=errors,
        output_path=output,
        per_query_counts={
            model_id: {query_id: len(items) for query_id, items in per_query.items()}
            for model_id, per_query in rankings.items()
        },
        model_metadata=model_metadata,
        evaluated_candidates=evaluated,
        final_output_path=final_output,
        execution_counters={
            "images_decoded": 0,
            "siglip2_images_encoded": 0,
            "clip_images_encoded": 0,
        },
        timings=timings,
    )
