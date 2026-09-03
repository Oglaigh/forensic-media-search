"""Sequential two-model forensic search orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

from .models import VisionLanguageModel
from .queries import QuerySpec
from .ranking import FusedCandidate, PerQueryTopK, fuse_rankings
from .reporting import (
    ErrorJournal,
    build_rows,
    display_path,
    utc_timestamp,
    write_all_results_csv,
    write_csv,
)
from .scanner import (
    ManifestEntry,
    ScanMetrics,
    iter_decoded_batches,
    read_manifest,
    validate_artifact_path,
    write_manifest,
)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    evidence_root: Path
    output_path: Path
    queries: tuple[QuerySpec, ...]
    top_k: int
    batch_size: int
    rrf_constant: int
    display_root: str | None = None
    max_images: int | None = None


@dataclass(frozen=True, slots=True)
class AllResultsConfig:
    evidence_root: Path
    output_path: Path
    queries: tuple[QuerySpec, ...]
    batch_size: int
    display_root: str | None = None


@dataclass(slots=True)
class SearchResult:
    candidates: list[FusedCandidate]
    metrics: ScanMetrics
    preparation_seconds: dict[str, float]
    total_seconds: float
    manifest_path: Path
    error_log_path: Path
    output_path: Path
    per_query_counts: dict[str, dict[str, int]]
    model_metadata: dict[str, dict[str, str | None]]


@dataclass(frozen=True, slots=True)
class AllResultsRecord:
    file_id: int
    query_id: str
    score: float
    rank: int


@dataclass(slots=True)
class AllResultsResult:
    records: list[AllResultsRecord]
    metrics: ScanMetrics
    preparation_seconds: float
    total_seconds: float
    manifest_path: Path
    error_log_path: Path
    output_path: Path
    model_metadata: dict[str, str | None]


def _cuda_sync(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def run_model_pass(
    *,
    model: VisionLanguageModel,
    manifest_path: Path,
    queries: Sequence[QuerySpec],
    top_k: int,
    batch_size: int,
    metrics: ScanMetrics,
    journal: ErrorJournal,
    progress: Callable[[str], None],
    total_images: int,
) -> tuple[dict[str, tuple], float]:
    """Load one model, scan the fixed manifest, and retain per-query Top-K."""

    model_id = model.model_id
    preparation_started = perf_counter()
    model.load()
    try:
        query_embeddings = model.encode_queries(queries)
        preparation_seconds = perf_counter() - preparation_started
        ranking = PerQueryTopK(model_id, queries, k=top_k)
        _cuda_sync(model.info.device)
        metrics.begin_pass(model_id)
        examined = 0
        try:
            for decoded_batch in iter_decoded_batches(
                read_manifest(manifest_path), batch_size=batch_size
            ):
                examined += len(decoded_batch.images) + len(decoded_batch.errors)
                for error in decoded_batch.errors:
                    metrics.record_decode_error(model_id, error.entry.ordinal)
                    journal.record(phase="decode", model_id=model_id, error=error)

                if not decoded_batch.images:
                    continue

                try:
                    images = [item.image for item in decoded_batch.images]
                    file_ids = [item.entry.ordinal for item in decoded_batch.images]
                    try:
                        image_embeddings = model.encode_images(images)
                        similarities = model.similarity(image_embeddings, query_embeddings)
                    except RuntimeError as error:
                        detail = str(error)
                        if "out of memory" in detail.casefold():
                            raise RuntimeError(
                                f"{model_id} CUDA out of memory while processing a batch "
                                f"of {len(images)} images; reduce --batch-size"
                            ) from error
                        raise RuntimeError(
                            f"{model_id} inference failed for manifest ordinals "
                            f"{file_ids[0]}..{file_ids[-1]}: {detail}"
                        ) from error
                    ranking.update_batch(file_ids, similarities)
                    for file_id in file_ids:
                        metrics.record_processed(model_id, file_id)
                    progress(f"@@PROGRESS\t{model_id}\t{examined}\t{total_images}")
                finally:
                    for item in decoded_batch.images:
                        item.image.close()
            progress(f"@@PROGRESS\t{model_id}\t{total_images}\t{total_images}")
        finally:
            _cuda_sync(model.info.device)
            metrics.finish_pass(model_id)
        return ranking.ranked(), preparation_seconds
    finally:
        model.unload()


def _candidate_entries(
    manifest_path: Path, candidates: Sequence[FusedCandidate]
) -> dict[int, ManifestEntry]:
    return _entries_for_file_ids(
        manifest_path, {candidate.file_id for candidate in candidates}
    )


def _entries_for_file_ids(
    manifest_path: Path, wanted: set[int]
) -> dict[int, ManifestEntry]:
    found: dict[int, ManifestEntry] = {}
    for entry in read_manifest(manifest_path):
        if entry.ordinal in wanted:
            found[entry.ordinal] = entry
            if len(found) == len(wanted):
                break
    if len(found) != len(wanted):
        missing = sorted(wanted.difference(found))
        raise RuntimeError(f"Candidate ordinals missing from manifest: {missing[:10]}")
    return found


def _score_rows(score_matrix: object) -> list[list[float]]:
    if hasattr(score_matrix, "detach"):
        score_matrix = score_matrix.detach().to("cpu").tolist()
    elif hasattr(score_matrix, "tolist"):
        score_matrix = score_matrix.tolist()
    return [[float(score) for score in row] for row in score_matrix]


def run_all_results(
    config: AllResultsConfig,
    model: VisionLanguageModel,
    *,
    progress: Callable[[str], None] = print,
) -> AllResultsResult:
    """Export every processable image ranked independently for every query."""

    if model.model_id != "siglip2":
        raise ValueError("--all-results requires the SigLIP2 adapter")
    root = config.evidence_root.resolve(strict=True)
    output = config.output_path.resolve(strict=False)
    validate_artifact_path(root, output)
    manifest = output.with_suffix(output.suffix + ".manifest.jsonl")
    errors = output.with_suffix(output.suffix + ".errors.jsonl")
    validate_artifact_path(root, manifest)
    validate_artifact_path(root, errors)

    started = perf_counter()
    scores: dict[str, list[tuple[float, int]]] = {
        query.query_id: [] for query in config.queries
    }
    with ErrorJournal(errors) as journal:
        discovery = write_manifest(
            root,
            manifest,
            on_error=lambda error: journal.record(
                phase="discovery", model_id=None, error=error
            ),
        )
        metrics = ScanMetrics(
            images_discovered=discovery.images_discovered,
            filesystem_errors=discovery.filesystem_errors,
        )
        preparation_started = perf_counter()
        model.load()
        try:
            query_embeddings = model.encode_queries(config.queries)
            preparation_seconds = perf_counter() - preparation_started
            _cuda_sync(model.info.device)
            metrics.begin_pass(model.model_id)
            progress_count = 0
            try:
                for decoded_batch in iter_decoded_batches(
                    read_manifest(manifest), batch_size=config.batch_size
                ):
                    for error in decoded_batch.errors:
                        metrics.record_decode_error(model.model_id, error.entry.ordinal)
                        journal.record(phase="decode", model_id=model.model_id, error=error)
                    if not decoded_batch.images:
                        continue
                    try:
                        images = [item.image for item in decoded_batch.images]
                        file_ids = [item.entry.ordinal for item in decoded_batch.images]
                        try:
                            image_embeddings = model.encode_images(images)
                            rows = _score_rows(
                                model.similarity(image_embeddings, query_embeddings)
                            )
                        except RuntimeError as error:
                            detail = str(error)
                            if "out of memory" in detail.casefold():
                                raise RuntimeError(
                                    "siglip2 CUDA out of memory while processing a batch "
                                    f"of {len(images)} images; reduce --batch-size"
                                ) from error
                            raise
                        if len(rows) != len(file_ids):
                            raise ValueError("score row count must match decoded images")
                        for file_id, row in zip(file_ids, rows):
                            if len(row) != len(config.queries):
                                raise ValueError("each score row must contain every query")
                            for query, score in zip(config.queries, row):
                                if not math.isfinite(score):
                                    raise ValueError("SigLIP2 scores must be finite")
                                scores[query.query_id].append((score, file_id))
                            metrics.record_processed(model.model_id, file_id)
                        progress_count += len(file_ids)
                        if progress_count >= 1000:
                            progress(
                                "siglip2: processed "
                                f"{len(metrics.passes[model.model_id].processed)} images"
                            )
                            progress_count = 0
                    finally:
                        for item in decoded_batch.images:
                            item.image.close()
            finally:
                _cuda_sync(model.info.device)
                metrics.finish_pass(model.model_id)
        finally:
            model.unload()

    records: list[AllResultsRecord] = []
    for query in config.queries:
        ordered = sorted(scores[query.query_id], key=lambda item: (-item[0], item[1]))
        records.extend(
            AllResultsRecord(file_id=file_id, query_id=query.query_id, score=score, rank=rank)
            for rank, (score, file_id) in enumerate(ordered, start=1)
        )

    entries = _entries_for_file_ids(
        manifest, {record.file_id for record in records}
    )
    query_map = {query.query_id: query for query in config.queries}
    write_all_results_csv(
        (
            {
                "FilePath": display_path(entries[record.file_id], config.display_root),
                "Query": query_map[record.query_id].matched_query,
                "SigLIP2Score": record.score,
                "Rank": record.rank,
                "Model": model.info.name,
                "Revision": model.info.revision,
            }
            for record in records
        ),
        output,
    )
    return AllResultsResult(
        records=records,
        metrics=metrics,
        preparation_seconds=preparation_seconds,
        total_seconds=perf_counter() - started,
        manifest_path=manifest,
        error_log_path=errors,
        output_path=output,
        model_metadata={
            "name": model.info.name,
            "revision": model.info.revision,
            "dtype": model.info.dtype,
            "device": model.info.device,
        },
    )


def run_search(
    config: SearchConfig,
    models: Sequence[VisionLanguageModel],
    *,
    progress: Callable[[str], None] = print,
) -> SearchResult:
    if [model.model_id for model in models] != ["siglip2", "clip"]:
        raise ValueError("models must be ordered as SigLIP2 followed by CLIP")

    root = config.evidence_root.resolve(strict=True)
    output = config.output_path.resolve(strict=False)
    validate_artifact_path(root, output)
    manifest = output.with_suffix(output.suffix + ".manifest.jsonl")
    errors = output.with_suffix(output.suffix + ".errors.jsonl")
    validate_artifact_path(root, manifest)
    validate_artifact_path(root, errors)

    started = perf_counter()
    model_metadata: dict[str, dict[str, str | None]] = {}
    rankings: dict[str, Mapping[str, tuple]] = {}
    preparation: dict[str, float] = {}

    with ErrorJournal(errors) as journal:
        discovery = write_manifest(
            root,
            manifest,
            max_images=config.max_images,
            on_error=lambda error: journal.record(
                phase="discovery", model_id=None, error=error
            ),
        )
        metrics = ScanMetrics(
            images_discovered=discovery.images_discovered,
            filesystem_errors=discovery.filesystem_errors,
        )
        progress(
            f"@@PROGRESS\tdiscovery\t{discovery.images_discovered}\t"
            f"{discovery.images_discovered}"
        )
        for model in models:
            progress(f"PASS: {model.model_id}")
            ranked, prep_seconds = run_model_pass(
                model=model,
                manifest_path=manifest,
                queries=config.queries,
                top_k=config.top_k,
                batch_size=config.batch_size,
                metrics=metrics,
                journal=journal,
                progress=progress,
                total_images=discovery.images_discovered,
            )
            progress(f"@@MODEL_COMPLETE\t{model.model_id}")
            rankings[model.model_id] = ranked
            preparation[model.model_id] = prep_seconds
            model_metadata[model.model_id] = {
                "name": model.info.name,
                "revision": model.info.revision,
                "dtype": model.info.dtype,
                "device": model.info.device,
            }

    candidates = fuse_rankings(
        rankings,
        rrf_constant=config.rrf_constant,
        query_order=[query.query_id for query in config.queries],
    )
    entries = _candidate_entries(manifest, candidates)
    rows = build_rows(
        candidates,
        entries,
        {query.query_id: query for query in config.queries},
        display_root=config.display_root,
        model_metadata=model_metadata,
        scan_timestamp=utc_timestamp(),
    )
    write_csv(rows, output)
    total_seconds = perf_counter() - started
    return SearchResult(
        candidates=candidates,
        metrics=metrics,
        preparation_seconds=preparation,
        total_seconds=total_seconds,
        manifest_path=manifest,
        error_log_path=errors,
        output_path=output,
        per_query_counts={
            model_id: {query_id: len(items) for query_id, items in per_query.items()}
            for model_id, per_query in rankings.items()
        },
        model_metadata=model_metadata,
    )
