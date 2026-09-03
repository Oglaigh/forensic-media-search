"""Sequential two-model forensic search orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

from .models import VisionLanguageModel
from .queries import QuerySpec
from .ranking import FusedCandidate, PerQueryTopK, fuse_rankings
from .reporting import ErrorJournal, build_rows, utc_timestamp, write_csv
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
        processed_since_progress = 0
        try:
            for decoded_batch in iter_decoded_batches(
                read_manifest(manifest_path), batch_size=batch_size
            ):
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
                    processed_since_progress += len(file_ids)
                    if processed_since_progress >= 1000:
                        progress(
                            f"{model_id}: processed "
                            f"{len(metrics.passes[model_id].processed)} images"
                        )
                        processed_since_progress = 0
                finally:
                    for item in decoded_batch.images:
                        item.image.close()
        finally:
            _cuda_sync(model.info.device)
            metrics.finish_pass(model_id)
        return ranking.ranked(), preparation_seconds
    finally:
        model.unload()


def _candidate_entries(
    manifest_path: Path, candidates: Sequence[FusedCandidate]
) -> dict[int, ManifestEntry]:
    wanted = {candidate.file_id for candidate in candidates}
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
            )
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
