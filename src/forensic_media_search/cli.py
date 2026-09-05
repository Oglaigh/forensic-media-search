"""Command-line interface for the forensic ensemble search."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from .indexing import IndexBuildConfig, build_index, verify_index
from .models import OpenAIClipModel, SigLIP2Model
from .pipeline import (
    AllResultsConfig,
    AllResultsResult,
    SearchConfig,
    SearchResult,
    run_all_results,
    run_search,
)
from .queries import IdentityQueryProcessor
from .scanner import validate_artifact_path
from .search import IndexedSearchConfig, run_indexed_search

load_dotenv(dotenv_path=Path.cwd() / ".env")

SIGLIP2_REVISION = os.environ.get(
    "SIGLIP2_REVISION", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
)
OPENAI_CLIP_REVISION = os.environ.get(
    "OPENAI_CLIP_REVISION", "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline forensic candidate search using SigLIP2 and OpenAI CLIP."
    )
    parser.add_argument("--directory", required=True, help="Evidence directory.")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Candidates retained independently per model and query. Default: 5000.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="/output/report.csv")
    parser.add_argument("--final-output", default=None)
    parser.add_argument("--evaluator-top-k", type=int, default=None)
    parser.add_argument("--evaluator-rrf-constant", type=int, default=None)
    parser.add_argument("--display-root", default=None)
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--clip-model", "--model", dest="clip_model", default=None)
    parser.add_argument("--rrf-constant", type=int, default=None)
    parser.add_argument(
        "--all-results", action="store_true",
        help="Diagnostic SigLIP2-only full ranking; disables Top-K, CLIP, and RRF.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Deterministic partial-scan limit intended for smoke tests.",
    )
    parser.add_argument("--min-percent", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reference-cos", type=float, default=None, help=argparse.SUPPRESS)
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--clip-model", "--model", dest="clip_model", default="ViT-B/32")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")


def _build_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or resume a persistent forensic image index.")
    parser.add_argument("--directory", required=True, help="Read-only evidence directory.")
    parser.add_argument("--index", required=True, help="Index directory outside evidence.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-rows", type=int, default=4096)
    parser.add_argument("--max-images", type=int, default=None)
    _add_model_arguments(parser)
    return parser


def _build_indexed_search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a COMPLETE persistent index exactly.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--output", default="/output/report.csv")
    parser.add_argument("--final-output", default=None)
    parser.add_argument("--evaluator-top-k", type=int, default=None)
    parser.add_argument("--evaluator-rrf-constant", type=int, default=None)
    parser.add_argument("--display-root", default=None)
    parser.add_argument("--rrf-constant", type=int, default=60)
    _add_model_arguments(parser)
    return parser


def _build_verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify persistent index integrity.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--full", action="store_true")
    return parser


def _require_cuda(parser: argparse.ArgumentParser, device: str) -> None:
    if device != "cuda":
        return
    import torch
    if not torch.cuda.is_available():
        parser.error(
            "CUDA was requested but is not available; use --device cpu explicitly "
            "only when CPU execution is intended"
        )


def _models(args: argparse.Namespace) -> tuple[SigLIP2Model, OpenAIClipModel]:
    return (
        SigLIP2Model(
            args.siglip_model, device=args.device, revision=SIGLIP2_REVISION,
            cache_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
        ),
        OpenAIClipModel(
            args.clip_model, device=args.device, revision=OPENAI_CLIP_REVISION,
            download_root=os.environ.get("CLIP_CACHE_DIR", "/root/.cache/clip"),
        ),
    )


def _main_index(argv: list[str]) -> int:
    parser = _build_index_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.checkpoint_rows <= 0:
        parser.error("--batch-size and --checkpoint-rows must be greater than zero")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be greater than zero")
    evidence = Path(args.directory)
    if not evidence.is_dir():
        parser.error(f"evidence directory does not exist or is not a directory: {evidence}")
    try:
        validate_artifact_path(evidence.resolve(), Path(args.index).resolve(strict=False))
    except ValueError:
        parser.error("--index must resolve outside the evidence directory")
    _require_cuda(parser, args.device)
    result = build_index(
        IndexBuildConfig(
            evidence_root=evidence, index_path=Path(args.index), batch_size=args.batch_size,
            checkpoint_rows=args.checkpoint_rows, max_images=args.max_images, resume=args.resume,
        ),
        _models(args),
    )
    print(f"Index COMPLETE: {result.index_path} ({result.image_count} images)")
    print(f"Total wall time        : {_duration(result.total_seconds)}")
    print(f"SigLIP2 indexing       : {_duration(result.model_seconds['siglip2'])}")
    print(f"CLIP indexing          : {_duration(result.model_seconds['clip'])}")
    print(f"Throughput             : {result.throughput_images_per_second:.1f} images/sec")
    print(f"Index size             : {result.index_size_bytes} bytes")
    return 0


def _main_indexed_search(argv: list[str]) -> int:
    parser = _build_indexed_search_parser()
    args = parser.parse_args(argv)
    if args.top_k <= 0 or args.chunk_rows <= 0:
        parser.error("--top-k and --chunk-rows must be greater than zero")
    if args.rrf_constant < 0:
        parser.error("--rrf-constant must be non-negative")
    if (args.final_output is None) != (args.evaluator_top_k is None):
        parser.error("--final-output and --evaluator-top-k must be provided together")
    if args.evaluator_top_k is not None and args.evaluator_top_k <= 0:
        parser.error("--evaluator-top-k must be greater than zero")
    if args.evaluator_rrf_constant is not None and args.evaluator_rrf_constant < 0:
        parser.error("--evaluator-rrf-constant must be non-negative")
    _require_cuda(parser, args.device)
    queries = IdentityQueryProcessor().process(args.query)
    result = run_indexed_search(
        IndexedSearchConfig(
            index_path=Path(args.index), output_path=Path(args.output), queries=queries,
            top_k=args.top_k, chunk_rows=args.chunk_rows, rrf_constant=args.rrf_constant,
            display_root=args.display_root,
            final_output_path=Path(args.final_output) if args.final_output else None,
            evaluator_top_k=args.evaluator_top_k,
            evaluator_rrf_constant=args.evaluator_rrf_constant,
        ),
        _models(args),
    )
    _print_summary(result, {query.query_id: query.original_query for query in queries})
    return 0


def _main_verify(argv: list[str]) -> int:
    parser = _build_verify_parser()
    args = parser.parse_args(argv)
    result = verify_index(Path(args.index), full=args.full)
    level = "full" if args.full else "structural"
    print(f"Index verification passed ({level}): {result.index_path} [{result.state}]")
    return 0


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.all_results:
        incompatible = []
        if args.top_k is not None:
            incompatible.append("--top-k")
        if args.clip_model is not None:
            incompatible.append("--clip-model/--model")
        if args.rrf_constant is not None:
            incompatible.append("--rrf-constant")
        if args.max_images is not None:
            incompatible.append("--max-images")
        if args.final_output is not None:
            incompatible.append("--final-output")
        if args.evaluator_top_k is not None:
            incompatible.append("--evaluator-top-k")
        if args.evaluator_rrf_constant is not None:
            incompatible.append("--evaluator-rrf-constant")
        if incompatible:
            parser.error(
                "--all-results is incompatible with " + ", ".join(incompatible)
            )
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.rrf_constant is not None and args.rrf_constant < 0:
        parser.error("--rrf-constant must be non-negative")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be greater than zero")
    if (args.final_output is None) != (args.evaluator_top_k is None):
        parser.error("--final-output and --evaluator-top-k must be provided together")
    if args.evaluator_top_k is not None and args.evaluator_top_k <= 0:
        parser.error("--evaluator-top-k must be greater than zero")
    if args.evaluator_rrf_constant is not None and args.evaluator_rrf_constant < 0:
        parser.error("--evaluator-rrf-constant must be non-negative")
    if args.evaluator_rrf_constant is not None and args.final_output is None:
        parser.error("--evaluator-rrf-constant requires --final-output")
    if args.min_percent is not None or args.reference_cos is not None:
        parser.error(
            "--min-percent and --reference-cos were removed: use per-query --top-k; "
            "ensemble scores are not calibrated percentages"
        )
    root = Path(args.directory)
    if not root.exists() or not root.is_dir():
        parser.error(f"evidence directory does not exist or is not a directory: {root}")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            parser.error(
                "CUDA was requested but is not available; use --device cpu explicitly "
                "only when CPU execution is intended"
            )

    evidence_root = root.resolve(strict=True)
    output_path = Path(args.output).resolve(strict=False)
    try:
        validate_artifact_path(evidence_root, output_path)
    except ValueError:
        parser.error(f"--output must resolve outside the evidence directory: {args.output}")
    if args.final_output is not None:
        final_output = Path(args.final_output).resolve(strict=False)
        try:
            validate_artifact_path(evidence_root, final_output)
        except ValueError:
            parser.error(
                "--final-output must resolve outside the evidence directory: "
                f"{args.final_output}"
            )
        if final_output == output_path:
            parser.error("--final-output must be different from --output")
    cache_locations = [
        ("HF_HOME", os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    ]
    if not args.all_results:
        cache_locations.append(
            ("CLIP_CACHE_DIR", os.environ.get("CLIP_CACHE_DIR", "/root/.cache/clip"))
        )
    for label, value in cache_locations:
        try:
            validate_artifact_path(evidence_root, Path(value).resolve(strict=False))
        except ValueError:
            parser.error(f"{label} must resolve outside the evidence directory: {value}")


def _duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_startup(args: argparse.Namespace) -> None:
    print("=" * 78)
    print("FORENSIC MEDIA SEARCH - SigLIP2 + OpenAI CLIP")
    print("=" * 78)
    print(f"Directory          : {Path(args.directory).resolve()}")
    print(f"Device             : {args.device}")
    if args.device == "cuda":
        import torch
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"SigLIP2 model      : {args.siglip_model}")
    print(f"OpenAI CLIP model  : {args.clip_model}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Top-K              : {args.top_k} per model and query")
    print(f"RRF constant       : {args.rrf_constant}")
    print(f"Audit report       : {Path(args.output).resolve()}")
    if args.final_output is not None:
        print("Final evaluation   : enabled")
        print(f"Evaluator Top-K    : {args.evaluator_top_k} per query")
        print(f"Evaluator RRF const: {args.evaluator_rrf_constant}")
        print(f"Final report       : {Path(args.final_output).resolve()}")
    else:
        print("Final evaluation   : disabled")
    if args.max_images is not None:
        print(f"PARTIAL SCAN       : first {args.max_images} discovered images")
    print("Queries:")
    for query in args.query:
        print(f"  - {query}")
    print()


def _print_summary(result: SearchResult, queries_by_id: dict[str, str]) -> None:
    siglip_pass = result.metrics.passes["siglip2"]
    clip_pass = result.metrics.passes["clip"]
    intersection = sum(1 for candidate in result.candidates if len(candidate.ranks) == 2)
    processing_seconds = siglip_pass.processing_seconds + clip_pass.processing_seconds
    throughput = result.metrics.images_processed / processing_seconds if processing_seconds else 0.0
    print()
    print("=" * 78)
    print("SCAN SUMMARY")
    print("=" * 78)
    print(f"Images discovered          : {result.metrics.images_discovered}")
    print(f"Images processed           : {result.metrics.images_processed}")
    print(f"Filesystem errors          : {result.metrics.filesystem_errors}")
    print(f"Decode errors (unique)     : {result.metrics.decode_errors}")
    print(f"SigLIP2 processed          : {len(siglip_pass.processed)}")
    print(f"CLIP processed             : {len(clip_pass.processed)}")
    for model_id, counts in result.per_query_counts.items():
        display = "SigLIP2" if model_id == "siglip2" else "CLIP"
        print(f"{display} candidates        : {sum(counts.values())}")
        for query_id, count in counts.items():
            print(f"  {query_id} ({queries_by_id[query_id]}) : {count}")
    print(f"Intersection (file+query)  : {intersection}")
    print(f"Final union (file+query)   : {len(result.candidates)}")
    print(f"SigLIP2 preparation       : {_duration(result.preparation_seconds['siglip2'])}")
    print(f"CLIP preparation          : {_duration(result.preparation_seconds['clip'])}")
    print(f"SigLIP2 processing        : {_duration(siglip_pass.processing_seconds)}")
    print(f"CLIP processing           : {_duration(clip_pass.processing_seconds)}")
    print(f"Total processing          : {_duration(processing_seconds)}")
    print(f"Total wall time           : {_duration(result.total_seconds)}")
    if result.timings is not None:
        print(f"Index open                : {_duration(result.timings['index_open_seconds'])}")
        print(
            "Text encoding SigLIP2    : "
            f"{_duration(result.timings['text_encoding_siglip2_seconds'])}"
        )
        print(
            "Text encoding CLIP       : "
            f"{_duration(result.timings['text_encoding_clip_seconds'])}"
        )
        print(
            "SigLIP2 vector search    : "
            f"{_duration(result.timings['vector_search_siglip2_seconds'])}"
        )
        print(
            "CLIP vector search       : "
            f"{_duration(result.timings['vector_search_clip_seconds'])}"
        )
        print(f"RRF                       : {_duration(result.timings['rrf_seconds'])}")
        print(f"Evaluator                 : {_duration(result.timings['evaluator_seconds'])}")
    if result.execution_counters is not None:
        print(f"Images decoded           : {result.execution_counters['images_decoded']}")
        print(
            "SigLIP2 image encoded    : "
            f"{result.execution_counters['siglip2_images_encoded']}"
        )
        print(
            "CLIP image encoded       : "
            f"{result.execution_counters['clip_images_encoded']}"
        )
    print(f"Throughput                : {throughput:.1f} images/sec")
    print(f"CSV                       : {result.output_path}")
    if result.final_output_path is not None:
        print(f"Final files              : {len(result.evaluated_candidates)}")
        print(f"Final CSV                : {result.final_output_path}")
    print(f"Manifest                  : {result.manifest_path}")
    print(f"Error journal             : {result.error_log_path}")
    print("Scores and FusionScore are ranking measures, not probabilities.")
    print("=" * 78)


def _print_all_results_startup(args: argparse.Namespace) -> None:
    print("=" * 78)
    print("FORENSIC MEDIA SEARCH - SigLIP2 ALL RESULTS DIAGNOSTIC")
    print("=" * 78)
    print(f"Directory          : {Path(args.directory).resolve()}")
    print(f"Device             : {args.device}")
    if args.device == "cuda":
        import torch
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"SigLIP2 model      : {args.siglip_model}")
    print(f"Batch size         : {args.batch_size}")
    print("Selection          : all processable images per query")
    print("CLIP / Top-K / RRF : disabled")
    print("Queries:")
    for query in args.query:
        print(f"  - {query}")
    print()


def _print_all_results_summary(result: AllResultsResult, query_count: int) -> None:
    model_pass = result.metrics.passes["siglip2"]
    expected_rows = result.metrics.images_processed * query_count
    print()
    print("=" * 78)
    print("ALL RESULTS DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(f"Images discovered      : {result.metrics.images_discovered}")
    print(f"Images processed       : {result.metrics.images_processed}")
    print(f"Decode errors          : {result.metrics.decode_errors}")
    print(f"Queries                : {query_count}")
    print(f"Ranking rows           : {len(result.records)}")
    print(f"Expected rows          : {expected_rows}")
    print(f"SigLIP2 preparation    : {_duration(result.preparation_seconds)}")
    print(f"SigLIP2 processing     : {_duration(model_pass.processing_seconds)}")
    print(f"Total wall time        : {_duration(result.total_seconds)}")
    print(f"CSV                    : {result.output_path}")
    print(f"Manifest               : {result.manifest_path}")
    print(f"Error journal          : {result.error_log_path}")
    print("Every row is diagnostic ranking output, not necessarily a candidate.")
    print("SigLIP2Score is cosine similarity, not a probability.")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv:
        command = effective_argv[0]
        if command == "index":
            return _main_index(effective_argv[1:])
        if command == "verify":
            return _main_verify(effective_argv[1:])
        if command == "search":
            search_argv = effective_argv[1:]
            if "--direct" in search_argv:
                search_argv = [item for item in search_argv if item != "--direct"]
                effective_argv = search_argv
            else:
                return _main_indexed_search(search_argv)
    parser = build_parser()
    args = parser.parse_args(effective_argv)
    validate_args(parser, args)
    queries = IdentityQueryProcessor().process(args.query)
    if args.all_results:
        _print_all_results_startup(args)
        result = run_all_results(
            AllResultsConfig(
                evidence_root=Path(args.directory),
                output_path=Path(args.output),
                queries=queries,
                batch_size=args.batch_size,
                display_root=args.display_root,
            ),
            SigLIP2Model(
                args.siglip_model,
                device=args.device,
                revision=SIGLIP2_REVISION,
                cache_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
            ),
        )
        _print_all_results_summary(result, len(queries))
        return 0

    args.top_k = 5000 if args.top_k is None else args.top_k
    args.clip_model = "ViT-B/32" if args.clip_model is None else args.clip_model
    args.rrf_constant = 60 if args.rrf_constant is None else args.rrf_constant
    if args.final_output is not None and args.evaluator_rrf_constant is None:
        args.evaluator_rrf_constant = 60
    _print_startup(args)
    models = _models(args)
    result = run_search(
        SearchConfig(
            evidence_root=Path(args.directory), output_path=Path(args.output),
            queries=queries, top_k=args.top_k, batch_size=args.batch_size,
            rrf_constant=args.rrf_constant, display_root=args.display_root,
            max_images=args.max_images,
            final_output_path=(Path(args.final_output) if args.final_output else None),
            evaluator_top_k=args.evaluator_top_k,
            evaluator_rrf_constant=args.evaluator_rrf_constant,
        ),
        models,
    )
    _print_summary(result, {query.query_id: query.original_query for query in queries})
    return 0
