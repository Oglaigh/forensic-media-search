"""Command-line interface for the forensic ensemble search."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .models import OpenAIClipModel, SigLIP2Model
from .pipeline import SearchConfig, SearchResult, run_search
from .queries import IdentityQueryProcessor
from .scanner import validate_artifact_path

SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
OPENAI_CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline forensic candidate search using SigLIP2 and OpenAI CLIP."
    )
    parser.add_argument("--directory", required=True, help="Evidence directory.")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument(
        "--top-k", type=int, default=5000,
        help="Candidates retained independently per model and query.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="/output/report.csv")
    parser.add_argument("--display-root", default=None)
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--clip-model", "--model", dest="clip_model", default="ViT-B/32")
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Deterministic partial-scan limit intended for smoke tests.",
    )
    parser.add_argument("--min-percent", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--reference-cos", type=float, default=None, help=argparse.SUPPRESS)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.rrf_constant < 0:
        parser.error("--rrf-constant must be non-negative")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be greater than zero")
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
    for label, value in (
        ("HF_HOME", os.environ.get("HF_HOME", "/root/.cache/huggingface")),
        ("CLIP_CACHE_DIR", os.environ.get("CLIP_CACHE_DIR", "/root/.cache/clip")),
    ):
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
    print(f"Throughput                : {throughput:.1f} images/sec")
    print(f"CSV                       : {result.output_path}")
    print(f"Manifest                  : {result.manifest_path}")
    print(f"Error journal             : {result.error_log_path}")
    print("Scores and FusionScore are ranking measures, not probabilities.")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    queries = IdentityQueryProcessor().process(args.query)
    _print_startup(args)
    models = (
        SigLIP2Model(
            args.siglip_model, device=args.device, revision=SIGLIP2_REVISION,
            cache_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
        ),
        OpenAIClipModel(
            args.clip_model, device=args.device, revision=OPENAI_CLIP_REVISION,
            download_root=os.environ.get("CLIP_CACHE_DIR", "/root/.cache/clip"),
        ),
    )
    result = run_search(
        SearchConfig(
            evidence_root=Path(args.directory), output_path=Path(args.output),
            queries=queries, top_k=args.top_k, batch_size=args.batch_size,
            rrf_constant=args.rrf_constant, display_root=args.display_root,
            max_images=args.max_images,
        ),
        models,
    )
    _print_summary(result, {query.query_id: query.original_query for query in queries})
    return 0
