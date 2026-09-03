"""Per-query Top-K and rank-fusion public API."""

from .evaluator import EvaluatedCandidate, EvaluatorQueryMatch, evaluate_files
from .fusion import FusedCandidate, fuse_rankings
from .topk import PerQueryTopK, RankedCandidate

__all__ = [
    "EvaluatedCandidate",
    "EvaluatorQueryMatch",
    "FusedCandidate",
    "PerQueryTopK",
    "RankedCandidate",
    "evaluate_files",
    "fuse_rankings",
]
