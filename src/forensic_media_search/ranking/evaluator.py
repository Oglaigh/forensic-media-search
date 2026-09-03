"""Final rank-only evaluation across independent queries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .fusion import FusedCandidate


@dataclass(frozen=True, slots=True)
class EvaluatorQueryMatch:
    query_id: str
    final_rank: int
    reciprocal_contribution: float


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    file_id: int
    evaluator_rank: int
    evaluator_score: float
    strong_query_count: int
    best_query_id: str
    best_query_rank: int
    query_matches: tuple[EvaluatorQueryMatch, ...]


def evaluate_files(
    candidates: Sequence[FusedCandidate],
    *,
    evaluator_top_k: int,
    rrf_constant: int = 60,
    query_order: Sequence[str] | None = None,
) -> list[EvaluatedCandidate]:
    """Collapse strong file-query results to one prioritized row per file."""
    if evaluator_top_k <= 0:
        raise ValueError("evaluator_top_k must be greater than zero")
    if rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")
    ordered_queries = tuple(query_order or ())
    if len(ordered_queries) != len(set(ordered_queries)):
        raise ValueError("query_order values must be unique")
    positions = {query_id: index for index, query_id in enumerate(ordered_queries)}
    unknown_position = len(positions)
    by_file: dict[int, dict[str, EvaluatorQueryMatch]] = {}
    seen: set[tuple[int, str]] = set()
    for candidate in candidates:
        key = (candidate.file_id, candidate.query_id)
        if key in seen:
            raise ValueError("candidates must be unique by file_id and query_id")
        seen.add(key)
        if candidate.final_rank <= 0:
            raise ValueError("candidate final ranks must be one-based")
        if candidate.final_rank > evaluator_top_k:
            continue
        by_file.setdefault(candidate.file_id, {})[candidate.query_id] = EvaluatorQueryMatch(
            candidate.query_id,
            candidate.final_rank,
            1.0 / (rrf_constant + candidate.final_rank),
        )
    evaluated: list[EvaluatedCandidate] = []
    for file_id, matches_by_query in by_file.items():
        matches = tuple(sorted(matches_by_query.values(), key=lambda item: (
            item.final_rank,
            positions.get(item.query_id, unknown_position),
            item.query_id,
        )))
        best = matches[0]
        evaluated.append(EvaluatedCandidate(
            file_id=file_id,
            evaluator_rank=0,
            evaluator_score=sum(item.reciprocal_contribution for item in matches),
            strong_query_count=len(matches),
            best_query_id=best.query_id,
            best_query_rank=best.final_rank,
            query_matches=matches,
        ))
    evaluated.sort(key=lambda item: (
        -item.strong_query_count,
        -item.evaluator_score,
        item.best_query_rank,
        item.file_id,
    ))
    return [replace(item, evaluator_rank=rank) for rank, item in enumerate(evaluated, 1)]
