"""Rank-based fusion without comparing model score scales."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .topk import RankedCandidate


MODEL_DISPLAY_NAMES = {"siglip2": "SigLIP2", "clip": "CLIP"}


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """A candidate identified by the composite key (file_id, query_id)."""

    file_id: int
    query_id: str
    scores: Mapping[str, float]
    ranks: Mapping[str, int]
    models_matched: str
    fusion_score: float
    final_rank: int

    @property
    def siglip2_score(self) -> float | None:
        return self.scores.get("siglip2")

    @property
    def clip_cos(self) -> float | None:
        return self.scores.get("clip")

    @property
    def siglip2_rank(self) -> int | None:
        return self.ranks.get("siglip2")

    @property
    def clip_rank(self) -> int | None:
        return self.ranks.get("clip")


def _display_name(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def _matched_models(model_ids: Sequence[str]) -> str:
    priority = {"siglip2": 0, "clip": 1}
    ordered = sorted(model_ids, key=lambda item: (priority.get(item, 2), item))
    return "+".join(_display_name(model_id) for model_id in ordered)


def fuse_rankings(
    rankings: Mapping[str, Mapping[str, Sequence[RankedCandidate]]],
    *,
    rrf_constant: int = 60,
    query_order: Sequence[str] | None = None,
) -> list[FusedCandidate]:
    """Union and RRF model Top-K lists for the same file and query.

    Each model contributes ``1 / (rrf_constant + rank)`` only when it retained
    that file in the Top-K for that exact query. Raw scores remain separate.
    """

    if rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")

    discovered_queries: list[str] = []
    accumulated: dict[tuple[int, str], dict[str, RankedCandidate]] = {}

    for model_id, per_query in rankings.items():
        for query_id, candidates in per_query.items():
            if query_id not in discovered_queries:
                discovered_queries.append(query_id)
            for candidate in candidates:
                if candidate.model_id != model_id:
                    raise ValueError("candidate model_id does not match ranking key")
                if candidate.query_id != query_id:
                    raise ValueError("candidate query_id does not match ranking key")
                if candidate.rank <= 0:
                    raise ValueError("candidate ranks must be one-based")

                by_model = accumulated.setdefault((candidate.file_id, query_id), {})
                previous = by_model.get(model_id)
                if previous is None or (candidate.rank, -candidate.score) < (
                    previous.rank,
                    -previous.score,
                ):
                    by_model[model_id] = candidate

    ordered_queries = list(query_order) if query_order is not None else discovered_queries
    if len(ordered_queries) != len(set(ordered_queries)):
        raise ValueError("query_order values must be unique")
    query_positions = {query_id: index for index, query_id in enumerate(ordered_queries)}
    unknown_position = len(query_positions)

    fused: list[FusedCandidate] = []
    for (file_id, query_id), by_model in accumulated.items():
        scores = {model_id: item.score for model_id, item in by_model.items()}
        ranks = {model_id: item.rank for model_id, item in by_model.items()}
        fusion_score = sum(
            1.0 / (rrf_constant + rank)
            for rank in ranks.values()
        )
        fused.append(
            FusedCandidate(
                file_id=file_id,
                query_id=query_id,
                scores=MappingProxyType(scores),
                ranks=MappingProxyType(ranks),
                models_matched=_matched_models(tuple(by_model)),
                fusion_score=fusion_score,
                final_rank=0,
            )
        )

    infinity = float("inf")

    def rank_key(item: FusedCandidate) -> tuple:
        return (
            -item.fusion_score,
            -len(item.ranks),
            min(item.ranks.values()),
            item.siglip2_rank if item.siglip2_rank is not None else infinity,
            item.clip_rank if item.clip_rank is not None else infinity,
            item.file_id,
        )

    per_query: dict[str, list[FusedCandidate]] = {}
    for item in fused:
        per_query.setdefault(item.query_id, []).append(item)

    ranked: list[FusedCandidate] = []
    for query_id in sorted(
        per_query,
        key=lambda value: (query_positions.get(value, unknown_position), value),
    ):
        ordered = sorted(per_query[query_id], key=rank_key)
        ranked.extend(
            replace(item, final_rank=rank)
            for rank, item in enumerate(ordered, start=1)
        )
    return ranked
