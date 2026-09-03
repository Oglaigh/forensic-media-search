"""Memory-bounded Top-K ranking, independent per model and query."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    model_id: str
    query_id: str
    file_id: int
    score: float
    rank: int


@dataclass(order=True, frozen=True, slots=True)
class _HeapEntry:
    """Heap order puts the worst retained candidate at index zero."""

    score: float
    negative_file_id: int
    file_id: int


def _query_id(query: Any) -> str:
    value = query if isinstance(query, str) else query.query_id
    if not value:
        raise ValueError("query_id must not be empty")
    return value


class PerQueryTopK:
    """One bounded heap for every query of one model.

    A separate instance is used for each model, yielding storage bounded by
    O(models * queries * K). Only ordinal, score and tie-break metadata live in
    the heaps; images and embeddings are never retained.
    """

    def __init__(self, model_id: str, queries: Sequence[Any], k: int) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not queries:
            raise ValueError("at least one query is required")

        query_ids = tuple(_query_id(query) for query in queries)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique")

        self.model_id = model_id
        self.query_ids = query_ids
        self.k = k
        self._heaps: dict[str, list[_HeapEntry]] = {
            query_id: [] for query_id in query_ids
        }

    @property
    def retained_count(self) -> int:
        return sum(len(heap) for heap in self._heaps.values())

    def counts(self) -> Mapping[str, int]:
        return {query_id: len(heap) for query_id, heap in self._heaps.items()}

    @staticmethod
    def _rows(score_matrix: Any) -> list[list[float]]:
        if hasattr(score_matrix, "detach"):
            score_matrix = score_matrix.detach().to("cpu").tolist()
        elif hasattr(score_matrix, "tolist"):
            score_matrix = score_matrix.tolist()
        return [[float(score) for score in row] for row in score_matrix]

    def update_batch(self, file_ids: Sequence[int], score_matrix: Any) -> None:
        """Update every query heap from a full [images, queries] matrix."""

        rows = self._rows(score_matrix)
        if len(rows) != len(file_ids):
            raise ValueError("score row count must match file_ids")

        for file_id, row in zip(file_ids, rows):
            if file_id < 0:
                raise ValueError("file_id ordinals must be non-negative")
            if len(row) != len(self.query_ids):
                raise ValueError("each score row must contain one value per query")

            for query_id, score in zip(self.query_ids, row):
                if not math.isfinite(score):
                    raise ValueError("similarity scores must be finite")
                self._offer(query_id, int(file_id), score)

    def _offer(self, query_id: str, file_id: int, score: float) -> None:
        entry = _HeapEntry(score, -file_id, file_id)
        heap = self._heaps[query_id]
        if len(heap) < self.k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    def ranked(self) -> dict[str, tuple[RankedCandidate, ...]]:
        result: dict[str, tuple[RankedCandidate, ...]] = {}
        for query_id in self.query_ids:
            ordered = sorted(
                self._heaps[query_id],
                key=lambda entry: (-entry.score, entry.file_id),
            )
            result[query_id] = tuple(
                RankedCandidate(
                    model_id=self.model_id,
                    query_id=query_id,
                    file_id=entry.file_id,
                    score=entry.score,
                    rank=rank,
                )
                for rank, entry in enumerate(ordered, start=1)
            )
        return result

