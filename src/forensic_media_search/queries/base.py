"""Traceable query processing contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """A model query with an immutable link to the investigator's input."""

    query_id: str
    original_query: str
    model_query: str

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must not be empty")
        if not self.original_query.strip():
            raise ValueError("original_query must not be empty")
        if not self.model_query.strip():
            raise ValueError("model_query must not be empty")

    @property
    def matched_query(self) -> str:
        """Text actually submitted to the embedding model."""

        return self.model_query


class QueryProcessor(ABC):
    """Turns investigator input into traceable model queries."""

    @abstractmethod
    def process(self, queries: Sequence[str]) -> tuple[QuerySpec, ...]:
        """Process a complete query set without losing source traceability."""

