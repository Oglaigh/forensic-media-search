"""Identity query processing for the first ensemble iteration."""

from __future__ import annotations

from typing import Sequence

from .base import QueryProcessor, QuerySpec


class IdentityQueryProcessor(QueryProcessor):
    """Preserve user text exactly while assigning stable query identifiers."""

    def process(self, queries: Sequence[str]) -> tuple[QuerySpec, ...]:
        if not queries:
            raise ValueError("at least one query is required")

        return tuple(
            QuerySpec(
                query_id=f"q{index:04d}",
                original_query=query,
                model_query=query,
            )
            for index, query in enumerate(queries, start=1)
        )

