"""Bounded batching helpers used by scanner passes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def batched(items: Iterable[T], batch_size: int) -> Iterator[tuple[T, ...]]:
    """Yield fixed-size tuples plus a final partial batch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    pending: list[T] = []
    for item in items:
        pending.append(item)
        if len(pending) == batch_size:
            yield tuple(pending)
            pending.clear()
    if pending:
        yield tuple(pending)
