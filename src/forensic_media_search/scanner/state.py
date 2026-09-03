"""Memory-bounded counters for repeatable multi-model scan passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


class OrdinalBitmap:
    """A compact fixed-capacity set for manifest ordinals."""

    __slots__ = ("_bits", "_capacity", "_count")

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self._capacity = capacity
        self._bits = bytearray((capacity + 7) // 8)
        self._count = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._count

    def __contains__(self, ordinal: object) -> bool:
        if not isinstance(ordinal, int) or ordinal < 0 or ordinal >= self._capacity:
            return False
        byte, bit = divmod(ordinal, 8)
        return bool(self._bits[byte] & (1 << bit))

    def add(self, ordinal: int) -> bool:
        """Add an ordinal and return ``True`` only when it was newly set."""

        if ordinal < 0 or ordinal >= self._capacity:
            raise IndexError(
                f"ordinal {ordinal} outside manifest capacity {self._capacity}"
            )
        byte, bit = divmod(ordinal, 8)
        mask = 1 << bit
        if self._bits[byte] & mask:
            return False
        self._bits[byte] |= mask
        self._count += 1
        return True


@dataclass(slots=True)
class PassMetrics:
    model_id: str
    capacity: int
    processed: OrdinalBitmap = field(init=False)
    decode_errors: OrdinalBitmap = field(init=False)
    started_at: float | None = None
    finished_at: float | None = None

    def __post_init__(self) -> None:
        self.processed = OrdinalBitmap(self.capacity)
        self.decode_errors = OrdinalBitmap(self.capacity)

    @property
    def processing_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        endpoint = self.finished_at if self.finished_at is not None else perf_counter()
        return max(0.0, endpoint - self.started_at)

    def start(self) -> None:
        if self.started_at is not None:
            raise RuntimeError(f"Pass already started: {self.model_id}")
        self.started_at = perf_counter()

    def finish(self) -> None:
        if self.started_at is None:
            raise RuntimeError(f"Pass not started: {self.model_id}")
        if self.finished_at is not None:
            raise RuntimeError(f"Pass already finished: {self.model_id}")
        self.finished_at = perf_counter()

    def record_processed(self, ordinal: int) -> bool:
        return self.processed.add(ordinal)

    def record_decode_error(self, ordinal: int) -> bool:
        return self.decode_errors.add(ordinal)


@dataclass(slots=True)
class ScanMetrics:
    """Bounded state shared by sequential model passes."""

    images_discovered: int
    filesystem_errors: int = 0
    passes: dict[str, PassMetrics] = field(default_factory=dict)
    _processed_any: OrdinalBitmap = field(init=False, repr=False)
    _decode_errors_any: OrdinalBitmap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._processed_any = OrdinalBitmap(self.images_discovered)
        self._decode_errors_any = OrdinalBitmap(self.images_discovered)

    @property
    def images_processed(self) -> int:
        return len(self._processed_any)

    @property
    def decode_errors(self) -> int:
        return len(self._decode_errors_any)

    def begin_pass(self, model_id: str) -> PassMetrics:
        if model_id in self.passes:
            raise ValueError(f"Duplicate model pass: {model_id}")
        metrics = PassMetrics(model_id=model_id, capacity=self.images_discovered)
        metrics.start()
        self.passes[model_id] = metrics
        return metrics

    def record_processed(self, model_id: str, ordinal: int) -> None:
        metrics = self.passes[model_id]
        metrics.record_processed(ordinal)
        self._processed_any.add(ordinal)

    def record_decode_error(self, model_id: str, ordinal: int) -> None:
        metrics = self.passes[model_id]
        metrics.record_decode_error(ordinal)
        self._decode_errors_any.add(ordinal)

    def finish_pass(self, model_id: str) -> None:
        self.passes[model_id].finish()
