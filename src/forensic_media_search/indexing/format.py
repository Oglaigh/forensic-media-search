"""Versioned on-disk index schema and immutable row contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INDEX_FORMAT = "forensic-media-search"
INDEX_VERSION = 1
STATE_BUILDING = "BUILDING"
STATE_COMPLETE = "COMPLETE"
ROW_PENDING = 0
ROW_VALID = 1
ROW_DECODE_ERROR = 2
VALID_ROW_STATES = frozenset((ROW_PENDING, ROW_VALID, ROW_DECODE_ERROR))
MODEL_IDS = ("siglip2", "clip")


@dataclass(frozen=True, slots=True)
class IndexPaths:
    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / "index.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def discovery_errors(self) -> Path:
        return self.root / "discovery-errors.jsonl"

    @property
    def journal(self) -> Path:
        return self.root / "build-journal.jsonl"

    def model_dir(self, model_id: str) -> Path:
        if model_id not in MODEL_IDS:
            raise ValueError(f"unsupported model_id: {model_id}")
        return self.root / "models" / model_id

    def embeddings(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "embeddings.npy"

    def rows(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "rows.u8"

    def errors(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "errors.jsonl"
