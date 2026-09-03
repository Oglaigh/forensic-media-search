from __future__ import annotations

import sys
from typing import Any, Sequence

import pytest

from forensic_media_search.models import (
    ModelNotLoadedError,
    OpenAIClipModel,
    SigLIP2Model,
    VisionLanguageModel,
)
from forensic_media_search.queries import IdentityQueryProcessor, QuerySpec


class FakeModel(VisionLanguageModel):
    def __init__(self) -> None:
        super().__init__(model_id="fake", name="fake-v1", device="cpu")
        self.query_encode_calls = 0

    def load(self) -> None:
        self._loaded = True

    def encode_queries(self, queries: Sequence[QuerySpec | str]) -> Any:
        self._require_loaded()
        self.query_encode_calls += 1
        return self.query_texts(queries)

    def encode_images(self, images: Sequence[Any]) -> Any:
        self._require_loaded()
        return list(images)

    def unload(self) -> None:
        self._loaded = False


def test_identity_queries_are_stable_and_traceable() -> None:
    original = ["Es un perro", "Es un robot", "Es un perro"]
    processed = IdentityQueryProcessor().process(original)

    assert [item.query_id for item in processed] == ["q0001", "q0002", "q0003"]
    assert [item.original_query for item in processed] == original
    assert [item.model_query for item in processed] == original


def test_model_contract_requires_explicit_lifecycle() -> None:
    model = FakeModel()
    queries = IdentityQueryProcessor().process(["perro", "robot"])

    with pytest.raises(ModelNotLoadedError):
        model.encode_queries(queries)

    with model:
        embeddings = model.encode_queries(queries)
        assert embeddings == ["perro", "robot"]
        assert model.query_encode_calls == 1
        assert model.is_loaded

    assert not model.is_loaded


def test_adapter_construction_does_not_import_or_download_backends() -> None:
    clip_was_loaded = "clip" in sys.modules
    transformers_was_loaded = "transformers" in sys.modules

    clip_model = OpenAIClipModel(device="cpu")
    siglip_model = SigLIP2Model(device="cpu")

    assert not clip_model.is_loaded
    assert not siglip_model.is_loaded
    assert ("clip" in sys.modules) is clip_was_loaded
    assert ("transformers" in sys.modules) is transformers_was_loaded


def test_query_specs_reject_empty_text_without_rewriting_it() -> None:
    with pytest.raises(ValueError, match="original_query"):
        QuerySpec("q1", "  ", "valid")
    with pytest.raises(ValueError, match="model_query"):
        QuerySpec("q1", "valid", "  ")

