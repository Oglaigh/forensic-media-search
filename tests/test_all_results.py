from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

import forensic_media_search.pipeline as pipeline_module
from forensic_media_search.models import VisionLanguageModel
from forensic_media_search.pipeline import AllResultsConfig, run_all_results
from forensic_media_search.queries import IdentityQueryProcessor, QuerySpec


class FakeSigLIP2(VisionLanguageModel):
    def __init__(self, scores: dict[int, list[float]]) -> None:
        super().__init__(model_id="siglip2", name="fake", device="cpu", revision="test")
        self.scores = scores
        self.query_encode_calls = 0

    def load(self) -> None:
        self._loaded = True

    def encode_queries(self, queries: Sequence[QuerySpec | str]) -> Any:
        self._require_loaded()
        self.query_encode_calls += 1
        return list(range(len(queries)))

    def encode_images(self, images: Sequence[Any]) -> Any:
        self._require_loaded()
        return [image.getpixel((0, 0))[0] for image in images]

    def similarity(self, image_embeddings: Any, text_embeddings: Any) -> Any:
        return [self.scores[value] for value in image_embeddings]

    def unload(self) -> None:
        self._loaded = False


def test_all_results_ranks_every_processable_image_per_query_without_rrf(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for value in (10, 20, 30):
        Image.new("RGB", (1, 1), (value, value, value)).save(evidence / f"{value}.png")
    (evidence / "corrupt.jpg").write_bytes(b"not an image")
    queries = IdentityQueryProcessor().process(["perro", "robot"])
    model = FakeSigLIP2({10: [0.9, 0.1], 20: [0.9, 0.8], 30: [0.2, 0.7]})
    monkeypatch.setattr(
        pipeline_module, "fuse_rankings",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RRF called")),
    )
    output = tmp_path / "output" / "all.csv"
    result = run_all_results(
        AllResultsConfig(evidence, output, queries, batch_size=2),
        model,
        progress=lambda _: None,
    )

    assert model.query_encode_calls == 1
    assert result.metrics.images_discovered == 4
    assert result.metrics.images_processed == 3
    assert result.metrics.decode_errors == 1
    assert len(result.records) == 6
    by_query = {
        query_id: [record for record in result.records if record.query_id == query_id]
        for query_id in ("q0001", "q0002")
    }
    assert [record.rank for record in by_query["q0001"]] == [1, 2, 3]
    assert [record.file_id for record in by_query["q0001"]] == [0, 1, 2]
    assert [record.file_id for record in by_query["q0002"]] == [1, 2, 0]
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert list(rows[0]) == [
        "FilePath", "Query", "SigLIP2Score", "Rank", "Model", "Revision"
    ]
