from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pytest
from PIL import Image

from forensic_media_search.models import VisionLanguageModel
from forensic_media_search.pipeline import SearchConfig, run_search
from forensic_media_search.queries import IdentityQueryProcessor, QuerySpec


class FakeModel(VisionLanguageModel):
    def __init__(self, model_id: str, scores: dict[int, list[float]]) -> None:
        super().__init__(model_id=model_id, name=f"fake-{model_id}", device="cpu", revision="test")
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


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_fuses_by_file_and_query_and_writes_empty_model_fields(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    output = tmp_path / "output" / "report.csv"
    evidence.mkdir()
    paths = []
    for value in (10, 20, 30):
        path = evidence / f"{value}.png"
        Image.new("RGB", (1, 1), (value, value, value)).save(path)
        paths.append(path)
    before = {path: (_digest(path), path.stat().st_mtime_ns) for path in paths}
    queries = IdentityQueryProcessor().process(["Es un perro", "Es un robot"])

    siglip = FakeModel("siglip2", {10: [0.9, 0.8], 20: [0.7, 0.1], 30: [0.2, 0.7]})
    clip = FakeModel("clip", {10: [0.8, 0.1], 20: [0.6, 0.9], 30: [0.1, 0.8]})
    result = run_search(
        SearchConfig(evidence, output, queries, top_k=1, batch_size=2, rrf_constant=60),
        (siglip, clip),
        progress=lambda _: None,
    )

    assert siglip.query_encode_calls == clip.query_encode_calls == 1
    assert {(item.file_id, item.query_id) for item in result.candidates} == {
        (0, "q0001"), (0, "q0002"), (1, "q0002")
    }
    assert {path: (_digest(path), path.stat().st_mtime_ns) for path in paths} == before
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    clip_only = next(row for row in rows if row["FilePath"].endswith("20.png"))
    assert clip_only["SigLIP2Score"] == ""
    assert clip_only["SigLIP2Rank"] == ""
    assert clip_only["CLIPCos"] != ""
    assert clip_only["OriginalQuery"] == "Es un robot"


def test_pipeline_rejects_report_inside_evidence_before_writing(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    queries = IdentityQueryProcessor().process(["query"])
    models = (FakeModel("siglip2", {}), FakeModel("clip", {}))
    output = evidence / "report.csv"
    with pytest.raises(ValueError, match="outside evidence"):
        run_search(
            SearchConfig(evidence, output, queries, 1, 1, 60),
            models,
            progress=lambda _: None,
        )
    assert not output.exists()


def test_pipeline_writes_audit_then_one_row_per_file_final_report(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for value in (10, 20, 30):
        Image.new("RGB", (1, 1), (value, value, value)).save(evidence / f"{value}.png")
    audit = tmp_path / "output" / "audit.csv"
    final = tmp_path / "output" / "final.csv"
    queries = IdentityQueryProcessor().process(["gato", "perro"])
    siglip = FakeModel("siglip2", {10: [0.9, 0.9], 20: [0.8, 0.1], 30: [0.1, 0.8]})
    clip = FakeModel("clip", {10: [0.9, 0.9], 20: [0.8, 0.1], 30: [0.1, 0.8]})
    result = run_search(
        SearchConfig(
            evidence, audit, queries, top_k=3, batch_size=2, rrf_constant=60,
            final_output_path=final, evaluator_top_k=1, evaluator_rrf_constant=60,
        ),
        (siglip, clip), progress=lambda _: None,
    )
    assert audit.exists() and final.exists()
    with audit.open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 6
    with final.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["StrongQueryCount"] == "2"
    assert rows[0]["BestQueryRank"] == "1"
    assert rows[0]["EvaluatorTopK"] == "1"
    assert rows[0]["EvaluatorRRFConstant"] == "60"
    assert rows[0]["AuditReport"] == str(audit.resolve())
    matches = json.loads(rows[0]["QueryMatches"])
    assert [item["QueryId"] for item in matches] == ["q0001", "q0002"]
    assert len(result.evaluated_candidates) == 1


@pytest.mark.parametrize("suffix", ["", ".manifest.jsonl", ".errors.jsonl"])
def test_pipeline_rejects_colliding_final_destination(
    tmp_path: Path, suffix: str
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    queries = IdentityQueryProcessor().process(["query"])
    output = tmp_path / "report.csv"
    final = output.with_suffix(output.suffix + suffix) if suffix else output
    with pytest.raises(ValueError, match="different"):
        run_search(
            SearchConfig(
                evidence, output, queries, 1, 1, 60,
                final_output_path=final, evaluator_top_k=1,
            ),
            (FakeModel("siglip2", {}), FakeModel("clip", {})),
            progress=lambda _: None,
        )
