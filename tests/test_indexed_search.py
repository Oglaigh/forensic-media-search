from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from PIL import Image

from forensic_media_search.indexing import IndexBuildConfig, build_index, verify_index
from forensic_media_search.indexing.format import ROW_DECODE_ERROR, ROW_VALID
from forensic_media_search.models import VisionLanguageModel
from forensic_media_search.pipeline import SearchConfig, run_search
from forensic_media_search.queries import IdentityQueryProcessor, QuerySpec
from forensic_media_search.search import IndexedSearchConfig, run_indexed_search


class _State:
    def state_dict(self) -> dict:
        return {}


class FakeEmbeddingModel(VisionLanguageModel):
    def __init__(self, model_id: str, *, fail_after: int | None = None) -> None:
        super().__init__(
            model_id=model_id, name=f"fake-{model_id}", device="cpu", revision="test-v1"
        )
        self._model: Any = None
        self.fail_after = fail_after
        self.image_calls = 0

    def load(self) -> None:
        self._model = _State()
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        self._require_loaded()
        self.image_calls += 1
        if self.fail_after is not None and self.image_calls > self.fail_after:
            raise RuntimeError("injected inference failure")
        values = np.asarray([
            [image.getpixel((0, 0))[0], 255 - image.getpixel((0, 0))[0]]
            for image in images
        ], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def encode_queries(self, queries: Sequence[QuerySpec | str]) -> np.ndarray:
        self._require_loaded()
        basis = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        return basis[:len(queries)]

    def similarity(self, images: Any, texts: Any) -> np.ndarray:
        images = np.asarray(images, dtype=np.float32)
        texts = np.asarray(texts, dtype=np.float32)
        images = images / np.linalg.norm(images, axis=1, keepdims=True)
        texts = texts / np.linalg.norm(texts, axis=1, keepdims=True)
        return images @ texts.T


def _models(**kwargs: Any) -> tuple[FakeEmbeddingModel, FakeEmbeddingModel]:
    return FakeEmbeddingModel("siglip2", **kwargs), FakeEmbeddingModel("clip")


def _evidence(tmp_path: Path, *, corrupt: bool = False) -> tuple[Path, list[Path]]:
    root = tmp_path / "evidence"
    root.mkdir()
    paths: list[Path] = []
    for index, value in enumerate((10, 100, 240)):
        path = root / f"{index}.png"
        Image.new("RGB", (2, 2), (value, value, value)).save(path)
        paths.append(path)
    if corrupt:
        path = root / "3.png"
        path.write_bytes(b"not-an-image")
        paths.append(path)
    return root, paths


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int]]:
    return {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in paths
    }


def test_build_verify_and_exact_multi_chunk_search(tmp_path: Path) -> None:
    evidence, evidence_paths = _evidence(tmp_path, corrupt=True)
    before = _snapshot(evidence_paths)
    index = tmp_path / "case.index"

    built = build_index(
        IndexBuildConfig(evidence, index, batch_size=2, checkpoint_rows=2),
        _models(), progress=lambda _: None,
    )
    assert built.image_count == 4
    assert verify_index(index, full=True, require_complete=True).state == "COMPLETE"
    assert _snapshot(evidence_paths) == before

    metadata = json.loads((index / "index.json").read_text(encoding="utf-8"))
    assert metadata["row_mapping"] == "row_equals_manifest_ordinal"
    for model_id in ("siglip2", "clip"):
        embeddings = np.load(index / "models" / model_id / "embeddings.npy", mmap_mode="r")
        states = np.memmap(index / "models" / model_id / "rows.u8", dtype=np.uint8, mode="r")
        assert embeddings.dtype == np.float32 and embeddings.shape == (4, 2)
        assert states.tolist() == [ROW_VALID, ROW_VALID, ROW_VALID, ROW_DECODE_ERROR]

    queries = IdentityQueryProcessor().process(("bright", "dark"))
    output = tmp_path / "reports" / "audit.csv"
    search_models = _models()
    result = run_indexed_search(
        IndexedSearchConfig(index, output, queries, top_k=2, chunk_rows=2, rrf_constant=60),
        search_models, progress=lambda _: None,
    )
    assert output.is_file()
    assert output.with_suffix(output.suffix + ".run.json").is_file()
    assert {(item.file_id, item.query_id) for item in result.candidates} == {
        (2, "q0001"), (1, "q0001"), (0, "q0002"), (1, "q0002")
    }
    assert all(model.image_calls == 0 for model in search_models)
    assert result.execution_counters == {
        "images_decoded": 0,
        "siglip2_images_encoded": 0,
        "clip_images_encoded": 0,
    }
    assert _snapshot(evidence_paths) == before


def test_direct_and_indexed_rankings_are_identical(tmp_path: Path) -> None:
    evidence, _ = _evidence(tmp_path)
    queries = IdentityQueryProcessor().process(("bright", "dark"))
    direct = run_search(
        SearchConfig(
            evidence, tmp_path / "direct-audit.csv", queries,
            top_k=2, batch_size=2, rrf_constant=60,
            final_output_path=tmp_path / "direct-final.csv",
            evaluator_top_k=2, evaluator_rrf_constant=60,
        ),
        _models(), progress=lambda _: None,
    )
    index = tmp_path / "case.index"
    build_index(
        IndexBuildConfig(evidence, index, batch_size=2, checkpoint_rows=2),
        _models(), progress=lambda _: None,
    )
    indexed = run_indexed_search(
        IndexedSearchConfig(
            index, tmp_path / "indexed-audit.csv", queries,
            top_k=2, chunk_rows=2, rrf_constant=60,
            final_output_path=tmp_path / "indexed-final.csv",
            evaluator_top_k=2, evaluator_rrf_constant=60,
        ),
        _models(), progress=lambda _: None,
    )
    assert indexed.candidates == direct.candidates
    assert indexed.evaluated_candidates == direct.evaluated_candidates


def test_resume_uses_only_durable_committed_prefix(tmp_path: Path) -> None:
    evidence, _ = _evidence(tmp_path)
    index = tmp_path / "case.index"
    with pytest.raises(RuntimeError, match="injected"):
        build_index(
            IndexBuildConfig(evidence, index, batch_size=1, checkpoint_rows=1),
            _models(fail_after=1), progress=lambda _: None,
        )
    metadata = json.loads((index / "index.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "BUILDING"
    assert metadata["models"]["siglip2"]["committed_prefix"] == 1
    with (index / "build-journal.jsonl").open("ab") as stream:
        stream.write(b'{"crash_truncated":')

    result = build_index(
        IndexBuildConfig(
            evidence, index, batch_size=1, checkpoint_rows=1, resume=True
        ),
        _models(), progress=lambda _: None,
    )
    assert result.resumed
    assert verify_index(index, full=True, require_complete=True).state == "COMPLETE"


def test_complete_index_is_immutable_and_corruption_is_detected(tmp_path: Path) -> None:
    evidence, _ = _evidence(tmp_path)
    index = tmp_path / "case.index"
    build_index(
        IndexBuildConfig(evidence, index, batch_size=2, checkpoint_rows=2),
        _models(), progress=lambda _: None,
    )
    with pytest.raises(ValueError, match="BUILDING"):
        build_index(
            IndexBuildConfig(evidence, index, resume=True),
            _models(), progress=lambda _: None,
        )

    rows = index / "models" / "clip" / "rows.u8"
    with rows.open("r+b") as stream:
        stream.seek(0)
        stream.write(bytes((99,)))
    with pytest.raises(ValueError, match="clip rows hash mismatch"):
        verify_index(index)


def test_quick_verify_hashes_a_changed_complete_file(tmp_path: Path) -> None:
    evidence, _ = _evidence(tmp_path)
    index = tmp_path / "case.index"
    build_index(
        IndexBuildConfig(evidence, index, batch_size=2, checkpoint_rows=2),
        _models(), progress=lambda _: None,
    )
    embeddings = index / "models" / "siglip2" / "embeddings.npy"
    with embeddings.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes((value[0] ^ 1,)))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_index(index)


def test_resume_authenticates_manifest_before_loading_models(tmp_path: Path) -> None:
    evidence, _ = _evidence(tmp_path)
    index = tmp_path / "case.index"
    with pytest.raises(RuntimeError):
        build_index(
            IndexBuildConfig(evidence, index, batch_size=1, checkpoint_rows=1),
            _models(fail_after=1), progress=lambda _: None,
        )
    with (index / "manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    resume_models = _models()
    with pytest.raises(ValueError, match="manifest size mismatch"):
        build_index(
            IndexBuildConfig(evidence, index, batch_size=1, checkpoint_rows=1, resume=True),
            resume_models, progress=lambda _: None,
        )
    assert all(not model.is_loaded for model in resume_models)


def test_search_rejects_building_index(tmp_path: Path) -> None:
    evidence, _ = _evidence(tmp_path)
    index = tmp_path / "case.index"
    with pytest.raises(RuntimeError):
        build_index(
            IndexBuildConfig(evidence, index, batch_size=1, checkpoint_rows=1),
            _models(fail_after=0), progress=lambda _: None,
        )
    queries = IdentityQueryProcessor().process(("query",))
    with pytest.raises(ValueError, match="COMPLETE"):
        run_indexed_search(
            IndexedSearchConfig(
                index, tmp_path / "audit.csv", queries,
                top_k=1, chunk_rows=1, rrf_constant=60,
            ),
            _models(), progress=lambda _: None,
        )
