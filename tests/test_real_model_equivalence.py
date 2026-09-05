"""Opt-in CUDA equivalence test using the pinned production adapters.

Run with RUN_REAL_MODEL_EQUIVALENCE=1 when model weights are available offline.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from PIL import Image

from forensic_media_search.cli import OPENAI_CLIP_REVISION, SIGLIP2_REVISION
from forensic_media_search.indexing import IndexBuildConfig, build_index
from forensic_media_search.models import OpenAIClipModel, SigLIP2Model
from forensic_media_search.pipeline import SearchConfig, run_search
from forensic_media_search.queries import IdentityQueryProcessor
from forensic_media_search.search import IndexedSearchConfig, run_indexed_search


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_EQUIVALENCE") != "1",
    reason="set RUN_REAL_MODEL_EQUIVALENCE=1 with offline model caches available",
)


def _models(device: str):
    return (
        SigLIP2Model(
            device=device, revision=SIGLIP2_REVISION,
            cache_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
        ),
        OpenAIClipModel(
            "ViT-B/32", device=device, revision=OPENAI_CLIP_REVISION,
            download_root=os.environ.get("CLIP_CACHE_DIR", "/root/.cache/clip"),
        ),
    )


def _ranking_signature(candidate):
    return (
        candidate.file_id, candidate.query_id,
        candidate.siglip2_rank, candidate.clip_rank,
        candidate.models_matched, candidate.final_rank, candidate.fusion_score,
    )


def test_real_direct_indexed_equivalence(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("real equivalence baseline requires CUDA")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    colors = ((0, 0, 0), (255, 255, 255), (255, 0, 0), (254, 0, 0), (0, 0, 255))
    for ordinal, color in enumerate(colors):
        Image.new("RGB", (224, 224), color).save(evidence / f"{ordinal:02d}.png")
    queries = IdentityQueryProcessor().process(("a red square", "a dark image"))

    direct = run_search(
        SearchConfig(
            evidence, tmp_path / "direct.csv", queries,
            top_k=len(colors), batch_size=2, rrf_constant=60,
            final_output_path=tmp_path / "direct-final.csv",
            evaluator_top_k=3, evaluator_rrf_constant=60,
        ),
        _models("cuda"), progress=lambda _: None,
    )
    index = tmp_path / "case.index"
    build_index(
        IndexBuildConfig(evidence, index, batch_size=3, checkpoint_rows=4),
        _models("cuda"), progress=lambda _: None,
    )
    indexed = run_indexed_search(
        IndexedSearchConfig(
            index, tmp_path / "indexed.csv", queries,
            top_k=len(colors), chunk_rows=4, rrf_constant=60,
            final_output_path=tmp_path / "indexed-final.csv",
            evaluator_top_k=3, evaluator_rrf_constant=60,
        ),
        _models("cuda"), progress=lambda _: None,
    )

    assert [_ranking_signature(item) for item in indexed.candidates] == [
        _ranking_signature(item) for item in direct.candidates
    ]
    assert indexed.evaluated_candidates == direct.evaluated_candidates
    direct_by_key = {(item.file_id, item.query_id): item for item in direct.candidates}
    deltas: list[float] = []
    for actual in indexed.candidates:
        expected = direct_by_key[(actual.file_id, actual.query_id)]
        for left, right in (
            (actual.siglip2_score, expected.siglip2_score),
            (actual.clip_cos, expected.clip_cos),
        ):
            if left is not None and right is not None:
                delta = abs(left - right)
                assert math.isfinite(delta)
                deltas.append(delta)
    print(f"DIRECT/INDEXED maximum absolute score delta: {max(deltas, default=0.0):.12g}")
