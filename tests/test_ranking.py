from __future__ import annotations

import pytest

from forensic_media_search.queries import IdentityQueryProcessor
from forensic_media_search.ranking import PerQueryTopK, RankedCandidate, fuse_rankings


QUERIES = IdentityQueryProcessor().process(["Es un perro", "Es un robot"])


def test_dominant_query_cannot_displace_other_query_candidates() -> None:
    topk = PerQueryTopK("siglip2", QUERIES, k=2)
    topk.update_batch(
        [0, 1, 2, 3],
        [
            [0.99, 0.10],
            [0.98, 0.20],
            [0.97, 0.90],
            [0.96, 0.80],
        ],
    )

    ranked = topk.ranked()
    assert [item.file_id for item in ranked["q0001"]] == [0, 1]
    assert [item.file_id for item in ranked["q0002"]] == [2, 3]


def test_each_model_query_heap_retains_up_to_k() -> None:
    for model_id in ("siglip2", "clip"):
        topk = PerQueryTopK(model_id, QUERIES, k=3)
        topk.update_batch(
            range(10),
            [[index / 10, (10 - index) / 10] for index in range(10)],
        )
        assert topk.counts() == {"q0001": 3, "q0002": 3}
        assert topk.retained_count == 6


def test_same_image_can_rank_for_multiple_queries() -> None:
    topk = PerQueryTopK("clip", QUERIES, k=2)
    topk.update_batch([7, 8], [[0.8, 0.7], [0.2, 0.1]])

    ranked = topk.ranked()
    assert ranked["q0001"][0].file_id == 7
    assert ranked["q0002"][0].file_id == 7


def test_fusion_key_is_file_and_query_not_file_alone() -> None:
    siglip = {
        "q0001": (RankedCandidate("siglip2", "q0001", 42, 0.31, 12),),
        "q0002": (RankedCandidate("siglip2", "q0002", 42, 0.22, 820),),
    }
    clip = {
        "q0001": (RankedCandidate("clip", "q0001", 42, 0.29, 19),),
        "q0002": (),
    }

    fused = fuse_rankings(
        {"siglip2": siglip, "clip": clip},
        query_order=["q0001", "q0002"],
    )

    assert {(item.file_id, item.query_id) for item in fused} == {
        (42, "q0001"),
        (42, "q0002"),
    }
    by_query = {item.query_id: item for item in fused}
    assert by_query["q0001"].models_matched == "SigLIP2+CLIP"
    assert by_query["q0001"].fusion_score == pytest.approx(1 / 72 + 1 / 79)
    assert by_query["q0002"].models_matched == "SigLIP2"
    assert by_query["q0002"].clip_rank is None
    assert by_query["q0002"].clip_cos is None


def test_topk_and_fusion_are_deterministic_on_ties() -> None:
    topk = PerQueryTopK("siglip2", QUERIES, k=2)
    topk.update_batch([9, 2, 5], [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    ranked = topk.ranked()
    assert [item.file_id for item in ranked["q0001"]] == [2, 5]
    assert [item.file_id for item in ranked["q0002"]] == [2, 5]

    first = fuse_rankings(
        {"siglip2": ranked},
        query_order=["q0001", "q0002"],
    )
    second = fuse_rankings(
        {"siglip2": ranked},
        query_order=["q0001", "q0002"],
    )
    assert [(item.query_id, item.file_id) for item in first] == [
        ("q0001", 2),
        ("q0001", 5),
        ("q0002", 2),
        ("q0002", 5),
    ]
    assert [(item.query_id, item.final_rank) for item in first] == [
        ("q0001", 1),
        ("q0001", 2),
        ("q0002", 1),
        ("q0002", 2),
    ]
    assert first == second


def test_invalid_scores_and_dimensions_are_rejected() -> None:
    topk = PerQueryTopK("clip", QUERIES, k=1)
    with pytest.raises(ValueError, match="one value per query"):
        topk.update_batch([1], [[0.1]])
    with pytest.raises(ValueError, match="finite"):
        topk.update_batch([1], [[float("nan"), 0.2]])
