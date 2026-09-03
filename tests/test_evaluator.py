from __future__ import annotations

from types import MappingProxyType

import pytest

from forensic_media_search.ranking import FusedCandidate, evaluate_files


def candidate(file_id: int, query_id: str, final_rank: int) -> FusedCandidate:
    return FusedCandidate(
        file_id=file_id,
        query_id=query_id,
        scores=MappingProxyType({"siglip2": float(file_id)}),
        ranks=MappingProxyType({"siglip2": 999 - file_id}),
        models_matched="SigLIP2",
        fusion_score=float(file_id * 100),
        final_rank=final_rank,
    )


def test_consensus_precedes_a_single_excellent_query() -> None:
    evaluated = evaluate_files(
        [candidate(1, "dog", 1), candidate(2, "cat", 95), candidate(2, "dog", 96)],
        evaluator_top_k=100,
        query_order=("cat", "dog"),
    )
    assert [item.file_id for item in evaluated] == [2, 1]
    assert [item.strong_query_count for item in evaluated] == [2, 1]


def test_or_inclusion_and_cutoff_ignore_weak_queries() -> None:
    evaluated = evaluate_files(
        [candidate(3, "cat", 101), candidate(3, "dog", 12)],
        evaluator_top_k=100,
        query_order=("cat", "dog"),
    )
    assert len(evaluated) == 1
    assert evaluated[0].file_id == 3
    assert [match.query_id for match in evaluated[0].query_matches] == ["dog"]


def test_reciprocal_quality_orders_within_same_consensus_level() -> None:
    evaluated = evaluate_files(
        [
            candidate(1, "cat", 2), candidate(1, "dog", 50),
            candidate(2, "cat", 10), candidate(2, "dog", 50),
        ],
        evaluator_top_k=100,
    )
    assert [item.file_id for item in evaluated] == [1, 2]


def test_file_is_deduplicated_and_ties_use_manifest_ordinal() -> None:
    evaluated = evaluate_files(
        [candidate(9, "cat", 5), candidate(9, "dog", 6),
         candidate(2, "cat", 5), candidate(2, "dog", 6)],
        evaluator_top_k=10,
    )
    assert [item.file_id for item in evaluated] == [2, 9]
    assert all(len(item.query_matches) == 2 for item in evaluated)


def test_evaluator_depends_only_on_final_ranks() -> None:
    first = evaluate_files([candidate(4, "dog", 7)], evaluator_top_k=10)
    changed = candidate(4, "dog", 7)
    changed = FusedCandidate(
        changed.file_id, changed.query_id,
        MappingProxyType({"clip": -999.0}), MappingProxyType({"clip": 1}),
        "CLIP", -1000.0, changed.final_rank,
    )
    second = evaluate_files([changed], evaluator_top_k=10)
    assert first == second


@pytest.mark.parametrize("top_k,constant", [(0, 60), (10, -1)])
def test_invalid_evaluator_configuration_is_rejected(top_k: int, constant: int) -> None:
    with pytest.raises(ValueError):
        evaluate_files([], evaluator_top_k=top_k, rrf_constant=constant)
