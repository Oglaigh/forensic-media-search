---
name: embedding-equivalence
description: Validate that persistent INDEXED search reproduces DIRECT SigLIP2 and CLIP scores, Top-K membership, ranks, RRF, and final evaluator ordering.
---

# Embedding Equivalence Validation

## Goal

Demonstrate that INDEXED search preserves DIRECT search semantics.

Do not accept merely similar-looking rankings.

## Procedure

1. Select deterministic evidence.
2. Record manifest order.
3. Choose representative single- and multi-query runs.
4. Run DIRECT mode.
5. Build the index from the same evidence.
6. Run INDEXED mode with identical configuration.
7. Compare all outputs.

## Compare

- SigLIP2 scores
- CLIP scores
- SigLIP2 ranks
- CLIP ranks
- Top-K membership per model + query
- RRF/FusionScore
- FinalRank per file+query
- audit rows
- Evaluator/final report when enabled

## Acceptance

Baseline expectation:

- identical Top-K membership;
- identical ranks;
- identical RRF ordering;
- identical final ordering.

If scores differ, quantify maximum error, explain the cause, and prove ranking is unchanged.

## Edge cases

Test:

- K < N
- K == N
- K > N
- one query
- multiple queries
- ties or near-ties
- final partial chunk

## Forbidden shortcuts

Do not compare only a small prefix, use float16 for baseline equivalence, or use approximate search.
