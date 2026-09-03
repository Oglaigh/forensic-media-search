---
name: clip-query-design
description: Use when converting written forensic visual specifications into CLIP-compatible search queries and combining their similarity results.
---

# CLIP Query Design

## Goal

Convert human visual specifications into traceable model queries.

Example specification:

    Houses with red roofs or black doors

Possible query set:

    a house with a red roof
    a house with a black door

## Rules

Preserve the original specification.

Record every generated query.

Do not silently change AND into OR or OR into AND.

For OR semantics:

    score(image) = max(score(image, query_i))

unless another aggregation method is explicitly selected.

Generate text embeddings once for the complete query set.

## Scoring

Cosine similarity is a model similarity measure.

Never describe cosine similarity as probability.

Softmax across candidate queries is relative to those specific queries and is
not an absolute probability that the image matches the specification.

Any percentage shown to a user must have an explicitly documented meaning.

## Verification

For every result it must be possible to determine:

- original specification;
- generated query;
- cosine score;
- aggregation rule.