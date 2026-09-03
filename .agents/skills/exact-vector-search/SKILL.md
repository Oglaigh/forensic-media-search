---
name: exact-vector-search
description: Implement or validate exact chunked search over persisted normalized image embeddings without decoding images or using approximate nearest-neighbor indexes.
---

# Exact Vector Search

## Goal

Search millions of persisted embeddings exactly, without decoding evidence images and without ANN.

## Preconditions

- index state is COMPLETE;
- metadata is compatible;
- image embeddings match DIRECT mode;
- text embeddings use the same normalization as DIRECT mode.

## Similarity

For L2-normalized embeddings:

```text
cosine(image, text) == image_embedding @ text_embedding
```

Validate normalization before relying on this equality.

## Chunked search

```text
for embedding_chunk in mmap_matrix:
    scores = embedding_chunk @ query_matrix.T
    update exact Top-K per model + query
```

Requirements:

- exact results;
- Top-K independent per model + query;
- deterministic tie handling;
- bounded memory;
- no requirement to store all scores.

## Multiple queries

Prefer one matrix multiplication per chunk:

```text
scores = chunk @ query_matrix.T
```

when this preserves current ranking semantics.

## GPU

CUDA may process chunks without requiring the full index in VRAM.

## Forbidden baseline methods

Do not use ANN, HNSW, IVF, PQ, vector quantization or approximate GPU indexes.

## Validation

Compare chunked exact search against full-matrix exact search where feasible and against DIRECT search.
