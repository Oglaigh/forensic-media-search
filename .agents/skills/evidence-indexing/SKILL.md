---
name: evidence-indexing
description: Build or review persistent forensic image indexes with deterministic manifest mapping, read-only evidence, crash-safe resume, and index integrity checks.
---

# Evidence Indexing

Use this skill when creating or modifying persistent image indexing.

## Goal

Build a persistent derived index over read-only evidence so image encoders run once per evidence item instead of once per search.

## Invariants

- Evidence remains read-only.
- Index/output is outside evidence.
- Discovery order is deterministic.
- One manifest ordinal maps to exactly one embedding row per model.
- An incomplete index is never exposed as COMPLETE.
- Resume never duplicates committed rows.
- A partial row is never considered committed.

## Recommended logical layout

```text
case.index/
    index.json
    manifest.jsonl
    journal.jsonl
    siglip2/
        embeddings.f32
    clip/
        embeddings.f32
```

Do not create one embedding file per image.

## Metadata

Record at minimum:

- index_version
- state: BUILDING or COMPLETE
- created_at / completed_at
- image_count
- committed_rows
- manifest identity/hash
- embedding dtype
- normalization
- embedding dimension per model
- model name
- model revision
- preprocessor identity/version
- weight SHA256 when available

## Build

For each deterministic manifest item:

1. open evidence read-only;
2. decode with the current preprocessing path;
3. generate the same image feature used by DIRECT mode;
4. normalize exactly as DIRECT mode;
5. write the row matching the manifest ordinal;
6. durably checkpoint only after the row is committed.

## Resume

Before continuing, validate index version, manifest identity, model/revision/preprocessing/dimension/dtype, committed row count and storage length.

Fail closed on ambiguity.

## Completion

Mark COMPLETE only after every expected row is committed and storage shape/size matches metadata.

SEARCH must reject BUILDING indexes.
