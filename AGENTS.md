# forensic-media-search

## Purpose

`forensic-media-search` is an offline forensic-media triage system for ranking candidate images for human review.

Model scores, ranks, fusion scores, and evaluation ranks are not probabilities and are not evidentiary conclusions.

## Critical invariants

- Evidence is strictly read-only.
- Generated artifacts must be outside the evidence directory.
- Never modify evidence files, names, contents, metadata, or timestamps.
- Prefer recall over precision.
- Preserve deterministic ordering whenever practical.
- Never silently fall back from CUDA to CPU when CUDA is required.
- Model names, revisions, preprocessing, and weight hashes must remain traceable.
- Do not compare raw model scores across incompatible models.
- Similarity scores are not probabilities.

## Current pipeline

```text
DISCOVERY
    ↓
DIRECT MODEL INFERENCE
    ↓
TOP-K PER MODEL + QUERY
    ↓
RRF PER FILE + QUERY
    ↓
AUDIT REPORT
    ↓
QUERY EVALUATOR
    ↓
FINAL REPORT
```

The audit report intentionally has one row per `file + query`.

The final report intentionally has one row per file.

For multiple queries:

- inclusion is OR-based;
- matching multiple queries increases priority;
- failing one query must not eliminate a strong hit on another query.

## Target architecture

```text
EVIDENCE
   ↓
INDEX ONCE
   ↓
PERSISTENT IMAGE EMBEDDINGS
   ↓
SEARCH MANY TIMES
   ↓
TOP-K PER MODEL + QUERY
   ↓
RRF PER FILE + QUERY
   ↓
AUDIT REPORT
   ↓
QUERY EVALUATOR
   ↓
FINAL REPORT
```

SEARCH must not decode evidence images or run image encoders.

The first indexed-search implementation must be exact.

Do not introduce ANN, HNSW, IVF, PQ, quantization, lower-resolution inference, smaller models, thumbnails, or sampling unless explicitly requested.

## Direct vs indexed equivalence

Keep DIRECT mode for validation.

For identical evidence, manifest, models, revisions, preprocessing, queries, Top-K, RRF and Evaluator configuration:

- Top-K membership must match;
- model ranks must match;
- RRF ordering must match;
- FinalRank must match;
- final Evaluator ordering must match.

Any floating-point score difference must be measured and explained.

## Delegation

Use custom agents for focused read-heavy analysis.

- `forensic_architect`: architecture, index layout, metadata, lifecycle, resume, forensic invariants.
- `vision_model_specialist`: SigLIP2/CLIP feature semantics and DIRECT-vs-INDEXED numerical equivalence.
- `indexing_engineer`: memmap/mmap, chunked exact search, resume, bounded memory and throughput.
- `forensic_reviewer`: final read-only forensic/correctness review.

The primary agent owns implementation and integration.

Do not have multiple agents modify overlapping files concurrently.

Preferred workflow:

1. delegate analysis to specialists;
2. wait for all specialist results;
3. primary agent proposes one architecture;
4. primary agent implements;
5. specialists validate;
6. forensic reviewer performs final read-only review.

## Skills

Use these reusable procedures when applicable:

- `evidence-indexing`
- `exact-vector-search`
- `embedding-equivalence`
- `forensic-validation`

## Scope

For the indexed-search architecture iteration, stay within:

```text
PERSISTENT IMAGE EMBEDDINGS
+
EXACT INDEXED SEARCH
+
RESUMABLE INDEX BUILD
+
DIRECT/INDEXED EQUIVALENCE VALIDATION
```

Do not add unrelated databases, APIs, Kafka, Redis, UI, OCR, video processing, cloud inference or microservices.
