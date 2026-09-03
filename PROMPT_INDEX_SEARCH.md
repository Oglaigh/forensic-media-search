# Prompt — Persistent INDEX + Exact SEARCH

Implement persistent INDEX + exact SEARCH for `forensic-media-search`.

Follow `AGENTS.md`.

Use specialized agents where appropriate:

- `forensic-architect` for index architecture, metadata, lifecycle, compatibility and forensic invariants;
- `vision-model-specialist` for the exact persisted representation and DIRECT vs INDEXED equivalence;
- `indexing-engineer` for persistent storage, mmap/memmap, bounded-memory writers, chunked exact search, resume and performance;
- `forensic-reviewer` after implementation for a read-only final review.

Use the relevant skills:

- `.agents/skills/evidence-indexing/SKILL.md`
- `.agents/skills/exact-vector-search/SKILL.md`
- `.agents/skills/embedding-equivalence/SKILL.md`
- `.agents/skills/forensic-validation/SKILL.md`

## Objective

Change the expensive workflow from:

```text
every search
    -> discover/read/decode every image
    -> run SigLIP2 image encoder
    -> run CLIP image encoder
    -> score queries
```

to:

```text
INDEX once
    -> discover/read/decode every image
    -> persist normalized SigLIP2 image embeddings
    -> persist normalized CLIP image embeddings

SEARCH many times
    -> encode text only
    -> exact similarity against persisted image embeddings
    -> existing Top-K
    -> existing model RRF
    -> existing audit report
    -> existing Evaluator
    -> final report
```

The baseline indexed implementation must not reduce quality.

## Constraints

Keep the existing SigLIP2/CLIP models, revisions, preprocessing, normalization, Top-K, RRF, audit-report semantics, Evaluator semantics, and DIRECT path.

Persist image embeddings initially as normalized `float32`.

Do not introduce ANN, HNSW, IVF, PQ, vector quantization, lower-resolution inference, smaller models, thumbnails, or sampling.

## INDEX requirements

Create a persistent derived index outside evidence.

Use deterministic manifest ordinal -> embedding row mapping.

Prefer a simple contiguous mmap/memmap-compatible format.

Record index version, BUILDING/COMPLETE state, manifest identity, model names/revisions, weight hashes when available, preprocessing identity, dimensions, dtype, normalization, row counts and timestamps.

Support safe resume. A partially written row must never be treated as committed.

## SEARCH requirements

SEARCH over an existing COMPLETE index.

SEARCH must not open/decode evidence images and must not invoke image encoders.

Implement exact chunked search suitable for millions of embeddings:

```text
for chunk:
    scores = chunk @ query_matrix.T
    update exact Top-K per model + query
```

Do not require the full index to fit in RAM or VRAM.

Use CUDA for chunk matrix multiplication when beneficial while preserving exact ranking semantics.

Reject incompatible or incomplete indexes.

## CLI

Inspect the existing CLI first and choose the smallest coherent extension.

Preferred conceptual interface:

```powershell
python ... index `
  --directory /evidence `
  --index /output/case-001.index `
  --batch-size 64
```

```powershell
python ... search `
  --index /output/case-001.index `
  --query "es un perro" `
  --query "es un gato" `
  --top-k 5000 `
  --evaluation-top-k 300 `
  --final-top-k 200 `
  --output /output/report.csv
```

Do not force subcommands if they create unnecessary refactoring; explain the chosen interface.

## Required equivalence test

Run the same evidence and queries through DIRECT and INDEXED modes.

Compare SigLIP2 scores, CLIP scores, SigLIP2 ranks, CLIP ranks, Top-K membership, RRF, FinalRank, audit report and Evaluator/final report when enabled.

Top-K membership and ranks must be identical.

Any floating-point score difference must have an explicit measured tolerance and root-cause explanation.

## Performance reporting

Report INDEX separately from SEARCH.

INDEX:
- images discovered;
- images indexed;
- decode errors;
- model processing time;
- wall time;
- images/sec.

SEARCH:
- index open time;
- text encoding time;
- SigLIP2 exact-search time;
- CLIP exact-search time;
- RRF time;
- Evaluator time;
- total search time.

## Work order

1. Inspect current architecture.
2. Ask `forensic-architect` for a focused design.
3. Ask `vision-model-specialist` to define exact persisted embeddings and equivalence requirements.
4. Produce a short implementation plan.
5. Implement the minimum necessary changes.
6. Ask `indexing-engineer` to validate storage/search/resume/performance.
7. Run tests.
8. Run DIRECT vs INDEXED equivalence.
9. Ask `forensic-reviewer` for a read-only review.
10. Resolve in-scope Critical/High findings.
11. Report results.

## Final report

Include:
1. architecture chosen;
2. specialist recommendations;
3. files created/modified;
4. exact on-disk index format;
5. metadata stored;
6. resume strategy;
7. exact search algorithm;
8. memory/RAM/VRAM strategy;
9. INDEX command;
10. SEARCH command;
11. DIRECT vs INDEXED equivalence results;
12. index benchmark;
13. search benchmark;
14. tests executed;
15. forensic-reviewer findings;
16. numerical differences, if any.
