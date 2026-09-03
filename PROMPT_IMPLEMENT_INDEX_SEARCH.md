# Phase 2 — Implement the approved INDEX + SEARCH architecture

Implement the INDEX + exact SEARCH architecture approved in the previous architecture phase.

Follow `AGENTS.md` and the relevant skills:

- evidence-indexing
- exact-vector-search
- embedding-equivalence
- forensic-validation

The main thread owns all application code changes.

Use `indexing_engineer` and `vision_model_specialist` as read-only validators while implementing.

Requirements:

- persist normalized float32 SigLIP2 and CLIP image embeddings;
- deterministic manifest ordinal -> embedding row;
- BUILDING/COMPLETE lifecycle;
- crash-safe resume;
- exact chunked search;
- bounded RAM/VRAM;
- multi-query matrix scoring when safe;
- preserve Top-K per model+query;
- preserve RRF per file+query;
- preserve audit report;
- preserve current Evaluator;
- keep DIRECT mode.

After implementation:

1. run unit/integration tests;
2. execute DIRECT vs INDEXED equivalence using `embedding-equivalence`;
3. benchmark INDEX separately from SEARCH;
4. ask `forensic_reviewer` for a read-only review;
5. resolve in-scope Critical and High findings;
6. report final architecture, files changed, commands, equivalence results, benchmark and reviewer findings.
