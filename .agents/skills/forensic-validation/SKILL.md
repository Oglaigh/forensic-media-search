---
name: forensic-validation
description: Review forensic-media indexing and search changes for evidence immutability, traceability, index integrity, reproducibility, and ranking-semantic regressions.
---

# Forensic Validation

## Evidence safety

Confirm:

- evidence is read-only;
- generated artifacts are outside evidence;
- files are not renamed;
- evidence contents/timestamps/metadata are not intentionally modified.

## Traceability

Confirm derived artifacts identify:

- evidence source/manifest;
- deterministic ordinal/relative path;
- model name/revision;
- weight SHA256 when available;
- preprocessing identity/version;
- embedding dtype/dimension;
- normalization;
- index version;
- timestamps;
- relevant search configuration.

## Index integrity

Confirm:

- BUILDING vs COMPLETE state;
- SEARCH rejects incomplete indexes;
- row count matches storage;
- manifest mapping is deterministic;
- resume cannot duplicate rows;
- partial writes are not committed;
- incompatible indexes are rejected.

## Semantic correctness

Confirm:

- similarity is not called probability;
- EvaluationRank is not called confidence;
- audit report remains file+query;
- final report remains unique-file;
- query OR inclusion and consensus prioritization remain correct;
- incompatible raw model scores are not directly compared.

## Findings

Classify as Critical, High, Medium, or Low.
