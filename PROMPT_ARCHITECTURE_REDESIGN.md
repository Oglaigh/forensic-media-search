# Phase 1 — Architecture only

Redefine the architecture of `forensic-media-search` so millions of evidence images are encoded once and later queries reuse persistent image embeddings.

Do not implement yet.

Follow `AGENTS.md`.

Delegate read-only analysis in parallel to:

- `forensic_architect`
- `vision_model_specialist`
- `indexing_engineer`

Ask each agent to inspect the current repository from its specialty.

Wait for all three results before deciding anything.

Then, in the main thread:

1. compare the three analyses;
2. identify agreements and conflicts;
3. propose one minimal target architecture;
4. define the on-disk index format;
5. define BUILDING/COMPLETE and resume semantics;
6. define the exact DIRECT-vs-INDEXED equivalence contract;
7. define CLI changes;
8. identify exact files/modules likely to change;
9. list risks and open decisions.

Constraints:

- evidence remains read-only;
- embeddings are initially normalized float32;
- SEARCH must not decode evidence images;
- SEARCH must not run image encoders;
- search is exact;
- no ANN/HNSW/IVF/PQ/quantization;
- current DIRECT path remains available;
- current Top-K, RRF, audit report and Evaluator semantics remain intact.

Stop after presenting the architecture proposal.

Do not edit application code in this phase.
