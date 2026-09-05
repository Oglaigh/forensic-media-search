# Corrected Codex Agents + Skills pack

This version matches the current Codex custom-agent and skill schemas.

## Important fixes

Custom agent TOML files use the required:

```toml
name = "..."
description = "..."
developer_instructions = "..."
```

Skills begin with required YAML frontmatter:

```yaml
---
name: skill-name
description: When the skill should be used.
---
```

## Layout

```text
AGENTS.md

.codex/
└── agents/
    ├── forensic-architect.toml
    ├── indexing-engineer.toml
    ├── vision-model-specialist.toml
    └── forensic-reviewer.toml

.agents/
└── skills/
    ├── evidence-indexing/SKILL.md
    ├── exact-vector-search/SKILL.md
    ├── embedding-equivalence/SKILL.md
    └── forensic-validation/SKILL.md

PROMPT_ARCHITECTURE_REDESIGN.md
PROMPT_IMPLEMENT_INDEX_SEARCH.md
```

## Recommended use

1. Finish and commit current Evaluator work.
2. Copy this pack into the repository root, replacing the malformed versions from the previous bundle.
3. Restart Codex CLI.
4. Run Phase 1 first with `PROMPT_ARCHITECTURE_REDESIGN.md`.
5. Inspect subagents using `/agent`.
6. Review the architecture before implementation.
7. Run Phase 2 using `PROMPT_IMPLEMENT_INDEX_SEARCH.md`.

The specialist agents in this pack are deliberately read-only. The main Codex thread owns integration and file edits.

## Persistent exact index

The runtime now exposes three compatible workflows:

```text
# Build a new immutable index outside evidence.
python -m forensic_media_search index --directory EVIDENCE --index CASE.index

# Resume only an incomplete BUILDING index.
python -m forensic_media_search index --directory EVIDENCE --index CASE.index --resume

# Search every valid embedding exactly, without opening evidence images.
python -m forensic_media_search search --index CASE.index --query "description" --output report.csv

# Recalculate complete index hashes and validate normalized rows.
python -m forensic_media_search verify --index CASE.index --full
```

The legacy command line remains the DIRECT validation mode. Indexed SEARCH uses
memory-mapped float32 `.npy` matrices, preserves `row == manifest ordinal`, and
reuses the existing per-model/query Top-K, RRF, audit and Evaluator semantics.
Scores and ranks remain ranking measures, not probabilities or evidentiary conclusions.
