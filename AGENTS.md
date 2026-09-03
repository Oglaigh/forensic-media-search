# Forensic Media Search

## Project purpose

Build an offline forensic media search system that scans a directory or mounted
disk recursively, evaluates image files against written visual specifications,
and produces a report containing candidate matches.

The primary use case is:

1. Receive a directory or disk.
2. Receive one or more written visual specifications.
3. Convert those specifications into one or more model queries.
4. Iterate all supported image files recursively.
5. Evaluate every image using the configured vision-language model.
6. Keep candidate matches according to configured scoring rules.
7. Generate a report.

Minimum report contract:

FilePath | % | Cos

Additional columns may be added when they improve traceability.

---

## Current architecture

The current baseline stack is:

- Windows host
- Docker Desktop
- Linux containers through WSL2
- NVIDIA CUDA
- NVIDIA GeForce RTX 4080 16 GB
- Python 3.11
- PyTorch
- OpenAI CLIP
- ViT-B/32 baseline model

The model execution must use CUDA when available.

Do not silently fall back to CPU when the workflow explicitly expects GPU
execution. Report the selected device at startup.

---

## Forensic invariants

Evidence is immutable.

Never:

- modify an evidence file;
- rename an evidence file;
- delete an evidence file;
- rewrite EXIF metadata;
- create thumbnails inside the evidence directory;
- write temporary files into the evidence directory.

Evidence volumes must be mounted read-only whenever possible.

Example:

    ./evidence:/evidence:ro

All generated artifacts must be written outside the evidence tree, normally
under:

    /output

Preserve the original file path in all reports.

An unreadable or corrupt image must not stop the complete scan. Record the
error and continue processing.

---

## Search behavior

The system is a forensic candidate-search engine, not a definitive classifier.

A model score means that an image is a candidate for human review.

Do not state that a model result proves that an object or attribute exists.

Written specifications may produce multiple independent queries.

Example:

    Houses with red roofs or black doors

may become:

    a house with a red roof
    a house with a black door

For OR semantics, the best matching query may determine the candidate score.

Keep the original specification and the generated queries traceable.

---

## Scores

Cosine similarity is the primary raw model score.

Do not describe cosine similarity as a probability.

Do not convert:

    cosine * 100

and call it confidence.

If a percentage is exposed before statistical calibration, it must be clearly
identified as a relative/ranking score and its formula must be documented.

A future calibrated percentage may be introduced only after validation against
a labeled dataset.

---

## Image discovery

Scan recursively.

Initial supported formats:

- .jpg
- .jpeg
- .png
- .bmp
- .webp
- .tif
- .tiff

Format detection should eventually rely on actual decodability in addition to
file extension.

Do not assume every file with a valid extension is a valid image.

---

## GPU processing

Do not process large collections one image at a time when batching is possible.

The scanning implementation should support configurable batch sizes.

Default development target:

    batch_size = 64

Batch size must remain configurable because memory consumption depends on the
model.

Use inference-only execution:

    model.eval()

and:

    torch.inference_mode()

or equivalent.

---

## Performance architecture

Separate these concepts:

    media discovery
    image decoding
    embedding generation
    query embedding
    similarity calculation
    candidate filtering
    report generation

Avoid coupling filesystem traversal directly to CLIP internals.

The query embedding must be generated once per query set, not once per image.

---

## Model architecture

OpenAI CLIP ViT-B/32 is the initial baseline, not a permanent architectural
dependency.

Model loading must remain replaceable so future implementations can evaluate:

- ViT-B/16
- ViT-L/14
- OpenCLIP
- SigLIP
- other compatible vision-language embedding models

Do not spread CLIP-specific calls throughout the filesystem scanning code.

Use an abstraction around the embedding model.

---

## Output

Primary CSV report must contain at least:

    FilePath
    Percent
    Cos

Prefer also retaining internally:

    MatchedQuery
    Model
    ModelVersion
    ScanTimestamp

Future forensic manifests may additionally contain SHA256.

The report must be sorted from strongest candidate to weakest candidate unless
the user requests another ordering.

---

## Code quality

Prefer small modules with explicit responsibilities.

Suggested structure:

    src/
      scanner/
      models/
      queries/
      reporting/
      cli/

Do not introduce databases, FAISS, APIs, web interfaces, video analysis, OCR,
or object detection unless the task explicitly requires them.

We are currently building the simplest reliable end-to-end image scanning
pipeline.

---

## Docker

Keep evidence mounts read-only.

Do not bake evidence or generated reports into Docker images.

The source tree may be bind-mounted read-only during development.

Models may use a persistent cache volume.

---

## Validation

After relevant changes:

1. Verify Python imports.
2. Verify Docker build when Docker-related files changed.
3. Verify CUDA is available for GPU workflows.
4. Run the smallest applicable scan test.
5. Confirm evidence files were not modified.
6. Confirm the report is generated outside the evidence directory.

---

## Agent delegation

Use specialized subagents when a task materially benefits from separation of
concerns.

Preferred roles:

- forensic_architect: architecture and forensic invariants
- scan_engineer: scanner, batching, filesystem and implementation
- clip_specialist: model queries, embeddings and scoring
- forensic_reviewer: final read-only review

For small changes, do not delegate unnecessarily.

For architecture or large changes, prefer:

    forensic_architect
            ↓
    scan_engineer / clip_specialist
            ↓
    forensic_reviewer