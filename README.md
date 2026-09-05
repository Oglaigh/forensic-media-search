# Forensic Media Search

Offline forensic triage tool for inventorying digital evidence and searching visual content using vision-language models.

> **Important:** this application is a triage and prioritization tool. Model scores, rankings, and results are not probabilities and do not constitute forensic conclusions by themselves.

---

## How the application works

The application is organized around **cases / examinations**.

The intended workflow is:

```text
Create / open case
        ↓
Select evidence
        ↓
Identify evidence source
        ↓
Scan the content once
        ↓
Create persistent inventory
        ↓
Choose what to analyze
        ↓
Build or reuse index
        ↓
Run searches
```

### 1. Create or open a case

When the application starts, the investigator creates a new case or opens an existing one.

Each case preserves:

- evidence identification;
- last known evidence location;
- file inventory;
- scan state;
- generated indexes;
- search reports.

Closing the application does not delete this information.

---

### 2. Select evidence

Initially, the evidence source may be a disk or directory accessible from the operating system.

A path such as:

```text
E:\
```

is treated as a **locator**, not as the permanent identity of the disk.

The case stores evidence identifiers so the application can recognize the same source even if Windows assigns it a different drive letter later.

Evidence should be accessed in **read-only mode** whenever possible.

---

### 3. Evidence scan

The application performs a full scan before semantic analysis.

The result is a persistent master inventory containing the discovered files and their classification.

Example:

```text
Files / entries : 1,323,932

Images          : 182,415
Videos          : 14,210
Documents       : 93,812
Text            : 55,091
Archives        : 18,422
Other           : ...
```

The scan:

- does not run SigLIP2 or CLIP;
- does not perform OCR;
- does not analyze video;
- does not perform semantic search;
- does not modify evidence.

Files may also be identified by content instead of relying only on filename extensions.

---

### 4. ZIP / RAR

Compressed files are part of the inventory and may also contain additional evidence.

Example:

```text
backup.zip
backup.zip::Photos\image.jpg
backup.zip::Documents\file.pdf
```

ZIP/RAR members are recorded as logical evidence entries related to their parent container.

The application avoids permanently extracting files into the evidence source.

Files that are:

- encrypted;
- corrupt;
- partially processed;
- unsupported;

must remain explicitly recorded.

---

### 5. Select analysis type

After the scan, the investigator chooses what to analyze.

```text
[1] Images
[2] Videos
[3] Documents / Text
[4] Inventory
```

The currently implemented semantic analysis is **Images**.

Video and document/text analysis are planned for later stages.

---

## Image analysis

When **Images** is selected, the application uses all inventory entries classified as images.

This includes:

- regular filesystem images;
- images found inside ZIP/RAR files.

### First run

If no compatible visual index exists yet:

```text
Images
   ↓
SigLIP2
   ↓
CLIP
   ↓
Persistent embeddings
```

This process may take minutes or hours depending on the number of files.

This cost is paid **once**.

### Subsequent searches

Once the index has been built:

```text
Query
   ↓
Text embedding
   ↓
Exact search over index
   ↓
Top-K
   ↓
RRF
   ↓
Evaluator
   ↓
Report
```

Images are not decoded or processed again by the image encoders.

The index is reused even after closing and reopening the application.

---

## Visual models

The current visual pipeline uses:

- **SigLIP2**
  - `google/siglip2-base-patch16-224`

- **OpenAI CLIP**
  - `ViT-B/32`

Visual embeddings are stored persistently.

Search is exact and does not use, by default:

- ANN;
- HNSW;
- IVF;
- PQ;
- quantization.

---

## Persistence

A case may use a structure similar to:

```text
CASE-2026-001/
├── case.json
├── scan/
│   ├── manifest.jsonl
│   ├── summary.json
│   ├── scan-state.json
│   └── errors.jsonl
├── indexes/
│   └── images/
│       ├── index.json
│       ├── siglip2/
│       │   └── embeddings.f32
│       └── clip/
│           └── embeddings.f32
├── reports/
└── temp/
```

When reopening a case:

```text
Scan          COMPLETE
Image Index   COMPLETE
```

the application reuses both artifacts.

There is no need to rescan the evidence or recompute image embeddings.

---

# Requirements

## Operating system

Currently validated environment:

- Windows 10/11 64-bit
- Docker Desktop
- WSL2 backend

## GPU

An NVIDIA GPU with CUDA support is recommended.

Currently validated environment:

- NVIDIA GeForce RTX 4080
- 16 GB VRAM

For large collections, recommended:

- **12 GB VRAM or more**
- 16 GB VRAM recommended

The application must not silently fall back to CPU when CUDA is explicitly requested.

## CPU

Minimum recommended:

- 4 cores

Recommended for large-scale indexing:

- 8 cores or more

The CPU is especially involved in:

- filesystem traversal;
- file reading;
- hashing;
- decompression;
- image decoding;
- batch preparation.

## RAM

Minimum recommended:

- 16 GB

Recommended:

- 32 GB or more

For very large collections, 64 GB may improve operating-system cache efficiency, although the application is designed to keep memory usage bounded.

## Storage

Separate storage outside the evidence source is required for:

- inventory;
- indexes;
- model files;
- reports;
- temporary files.

As a reference, the current visual index uses approximately **5 KB per image** with SigLIP2 + CLIP stored as `float32`.

Approximate estimate:

```text
100,000 images     ≈ 500 MB
1,000,000 images   ≈ 5 GB
10,000,000 images  ≈ 50 GB
```

Additional space is required for manifests, reports, caches, and temporary files.

---

# Installation

## 1. Clone the repository

```powershell
git clone <REPOSITORY_URL>
cd forensic-media-search
```

---

## 2. Install Docker Desktop

Install Docker Desktop with the WSL2 backend.

Verify:

```powershell
docker version
docker ps
```

---

## 3. Install or update NVIDIA drivers

Verify that Windows detects the GPU correctly:

```powershell
nvidia-smi
```

---

## 4. Verify GPU access from Docker

```powershell
docker run --rm --gpus all ubuntu nvidia-smi
```

The NVIDIA GPU should appear inside the container.

---

## 5. Build the application image

From the repository root:

```powershell
docker build -t forensic-media-search:dev .
```

If the project uses Docker Compose:

```powershell
docker compose build
```

---

## 6. Start the console application

From the repository root:

```powershell
python .\console.py
```

The console guides the investigator through:

1. creating or opening a case;
2. selecting evidence;
3. scanning;
4. inventory review;
5. selecting the analysis type;
6. building or reusing indexes;
7. searches;
8. report generation.

---

# First examination

Conceptual example:

```text
FORENSIC MEDIA SEARCH

[1] New case
[2] Open case

> 1

Name / ID:
CASE-2026-001

Select evidence:
E:\

Scanning...

✓ Inventory COMPLETE

Images        182,415
Videos         14,210
Documents      93,812
...

What do you want to analyze?

[1] Images
[2] Videos
[3] Documents / Text
[4] Inventory
```

When Images is selected, if no image index exists yet:

```text
Building visual index...

SigLIP2
CLIP

✓ Image Index COMPLETE
```

Then:

```text
Query 01 > there is a dog
Query 02 > there is a cat
```

---

# Reopening a case

On a later run:

```text
FORENSIC MEDIA SEARCH

[1] New case
[2] Open case

> 2

CASE-2026-001
```

The application restores:

- registered evidence;
- inventory;
- indexes;
- reports.

If Windows assigned the disk a different drive letter, the application should try to identify it by its persisted identity.

Example:

```text
Previous location : E:\
Current location  : F:\

✓ Evidence EV-001 verified
```

---

# Evidence disconnected

A case can still be opened when the original disk is not connected.

If the visual index is available:

```text
Evidence      OFFLINE
Inventory     COMPLETE
Image Index   COMPLETE
```

searches can still run against persisted embeddings.

However, original evidence bytes cannot be opened until the source is available again and successfully verified.

---

# Outputs

The application generates derived artifacts outside the evidence source.

These may include:

- master manifest;
- scan summary;
- error journal;
- persistent indexes;
- audit CSV;
- final candidate CSV.

The audit report keeps information per:

```text
file + query
```

The final report consolidates results per file.

---

# Forensic considerations

- Evidence must remain immutable.
- Use read-only evidence sources whenever possible.
- Indexes and reports are derived artifacts.
- Models do not replace human review.
- Scores are not probabilities.
- Results should remain reproducible from:
  - evidence;
  - manifest;
  - models;
  - revisions;
  - configuration;
  - indexes.

---

# Current status

Implemented:

- visual analysis with SigLIP2 + CLIP;
- persistent embeddings;
- exact indexed search;
- Top-K per model/query;
- RRF;
- multi-query Evaluator;
- audit report;
- final report.

Currently being refactored / evolved:

- case persistence;
- persistent evidence identification;
- master evidence manifest;
- full-disk scan;
- file-type classification;
- ZIP/RAR handling;
- modality-based interactive flow.

Planned for later stages:

- video analysis;
- document/text analysis;
- OCR;
- E01 / RAW / DD;
- deleted files;
- carving;
- unallocated-space analysis.
