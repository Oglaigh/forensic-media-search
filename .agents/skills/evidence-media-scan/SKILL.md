---
name: evidence-media-scan
description: Use when implementing, debugging or running recursive image scanning over a forensic evidence directory or mounted disk.
---

# Evidence Media Scan

## Goal

Process all supported images in an evidence tree without modifying evidence.

## Inputs

- evidence root
- supported extensions
- batch size
- model
- queries
- candidate threshold
- output path

## Workflow

1. Confirm the evidence path.
2. Treat the evidence path as read-only.
3. Enumerate files recursively.
4. Identify candidate image files.
5. Decode images safely.
6. Record unreadable files and continue.
7. Process valid images in batches.
8. Calculate similarity against precomputed query embeddings.
9. Retain candidates according to the configured criteria.
10. Sort results by score.
11. Write results outside the evidence tree.

## Performance

Never load every decoded image into memory simultaneously.

Prefer bounded batches.

Use GPU inference when configured.

## Evidence safety

Never:

- modify source images;
- change metadata;
- write temporary images beside evidence;
- rename files;
- delete files.

## Verification

Confirm:

- evidence remains unchanged;
- corrupt files do not abort the scan;
- all generated files are outside evidence;
- the report contains original file paths.