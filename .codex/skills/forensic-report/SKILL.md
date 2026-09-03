---
name: forensic-report
description: Use when generating candidate-match reports from a forensic media scan, especially CSV output containing file path, percentage score and cosine similarity.
---

# Forensic Report

## Required output

The primary report must contain at least:

    FilePath | Percent | Cos

Recommended additional fields:

    MatchedQuery
    Model
    ScanTimestamp

## FilePath

Preserve the original evidence path.

Do not replace it with a temporary container path in the final exported report
when the original path mapping is known.

## Percent

The meaning of Percent must be documented.

Do not label a raw cosine-derived number as probability or confidence.

Before calibration exists, use a clearly defined relative score.

## Cos

Store sufficient precision for later review.

Recommended:

    6 decimal places

Do not round the internal score before ranking.

## Ordering

Default ordering:

    Cos DESC

unless the configured scoring system defines another authoritative ranking.

## Evidence

Report generation must never modify evidence files.

Write reports only to the configured output directory.