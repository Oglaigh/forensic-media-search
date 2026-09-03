from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from forensic_media_search.scanner.discovery import (  # noqa: E402
    discover_images,
    read_manifest,
    write_manifest,
)


def test_discovery_is_recursive_filtered_and_deterministic(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "B-dir").mkdir(parents=True)
    (evidence / "a-dir").mkdir()
    (evidence / "z.JPG").write_bytes(b"candidate")
    (evidence / "A.png").write_bytes(b"candidate")
    (evidence / "ignore.txt").write_text("not an image", encoding="utf-8")
    (evidence / "a-dir" / "inside.webp").write_bytes(b"candidate")
    (evidence / "B-dir" / "other.tiff").write_bytes(b"candidate")

    first = list(discover_images(evidence))
    second = list(discover_images(evidence))

    assert first == second
    assert [entry.ordinal for entry in first] == [0, 1, 2, 3]
    assert [entry.relative_path for entry in first] == [
        "A.png",
        "z.JPG",
        "a-dir/inside.webp",
        "B-dir/other.tiff",
    ]
    assert all(Path(entry.source_path).is_absolute() for entry in first)


def test_discovery_skips_file_and_directory_symlinks(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    outside = tmp_path / "outside"
    evidence.mkdir()
    outside.mkdir()
    (outside / "outside.jpg").write_bytes(b"outside")
    (evidence / "inside.jpg").write_bytes(b"inside")

    try:
        (evidence / "linked-file.jpg").symlink_to(outside / "outside.jpg")
        (evidence / "linked-directory").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable in this environment: {error}")

    assert [entry.relative_path for entry in discover_images(evidence)] == [
        "inside.jpg"
    ]


def test_manifest_is_jsonl_streamable_and_stable(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    evidence.mkdir()
    (evidence / "b.jpeg").write_bytes(b"b")
    (evidence / "a.bmp").write_bytes(b"a")

    manifest = output / "manifest.jsonl"
    stats = write_manifest(evidence, manifest)
    raw_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    entries = list(read_manifest(manifest))

    assert stats.images_discovered == 2
    assert stats.filesystem_errors == 0
    assert [row["relative_path"] for row in raw_rows] == ["a.bmp", "b.jpeg"]
    assert [entry.ordinal for entry in entries] == [0, 1]
    assert not list(output.glob("*.tmp"))


def test_manifest_cannot_be_written_inside_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    with pytest.raises(ValueError, match="outside evidence"):
        write_manifest(evidence, evidence / "manifest.jsonl")

    assert not (evidence / "manifest.jsonl").exists()


def test_discovery_max_images_is_stable(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ("c.jpg", "a.jpg", "b.jpg"):
        (evidence / name).write_bytes(b"candidate")

    assert [entry.relative_path for entry in discover_images(evidence, max_images=2)] == [
        "a.jpg",
        "b.jpg",
    ]
    assert list(discover_images(evidence, max_images=0)) == []


def test_read_manifest_rejects_non_contiguous_ordinals(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"ordinal": 2, "relative_path": "a.jpg", "source_path": "/evidence/a.jpg"}
        )
        + os.linesep,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        list(read_manifest(manifest))
