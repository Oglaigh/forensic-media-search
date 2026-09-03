from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from forensic_media_search.scanner import (  # noqa: E402
    ManifestEntry,
    OrdinalBitmap,
    ScanMetrics,
    batched,
    decode_batch,
    decode_image,
    iter_decoded_batches,
)


def entry(path: Path, ordinal: int = 0) -> ManifestEntry:
    return ManifestEntry(ordinal=ordinal, relative_path=path.name, source_path=str(path))


def test_decoder_applies_exif_orientation_and_returns_rgb(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    source = Image.new("RGB", (2, 1), color=(200, 10, 10))
    exif = Image.Exif()
    exif[274] = 6
    source.save(path, exif=exif)

    decoded = decode_image(entry(path))
    try:
        assert decoded.image.mode == "RGB"
        assert decoded.image.size == (1, 2)
        # Pixel data remains available after the evidence handle has closed.
        decoded.image.getpixel((0, 0))
    finally:
        decoded.image.close()


def test_corrupt_image_is_recorded_and_next_image_continues(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    valid = tmp_path / "valid.png"
    corrupt.write_bytes(b"not an image")
    Image.new("L", (2, 2), color=123).save(valid)

    result = decode_batch((entry(corrupt, 0), entry(valid, 1)))
    try:
        assert [item.entry.ordinal for item in result.images] == [1]
        assert [error.entry.ordinal for error in result.errors] == [0]
        assert result.errors[0].error_type
    finally:
        for item in result.images:
            item.image.close()


def test_evidence_is_only_opened_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (1, 1)).save(path)
    original_open = Path.open
    modes: list[str] = []

    def observed_open(self: Path, mode: str = "r", *args, **kwargs):
        if self == path:
            modes.append(mode)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    decoded = decode_image(entry(path))
    decoded.image.close()

    assert modes == ["rb"]


def test_batched_is_bounded_and_keeps_final_partial_batch() -> None:
    assert list(batched(range(5), 2)) == [(0, 1), (2, 3), (4,)]
    with pytest.raises(ValueError, match="batch_size"):
        list(batched(range(2), 0))


def test_iter_decoded_batches_preserves_batch_boundaries(tmp_path: Path) -> None:
    entries = []
    for ordinal in range(3):
        path = tmp_path / f"{ordinal}.png"
        Image.new("RGB", (1, 1)).save(path)
        entries.append(entry(path, ordinal))

    results = list(iter_decoded_batches(entries, batch_size=2))
    try:
        assert [len(result.images) for result in results] == [2, 1]
        assert all(not result.errors for result in results)
    finally:
        for result in results:
            for item in result.images:
                item.image.close()


def test_ordinal_bitmap_is_fixed_capacity_and_idempotent() -> None:
    bitmap = OrdinalBitmap(10)
    assert bitmap.add(9)
    assert not bitmap.add(9)
    assert 9 in bitmap
    assert len(bitmap) == 1
    with pytest.raises(IndexError):
        bitmap.add(10)


def test_scan_metrics_track_unique_state_across_two_passes() -> None:
    metrics = ScanMetrics(images_discovered=3, filesystem_errors=1)
    metrics.begin_pass("siglip2")
    metrics.record_processed("siglip2", 0)
    metrics.record_processed("siglip2", 1)
    metrics.record_decode_error("siglip2", 2)
    metrics.finish_pass("siglip2")

    metrics.begin_pass("clip")
    metrics.record_processed("clip", 0)
    metrics.record_processed("clip", 2)
    metrics.record_decode_error("clip", 1)
    metrics.finish_pass("clip")

    assert metrics.images_processed == 3
    assert metrics.decode_errors == 2
    assert len(metrics.passes["siglip2"].processed) == 2
    assert len(metrics.passes["clip"].processed) == 2
    assert metrics.passes["siglip2"].processing_seconds >= 0
