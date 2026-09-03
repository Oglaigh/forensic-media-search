"""Read-only, failure-tolerant image decoding."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .batching import batched
from .discovery import ManifestEntry


@dataclass(slots=True)
class DecodedImage:
    entry: ManifestEntry
    image: Image.Image


@dataclass(frozen=True, slots=True)
class DecodeError:
    entry: ManifestEntry
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class DecodedBatch:
    images: tuple[DecodedImage, ...]
    errors: tuple[DecodeError, ...]


def decode_image(entry: ManifestEntry) -> DecodedImage:
    """Decode one image without ever opening its evidence path for writing."""

    # Opening the file handle ourselves makes the read-only mode explicit and
    # prevents a decoder from choosing a mode based on the path.
    with Path(entry.source_path).open("rb") as source:
        with Image.open(source) as encoded:
            oriented = ImageOps.exif_transpose(encoded)
            try:
                rgb = oriented.convert("RGB")
                rgb.load()
                # Detach all pixel data from the source stream before it closes.
                detached = rgb.copy()
            finally:
                if "rgb" in locals():
                    rgb.close()
                if oriented is not encoded:
                    oriented.close()
    return DecodedImage(entry=entry, image=detached)


def decode_batch(entries: Iterable[ManifestEntry]) -> DecodedBatch:
    """Decode valid images and return corrupt/unreadable files as data."""

    images: list[DecodedImage] = []
    errors: list[DecodeError] = []
    for entry in entries:
        try:
            images.append(decode_image(entry))
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            errors.append(
                DecodeError(
                    entry=entry,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
    return DecodedBatch(images=tuple(images), errors=tuple(errors))


def iter_decoded_batches(
    entries: Iterable[ManifestEntry], batch_size: int
) -> Iterator[DecodedBatch]:
    """Stream evidence entries through bounded decode batches."""

    for entry_batch in batched(entries, batch_size):
        yield decode_batch(entry_batch)
