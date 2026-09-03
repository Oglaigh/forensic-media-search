"""Filesystem scanning primitives for forensic evidence."""

from .batching import batched
from .decoder import (
    DecodeError,
    DecodedBatch,
    DecodedImage,
    decode_batch,
    decode_image,
    iter_decoded_batches,
)
from .discovery import (
    SUPPORTED_IMAGE_EXTENSIONS,
    DiscoveryError,
    DiscoveryStats,
    ManifestEntry,
    discover_images,
    is_within,
    read_manifest,
    validate_artifact_path,
    write_manifest,
)
from .state import OrdinalBitmap, PassMetrics, ScanMetrics

__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "DecodeError",
    "DecodedBatch",
    "DecodedImage",
    "DiscoveryError",
    "DiscoveryStats",
    "ManifestEntry",
    "OrdinalBitmap",
    "PassMetrics",
    "ScanMetrics",
    "batched",
    "decode_batch",
    "decode_image",
    "discover_images",
    "is_within",
    "iter_decoded_batches",
    "read_manifest",
    "validate_artifact_path",
    "write_manifest",
]
