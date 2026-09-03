"""Deterministic, read-only discovery of image evidence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """A stable reference to one source image.

    ``ordinal`` is assigned in deterministic discovery order. ``relative_path``
    always uses POSIX separators so manifests are stable across host platforms.
    ``source_path`` retains the absolute path that must be opened read-only.
    """

    ordinal: int
    relative_path: str
    source_path: str


@dataclass(frozen=True, slots=True)
class DiscoveryError:
    path: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class DiscoveryStats:
    images_discovered: int
    filesystem_errors: int


DiscoveryErrorHandler = Callable[[DiscoveryError], None]


def _sort_key(path: Path) -> tuple[str, str]:
    # casefold gives consistent ordering on case-insensitive evidence volumes;
    # the original spelling is a deterministic tie breaker.
    return (path.name.casefold(), path.name)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def is_within(path: Path, directory: Path) -> bool:
    """Return whether *path* resolves to *directory* or one of its children."""

    resolved_path = path.resolve(strict=False)
    resolved_directory = directory.resolve(strict=False)
    try:
        return os.path.commonpath((resolved_path, resolved_directory)) == str(
            resolved_directory
        )
    except ValueError:
        # Different Windows drives cannot contain one another.
        return False


def validate_artifact_path(evidence_root: Path, artifact_path: Path) -> None:
    """Reject generated artifacts located anywhere in the evidence tree."""

    if is_within(artifact_path, evidence_root):
        raise ValueError(
            f"Generated artifact must be outside evidence: {artifact_path}"
        )


def discover_images(
    evidence_root: Path,
    *,
    extensions: frozenset[str] = SUPPORTED_IMAGE_EXTENSIONS,
    on_error: DiscoveryErrorHandler | None = None,
    max_images: int | None = None,
) -> Iterator[ManifestEntry]:
    """Yield supported evidence files recursively in deterministic order.

    Symbolic links are skipped whether they point to files or directories. The
    traversal only holds the entries for the current directory in memory.
    Filesystem errors are reported through ``on_error`` and do not stop sibling
    directory traversal.
    """

    root = _absolute(Path(evidence_root))
    if not root.exists():
        raise FileNotFoundError(f"Evidence directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Evidence path is not a directory: {root}")
    if max_images is not None and max_images < 0:
        raise ValueError("max_images must be >= 0")

    normalized_extensions = frozenset(extension.casefold() for extension in extensions)
    ordinal = 0

    def report(path: Path, error: OSError) -> None:
        if on_error is not None:
            on_error(
                DiscoveryError(
                    path=str(path),
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    def walk(directory: Path) -> Iterator[Path]:
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda entry: _sort_key(Path(entry.name)))
        except OSError as error:
            report(directory, error)
            return

        directories: list[Path] = []
        files: list[Path] = []
        for entry in entries:
            entry_path = directory / entry.name
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    directories.append(entry_path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(entry_path)
            except OSError as error:
                report(entry_path, error)

        # Emit files before descending so ordering does not depend on whether a
        # filesystem happens to interleave file and directory entries.
        for file_path in files:
            if file_path.suffix.casefold() in normalized_extensions:
                yield file_path
        for child in directories:
            yield from walk(child)

    if max_images == 0:
        return

    for source_path in walk(root):
        relative_path = source_path.relative_to(root).as_posix()
        yield ManifestEntry(
            ordinal=ordinal,
            relative_path=relative_path,
            source_path=str(source_path),
        )
        ordinal += 1
        if max_images is not None and ordinal >= max_images:
            return


def _write_manifest_rows(entries: Iterator[ManifestEntry], stream: TextIO) -> int:
    count = 0
    for entry in entries:
        json.dump(asdict(entry), stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        count += 1
    return count


def write_manifest(
    evidence_root: Path,
    manifest_path: Path,
    *,
    extensions: frozenset[str] = SUPPORTED_IMAGE_EXTENSIONS,
    on_error: DiscoveryErrorHandler | None = None,
    max_images: int | None = None,
) -> DiscoveryStats:
    """Discover images and atomically write a JSONL manifest outside evidence."""

    root = _absolute(Path(evidence_root))
    destination = _absolute(Path(manifest_path))
    validate_artifact_path(root, destination)

    error_count = 0

    def collect(error: DiscoveryError) -> None:
        nonlocal error_count
        error_count += 1
        if on_error is not None:
            on_error(error)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            count = _write_manifest_rows(
                discover_images(
                    root,
                    extensions=extensions,
                    on_error=collect,
                    max_images=max_images,
                ),
                temporary,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    return DiscoveryStats(images_discovered=count, filesystem_errors=error_count)


def read_manifest(manifest_path: Path) -> Iterator[ManifestEntry]:
    """Stream and validate entries from a JSONL discovery manifest."""

    expected_ordinal = 0
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                entry = ManifestEntry(
                    ordinal=int(raw["ordinal"]),
                    relative_path=str(raw["relative_path"]),
                    source_path=str(raw["source_path"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid manifest entry at line {line_number}: {error}"
                ) from error
            if entry.ordinal != expected_ordinal:
                raise ValueError(
                    "Manifest ordinals must be contiguous: "
                    f"expected {expected_ordinal}, got {entry.ordinal} at line {line_number}"
                )
            expected_ordinal += 1
            yield entry
