import argparse
import csv
import os
import sys
from pathlib import Path, PureWindowsPath

import clip
import torch
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def cosine_to_percent(cosine: float, reference_cos: float) -> float:
    """
    Heuristic ranking score.

    This is NOT a probability.
    """
    value = (cosine / reference_cos) * 100.0
    return max(0.0, min(100.0, value))


def find_images(root: Path):
    errors = []

    def on_error(error):
        errors.append(str(error))
        print(f"[FS ERROR] {error}", file=sys.stderr)

    for directory, _, filenames in os.walk(
        root,
        topdown=True,
        onerror=on_error,
        followlinks=False,
    ):
        directory_path = Path(directory)

        for filename in filenames:
            path = directory_path / filename

            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def display_path(
    path: Path,
    root: Path,
    display_root: str | None,
) -> str:

    if not display_root:
        return str(path)

    try:
        relative = path.relative_to(root)

        windows_path = PureWindowsPath(
            display_root
        ).joinpath(*relative.parts)

        return str(windows_path)

    except ValueError:
        return str(path)


def load_image_tensor(path: Path, preprocess):
    with Image.open(path) as image:

        # Respect EXIF orientation without modifying the file.
        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        return preprocess(image)


def process_batch(
    paths,
    *,
    model,
    preprocess,
    text_features,
    queries,
    device,
    root,
    display_root,
    min_percent,
    reference_cos,
):

    tensors = []
    valid_paths = []
    errors = 0

    for path in paths:

        try:
            tensor = load_image_tensor(
                path,
                preprocess
            )

            tensors.append(tensor)
            valid_paths.append(path)

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            errors += 1

            print(
                f"[IMAGE ERROR] {path}: {error}",
                file=sys.stderr,
            )

    if not tensors:
        return [], errors

    image_batch = torch.stack(tensors).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():

        image_features = model.encode_image(
            image_batch
        )

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True,
        )

        similarities = (
            image_features @ text_features.T
        )

    results = []

    for index, path in enumerate(valid_paths):

        row = similarities[index]

        best_query_index = int(
            row.argmax().item()
        )

        cosine = float(
            row[best_query_index].item()
        )

        percent = cosine_to_percent(
            cosine,
            reference_cos,
        )

        if percent < min_percent:
            continue

        matched_query = queries[
            best_query_index
        ]

        file_path = display_path(
            path,
            root,
            display_root,
        )

        result = {
            "FilePath": file_path,
            "%": percent,
            "Cos": cosine,
            "MatchedQuery": matched_query,
        }

        results.append(result)

    return results, errors


def write_csv(results, output_path: Path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "FilePath",
                "%",
                "Cos",
                "MatchedQuery",
            ],
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "FilePath":
                    result["FilePath"],

                "%":
                    f'{result["%"]:.2f}',

                "Cos":
                    f'{result["Cos"]:.6f}',

                "MatchedQuery":
                    result["MatchedQuery"],
            })


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Recursive forensic image search "
            "using OpenAI CLIP."
        )
    )

    parser.add_argument(
        "--directory",
        required=True,
        help="Directory to scan recursively.",
    )

    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help=(
            "Visual query. "
            "Repeat --query for multiple classifications."
        ),
    )

    parser.add_argument(
        "--min-percent",
        type=float,
        default=75.0,
        help=(
            "Minimum heuristic match score. "
            "Default: 75."
        ),
    )

    parser.add_argument(
        "--reference-cos",
        type=float,
        default=0.35,
        help=(
            "Cosine value mapped to 100%%. "
            "Default: 0.35."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="GPU batch size. Default: 64.",
    )

    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="CLIP model. Default: ViT-B/32.",
    )

    parser.add_argument(
        "--output",
        default="/output/report.csv",
        help="CSV report path.",
    )

    parser.add_argument(
        "--display-root",
        default=None,
        help=(
            "Original host path used in reports."
        ),
    )

    args = parser.parse_args()

    root = Path(args.directory)

    if not root.exists():
        raise SystemExit(
            f"Directory does not exist: {root}"
        )

    if not 0 <= args.min_percent <= 100:
        raise SystemExit(
            "--min-percent must be between 0 and 100"
        )

    if args.reference_cos <= 0:
        raise SystemExit(
            "--reference-cos must be > 0"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("FORENSIC MEDIA SEARCH")
    print("=" * 78)

    print(f"Directory : {root}")
    print(f"Device    : {device}")

    if device == "cuda":
        print(
            f"GPU       : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Model     : {args.model}")
    print(f"Batch     : {args.batch_size}")

    min_cos = (
        args.min_percent
        / 100.0
        * args.reference_cos
    )

    print(
        f"Threshold : "
        f"{args.min_percent:.2f}% "
        f"(cos >= {min_cos:.4f})"
    )

    print()
    print("Queries:")

    for query in args.query:
        print(f"  - {query}")

    print()
    print("Loading CLIP...")

    model, preprocess = clip.load(
        args.model,
        device=device,
    )

    model.eval()

    print("Encoding queries...")

    tokens = clip.tokenize(
        args.query
    ).to(device)

    with torch.inference_mode():

        text_features = model.encode_text(
            tokens
        )

        text_features /= text_features.norm(
            dim=-1,
            keepdim=True,
        )

    print("Scanning...")
    print()

    scanned = 0
    errors = 0
    matches = []

    batch_paths = []

    for path in find_images(root):

        scanned += 1

        batch_paths.append(path)

        if len(batch_paths) < args.batch_size:
            continue

        batch_results, batch_errors = (
            process_batch(
                batch_paths,
                model=model,
                preprocess=preprocess,
                text_features=text_features,
                queries=args.query,
                device=device,
                root=root,
                display_root=args.display_root,
                min_percent=args.min_percent,
                reference_cos=args.reference_cos,
            )
        )

        matches.extend(batch_results)
        errors += batch_errors

        print(
            f"Processed: {scanned:>8} | "
            f"Matches: {len(matches):>6}"
        )

        batch_paths = []

    # Final incomplete batch

    if batch_paths:

        batch_results, batch_errors = (
            process_batch(
                batch_paths,
                model=model,
                preprocess=preprocess,
                text_features=text_features,
                queries=args.query,
                device=device,
                root=root,
                display_root=args.display_root,
                min_percent=args.min_percent,
                reference_cos=args.reference_cos,
            )
        )

        matches.extend(batch_results)
        errors += batch_errors

    matches.sort(
        key=lambda item: item["Cos"],
        reverse=True,
    )

    print()
    print("=" * 78)
    print("MATCHES")
    print("=" * 78)

    if matches:

        for result in matches:

            print(
                f'{result["%"]:6.2f}% | '
                f'cos={result["Cos"]:.6f} | '
                f'{result["FilePath"]}'
            )

            print(
                f'         | '
                f'{result["MatchedQuery"]}'
            )

    else:
        print("No matches found.")

    output_path = Path(args.output)

    write_csv(
        matches,
        output_path,
    )

    print()
    print("=" * 78)
    print(f"Images scanned : {scanned}")
    print(f"Matches        : {len(matches)}")
    print(f"Errors         : {errors}")
    print(f"CSV            : {output_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()