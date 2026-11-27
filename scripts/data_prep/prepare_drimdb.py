#!/usr/bin/env python3
"""
Prepare a small DRIMDB pilot subset for Medical AgenticIQA.

DRIMDB (Diabetic Retinopathy Image Database) is a public retinal fundus image
quality benchmark. The original release contains 216 images with three
categorical quality labels: Good (125), Bad (69), and Outlier (22).

This script intentionally uses only the original images at the top level of
each class directory. Some redistributions (including a Kaggle mirror) also
contain a nested ``iwebalbumfiles`` directory with thumbnails and album copies;
those are ignored to avoid duplicate or reduced-resolution samples.

Because DRIMDB does not provide a continuous MOS, this script maps the three
ordered quality classes to pseudo-MOS values for correlation analysis:

    Good    -> 5.0
    Bad     -> 3.0
    Outlier -> 1.0

The mapping is ordinal only. It is not a clinical quality score and should not
be interpreted as one. SRCC on the mapped labels is the primary metric; MAE and
accuracy are secondary and depend on this arbitrary spacing.

Usage:
    python scripts/data_prep/prepare_drimdb.py \
        --data-dir data/raw/drimdb/DRIMDB \
        --output data/processed/drimdb/manifest.jsonl \
        --per-class 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


CLASS_TO_MOS: Dict[str, float] = {
    "Good": 5.0,
    "Bad": 3.0,
    "Outlier": 1.0,
}

CLASS_ORDER = ["Good", "Bad", "Outlier"]

DEFAULT_QUERY = (
    "Assign a diagnostic quality score from 1 to 5 for this retinal fundus "
    "photograph, where 1 is unusable and 5 is diagnostic quality. "
    "Return only the numerical score."
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def resolve_data_dir(data_dir: Path) -> Path:
    """Return the directory that directly contains the class folders."""
    if all((data_dir / label).is_dir() for label in CLASS_ORDER):
        return data_dir

    nested = data_dir / "DRIMDB"
    if all((nested / label).is_dir() for label in CLASS_ORDER):
        return nested

    missing = [label for label in CLASS_ORDER if not (data_dir / label).is_dir()]
    raise FileNotFoundError(
        f"Could not find DRIMDB class directories {missing} under {data_dir}"
    )


def list_class_images(class_dir: Path) -> List[Path]:
    """List directly contained image files, excluding album/thumbnail trees."""
    return sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_image(path: Path) -> bool:
    """Return True if PIL can decode the image."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.width > 0 and image.height > 0
    except Exception as exc:  # pragma: no cover - defensive, depends on files
        logger.warning("Skipping unreadable image %s: %s", path, exc)
        return False


def select_class_images(
    images: Iterable[Path],
    per_class: int,
    rng: random.Random,
) -> List[Path]:
    """Select a deterministic subset of readable images from one class."""
    candidates = [p for p in images if validate_image(p)]
    if len(candidates) <= per_class:
        return candidates
    return sorted(rng.sample(candidates, per_class))


def choose_output_path(image_path: Path, output_dir: Path, relative_to: Path) -> str:
    """Choose a portable path for the manifest."""
    try:
        return str(output_dir.relative_to(relative_to))
    except ValueError:
        return str(output_dir)


def build_subset(
    data_dir: Path,
    output: Path,
    ground_truth_output: Path,
    images_dir: Path,
    per_class: int,
    seed: int,
    query: str,
    copy_images: bool,
    path_mode: str,
) -> Tuple[int, Dict[str, int]]:
    """Build the manifest and ground-truth files. Returns (count, class counts)."""
    if per_class < 1:
        raise ValueError("--per-class must be at least 1")

    data_dir = resolve_data_dir(data_dir)
    rng = random.Random(seed)
    repo_root = Path.cwd()

    output.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_output.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[dict] = []
    gt_entries: List[dict] = []
    class_counts: Dict[str, int] = {}

    for label in CLASS_ORDER:
        class_dir = data_dir / label
        pool = list_class_images(class_dir)
        if not pool:
            raise FileNotFoundError(f"No images found in {class_dir}")

        selected = select_class_images(pool, per_class, rng)
        class_counts[label] = len(selected)
        logger.info(
            "Class %s: %d available originals, selected %d",
            label,
            len(pool),
            len(selected),
        )

        for index, source in enumerate(selected, start=1):
            sample_id = f"drimdb_{label.lower()}_{index:03d}"

            if copy_images:
                destination = images_dir / f"{sample_id}{source.suffix.lower()}"
                shutil.copy2(source, destination)
                image_path = destination
            else:
                image_path = source

            if path_mode == "relative":
                manifest_image_path = choose_output_path(
                    image_path, image_path, repo_root
                )
            else:
                manifest_image_path = str(image_path.resolve())

            entry = {
                "sample_id": sample_id,
                "dataset": "drimdb",
                "split": "pilot",
                "query": query,
                "image_path": manifest_image_path,
                "mos": CLASS_TO_MOS[label],
                "quality_label": label,
                "metadata": {
                    "source_dataset": "DRIMDB",
                    "source_label": label,
                    "source_file": source.name,
                    "label_mapping": {
                        "Good": 5.0,
                        "Bad": 3.0,
                        "Outlier": 1.0,
                    },
                    "note": (
                        "Categorical DRIMDB quality label mapped to an ordinal "
                        "pseudo-MOS. Not a clinical MOS."
                    ),
                },
            }
            manifest_entries.append(entry)
            gt_entries.append(
                {
                    "sample_id": sample_id,
                    "mos": CLASS_TO_MOS[label],
                    "quality_label": label,
                    "source_file": source.name,
                }
            )

    with open(output, "w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(ground_truth_output, "w", encoding="utf-8") as handle:
        for entry in gt_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Wrote %d samples to %s", len(manifest_entries), output)
    logger.info("Wrote ground truth to %s", ground_truth_output)
    return len(manifest_entries), class_counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a stratified DRIMDB pilot subset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="DRIMDB root containing Good/, Bad/, Outlier/ (or its parent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/drimdb/manifest.jsonl"),
        help="Output manifest JSONL",
    )
    parser.add_argument(
        "--ground-truth-output",
        type=Path,
        default=Path("data/processed/drimdb/ground_truth.jsonl"),
        help="Output ground-truth JSONL",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/processed/drimdb/images"),
        help="Directory for copied pilot images",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10,
        help="Number of images to sample from each quality class (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Assessment query stored in every manifest entry",
    )
    parser.add_argument(
        "--path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Write manifest image paths relative to cwd or absolute",
    )
    parser.add_argument(
        "--no-copy-images",
        dest="copy_images",
        action="store_false",
        help="Reference original images instead of copying a flat subset",
    )
    parser.set_defaults(copy_images=True)

    args = parser.parse_args()

    try:
        count, counts = build_subset(
            data_dir=args.data_dir,
            output=args.output,
            ground_truth_output=args.ground_truth_output,
            images_dir=args.images_dir,
            per_class=args.per_class,
            seed=args.seed,
            query=args.query,
            copy_images=args.copy_images,
            path_mode=args.path_mode,
        )
    except Exception as exc:
        logger.error("Failed to prepare DRIMDB subset: %s", exc)
        return 1

    print("\nDRIMDB pilot prepared")
    print(f"  total samples: {count}")
    for label in CLASS_ORDER:
        print(f"  {label}: {counts.get(label, 0)}")
    print("  label mapping: Good=5, Bad=3, Outlier=1 (ordinal pseudo-MOS)")
    print(f"  manifest: {args.output}")
    print(f"  ground truth: {args.ground_truth_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
