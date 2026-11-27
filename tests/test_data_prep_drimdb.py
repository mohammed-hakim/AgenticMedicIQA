"""Tests for the DRIMDB pilot manifest preparation script."""

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.data_prep.prepare_drimdb import (
    CLASS_TO_MOS,
    build_subset,
    list_class_images,
    resolve_data_dir,
    select_class_images,
)


def _make_image(path: Path, color=(128, 128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="JPEG")


@pytest.fixture()
def drimdb_root(tmp_path: Path) -> Path:
    """Create a miniature DRIMDB-like tree with nested album copies."""
    root = tmp_path / "DRIMDB"
    colors = {
        "Good": (10, 200, 10),
        "Bad": (200, 200, 10),
        "Outlier": (200, 10, 10),
    }
    for label, color in colors.items():
        class_dir = root / label
        for index in range(3):
            _make_image(class_dir / f"{label.lower()}_{index}.jpg", color)

    # Mirror the nested album tree that must be ignored.
    _make_image(root / "Good" / "iwebalbumfiles" / "thumb_good_0.jpg")
    (root / "README.txt").write_text("synthetic test tree", encoding="utf-8")
    return root


def test_resolve_data_dir_accepts_parent_and_nested_root(tmp_path: Path, drimdb_root: Path):
    assert resolve_data_dir(drimdb_root) == drimdb_root
    assert resolve_data_dir(drimdb_root.parent) == drimdb_root


def test_resolve_data_dir_reports_missing_classes(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Could not find DRIMDB class directories"):
        resolve_data_dir(tmp_path / "empty")


def test_list_class_images_ignores_nested_album(drimdb_root: Path):
    images = list_class_images(drimdb_root / "Good")
    assert len(images) == 3
    assert all(image.parent == drimdb_root / "Good" for image in images)


def test_select_class_images_is_deterministic(drimdb_root: Path):
    import random

    images = list_class_images(drimdb_root / "Good")
    first = select_class_images(images, per_class=2, rng=random.Random(42))
    second = select_class_images(images, per_class=2, rng=random.Random(42))

    assert [p.name for p in first] == [p.name for p in second]
    assert len(first) == 2


def test_build_subset_writes_manifest_ground_truth_and_copies(
    tmp_path: Path, drimdb_root: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "processed" / "manifest.jsonl"
    ground_truth = tmp_path / "processed" / "ground_truth.jsonl"
    images_dir = tmp_path / "processed" / "images"

    count, class_counts = build_subset(
        data_dir=drimdb_root,
        output=output,
        ground_truth_output=ground_truth,
        images_dir=images_dir,
        per_class=2,
        seed=42,
        query="Assign a score from 1 to 5.",
        copy_images=True,
        path_mode="absolute",
    )

    assert count == 6
    assert class_counts == {"Good": 2, "Bad": 2, "Outlier": 2}

    manifest = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    gt = [json.loads(line) for line in ground_truth.read_text(encoding="utf-8").splitlines()]

    assert len(manifest) == len(gt) == 6
    assert {entry["sample_id"] for entry in manifest} == {entry["sample_id"] for entry in gt}
    assert {entry["quality_label"] for entry in manifest} == {"Good", "Bad", "Outlier"}

    for entry, ground_truth_entry in zip(manifest, gt):
        assert entry["sample_id"] == ground_truth_entry["sample_id"]
        assert entry["mos"] == ground_truth_entry["mos"]
        assert entry["mos"] == CLASS_TO_MOS[entry["quality_label"]]
        assert entry["dataset"] == "drimdb"
        assert entry["split"] == "pilot"
        assert Path(entry["image_path"]).is_file()
        assert Path(entry["image_path"]).parent == images_dir
        assert entry["metadata"]["source_label"] == entry["quality_label"]


def test_build_subset_rejects_invalid_per_class(tmp_path: Path, drimdb_root: Path):
    with pytest.raises(ValueError, match="at least 1"):
        build_subset(
            data_dir=drimdb_root,
            output=tmp_path / "manifest.jsonl",
            ground_truth_output=tmp_path / "gt.jsonl",
            images_dir=tmp_path / "images",
            per_class=0,
            seed=42,
            query="score",
            copy_images=False,
            path_mode="absolute",
        )
