from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def build_classification_manifest(root: str | Path, source_dataset: str, split: str = "train"):
    root = Path(root)
    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}
    rows = []
    for class_dir in class_dirs:
        class_id = class_to_idx[class_dir.name]
        for image_path in _iter_images(class_dir):
            rows.append(
                {
                    "image_path": str(image_path),
                    "class_id": class_id,
                    "class_name": class_dir.name,
                    "has_lesion_mask": False,
                    "has_leaf_mask": False,
                    "source_dataset": source_dataset,
                    "severity_label": 0,
                    "split": split,
                }
            )
    return pd.DataFrame(rows), class_to_idx


def build_segmentation_manifest(
    image_dir: str | Path,
    mask_dir: str | Path,
    source_dataset: str,
    class_name: str,
    class_id: int,
    split: str = "train",
):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    rows = []
    for image_path in _iter_images(image_dir):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "class_id": class_id,
                "class_name": class_name,
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": source_dataset,
                "severity_label": 0,
                "split": split,
                "lesion_mask_path": str(mask_path),
            }
        )
    return pd.DataFrame(rows)


def build_plantseg_manifest(root: str | Path):
    root = Path(root)
    metadata_path = root / "Metadatav2.csv"
    metadata = pd.read_csv(metadata_path)
    split_map = {
        "Training": "train",
        "Validation": "val",
        "Test": "test",
    }

    class_names = []
    for _, row in metadata.iterrows():
        class_names.append(f"{row['Plant']}___{row['Disease']}")
    class_to_idx = {name: idx for idx, name in enumerate(sorted(set(class_names)))}

    rows = []
    for _, row in metadata.iterrows():
        split = split_map[row["Split"]]
        class_name = f"{row['Plant']}___{row['Disease']}"
        image_path = root / "images" / split / row["Name"]
        mask_path = root / "annotations" / split / row["Label file"]
        rows.append(
            {
                "image_path": str(image_path),
                "class_id": class_to_idx[class_name],
                "class_name": class_name,
                "has_lesion_mask": mask_path.exists(),
                "has_leaf_mask": False,
                "source_dataset": "PlantSeg",
                "severity_label": 0,
                "split": split,
                "lesion_mask_path": str(mask_path),
            }
        )
    return pd.DataFrame(rows), class_to_idx
