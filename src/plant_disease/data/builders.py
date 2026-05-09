from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SEVERITY_THRESHOLDS = (0.01, 0.05, 0.20)


def _iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def compute_mask_severity_ratio(mask_path: str | Path) -> float:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask for severity: {mask_path}")
    total_pixels = float(mask.shape[0] * mask.shape[1])
    if total_pixels <= 0:
        return 0.0
    lesion_pixels = float((mask > 0).sum())
    return lesion_pixels / total_pixels


def severity_label_from_ratio(ratio: float, thresholds: tuple[float, float, float] = DEFAULT_SEVERITY_THRESHOLDS) -> int:
    low, medium, high = thresholds
    if ratio < low:
        return 0
    if ratio < medium:
        return 1
    if ratio < high:
        return 2
    return 3


def annotate_severity_columns(
    dataframe: pd.DataFrame,
    thresholds: tuple[float, float, float] = DEFAULT_SEVERITY_THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    for _, row in dataframe.iterrows():
        record = row.to_dict()
        has_mask = bool(record.get("has_lesion_mask", False))
        mask_path = record.get("lesion_mask_path")
        if has_mask and mask_path:
            ratio = compute_mask_severity_ratio(mask_path)
            record["severity_ratio"] = ratio
            record["severity_label"] = severity_label_from_ratio(ratio, thresholds=thresholds)
            record["has_valid_severity"] = True
        else:
            record["severity_ratio"] = -1.0
            record["severity_label"] = -1
            record["has_valid_severity"] = False
        rows.append(record)
    return pd.DataFrame(rows)


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
                    "severity_label": -1,
                    "severity_ratio": -1.0,
                    "has_valid_severity": False,
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
                "split": split,
                "lesion_mask_path": str(mask_path),
            }
        )
    return annotate_severity_columns(pd.DataFrame(rows))


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
                "split": split,
                "lesion_mask_path": str(mask_path),
            }
        )
    return annotate_severity_columns(pd.DataFrame(rows)), class_to_idx


def build_mixed_manifest(real_manifest_csv: str | Path, pseudo_manifest_csv: str | Path) -> pd.DataFrame:
    real_df = pd.read_csv(real_manifest_csv)
    pseudo_df = pd.read_csv(pseudo_manifest_csv)
    if "pseudo_label" not in real_df.columns:
        real_df["pseudo_label"] = False
    pseudo_df["pseudo_label"] = True
    combined = pd.concat([real_df, pseudo_df], ignore_index=True)
    return combined
