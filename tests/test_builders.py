from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from plant_disease.data.builders import (
    annotate_severity_columns,
    build_classification_manifest,
    build_mixed_manifest,
    build_segmentation_manifest,
)


def test_build_classification_manifest_reads_class_directories(tmp_path: Path):
    root = tmp_path / "plantvillage"
    (root / "Tomato___healthy").mkdir(parents=True)
    (root / "Tomato___Early_blight").mkdir(parents=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "Tomato___healthy" / "a.png"), image)
    cv2.imwrite(str(root / "Tomato___Early_blight" / "b.png"), image)

    df, class_to_idx = build_classification_manifest(root, source_dataset="PlantVillage", split="train")

    assert len(df) == 2
    assert set(df["split"]) == {"train"}
    assert set(class_to_idx) == {"Tomato___Early_blight", "Tomato___healthy"}
    assert df["has_lesion_mask"].sum() == 0
    assert set(df["severity_label"]) == {-1}
    assert set(df["has_valid_severity"]) == {False}


def test_build_segmentation_manifest_pairs_images_and_masks(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    cv2.imwrite(str(image_dir / "leaf_001.png"), image)
    cv2.imwrite(str(mask_dir / "leaf_001.png"), mask)

    df = build_segmentation_manifest(
        image_dir=image_dir,
        mask_dir=mask_dir,
        source_dataset="PlantSeg",
        class_name="Tomato___Leaf_Mold",
        class_id=4,
        split="train",
    )

    assert len(df) == 1
    assert bool(df.loc[0, "has_lesion_mask"]) is True
    assert Path(df.loc[0, "lesion_mask_path"]).name == "leaf_001.png"
    assert df.loc[0, "class_id"] == 4
    assert float(df.loc[0, "severity_ratio"]) == 0.0625
    assert int(df.loc[0, "severity_label"]) == 2
    assert bool(df.loc[0, "has_valid_severity"]) is True


def test_build_mixed_manifest_marks_pseudo_rows(tmp_path: Path):
    real_csv = tmp_path / "real.csv"
    pseudo_csv = tmp_path / "pseudo.csv"

    pd.DataFrame(
        [
            {
                "image_path": "a.png",
                "class_id": 0,
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantSeg",
                "severity_label": 0,
            }
        ]
    ).to_csv(real_csv, index=False)

    pd.DataFrame(
        [
            {
                "image_path": "b.png",
                "class_id": 1,
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantVillage_pseudo",
                "severity_label": 0,
                "pseudo_label": True,
            }
        ]
    ).to_csv(pseudo_csv, index=False)

    df = build_mixed_manifest(real_csv, pseudo_csv)

    assert len(df) == 2
    assert bool(df.loc[0, "pseudo_label"]) is False
    assert bool(df.loc[1, "pseudo_label"]) is True


def test_annotate_severity_columns_marks_invalid_rows_without_masks(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:1, :] = 255
    cv2.imwrite(str(mask_path), mask)

    df = pd.DataFrame(
        [
            {"has_lesion_mask": True, "lesion_mask_path": str(mask_path)},
            {"has_lesion_mask": False, "lesion_mask_path": ""},
        ]
    )

    enriched = annotate_severity_columns(df)

    assert float(enriched.loc[0, "severity_ratio"]) == 0.1
    assert int(enriched.loc[0, "severity_label"]) == 2
    assert bool(enriched.loc[0, "has_valid_severity"]) is True
    assert float(enriched.loc[1, "severity_ratio"]) == -1.0
    assert int(enriched.loc[1, "severity_label"]) == -1
    assert bool(enriched.loc[1, "has_valid_severity"]) is False
