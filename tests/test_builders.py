from pathlib import Path

import cv2
import numpy as np

from plant_disease.data.builders import build_classification_manifest, build_segmentation_manifest


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
