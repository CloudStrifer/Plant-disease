from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from plant_disease.data.dataset import PlantDiseaseDataset


def test_dataset_returns_mask_presence_flags(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    df = pd.DataFrame(
        [
            {
                "image_path": str(image_dir / "sample.png"),
                "class_id": 2,
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "unit",
                "severity_label": 1,
                "lesion_mask_path": str(mask_dir / "sample.png"),
            }
        ]
    )

    ds = PlantDiseaseDataset(df, transform=None)
    sample = ds[0]

    assert sample["image"].shape == (3, 8, 8)
    assert sample["class_id"] == 2
    assert sample["has_lesion_mask"] is True
    assert sample["lesion_mask"].shape == (1, 8, 8)
