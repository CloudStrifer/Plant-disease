from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from plant_disease.data.builders import build_plantseg_manifest


def test_build_plantseg_manifest_uses_metadata_and_split_dirs(tmp_path: Path):
    root = tmp_path / "plantseg"
    (root / "images" / "train").mkdir(parents=True)
    (root / "annotations" / "train").mkdir(parents=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    cv2.imwrite(str(root / "images" / "train" / "apple_black_rot_1.jpg"), image)
    cv2.imwrite(str(root / "annotations" / "train" / "apple_black_rot_1.png"), mask)

    metadata = pd.DataFrame(
        [
            {
                "Name": "apple_black_rot_1.jpg",
                "Plant": "Apple",
                "Disease": "apple black rot",
                "Label file": "apple_black_rot_1.png",
                "Split": "Training",
            }
        ]
    )
    metadata.to_csv(root / "Metadatav2.csv", index=False)

    df, class_to_idx = build_plantseg_manifest(root)

    assert len(df) == 1
    assert df.loc[0, "split"] == "train"
    assert bool(df.loc[0, "has_lesion_mask"]) is True
    assert df.loc[0, "class_name"] == "Apple___apple black rot"
    assert class_to_idx["Apple___apple black rot"] == df.loc[0, "class_id"]
