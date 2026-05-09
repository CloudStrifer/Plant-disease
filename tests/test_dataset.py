from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from plant_disease.data.dataset import PlantDiseaseDataset
from plant_disease.data.transforms import build_train_transform


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
                "class_name": "Tomato___Leaf_Mold",
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
    assert sample["class_name"] == "Tomato___Leaf_Mold"
    assert sample["source_dataset"] == "unit"
    assert sample["pseudo_label"] is False
    assert sample["seg_loss_weight"] == 1.0
    assert sample["concept_targets"].shape[0] == 4
    assert sample["concept_valid_mask"].shape[0] == 4
    assert sample["concept_weights"].shape[0] == 4
    assert float(sample["concept_targets"][2]) == 1.0


def test_dataset_transform_resizes_image_and_mask(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:24, 12:36] = 255

    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    df = pd.DataFrame(
        [
            {
                "image_path": str(image_dir / "sample.png"),
                "class_id": 1,
                "class_name": "Tomato___Early_blight",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "unit",
                "severity_label": 0,
                "lesion_mask_path": str(mask_dir / "sample.png"),
            }
        ]
    )

    transform = build_train_transform(image_size=64)
    ds = PlantDiseaseDataset(df, transform=transform)
    sample = ds[0]

    assert sample["image"].shape == (3, 64, 64)
    assert sample["lesion_mask"].shape == (1, 64, 64)
    assert sample["concept_targets"].shape[0] == 4


def test_dataset_reads_optional_segmentation_weight(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:7, 1:7] = 255

    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    df = pd.DataFrame(
        [
            {
                "image_path": str(image_dir / "sample.png"),
                "class_id": 0,
                "class_name": "Potato___Late_blight",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantVillage_pseudo",
                "severity_label": 0,
                "lesion_mask_path": str(mask_dir / "sample.png"),
                "pseudo_label": True,
                "seg_loss_weight": 0.3,
            }
        ]
    )

    sample = PlantDiseaseDataset(df, transform=None)[0]
    assert sample["pseudo_label"] is True
    assert sample["seg_loss_weight"] == 0.3
    assert sample["class_name"] == "Potato___Late_blight"


def test_dataset_resolves_paths_relative_to_manifest_dir(tmp_path: Path):
    manifest_dir = tmp_path / "manifests"
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "artifacts" / "pseudo"
    manifest_dir.mkdir()
    image_dir.mkdir()
    mask_dir.mkdir(parents=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    df = pd.DataFrame(
        [
            {
                "image_path": str(image_dir / "sample.png").replace("/", "\\"),
                "class_id": 0,
                "class_name": "Apple___Black_rot",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantVillage_pseudo",
                "severity_label": 0,
                "lesion_mask_path": "..\\artifacts\\pseudo\\sample.png",
                "__manifest_dir": str(manifest_dir.resolve()),
            }
        ]
    )

    sample = PlantDiseaseDataset(df, transform=None)[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["lesion_mask"].shape == (1, 8, 8)


def test_dataset_rebases_deep_manifest_relative_artifact_paths_to_repo_root(tmp_path: Path):
    repo_root = tmp_path / "repo"
    image_dir = repo_root / "src" / "plant_disease" / "dataset" / "plantseg" / "images" / "train"
    mask_dir = repo_root / "artifacts" / "plantvillage_pseudo"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    cv2.imwrite(str(image_dir / "sample.jpg"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    df = pd.DataFrame(
        [
            {
                "image_path": str(image_dir / "sample.jpg").replace("/", "\\"),
                "class_id": 0,
                "class_name": "Apple___Black_rot",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantVillage_pseudo",
                "severity_label": 0,
                "lesion_mask_path": "..\\artifacts\\plantvillage_pseudo\\sample.png",
                "__manifest_dir": str((repo_root / "artifacts" / "aligned_subsets" / "subset").resolve()),
            }
        ]
    )

    import plant_disease.data.dataset as dataset_module

    original_root = dataset_module.REPO_ROOT
    dataset_module.REPO_ROOT = repo_root
    try:
        sample = PlantDiseaseDataset(df, transform=None)[0]
    finally:
        dataset_module.REPO_ROOT = original_root

    assert sample["image"].shape == (3, 8, 8)
    assert sample["lesion_mask"].shape == (1, 8, 8)
