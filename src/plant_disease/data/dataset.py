from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from plant_disease.concepts.rules import derive_concept_supervision

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_data_path(raw_path: str | None, manifest_dir: str | None = None) -> str | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None

    normalized = text.replace("\\", "/")
    direct = Path(normalized)
    if direct.exists():
        return str(direct)

    if manifest_dir:
        manifest_candidate = (Path(manifest_dir) / normalized).resolve()
        if manifest_candidate.exists():
            return str(manifest_candidate)

    repo_candidate = (REPO_ROOT / normalized).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    lowered = normalized.lower()
    for anchor in ["src/plant_disease/dataset", "artifacts", "data"]:
        index = lowered.find(anchor)
        if index != -1:
            suffix = normalized[index:]
            suffix_path = Path(*[part for part in suffix.split("/") if part])
            rebased = (REPO_ROOT / suffix_path).resolve()
            if rebased.exists():
                return str(rebased)

    return normalized


class PlantDiseaseDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        manifest_dir = str(row["__manifest_dir"]) if "__manifest_dir" in row.index else None
        image_path = _resolve_data_path(row["image_path"], manifest_dir=manifest_dir)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image_path={row['image_path']} resolved_path={image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lesion_mask = None
        if bool(row["has_lesion_mask"]) and "lesion_mask_path" in row:
            mask_path = _resolve_data_path(row["lesion_mask_path"], manifest_dir=manifest_dir)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(
                    f"Failed to read lesion_mask_path={row['lesion_mask_path']} resolved_path={mask_path}"
                )
            lesion_mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            image, lesion_mask = self.transform(image, lesion_mask)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if lesion_mask is None:
            lesion_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        else:
            lesion_mask = torch.from_numpy(lesion_mask.astype(np.float32)).unsqueeze(0)

        class_name = str(row["class_name"]) if "class_name" in row else ""
        concept_targets, concept_valid_mask, concept_weights = derive_concept_supervision(image, lesion_mask, class_name)

        return {
            "image": image,
            "class_id": int(row["class_id"]),
            "class_name": class_name,
            "severity_label": int(row["severity_label"]),
            "has_lesion_mask": bool(row["has_lesion_mask"]),
            "lesion_mask": lesion_mask,
            "source_dataset": str(row["source_dataset"]) if "source_dataset" in row else "",
            "pseudo_label": bool(row["pseudo_label"]) if "pseudo_label" in row else False,
            "seg_loss_weight": float(row["seg_loss_weight"]) if "seg_loss_weight" in row else 1.0,
            "concept_targets": concept_targets,
            "concept_valid_mask": concept_valid_mask,
            "concept_weights": concept_weights,
        }
