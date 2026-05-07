from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        lesion_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        if bool(row["has_lesion_mask"]) and "lesion_mask_path" in row:
            mask = cv2.imread(row["lesion_mask_path"], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)
            lesion_mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image,
            "class_id": int(row["class_id"]),
            "severity_label": int(row["severity_label"]),
            "has_lesion_mask": bool(row["has_lesion_mask"]),
            "lesion_mask": lesion_mask,
        }
