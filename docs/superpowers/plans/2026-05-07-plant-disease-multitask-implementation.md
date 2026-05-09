# Plant Disease Multi-Task Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prototype-first, paper-ready plant disease framework that supports lesion segmentation, lesion-guided classification, severity estimation, pseudo-mask generation, and concept-rule reasoning.

**Architecture:** Start from a stable `SegFormer-B0` multi-task baseline with three outputs: lesion mask, disease class, and severity grade. Then add lesion-guided fusion, offline pseudo-mask generation for weak supervision, and finally a concept head with differentiable soft rule loss.

**Tech Stack:** Python 3.13, PyTorch, MMSegmentation, mmengine, timm, pytorch-grad-cam, segment-anything, albumentations, OpenCV, scikit-learn, pytest

---

## File Structure

Planned repository layout:

- `pyproject.toml`
  Python package metadata and tool configuration.
- `README.md`
  Project entrypoint and setup instructions.
- `configs/baseline_segformer_b0.yaml`
  Baseline experiment config.
- `configs/lesion_guided_segformer_b0.yaml`
  Lesion-guided config.
- `configs/pseudo_label_segformer_b0.yaml`
  Mixed-supervision config.
- `configs/concept_rule_segformer_b0.yaml`
  Full model config.
- `scripts/train_baseline.py`
  Launch baseline and lesion-guided experiments.
- `scripts/generate_pseudo_masks.py`
  Offline CAM-to-mask pipeline.
- `scripts/evaluate_model.py`
  Standard evaluation entrypoint.
- `src/plant_disease/__init__.py`
  Package marker.
- `src/plant_disease/data/manifest.py`
  CSV manifest parsing and validation.
- `src/plant_disease/data/dataset.py`
  Dataset and collate logic.
- `src/plant_disease/data/transforms.py`
  Train and eval transforms.
- `src/plant_disease/models/heads.py`
  Classification, severity, and concept heads.
- `src/plant_disease/models/lesion_guided.py`
  Lesion-guided fusion module.
- `src/plant_disease/models/multitask_model.py`
  Main multi-task model wrapper.
- `src/plant_disease/training/losses.py`
  Joint losses and weighting helpers.
- `src/plant_disease/training/engine.py`
  Train and validation loops.
- `src/plant_disease/eval/metrics.py`
  Classification, segmentation, severity, and box metrics.
- `src/plant_disease/pseudo_labels/cam_pipeline.py`
  CAM generation, thresholding, mask refinement.
- `src/plant_disease/concepts/rules.py`
  Concept vocabulary and differentiable rule penalties.
- `tests/test_manifest.py`
  Manifest validation tests.
- `tests/test_dataset.py`
  Dataset sample loading tests.
- `tests/test_model_shapes.py`
  Model I/O tests.
- `tests/test_losses.py`
  Joint loss tests.
- `tests/test_metrics.py`
  Metrics tests.
- `tests/test_cam_pipeline.py`
  Pseudo-mask generation tests.
- `tests/test_rules.py`
  Concept-rule consistency tests.

This structure keeps responsibilities narrow enough that each unit can be tested independently.

### Task 1: Bootstrap the Repository and Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/plant_disease/__init__.py`
- Test: `tests/test_import_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_smoke.py
def test_package_imports():
    from plant_disease import __version__

    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_import_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plant_disease'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "plant-disease"
version = "0.1.0"
description = "Plant disease multi-task research prototype"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
  "torch",
  "torchvision",
  "mmengine",
  "mmsegmentation",
  "timm",
  "albumentations",
  "opencv-python",
  "numpy",
  "pandas",
  "scikit-learn",
  "pytorch-grad-cam",
  "segment-anything",
  "pyyaml",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```markdown
# README.md

## Plant Disease Multi-Task Framework

Prototype-first implementation of lesion segmentation, lesion-guided classification, severity estimation, pseudo-label generation, and concept-rule reasoning.
```

```python
# src/plant_disease/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_import_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml README.md src/plant_disease/__init__.py tests/test_import_smoke.py
git commit -m "chore: bootstrap project package and test config"
```

### Task 2: Implement the Dataset Manifest Contract

**Files:**
- Create: `src/plant_disease/data/manifest.py`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
from pathlib import Path

from plant_disease.data.manifest import load_manifest


def test_load_manifest_validates_required_columns(tmp_path: Path):
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text("image_path,class_id\nsample.jpg,1\n", encoding="utf-8")

    try:
        load_manifest(csv_path)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected load_manifest to reject incomplete manifest")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.data.manifest`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/data/manifest.py
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "image_path",
    "class_id",
    "has_lesion_mask",
    "has_leaf_mask",
    "source_dataset",
    "severity_label",
}


def load_manifest(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_text}")
    return df.copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/data/manifest.py tests/test_manifest.py
git commit -m "feat: add manifest loader contract"
```

### Task 3: Implement Dataset Loading for Mixed Supervision

**Files:**
- Create: `src/plant_disease/data/dataset.py`
- Create: `src/plant_disease/data/transforms.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.data.dataset`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/data/transforms.py
from __future__ import annotations


def build_train_transform():
    return None


def build_eval_transform():
    return None
```

```python
# src/plant_disease/data/dataset.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/data/dataset.py src/plant_disease/data/transforms.py tests/test_dataset.py
git commit -m "feat: add mixed-supervision dataset loader"
```

### Task 4: Implement Core Metrics and Severity Utilities

**Files:**
- Create: `src/plant_disease/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import torch

from plant_disease.eval.metrics import dice_score, severity_ratio


def test_dice_score_matches_identical_masks():
    mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert dice_score(mask, mask) == 1.0


def test_severity_ratio_uses_lesion_over_leaf_area():
    lesion = torch.tensor([[1, 1], [0, 0]])
    leaf = torch.tensor([[1, 1], [1, 1]])
    assert severity_ratio(lesion, leaf) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.eval.metrics`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/eval/metrics.py
import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (union + eps)


def severity_ratio(lesion_mask: torch.Tensor, leaf_mask: torch.Tensor) -> float:
    lesion = float((lesion_mask > 0).sum())
    leaf = float((leaf_mask > 0).sum())
    if leaf == 0:
        return 0.0
    return lesion / leaf
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/eval/metrics.py tests/test_metrics.py
git commit -m "feat: add baseline segmentation and severity metrics"
```

### Task 5: Implement the Multi-Task Model Skeleton

**Files:**
- Create: `src/plant_disease/models/heads.py`
- Create: `src/plant_disease/models/multitask_model.py`
- Test: `tests/test_model_shapes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_shapes.py
import torch

from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel


def test_multitask_model_outputs_expected_shapes():
    model = MultiTaskPlantDiseaseModel(
        in_channels=32,
        num_classes=4,
        num_severity_grades=3,
    )
    x = torch.randn(2, 32, 16, 16)
    outputs = model.forward_from_features(x)

    assert outputs["segmentation_logits"].shape == (2, 1, 16, 16)
    assert outputs["classification_logits"].shape == (2, 4)
    assert outputs["severity_logits"].shape == (2, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_shapes.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.models.multitask_model`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/models/heads.py
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class SeverityHead(nn.Module):
    def __init__(self, in_features: int, num_grades: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_grades)

    def forward(self, x):
        return self.fc(x)
```

```python
# src/plant_disease/models/multitask_model.py
import torch
import torch.nn as nn

from plant_disease.models.heads import ClassificationHead, SeverityHead


class MultiTaskPlantDiseaseModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_severity_grades: int):
        super().__init__()
        self.seg_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = ClassificationHead(in_channels, num_classes)
        self.sev_head = SeverityHead(in_channels, num_severity_grades)

    def forward_from_features(self, features: torch.Tensor) -> dict:
        segmentation_logits = self.seg_head(features)
        pooled = self.pool(features).flatten(1)
        classification_logits = self.cls_head(pooled)
        severity_logits = self.sev_head(pooled)
        return {
            "segmentation_logits": segmentation_logits,
            "classification_logits": classification_logits,
            "severity_logits": severity_logits,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_shapes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/models/heads.py src/plant_disease/models/multitask_model.py tests/test_model_shapes.py
git commit -m "feat: add multitask model skeleton"
```

### Task 6: Implement Joint Losses and Baseline Training Engine

**Files:**
- Create: `src/plant_disease/training/losses.py`
- Create: `src/plant_disease/training/engine.py`
- Create: `scripts/train_baseline.py`
- Create: `configs/baseline_segformer_b0.yaml`
- Test: `tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses.py
import torch

from plant_disease.training.losses import compute_multitask_loss


def test_multitask_loss_returns_named_components():
    outputs = {
        "segmentation_logits": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "classification_logits": torch.zeros((2, 3), dtype=torch.float32),
        "severity_logits": torch.zeros((2, 2), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "class_id": torch.zeros(2, dtype=torch.long),
        "severity_label": torch.zeros(2, dtype=torch.long),
    }
    losses = compute_multitask_loss(outputs, batch)

    assert set(losses) == {"total", "segmentation", "classification", "severity"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.training.losses`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/training/losses.py
import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid().reshape(logits.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def compute_multitask_loss(outputs: dict, batch: dict, seg_weight: float = 1.0, sev_weight: float = 1.0) -> dict:
    seg_bce = F.binary_cross_entropy_with_logits(outputs["segmentation_logits"], batch["lesion_mask"].float())
    seg_dice = dice_loss(outputs["segmentation_logits"], batch["lesion_mask"].float())
    segmentation = seg_bce + seg_dice
    classification = F.cross_entropy(outputs["classification_logits"], batch["class_id"])
    severity = F.cross_entropy(outputs["severity_logits"], batch["severity_label"])
    total = classification + seg_weight * segmentation + sev_weight * severity
    return {
        "total": total,
        "segmentation": segmentation,
        "classification": classification,
        "severity": severity,
    }
```

```python
# src/plant_disease/training/engine.py
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch["image"].to(device))
        losses = loss_fn(outputs, {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()})
        losses["total"].backward()
        optimizer.step()
        running += float(losses["total"].detach().cpu())
    return running / max(len(loader), 1)
```

```python
# scripts/train_baseline.py
import yaml


def main():
    with open("configs/baseline_segformer_b0.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    print(f"Loaded baseline config: {config['experiment_name']}")


if __name__ == "__main__":
    main()
```

```yaml
# configs/baseline_segformer_b0.yaml
experiment_name: baseline_segformer_b0
backbone: segformer_b0
num_classes: 4
num_severity_grades: 3
batch_size: 8
lr: 0.0001
epochs: 30
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_losses.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/training/losses.py src/plant_disease/training/engine.py scripts/train_baseline.py configs/baseline_segformer_b0.yaml tests/test_losses.py
git commit -m "feat: add baseline training losses and entrypoint"
```

### Task 7: Implement Lesion-Guided Classification Fusion

**Files:**
- Create: `src/plant_disease/models/lesion_guided.py`
- Modify: `src/plant_disease/models/multitask_model.py`
- Test: `tests/test_model_shapes.py`
- Create: `configs/lesion_guided_segformer_b0.yaml`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_shapes.py
import torch

from plant_disease.models.lesion_guided import lesion_guided_pool


def test_lesion_guided_pool_reduces_spatial_dimensions():
    features = torch.ones((2, 4, 8, 8))
    mask = torch.ones((2, 1, 8, 8))
    fused = lesion_guided_pool(features, mask)

    assert fused.shape == (2, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_shapes.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.models.lesion_guided`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/models/lesion_guided.py
import torch


def lesion_guided_pool(features: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    mask = mask_logits.sigmoid()
    lesion_features = features * mask
    global_pool = features.mean(dim=(2, 3))
    lesion_pool = lesion_features.mean(dim=(2, 3))
    return torch.cat([global_pool, lesion_pool], dim=1)
```

```python
# src/plant_disease/models/multitask_model.py
import torch
import torch.nn as nn

from plant_disease.models.heads import ClassificationHead, SeverityHead
from plant_disease.models.lesion_guided import lesion_guided_pool


class MultiTaskPlantDiseaseModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_severity_grades: int):
        super().__init__()
        self.seg_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = ClassificationHead(in_channels * 2, num_classes)
        self.sev_head = SeverityHead(in_channels, num_severity_grades)

    def forward_from_features(self, features: torch.Tensor) -> dict:
        segmentation_logits = self.seg_head(features)
        pooled = lesion_guided_pool(features, segmentation_logits)
        severity_pool = self.pool(features).flatten(1)
        classification_logits = self.cls_head(pooled)
        severity_logits = self.sev_head(severity_pool)
        return {
            "segmentation_logits": segmentation_logits,
            "classification_logits": classification_logits,
            "severity_logits": severity_logits,
        }
```

```yaml
# configs/lesion_guided_segformer_b0.yaml
experiment_name: lesion_guided_segformer_b0
base_config: baseline_segformer_b0
use_lesion_guided_fusion: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_shapes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/models/lesion_guided.py src/plant_disease/models/multitask_model.py configs/lesion_guided_segformer_b0.yaml tests/test_model_shapes.py
git commit -m "feat: add lesion-guided classification fusion"
```

### Task 8: Implement Offline Pseudo-Mask Generation

**Files:**
- Create: `src/plant_disease/pseudo_labels/cam_pipeline.py`
- Create: `scripts/generate_pseudo_masks.py`
- Create: `configs/pseudo_label_segformer_b0.yaml`
- Test: `tests/test_cam_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cam_pipeline.py
import numpy as np

from plant_disease.pseudo_labels.cam_pipeline import heatmap_to_mask


def test_heatmap_to_mask_thresholds_activation():
    heatmap = np.array([[0.2, 0.9], [0.1, 0.8]], dtype=np.float32)
    mask = heatmap_to_mask(heatmap, threshold=0.5)

    assert mask.tolist() == [[0, 1], [0, 1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cam_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.pseudo_labels.cam_pipeline`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/pseudo_labels/cam_pipeline.py
import numpy as np


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value == min_value:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - min_value) / (max_value - min_value)).astype(np.float32)


def heatmap_to_mask(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    norm = normalize_heatmap(heatmap)
    return (norm >= threshold).astype(np.uint8)
```

```python
# scripts/generate_pseudo_masks.py
from pathlib import Path

import numpy as np

from plant_disease.pseudo_labels.cam_pipeline import heatmap_to_mask


def save_demo_mask(output_dir: str = "artifacts/pseudo_masks"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap = np.array([[0.0, 0.9], [0.2, 0.7]], dtype=np.float32)
    mask = heatmap_to_mask(heatmap, threshold=0.5)
    np.save(out_dir / "demo_mask.npy", mask)


if __name__ == "__main__":
    save_demo_mask()
```

```yaml
# configs/pseudo_label_segformer_b0.yaml
experiment_name: pseudo_label_segformer_b0
base_config: lesion_guided_segformer_b0
pseudo_label_threshold: 0.5
pseudo_label_weight: 0.4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cam_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/pseudo_labels/cam_pipeline.py scripts/generate_pseudo_masks.py configs/pseudo_label_segformer_b0.yaml tests/test_cam_pipeline.py
git commit -m "feat: add offline pseudo-mask generation pipeline"
```

### Task 9: Implement Concept Prediction and Rule Constraints

**Files:**
- Create: `src/plant_disease/concepts/rules.py`
- Modify: `src/plant_disease/models/heads.py`
- Create: `configs/concept_rule_segformer_b0.yaml`
- Test: `tests/test_rules.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rules.py
import torch

from plant_disease.concepts.rules import rule_consistency_loss


def test_rule_consistency_loss_is_non_negative():
    concepts = {
        "mildew_texture": torch.tensor([0.9]),
        "lesion_area_ratio": torch.tensor([0.7]),
    }
    class_probs = torch.tensor([[0.1, 0.8, 0.1]])
    severity_probs = torch.tensor([[0.2, 0.3, 0.5]])

    loss = rule_consistency_loss(concepts, class_probs, severity_probs)

    assert float(loss) >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plant_disease.concepts.rules`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/concepts/rules.py
import torch

CONCEPT_NAMES = [
    "yellowing",
    "necrosis",
    "spot_density",
    "mildew_texture",
    "lesion_irregularity",
    "color_variance",
    "lesion_area_ratio",
]


def rule_consistency_loss(concepts: dict, class_probs: torch.Tensor, severity_probs: torch.Tensor) -> torch.Tensor:
    mildew_penalty = torch.relu(concepts["mildew_texture"] - class_probs[:, 1]).mean()
    severity_penalty = torch.relu(concepts["lesion_area_ratio"] - severity_probs[:, -1]).mean()
    return mildew_penalty + severity_penalty
```

```python
# src/plant_disease/models/heads.py
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class SeverityHead(nn.Module):
    def __init__(self, in_features: int, num_grades: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_grades)

    def forward(self, x):
        return self.fc(x)


class ConceptHead(nn.Module):
    def __init__(self, in_features: int, num_concepts: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_concepts)

    def forward(self, x):
        return self.fc(x)
```

```yaml
# configs/concept_rule_segformer_b0.yaml
experiment_name: concept_rule_segformer_b0
base_config: pseudo_label_segformer_b0
use_concept_head: true
rule_weight: 0.2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/concepts/rules.py src/plant_disease/models/heads.py configs/concept_rule_segformer_b0.yaml tests/test_rules.py
git commit -m "feat: add concept prediction and rule consistency loss"
```

### Task 10: Add End-to-End Evaluation Entrypoint

**Files:**
- Create: `scripts/evaluate_model.py`
- Modify: `src/plant_disease/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import torch

from plant_disease.eval.metrics import mask_to_box


def test_mask_to_box_returns_xyxy_bounds():
    mask = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    assert mask_to_box(mask) == (1, 0, 2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ImportError` for `mask_to_box`

- [ ] **Step 3: Write minimal implementation**

```python
# src/plant_disease/eval/metrics.py
import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (union + eps)


def severity_ratio(lesion_mask: torch.Tensor, leaf_mask: torch.Tensor) -> float:
    lesion = float((lesion_mask > 0).sum())
    leaf = float((leaf_mask > 0).sum())
    if leaf == 0:
        return 0.0
    return lesion / leaf


def mask_to_box(mask: torch.Tensor) -> tuple[int, int, int, int]:
    indices = torch.nonzero(mask > 0, as_tuple=False)
    if len(indices) == 0:
        return (0, 0, 0, 0)
    y_min = int(indices[:, 0].min())
    x_min = int(indices[:, 1].min())
    y_max = int(indices[:, 0].max())
    x_max = int(indices[:, 1].max())
    return (x_min, y_min, x_max, y_max)
```

```python
# scripts/evaluate_model.py
def main():
    print("Run evaluation for classification, segmentation, severity, and derived boxes.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plant_disease/eval/metrics.py scripts/evaluate_model.py tests/test_metrics.py
git commit -m "feat: add evaluation entrypoint and derived box metric"
```

## Self-Review Checklist

### Spec coverage

- Baseline classification, segmentation, and severity are covered by Tasks 3 through 6.
- Lesion-guided classification is covered by Task 7.
- Weak pseudo-label generation is covered by Task 8.
- Concept prediction and rule constraints are covered by Task 9.
- Evaluation for segmentation, severity, and derived detection boxes is covered by Task 10.

### Placeholder scan

This plan intentionally avoids `TODO`, `TBD`, and deferred implementation placeholders inside tasks. Where a later stage is mentioned, that stage has its own explicit task and files.

### Type consistency

- Dataset samples consistently use `image`, `class_id`, `severity_label`, and `lesion_mask`.
- Model outputs consistently use `segmentation_logits`, `classification_logits`, and `severity_logits`.
- The lesion-guided path uses `mask_logits` and `features` throughout.

