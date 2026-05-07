# Plant Disease Multi-Task Framework

Prototype-first implementation of lesion segmentation, lesion-guided classification, severity estimation, pseudo-label generation, and concept-rule reasoning.

## Recommended Datasets

### 1. PlantSeg

Use as the primary segmentation supervision dataset. It is the best fit for lesion mask training and lesion-guided classification experiments.

### 2. PlantVillage

Use as a classification-heavy auxiliary dataset for weak supervision and pretraining. It is clean and large, but not representative of real field backgrounds.

### 3. PlantDoc

Use later as an external generalization benchmark for field robustness rather than as the main training source.

## Dataset Preparation

### PlantVillage manifest

```bash
python scripts/prepare_plantvillage.py --root "E:\Multi_modal_Code\Plant disease\src\plant_disease\dataset\plantvillage\color" --output data/manifests/plantvillage.csv
```

Expected input layout:

```text
<PLANTVILLAGE_CLASS_DIR>/
  Tomato___healthy/
  Tomato___Early_blight/
  ...
```

### Segmentation manifest for PlantSeg-style paired folders

```bash
python scripts/prepare_plantseg.py --root "E:\Multi_modal_Code\Plant disease\src\plant_disease\dataset\plantseg" --output data/manifests/plantseg.csv
```

This script reads:

- `images/train`, `images/val`, `images/test`
- `annotations/train`, `annotations/val`, `annotations/test`
- `Metadatav2.csv`

and generates one unified manifest with `split` values already assigned.

## Recommended Training Order

1. Train the first baseline on `PlantSeg` only
2. Use `PlantVillage` later for classification pretraining or weak supervision
3. Use `PlantDoc` later as an external robustness benchmark

## Train Baseline

Update `configs/baseline_segformer_b0.yaml` so that `train_manifest` points to your manifest CSV, then run:

```bash
python scripts/train_baseline.py --config configs/baseline_segformer_b0.yaml
```
