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
python scripts/prepare_plantvillage.py --root <PLANTVILLAGE_CLASS_DIR> --output data/manifests/plantvillage.csv
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
python scripts/prepare_segmentation_manifest.py ^
  --image-dir <IMAGE_DIR> ^
  --mask-dir <MASK_DIR> ^
  --class-name Tomato___Leaf_Mold ^
  --class-id 4 ^
  --source-dataset PlantSeg ^
  --split train ^
  --output data/manifests/plantseg_train.csv
```

## Train Baseline

Update `configs/baseline_segformer_b0.yaml` so that `train_manifest` points to your manifest CSV, then run:

```bash
python scripts/train_baseline.py --config configs/baseline_segformer_b0.yaml
```
