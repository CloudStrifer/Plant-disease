# Aligned Experiment Matrix

Date: 2026-05-08

## Purpose

This note consolidates the aligned-subset experiment entrypoints after taxonomy audit.
All configs below operate on the 14-class aligned subset derived from PlantSeg and PlantVillage pseudo labels.

## Aligned manifests

- `artifacts/aligned_subsets/plantseg_vs_plantvillage_pseudo/left_aligned_manifest.csv`
  Strong-supervision aligned subset from PlantSeg only.
- `artifacts/aligned_subsets/plantseg_vs_plantvillage_pseudo/mixed_aligned_manifest.csv`
  Aligned mixed manifest with full pseudo-mask segmentation supervision.
- `artifacts/aligned_subsets/plantseg_vs_plantvillage_pseudo/mixed_aligned_manifest_w03.csv`
  Same aligned mixed manifest, but pseudo-mask segmentation rows use `seg_loss_weight=0.3`.
- `artifacts/aligned_subsets/plantseg_vs_plantvillage_pseudo/mixed_aligned_manifest_w01.csv`
  Same aligned mixed manifest, but pseudo-mask segmentation rows use `seg_loss_weight=0.1`.
- `artifacts/aligned_subsets/plantseg_vs_plantvillage_pseudo/mixed_aligned_manifest_cls_only.csv`
  Same aligned mixed manifest, but pseudo rows do classification only (`seg_loss_weight=0.0`, `has_lesion_mask=False`).

## Config matrix

- `configs/baseline_resnet18_aligned_gpu.yaml`
  Aligned baseline on PlantSeg-only subset.
- `configs/mixed_supervision_resnet18_aligned_gpu.yaml`
  Aligned mixed supervision with full pseudo-mask segmentation influence.
- `configs/mixed_supervision_resnet18_aligned_w03_gpu.yaml`
  Aligned mixed supervision with moderate pseudo segmentation weight `0.3`.
- `configs/mixed_supervision_resnet18_aligned_w01_gpu.yaml`
  Aligned mixed supervision with lighter pseudo segmentation weight `0.1`.
- `configs/concept_rule_resnet18_aligned_gpu.yaml`
  Aligned concept/rule prototype on the aligned mixed manifest.

## Recommended future execution order

1. `baseline_resnet18_aligned_gpu.yaml`
2. `mixed_supervision_resnet18_aligned_gpu.yaml`
3. `mixed_supervision_resnet18_aligned_w03_gpu.yaml`
4. `mixed_supervision_resnet18_aligned_w01_gpu.yaml`
5. `concept_rule_resnet18_aligned_gpu.yaml`

## Interpretation goal

This aligned matrix is intended to answer three questions cleanly:

1. Does weak supervision still help once taxonomy mismatch is removed?
2. What pseudo-mask segmentation weight is most stable for aligned classes?
3. Does concept/rule regularization recover classification or segmentation quality on the aligned subset?
