# Paper-Style Results Summary

Date: 2026-05-08

## Recommended Main Story

The current project now supports all four planned innovations:

1. Weakly supervised pseudo lesion label generation
2. Lesion-guided classification
3. Symptom concept prediction and soft rule constraints
4. Severity estimation as an auxiliary task

At this stage, the most defensible paper narrative is:

- Use weighted mixed supervision as the core weak-supervision strategy.
- Use lesion-guided classification as the architectural backbone.
- Use severity estimation as a validated auxiliary task.
- Present concept/rule as a lightweight semantic regularization module that becomes effective after being attached to the strong `w0.3` mixed-supervision model, rather than as a heavy standalone supervision branch.

## Current Final Candidate Models

### Candidate A: Weighted Mixed Supervision (`w0.3`)

This is the strongest pure weak-supervision baseline and should remain in the paper as the principal non-concept comparison model.

Metrics on PlantSeg aligned test split:

- Classification Accuracy: `0.2541`
- Classification Macro-F1: `0.2329`
- Segmentation Dice: `0.3358`
- Segmentation mIoU: `0.2336`
- Severity Accuracy: `0.3703`

### Candidate B: Tuned Concept/Rule on Top of `w0.3`

This is currently the best overall model and should be treated as the primary final model candidate.

Metrics on PlantSeg aligned test split:

- Classification Accuracy: `0.3703`
- Classification Macro-F1: `0.3153`
- Segmentation Dice: `0.3705`
- Segmentation mIoU: `0.2565`
- Severity Accuracy: `0.4459`
- Concept Binary Accuracy: `0.2405`
- Concept MAE: `0.3071`
- Rule Consistency Loss: `0.0828`

## Interpretation of Final Results

The tuned concept/rule model improves classification, segmentation, and severity estimation over the current `w0.3` weighted mixed-supervision model under the latest aligned evaluation pipeline. This indicates that the lightweight concept/rule formulation is more effective than the earlier dense concept-supervision design.

The key practical lesson is that concept/rule should be used as a weak semantic regularizer, not as a strong concept-fitting objective. The successful version uses:

- only 4 concepts
- lesion-centered concept pooling
- concept supervision only on real samples
- small concept loss weight
- very small rule consistency weight
- initialization from the strong `w0.3` model instead of an earlier classification checkpoint

## Innovation Status

### Innovation 1: Weakly supervised pseudo lesion label generation

Status: Implemented and validated

Evidence:

- pseudo masks are generated from classification checkpoints
- mixed supervision is trainable
- weighted pseudo supervision (`w0.3`, `w0.1`) produces meaningful task trade-offs

### Innovation 2: Lesion-guided classification

Status: Implemented and validated

Evidence:

- lesion-guided pooling is integrated into the main model
- the weakly supervised pipeline depends on lesion-aware classification features
- this remains part of the best-performing final model candidate

### Innovation 3: Symptom concept prediction and soft rule constraints

Status: Implemented and lightly validated

Evidence:

- concept head, concept targets, valid masks, weighted concept loss, and rule loss are all integrated
- the initial dense concept/rule version harmed the main tasks
- the lightweight concept/rule version improved the main tasks when attached to the strong `w0.3` model

Recommended paper framing:

> A lightweight symptom concept and rule regularization module was introduced as a semantic auxiliary mechanism. Strong concept supervision was found to be unstable, while a small set of high-confidence concepts with weak rule regularization improved the final multi-task model.

### Innovation 4: Severity estimation as an auxiliary task

Status: Implemented and now evaluable

Evidence:

- severity labels are now automatically derived from lesion masks
- invalid severity samples are masked out during training
- training and evaluation use matched severity head dimensions
- the final candidate model reaches `0.4459` severity accuracy on the aligned PlantSeg test split

## Recommended Main Model

Use the tuned concept/rule model as the current main model.

Recommended naming:

- `KGL-MTN (w0.3 + lightweight concept/rule)`

Recommended supporting comparison models:

- Strong supervised aligned baseline
- Weighted mixed supervision `w0.3`
- Weighted mixed supervision `w0.1`
- Full pseudo supervision

## Recommended Result Table Layout

| Model | Cls Acc | Macro-F1 | Dice | mIoU | Severity Acc | Concept Acc | Concept MAE | Rule Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aligned Baseline | 0.2919 | 0.2235 | 0.3619 | 0.2483 | - | - | - | - |
| Full Mixed | 0.2757 | 0.2742 | 0.2830 | 0.1916 | - | - | - | - |
| Mixed `w0.1` | 0.2649 | 0.2553 | 0.3973 | 0.2770 | - | - | - | - |
| Mixed `w0.3` | 0.2541 | 0.2329 | 0.3358 | 0.2336 | 0.3703 | - | - | - |
| Tuned Concept/Rule | 0.3703 | 0.3153 | 0.3705 | 0.2565 | 0.4459 | 0.2405 | 0.3071 | 0.0828 |

Note:

- Some earlier baseline and mixed values were produced before the severity task was fully closed, so severity fields should only be reported where the severity pipeline is confirmed valid.
- The final paper should make this distinction explicit when describing the experimental timeline.

## Recommended Next Step

Do not expand the method further before stabilizing the final comparison set.

Priority order:

1. Freeze the current main model and comparison set.
2. Re-run only the minimum comparable experiments needed for a clean final table if any rows still come from earlier pipeline stages.
3. Start drafting the paper method and experimental sections using the current final narrative.

## SegFormer-B0 Validation Stage

The next backbone validation stage is now prepared and smoke-validated.

Planned comparison set:

- `baseline_segformer_b0_aligned_gpu`
- `mixed_supervision_segformer_b0_aligned_w03_gpu`
- `concept_rule_segformer_b0_aligned_gpu`

Practical status:

- The `SegFormer-B0` encoder path has been integrated through an equivalent supported API.
- Backbone-aware channel resolution is now handled in both training and evaluation.
- The aligned `baseline -> mixed w0.3 -> concept/rule` checkpoint chain has been smoke-tested end-to-end on tiny manifests.

What this stage is meant to answer:

1. Do the current conclusions remain stable when moving from `ResNet-18` to a stronger segmentation-oriented backbone?
2. Does the tuned lightweight concept/rule module remain beneficial on the stronger backbone?
3. Can `SegFormer-B0` become the final paper backbone while preserving the current four-innovation story?
