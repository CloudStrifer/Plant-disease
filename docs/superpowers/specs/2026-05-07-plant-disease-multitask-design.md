# Plant Disease Multi-Task Research Prototype Design

Date: 2026-05-07

## 1. Goal

Design a research-oriented but implementation-friendly framework for plant disease analysis under realistic field conditions. The system should support four coordinated outputs:

1. Plant disease classification
2. Lesion segmentation
3. Severity estimation
4. Interpretable concept and rule-guided reasoning

The design must balance paper-level completeness with prototype-first execution. The initial implementation should prioritize a strong, stable baseline and then progressively add the remaining innovation modules.

## 2. Problem Framing

The design is motivated by four practical gaps in existing plant disease methods:

1. Pixel-level lesion masks are expensive to obtain at scale.
2. Standard disease classifiers often rely on background or lighting cues instead of lesion regions.
3. Pure neural models are difficult to interpret in agricultural settings.
4. Disease recognition alone is not sufficient for downstream agronomic use; severity also matters.

The proposed framework addresses these gaps through four linked innovation points:

1. Weakly supervised pseudo lesion mask generation
2. Lesion-guided disease classification
3. Symptom concept prediction with soft rule constraints
4. Severity estimation as an auxiliary task

## 3. Recommended System Scope

The recommended project target is a unified multi-task model that predicts:

1. Disease class
2. Lesion mask
3. Bounding box derived from the lesion mask
4. Severity score or severity grade
5. Optional symptom concept scores for interpretability

This is intentionally broader than anomaly detection alone. The proposed system should be positioned as a disease classification and lesion understanding framework rather than a pure anomaly detector.

## 4. Architecture

### 4.1 Recommended variant

The recommended first implementation is:

- Backbone: `SegFormer-B0`
- Segmentation branch: standard segmentation decoder from `MMSegmentation`
- Classification branch: custom lesion-guided classifier head
- Severity branch: custom severity head
- Concept branch: custom concept head, added after the baseline is stable
- Rule module: soft rule consistency loss, added after the concept branch is stable

### 4.2 High-level data flow

```text
Input image
   ->
Shared backbone
   ->
Multi-scale feature fusion
   ->
Lesion segmentation branch -> lesion mask
   ->
Lesion-guided classification branch -> disease class
   ->
Severity branch -> severity score or grade
   ->
Concept branch -> symptom concept scores
   ->
Rule constraint module -> consistency regularization during training
```

### 4.3 Why this architecture

This structure separates the questions the system needs to answer:

- Classification answers what disease is present.
- Segmentation answers where the lesion is.
- Severity estimation answers how serious the infection is.
- Concept prediction and rule constraints answer why the prediction is plausible.

This creates a clear paper narrative and also produces an implementation order that can be staged safely.

## 5. Innovation Modules

### 5.1 Innovation 1: Weakly supervised pseudo lesion label generation

#### Purpose

Use samples with image-level disease labels but without pixel-level lesion labels.

#### Inputs

- Leaf image
- Disease class label
- Optional small subset of real lesion masks

#### Outputs

- Class activation heatmap
- Binary pseudo lesion mask
- Optional refined pseudo lesion mask

#### Recommended implementation

1. Train an image-level disease classifier.
2. Generate lesion-relevant heatmaps using `Grad-CAM`, `Score-CAM`, or similar methods.
3. Convert heatmaps into binary masks using thresholding.
4. Refine masks with simple morphology first.
5. Optionally refine boundaries later using `SAM`.
6. Assign confidence weights to pseudo masks and use them as weak segmentation supervision.

#### Recommended losses

- Classification: `CrossEntropy`
- Weak segmentation: weighted `BCE + Dice`

Pseudo labels should never be treated as equivalent to fully trusted masks.

### 5.2 Innovation 2: Lesion-guided classification

#### Purpose

Improve disease classification by forcing the classifier to focus on lesion regions rather than background or lighting artifacts.

#### Inputs

- Shared feature map from the backbone
- Predicted lesion probability map

#### Outputs

- Lesion-aware classification logits

#### Recommended implementation

Let:

- `F` be shared feature maps
- `M` be the predicted lesion mask

Then compute:

- `F_lesion = F x M`
- `F_global = GAP(F)`
- `F_lesion_pool = GAP(F_lesion)`
- `F_final = concat(F_global, F_lesion_pool)`

Then apply a fully connected classification head to `F_final`.

The first implementation should use simple concatenation or weighted addition. More complex fusion mechanisms such as cross-attention should be postponed until a stable baseline exists.

#### Recommended losses

- `L_cls = CrossEntropy`
- `L_seg = BCE + Dice`
- Joint objective: `L = L_cls + lambda_1 * L_seg`

This module is the primary innovation and should be emphasized in both implementation and ablation.

### 5.3 Innovation 3: Symptom concept prediction and soft rule constraints

#### Purpose

Provide interpretable intermediate evidence and encourage class predictions to remain consistent with disease-related visual concepts.

#### Recommended concept set

The initial concept vocabulary should stay compact:

- `yellowing`
- `necrosis`
- `spot_density`
- `mildew_texture`
- `lesion_irregularity`
- `color_variance`
- `lesion_area_ratio`

#### Inputs

- Lesion-aware feature representation
- Optional concept annotations

#### Outputs

- Concept score vector
- Rule consistency regularization during training

#### Recommended implementation

1. Attach a small multilayer concept head to lesion-aware features.
2. Predict concept probabilities or bounded scores.
3. Encode disease-specific soft rules as differentiable penalties.

Examples of soft rules:

- High mildew-like texture should increase powdery mildew probability.
- Large lesion area ratio should correlate with higher severity.
- High yellowing with irregular necrotic regions should raise some leaf spot or blight probabilities depending on the dataset taxonomy.

#### Recommended losses

- `L_concept`: `BCE` or `MSE`, depending on concept label format
- `L_rule`: custom differentiable consistency penalty

Combined objective at this stage:

`L = L_cls + lambda_1 * L_seg + lambda_2 * L_sev + lambda_3 * L_concept + lambda_4 * L_rule`

This module should be introduced only after the baseline and pseudo-label stages are stable.

### 5.4 Innovation 4: Severity estimation as an auxiliary task

#### Purpose

Align the method with agricultural decision support rather than limiting it to disease naming alone.

#### Inputs

- Lesion mask
- Optional leaf mask
- Lesion-aware features

#### Outputs

- Continuous severity ratio or severity grade

#### Recommended implementation

Preferred definition:

`severity = lesion_area / leaf_area`

If leaf masks are available, compute targets directly. If leaf masks are unavailable, either:

1. Introduce a leaf segmentation branch later, or
2. Use a separate preprocessing tool to estimate leaf region

The first prototype should use severity grading rather than fine-grained regression because class-based supervision is easier to stabilize.

#### Recommended losses

- Regression option: `MSE` or `SmoothL1`
- Classification option: `CrossEntropy`

The recommended first prototype uses graded severity classification.

## 6. Data Design

### 6.1 Supervision tiers

Organize samples by supervision availability:

- Tier A: class label + lesion mask
- Tier B: class label only
- Tier C: auxiliary or external evaluation data

### 6.2 Recommended metadata

Maintain a sample index table with:

- image path
- class id
- lesion mask availability
- leaf mask availability
- source dataset
- severity label

This makes mixed-supervision training easier to control and audit.

## 7. Training Strategy

### Stage 1: Strong supervised baseline

Use only Tier A data.

Train:

- backbone
- segmentation head
- classification head
- severity head

Target outputs:

- disease class
- lesion mask
- severity grade

This stage establishes the reference baseline.

### Stage 2: Add lesion-guided classification

Keep Tier A data as the main source.

Add lesion-guided feature fusion and compare:

- classification only
- standard multi-task classification + segmentation
- lesion-guided classification + segmentation

This stage tests the main innovation.

### Stage 3: Add weak pseudo masks

Use Tier A and Tier B together.

Workflow:

1. Train or reuse the disease classifier.
2. Generate pseudo lesion masks offline.
3. Filter or weight pseudo masks by confidence.
4. Mix real and pseudo supervision for segmentation training.

Pseudo-label generation should stay offline in the first version to reduce instability.

### Stage 4: Add concept and rule modules

Add:

- concept head
- soft rule loss

This stage validates interpretability and consistency improvements.

## 8. Evaluation Plan

### 8.1 Classification metrics

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1

### 8.2 Segmentation metrics

- mIoU
- Dice
- Pixel Accuracy
- Boundary F1

### 8.3 Detection metrics

If lesion boxes are derived from masks, report:

- mAP@0.5
- mAP@0.5:0.95
- Recall

### 8.4 Severity metrics

If severity is treated as classification:

- Accuracy
- Macro-F1

If severity is treated as regression:

- MAE
- RMSE

### 8.5 Interpretability evaluation

Recommended outputs:

- concept prediction accuracy or F1 where labels exist
- qualitative examples of concept activation
- qualitative examples of rule-consistent and rule-inconsistent predictions

## 9. Comparison and Ablation Plan

### 9.1 Baseline comparisons

Classification:

- ResNet50
- ConvNeXt-T
- EfficientNet
- Swin-T

Segmentation:

- U-Net
- DeepLabV3+
- SegFormer
- optional Mask2Former variant if resources allow

### 9.2 Model ablations

Required ablations:

1. Backbone choice
2. Lesion-guided classification on or off
3. Pseudo-label strategy variants
4. Concept head on or off
5. Rule loss on or off
6. Severity branch variants

### 9.3 Robustness analysis

Evaluate under:

- shadow variation
- blur
- occlusion
- color shift
- cluttered background

This is important for connecting the work back to real field conditions.

## 10. Error Handling and Risk Control

### 10.1 Main technical risks

1. Pseudo masks may be noisy.
2. Multi-task losses may compete with each other.
3. Concept labels may be sparse or partially subjective.
4. Severity labels may not be uniformly defined across datasets.

### 10.2 Mitigation

1. Use confidence-weighted pseudo supervision.
2. Add modules in stages rather than all at once.
3. Keep the concept set small and concrete in the first version.
4. Prefer severity grading before regression.
5. Maintain one stable baseline configuration for all later comparisons.

## 11. Recommended Software Stack

- `PyTorch`
- `MMSegmentation`
- `mmengine`
- `timm`
- `pytorch-grad-cam`
- `segment-anything`
- `opencv-python`
- `albumentations`
- `scikit-learn`

Optional later addition:

- `open_clip`

## 12. Recommended Implementation Roadmap

### Milestone M1

Build and verify the strong baseline:

- dataset reader
- segmentation
- classification
- severity

### Milestone M2

Add lesion-guided classification and run its ablation.

### Milestone M3

Add pseudo-label generation and mixed-supervision training.

### Milestone M4

Add concept prediction and rule-constrained learning.

This roadmap is the recommended execution path because it preserves both implementation stability and paper-quality narrative.

## 13. Success Criteria

The project is considered successful if it satisfies the following:

1. A stable baseline can jointly predict disease class, lesion mask, and severity.
2. Lesion-guided classification improves disease recognition over ordinary multi-task learning.
3. Pseudo-label augmentation improves or preserves segmentation quality when mask supervision is limited.
4. Concept and rule modules improve interpretability without destabilizing the main tasks.
5. The final method remains coherent enough to support a publishable agricultural AI paper narrative.

## 14. Recommendation

Proceed with the balanced strategy:

- design at full paper scope
- implement in prototype-first stages

The first code iteration should target:

- `SegFormer-B0`
- lesion segmentation
- lesion-guided classification
- severity grading

Pseudo-labeling and concept-rule reasoning should be added only after this baseline is running and measured.
