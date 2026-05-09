# SegFormer-B0 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `SegFormer-B0` as a second backbone into the current multi-task framework so the aligned baseline, `w0.3` mixed supervision, and lightweight concept/rule variants can be reproduced on a stronger segmentation-oriented encoder.

**Architecture:** Keep the current multi-task heads and training/evaluation pipeline intact, and replace only the encoder path first. The first SegFormer version should expose a single unified high-level feature map to the existing segmentation, lesion-guided classification, severity, and concept heads, so the backbone can be swapped without rewriting the rest of the system.

**Tech Stack:** Python, PyTorch, torchvision, timm, pytest, existing train/eval scripts and aligned manifests.

---

## File Structure

- Modify: `E:\Multi_modal_Code\Plant disease\src\plant_disease\models\multitask_model.py`
  - Add `segformer_b0` encoder branch and a unified feature projection path.
- Modify: `E:\Multi_modal_Code\Plant disease\scripts\train_baseline.py`
  - Allow `in_channels` to be inferred from the selected backbone or overridden explicitly.
- Modify: `E:\Multi_modal_Code\Plant disease\scripts\evaluate_model.py`
  - Mirror the same backbone and channel configuration path as training.
- Modify: `E:\Multi_modal_Code\Plant disease\tests\test_model_shapes.py`
  - Add SegFormer shape tests for forward and optional concept head outputs.
- Create: `E:\Multi_modal_Code\Plant disease\configs\baseline_segformer_b0_aligned_gpu.yaml`
  - Strong-supervised aligned baseline using `SegFormer-B0`.
- Create: `E:\Multi_modal_Code\Plant disease\configs\mixed_supervision_segformer_b0_aligned_w03_gpu.yaml`
  - Main weakly supervised aligned `w0.3` experiment using `SegFormer-B0`.
- Create: `E:\Multi_modal_Code\Plant disease\configs\concept_rule_segformer_b0_aligned_gpu.yaml`
  - Lightweight concept/rule experiment initialized from the SegFormer `w0.3` checkpoint.
- Modify: `E:\Multi_modal_Code\Plant disease\docs\superpowers\plans\2026-05-08-paper-results-summary.md`
  - Add a short note that SegFormer is the planned next-stage backbone validation.

## Task 1: Add Failing SegFormer Backbone Tests

**Files:**
- Modify: `E:\Multi_modal_Code\Plant disease\tests\test_model_shapes.py`

- [ ] **Step 1: Write the failing tests**

Add tests that instantiate `MultiTaskPlantDiseaseModel` with `backbone_name="segformer_b0"` and verify:

- `forward(images)` returns:
  - `segmentation_logits` with input spatial size
  - `classification_logits` with `num_classes`
  - `severity_logits` with `num_severity_grades`
- concept mode returns `concept_logits` with `num_concepts=4`

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests\test_model_shapes.py -q
```

Expected:

- FAIL because `segformer_b0` is not yet implemented in `multitask_model.py`

- [ ] **Step 3: Commit test-only red state**

```bash
git add tests/test_model_shapes.py
git commit -m "test: add failing segformer backbone shape coverage"
```

## Task 2: Implement SegFormer-B0 Encoder Path

**Files:**
- Modify: `E:\Multi_modal_Code\Plant disease\src\plant_disease\models\multitask_model.py`

- [ ] **Step 1: Implement the minimal backbone branch**

Use `timm.create_model("segformer_b0", features_only=True, pretrained=False)` or equivalent supported API. Keep the first version simple:

- collect the highest-level feature map
- project it to a single fixed channel count
- reuse existing heads and lesion-guided pooling unchanged

Recommended behavior:

- `resnet18 -> in_channels=512`
- `segformer_b0 -> in_channels=256`

- [ ] **Step 2: Run the SegFormer tests**

Run:

```bash
python -m pytest tests\test_model_shapes.py -q
```

Expected:

- PASS for the new SegFormer tests

- [ ] **Step 3: Run a broader safety check**

Run:

```bash
python -m pytest tests\test_model_shapes.py tests\test_losses.py tests\test_engine.py -q
```

Expected:

- PASS with no shape regressions in the current ResNet path

- [ ] **Step 4: Commit**

```bash
git add src/plant_disease/models/multitask_model.py tests/test_model_shapes.py
git commit -m "feat: add segformer-b0 backbone support"
```

## Task 3: Make Training and Evaluation Backbone-Aware

**Files:**
- Modify: `E:\Multi_modal_Code\Plant disease\scripts\train_baseline.py`
- Modify: `E:\Multi_modal_Code\Plant disease\scripts\evaluate_model.py`

- [ ] **Step 1: Add a small failing expectation**

Add or extend tests so a config using `backbone_name=segformer_b0` can build the model in both train and eval paths without requiring manual code edits.

- [ ] **Step 2: Run the target tests to see the current failure**

Run:

```bash
python -m pytest tests\test_train_paths.py tests\test_eval_reporting.py -q
```

Expected:

- FAIL or missing behavior for SegFormer-specific model creation

- [ ] **Step 3: Implement the minimal config-driven channel resolution**

Add shared logic so:

- `backbone_name=resnet18` uses current defaults
- `backbone_name=segformer_b0` uses the SegFormer channel count
- optional `in_channels` in config can override defaults

- [ ] **Step 4: Run verification**

Run:

```bash
python -m pytest tests\test_train_paths.py tests\test_eval_reporting.py tests\test_model_shapes.py -q
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_baseline.py scripts/evaluate_model.py
git commit -m "feat: make train and eval backbone-aware"
```

## Task 4: Add SegFormer Aligned Experiment Configs

**Files:**
- Create: `E:\Multi_modal_Code\Plant disease\configs\baseline_segformer_b0_aligned_gpu.yaml`
- Create: `E:\Multi_modal_Code\Plant disease\configs\mixed_supervision_segformer_b0_aligned_w03_gpu.yaml`
- Create: `E:\Multi_modal_Code\Plant disease\configs\concept_rule_segformer_b0_aligned_gpu.yaml`

- [ ] **Step 1: Write the config files**

Baseline config should point to:

- `train_manifest: artifacts/aligned_subsets/plantseg_vs_plantvillage_pseudo/left_aligned_manifest.csv`

Mixed config should point to:

- `train_manifest: artifacts/aligned_subsets/plantseg_vs_plantvillage_pseudo/mixed_aligned_manifest_w03.csv`

Concept/rule config should:

- use the same aligned `w0.3` manifest
- initialize from the SegFormer `w0.3` checkpoint
- keep the current lightweight settings:
  - `concept_weight: 0.05`
  - `rule_weight: 0.005`
  - `concept_real_only: true`
  - `num_concepts: 4`

- [ ] **Step 2: Validate the YAML loads**

Run:

```bash
python scripts/train_baseline.py --config configs/baseline_segformer_b0_aligned_gpu.yaml
```

Expected:

- startup logs print the config and model settings without immediate config/key errors

Use `Ctrl+C` after verifying the run enters training if you only need a smoke startup check at this stage.

- [ ] **Step 3: Commit**

```bash
git add configs/baseline_segformer_b0_aligned_gpu.yaml configs/mixed_supervision_segformer_b0_aligned_w03_gpu.yaml configs/concept_rule_segformer_b0_aligned_gpu.yaml
git commit -m "feat: add aligned segformer experiment configs"
```

## Task 5: Run a Minimal Smoke Validation

**Files:**
- No code changes required unless smoke checks fail

- [ ] **Step 1: Smoke test baseline startup**

Run:

```bash
python scripts/train_baseline.py --config configs/baseline_segformer_b0_aligned_gpu.yaml
```

Expected:

- model builds
- first epoch starts
- no backbone-related shape or device errors

- [ ] **Step 2: Smoke test mixed startup**

Run:

```bash
python scripts/train_baseline.py --config configs/mixed_supervision_segformer_b0_aligned_w03_gpu.yaml
```

Expected:

- checkpoint loads with either zero skips or only clearly justified head skips
- first epoch starts

- [ ] **Step 3: Smoke test concept/rule startup**

Run:

```bash
python scripts/train_baseline.py --config configs/concept_rule_segformer_b0_aligned_gpu.yaml
```

Expected:

- concept head builds
- first epoch starts
- no concept dimension mismatch

- [ ] **Step 4: If any smoke test fails, fix only the smallest necessary issue and rerun**

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "fix: stabilize segformer aligned experiment startup"
```

## Task 6: Document the New Comparison Stage

**Files:**
- Modify: `E:\Multi_modal_Code\Plant disease\docs\superpowers\plans\2026-05-08-paper-results-summary.md`

- [ ] **Step 1: Add a short section**

Document that the next validation stage is:

- reproduce `aligned baseline`
- reproduce `aligned mixed w0.3`
- reproduce tuned `concept/rule`

all on `SegFormer-B0`, to test whether the main conclusions are backbone-stable.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-05-08-paper-results-summary.md
git commit -m "docs: add segformer validation stage to paper summary"
```

## Self-Review

Spec coverage:

- Supports the original design recommendation that `SegFormer-B0` should become the stronger paper-oriented backbone.
- Preserves the current aligned `w0.3` and lightweight concept/rule story instead of redesigning the whole system.
- Keeps the first SegFormer version narrow in scope, avoiding an unnecessary decoder refactor.

Placeholder scan:

- No `TODO`, `TBD`, or “implement later” markers remain.
- Each task names exact files and exact verification commands.

Type consistency:

- `backbone_name=segformer_b0`
- `num_concepts=4`
- `concept_real_only=true`
- aligned `w0.3` remains the initialization base for concept/rule

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-segformer-b0-integration.md`.

Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
