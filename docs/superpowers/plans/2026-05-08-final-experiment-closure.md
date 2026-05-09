# Final Experiment Closure

Date: 2026-05-08

## Final Scope Decision

At the current stage, the project should be considered **methodologically closed** on the core four-innovation storyline. The recommended final scope is:

1. Keep the aligned `ResNet-18` experiment line as the main evidence chain.
2. Use the lightweight concept/rule model built on top of `w0.3` as the current main model.
3. Keep `SegFormer-B0` as a backbone transferability validation stage, not as the final paper backbone.

This means the project is no longer in the "prove the idea exists" stage. It has entered the "freeze the main narrative and refine only if needed" stage.

## Final Main Model

Recommended main model:

- `KGL-MTN (ResNet-18, w0.3, lightweight concept/rule, severity enabled)`

This model currently provides the best overall balance across classification, segmentation, severity estimation, and semantic regularization.

Metrics on the aligned PlantSeg test split:

- Classification Accuracy: `0.3703`
- Classification Macro-F1: `0.3153`
- Segmentation Dice: `0.3705`
- Segmentation mIoU: `0.2565`
- Severity Accuracy: `0.4459`
- Concept Binary Accuracy: `0.2405`
- Concept MAE: `0.3071`
- Rule Consistency Loss: `0.0828`

## Final Comparison Set

The cleanest comparison set to retain is:

1. `Aligned Baseline (ResNet-18)`
2. `Aligned Mixed w0.1 (ResNet-18)`
3. `Aligned Mixed w0.3 (ResNet-18)`
4. `Aligned Concept/Rule tuned on w0.3 (ResNet-18)`

Recommended interpretation:

- `Aligned Baseline`: strong supervised reference
- `Aligned Mixed w0.1`: segmentation-priority weak supervision variant
- `Aligned Mixed w0.3`: strongest pure weak-supervision baseline
- `Aligned Concept/Rule`: final semantic-regularized model

Optional supporting row:

- `Full Mixed`

This row can remain in the archive or appendix to show that full pseudo supervision is too aggressive for fine-grained lesion segmentation.

## Final Core Conclusions

### Conclusion 1: Weak supervision is valid after taxonomy cleanup

The aligned subset experiments show that pseudo-label-based weak supervision remains beneficial after cross-dataset taxonomy mismatch is reduced.

### Conclusion 2: Pseudo segmentation weight is a critical control variable

The weak-supervision pipeline is not monotonic:

- heavier pseudo-mask supervision helps classification but can hurt segmentation
- lighter pseudo-mask supervision helps segmentation but may weaken classification

This is why `w0.3` and `w0.1` should both remain in the final comparison matrix.

### Conclusion 3: Lightweight concept/rule is effective, heavy concept/rule is not

The dense concept/rule design harmed the main tasks. The successful version is the lightweight one:

- 4 concepts only
- lesion-centered concept pooling
- concept supervision only on real samples
- weak concept loss
- very weak rule loss
- initialized from the strong `w0.3` model

Therefore, concept/rule should be framed as a **light semantic regularization layer**, not as a heavy concept-fitting objective.

### Conclusion 4: Severity is now a real auxiliary task

Severity estimation is no longer a placeholder branch. The severity labels, training path, and evaluation path are now consistent enough to report severity as a real auxiliary task.

## Innovation Closure Status

### Innovation 1: Weakly supervised pseudo lesion label generation

Status: closed and validated

### Innovation 2: Lesion-guided classification

Status: closed and validated

### Innovation 3: Symptom concept prediction and soft rule constraints

Status: closed at the lightweight-effective level

Important nuance:

- the heavy version failed
- the lightweight version succeeded

This nuance should be preserved in the final write-up because it is one of the most useful experimental findings in the project.

### Innovation 4: Severity estimation as an auxiliary task

Status: closed and evaluable

Important nuance:

- current severity is a lesion-coverage-grade task derived from lesion masks
- it should be described honestly as a practical auxiliary severity estimate, not as expert-annotated agronomic severity

## SegFormer-B0 Closure Decision

SegFormer-B0 integration is technically successful:

- training works
- evaluation works
- checkpoint loading is consistent

However, the current `SegFormer-B0` results do not exceed the aligned `ResNet-18` line.

Observed pattern:

- `SegFormer-B0` baseline is weak
- `SegFormer-B0` weak supervision still helps
- `SegFormer-B0` lightweight concept/rule still improves over its own `w0.3` variant
- but the entire `SegFormer-B0` line remains below the current `ResNet-18` main model

Therefore:

- keep `SegFormer-B0` as a transferability / backbone-validation experiment
- do not promote it to the final main model in the current version

Most likely explanation:

- the current `SegFormer-B0` path is a lightweight encoder replacement
- it uses no strong pretrained segmentation setup
- it does not yet exploit a richer multi-scale decoder or transformer-specific optimization schedule

This means the result is still useful, but should be framed as:

> the proposed method remains portable to a transformer-style backbone, although the current lightweight SegFormer integration does not outperform the optimized ResNet-18 main line.

## Recommended Final Result Table

| Model | Cls Acc | Macro-F1 | Dice | mIoU | Severity Acc | Concept Acc | Concept MAE | Rule Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aligned Baseline | 0.2919 | 0.2235 | 0.3619 | 0.2483 | - | - | - | - |
| Mixed `w0.1` | 0.2649 | 0.2553 | 0.3973 | 0.2770 | - | - | - | - |
| Mixed `w0.3` | 0.2541 | 0.2329 | 0.3358 | 0.2336 | 0.3703 | - | - | - |
| Tuned Concept/Rule | 0.3703 | 0.3153 | 0.3705 | 0.2565 | 0.4459 | 0.2405 | 0.3071 | 0.0828 |

Optional appendix table:

| Model | Cls Acc | Macro-F1 | Dice | mIoU | Severity Acc |
|---|---:|---:|---:|---:|---:|
| Full Mixed | 0.2757 | 0.2742 | 0.2830 | 0.1916 | - |
| SegFormer Baseline | 0.1541 | 0.0404 | 0.0233 | 0.0147 | 0.4297 |
| SegFormer Mixed `w0.3` | 0.1811 | 0.1166 | 0.3185 | 0.2130 | 0.3757 |
| SegFormer Concept/Rule | 0.2405 | 0.2012 | 0.3125 | 0.2104 | 0.4000 |

## What Should Not Be Done Next

Do not immediately:

- expand to many new backbones
- add more training datasets into the main line
- redesign the whole architecture again
- keep heavily tuning dense concept supervision

Those changes would reopen the scope and blur the now-clean conclusion set.

## Recommended Next Step

The project should now move from exploratory implementation into controlled consolidation.

Recommended order:

1. Freeze the current final model and comparison set.
2. Keep only minimal reruns if strict same-pipeline comparability is still needed for a final table.
3. Start converting the current closure into paper-ready method, experiment, and ablation narratives.
