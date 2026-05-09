"""Compare native multitask outputs against inference-time mmseg mask replacement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.concepts.rules import CONCEPT_NAMES, build_rule_metadata, rule_consistency_loss
from plant_disease.eval.reporting import (
    binary_segmentation_stats,
    macro_f1_score,
    mean_absolute_error,
    multilabel_binary_accuracy,
    safe_mean,
)
from plant_disease_mmseg.predictions import build_prediction_index, load_mmseg_predictions
from plant_disease_mmseg.replacement import forward_from_features_with_external_masks, load_external_mask_batch
from scripts.evaluate_model import (
    ManifestEvalDataset,
    _build_model,
    _find_column,
    _infer_classification_logits,
    _infer_concept_logits,
    _infer_segmentation_logits,
    _infer_severity_logits,
    _load_checkpoint,
    _resolve_repo_path,
)


def _select_rows(dataframe: pd.DataFrame, split: str, source_dataset: str | None) -> pd.DataFrame:
    split_column = _find_column(list(dataframe.columns), ["split"])
    rows = dataframe[dataframe[split_column].astype(str).str.lower() == split.lower()].copy()
    if source_dataset:
        source_column = _find_column(list(rows.columns), ["source_dataset", "dataset"], required=False)
        if source_column is not None:
            rows = rows[rows[source_column].astype(str).str.lower() == source_dataset.lower()].copy()
    return rows


def _metric_state() -> dict[str, list]:
    return {
        "cls_true": [],
        "cls_pred": [],
        "seg_dice": [],
        "seg_iou": [],
        "sev_true": [],
        "sev_pred": [],
        "concept_acc": [],
        "concept_mae": [],
        "rule_scores": [],
    }


def _update_state(
    state: dict[str, list],
    outputs: Any,
    batch: dict[str, Any],
    rule_metadata: dict[str, list[int]],
) -> None:
    cls_logits = _infer_classification_logits(outputs)
    if cls_logits is None:
        raise KeyError("Model outputs do not contain classification logits")

    targets = batch["class_id"]
    predictions = cls_logits.argmax(dim=1).cpu()
    state["cls_true"].extend(targets.cpu().tolist())
    state["cls_pred"].extend(predictions.tolist())

    lesion_logits = _infer_segmentation_logits(outputs)
    if lesion_logits is not None:
        pred_masks = (torch.sigmoid(lesion_logits).cpu() > 0.5).float()
        gt_masks = batch["lesion_mask"].float()
        has_masks = batch["has_lesion_mask"].bool()
        for index in range(pred_masks.shape[0]):
            if not bool(has_masks[index].item()):
                continue
            dice, iou = binary_segmentation_stats(pred_masks[index], gt_masks[index])
            state["seg_dice"].append(dice)
            state["seg_iou"].append(iou)

    severity_logits = _infer_severity_logits(outputs, cls_logits=cls_logits)
    if severity_logits is not None:
        severity_targets = batch["severity"]
        valid = severity_targets >= 0
        if valid.any():
            state["sev_true"].extend(severity_targets[valid].cpu().tolist())
            if severity_logits.ndim > 1 and severity_logits.shape[1] > 1:
                sev_predictions = severity_logits.argmax(dim=1).cpu()
                state["sev_pred"].extend(sev_predictions[valid].tolist())
            else:
                sev_predictions = severity_logits.view(-1).round().long().cpu()
                state["sev_pred"].extend(sev_predictions[valid].tolist())

    concept_logits = _infer_concept_logits(outputs, cls_logits=cls_logits, severity_logits=severity_logits)
    if concept_logits is not None and "concept_targets" in batch:
        concept_probs = torch.sigmoid(concept_logits).cpu()
        concept_targets = batch["concept_targets"].float().cpu()
        concept_valid_mask = batch.get("concept_valid_mask")
        if concept_valid_mask is not None:
            concept_valid_mask = concept_valid_mask.float().cpu()
        state["concept_acc"].append(multilabel_binary_accuracy(concept_targets, concept_probs, valid_mask=concept_valid_mask))
        state["concept_mae"].append(mean_absolute_error(concept_targets, concept_probs, valid_mask=concept_valid_mask))
        if severity_logits is not None:
            class_probs = torch.softmax(cls_logits, dim=1)
            severity_probs = (
                torch.softmax(severity_logits, dim=1)
                if severity_logits.ndim > 1
                else torch.sigmoid(severity_logits.view(-1, 1))
            )
            rule_loss = float(
                rule_consistency_loss(concept_logits, class_probs, severity_probs, rule_metadata).detach().cpu().item()
            )
            state["rule_scores"].append(rule_loss)


def _finalize_state(state: dict[str, list]) -> dict[str, Any]:
    metrics = {
        "classification_accuracy": safe_mean([1.0 if t == p else 0.0 for t, p in zip(state["cls_true"], state["cls_pred"])]),
        "classification_macro_f1": macro_f1_score(state["cls_true"], state["cls_pred"]),
        "segmentation_dice": safe_mean(state["seg_dice"]),
        "segmentation_miou": safe_mean(state["seg_iou"]),
        "segmentation_samples": len(state["seg_dice"]),
    }
    if state["sev_true"]:
        metrics["severity_accuracy"] = safe_mean([1.0 if t == p else 0.0 for t, p in zip(state["sev_true"], state["sev_pred"])])
        metrics["severity_samples"] = len(state["sev_true"])
    if state["concept_acc"]:
        metrics["concept_binary_accuracy"] = safe_mean(state["concept_acc"])
        metrics["concept_mae"] = safe_mean(state["concept_mae"])
        metrics["concept_count"] = len(CONCEPT_NAMES)
    if state["rule_scores"]:
        metrics["rule_consistency_loss"] = safe_mean(state["rule_scores"])
    return metrics


def compare_native_and_replaced(
    config: dict[str, Any],
    checkpoint_path: Path,
    mmseg_predictions_path: Path,
    split: str,
    source_dataset: str | None,
    device_override: str | None,
    manifest_override: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_override or _resolve_repo_path(str(config["train_manifest"]))
    if manifest_path is None:
        raise ValueError("Config must include train_manifest")
    dataframe = pd.read_csv(manifest_path)
    rows = _select_rows(dataframe, split=split, source_dataset=source_dataset)
    if rows.empty:
        raise ValueError(f"No rows found for split={split!r} source_dataset={source_dataset!r}")

    class_column = _find_column(list(dataframe.columns), ["class_id", "label", "target"])
    num_classes = int(dataframe[class_column].max()) + 1
    image_size = int(config.get("image_size", 256))
    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    device = torch.device(device_override or config.get("device", "cpu"))

    dataset = ManifestEvalDataset(rows, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = _build_model(config, num_classes=num_classes).to(device)
    _load_checkpoint(model, checkpoint_path)
    model.eval()

    prediction_df = load_mmseg_predictions(mmseg_predictions_path)
    prediction_index = build_prediction_index(prediction_df)
    class_names = [str(name) for name in dataframe.sort_values(class_column)["class_name"].drop_duplicates().tolist()] if "class_name" in dataframe.columns else []
    rule_metadata = build_rule_metadata(class_names)

    native_state = _metric_state()
    replaced_state = _metric_state()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            native_outputs = model(images)
            _update_state(native_state, native_outputs, batch, rule_metadata)

            features = model.extract_features(images)
            external_masks = load_external_mask_batch(
                image_paths=list(batch["image_path"]),
                prediction_index=prediction_index,
                image_size=tuple(images.shape[-2:]),
            ).to(device)
            replaced_outputs = forward_from_features_with_external_masks(
                model,
                features,
                external_masks=external_masks,
                input_size=tuple(images.shape[-2:]),
            )
            _update_state(replaced_state, replaced_outputs, batch, rule_metadata)

    native_metrics = _finalize_state(native_state)
    replaced_metrics = _finalize_state(replaced_state)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "mmseg_predictions": str(mmseg_predictions_path),
        "split": split,
        "source_dataset": source_dataset,
        "num_samples": int(len(rows)),
    }
    for prefix, metrics in (("native", native_metrics), ("mmseg_replaced", replaced_metrics)):
        for key, value in metrics.items():
            result[f"{prefix}_{key}"] = value

    for metric_name in [
        "classification_accuracy",
        "classification_macro_f1",
        "segmentation_dice",
        "segmentation_miou",
        "severity_accuracy",
        "concept_binary_accuracy",
        "concept_mae",
        "rule_consistency_loss",
    ]:
        native_key = f"native_{metric_name}"
        replaced_key = f"mmseg_replaced_{metric_name}"
        if native_key in result and replaced_key in result:
            result[f"delta_{metric_name}"] = float(result[replaced_key]) - float(result[native_key])

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multitask-config", required=True)
    parser.add_argument("--multitask-checkpoint", required=True)
    parser.add_argument("--mmseg-predictions", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.multitask_config)
    checkpoint_path = _resolve_repo_path(args.multitask_checkpoint)
    prediction_path = _resolve_repo_path(args.mmseg_predictions)
    manifest_override = _resolve_repo_path(args.manifest)
    output_json = _resolve_repo_path(args.output_json)
    if config_path is None or checkpoint_path is None or prediction_path is None:
        raise ValueError("Config, checkpoint, and mmseg_predictions are required")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    metrics = compare_native_and_replaced(
        config=config,
        checkpoint_path=checkpoint_path,
        mmseg_predictions_path=prediction_path,
        split=args.split,
        source_dataset=args.source_dataset,
        device_override=args.device,
        manifest_override=manifest_override,
        output_json=output_json,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

