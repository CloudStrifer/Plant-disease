"""Run evaluation for classification, segmentation, severity, and derived boxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plant_disease.concepts.rules import CONCEPT_NAMES, build_rule_metadata, derive_concept_supervision, rule_consistency_loss
from plant_disease.data.dataset import _resolve_data_path
from plant_disease.eval.reporting import (
    binary_segmentation_stats,
    macro_f1_score,
    mean_absolute_error,
    multilabel_binary_accuracy,
    safe_mean,
)
from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel
from scripts.train_baseline import resolve_model_in_channels


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _find_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    column_set = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in column_set:
            return column_set[candidate.lower()]
    if required:
        raise KeyError(f"Expected one of columns {candidates}, found {columns}")
    return None


def _tensor_from_pil(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class ManifestEvalDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = int(image_size)
        columns = list(self.dataframe.columns)
        self.image_col = _find_column(columns, ["image_path", "image", "path"])
        self.class_col = _find_column(columns, ["class_id", "label", "target"])
        self.mask_col = _find_column(
            columns,
            ["lesion_mask_path", "mask_path", "lesion_mask", "mask", "segmentation_path", "annotation_path"],
            required=False,
        )
        self.has_mask_col = _find_column(columns, ["has_lesion_mask", "has_mask", "mask_available"], required=False)
        self.severity_col = _find_column(
            columns,
            ["severity_label", "severity", "severity_score", "severity_id"],
            required=False,
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def _load_mask(self, mask_path: str | None) -> Image.Image:
        if mask_path is None or not str(mask_path).strip():
            return Image.new("L", (self.image_size, self.image_size), 0)
        mask_image = Image.open(mask_path).convert("L")
        return mask_image.resize((self.image_size, self.image_size), Image.NEAREST)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        manifest_dir = str(row["__manifest_dir"]) if "__manifest_dir" in row.index else None
        image_path = Path(_resolve_data_path(str(row[self.image_col]), manifest_dir=manifest_dir))
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = _tensor_from_pil(image)

        mask_path_value = None
        if self.mask_col is not None:
            raw_mask = row[self.mask_col]
            if pd.notna(raw_mask):
                mask_path_value = _resolve_data_path(str(raw_mask), manifest_dir=manifest_dir)
        mask_image = self._load_mask(mask_path_value)
        mask_array = (np.asarray(mask_image, dtype=np.float32) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        has_mask = bool(mask_path_value) and Path(mask_path_value).exists()
        if self.has_mask_col is not None and pd.notna(row[self.has_mask_col]):
            has_mask = str(row[self.has_mask_col]).strip().lower() in {"1", "true", "yes"}

        severity_value = -1
        if self.severity_col is not None and pd.notna(row[self.severity_col]):
            severity_value = int(row[self.severity_col])

        class_name = ""
        if "class_name" in self.dataframe.columns and pd.notna(row["class_name"]):
            class_name = str(row["class_name"])
        concept_targets, concept_valid_mask, concept_weights = derive_concept_supervision(image_tensor, mask_tensor, class_name)

        return {
            "image": image_tensor,
            "class_id": torch.tensor(int(row[self.class_col]), dtype=torch.long),
            "class_name": class_name,
            "lesion_mask": mask_tensor,
            "has_lesion_mask": torch.tensor(has_mask, dtype=torch.bool),
            "severity": torch.tensor(severity_value, dtype=torch.long),
            "concept_targets": concept_targets,
            "concept_valid_mask": concept_valid_mask,
            "concept_weights": concept_weights,
            "image_path": str(image_path),
        }


def _build_model(config: dict[str, Any], num_classes: int) -> MultiTaskPlantDiseaseModel:
    kwargs: dict[str, Any] = {
        "num_classes": int(num_classes),
        "in_channels": resolve_model_in_channels(config),
        "num_severity_grades": int(config.get("num_severity_grades", 4)),
        "backbone_name": config.get("backbone_name", "resnet18"),
        "use_concept_head": bool(config.get("use_concept_head", False)),
        "num_concepts": int(config.get("num_concepts", len(CONCEPT_NAMES))),
    }
    if "fusion_mode" in config:
        kwargs["fusion_mode"] = config["fusion_mode"]
    return MultiTaskPlantDiseaseModel(**kwargs)


def _extract_output(outputs: Any, candidates: list[str]) -> torch.Tensor | None:
    if isinstance(outputs, dict):
        for candidate in candidates:
            if candidate in outputs:
                return outputs[candidate]
    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _iter_output_tensors(outputs: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(outputs, dict):
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                tensors.append(value)
    elif isinstance(outputs, (list, tuple)):
        for value in outputs:
            if isinstance(value, torch.Tensor):
                tensors.append(value)
    elif isinstance(outputs, torch.Tensor):
        tensors.append(outputs)
    return tensors


def _infer_classification_logits(outputs: Any) -> torch.Tensor | None:
    tensor = _extract_output(outputs, ["cls_logits", "class_logits", "classification_logits", "logits", "disease_logits"])
    if tensor is not None:
        return tensor
    candidates = [item for item in _iter_output_tensors(outputs) if item.ndim == 2 and item.shape[1] > 1]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.shape[1], reverse=True)
    return candidates[0]


def _infer_segmentation_logits(outputs: Any) -> torch.Tensor | None:
    tensor = _extract_output(outputs, ["lesion_logits", "segmentation_logits", "mask_logits", "seg_logits", "lesion_mask_logits"])
    if tensor is not None:
        return tensor
    candidates = [item for item in _iter_output_tensors(outputs) if item.ndim == 4]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.numel(), reverse=True)
    return candidates[0]


def _infer_severity_logits(outputs: Any, cls_logits: torch.Tensor | None = None) -> torch.Tensor | None:
    tensor = _extract_output(outputs, ["severity_logits", "sev_logits", "severity_output"])
    if tensor is not None:
        return tensor
    candidates = [item for item in _iter_output_tensors(outputs) if item.ndim in {1, 2}]
    if cls_logits is not None:
        candidates = [item for item in candidates if item is not cls_logits]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda item: item.shape[1] if item.ndim == 2 else 1)
    return candidates[0]


def _infer_concept_logits(
    outputs: Any,
    cls_logits: torch.Tensor | None = None,
    severity_logits: torch.Tensor | None = None,
) -> torch.Tensor | None:
    tensor = _extract_output(outputs, ["concept_logits", "concept_scores", "concept_output"])
    if tensor is not None:
        return tensor
    candidates = [item for item in _iter_output_tensors(outputs) if item.ndim == 2 and item.shape[1] == len(CONCEPT_NAMES)]
    if cls_logits is not None:
        candidates = [item for item in candidates if item is not cls_logits]
    if severity_logits is not None:
        candidates = [item for item in candidates if item is not severity_logits]
    if not candidates:
        return None
    return candidates[0]


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in state.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(
                (key, f"shape_mismatch ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}")
            )
            continue
        compatible_state[key] = value

    model.load_state_dict(compatible_state, strict=False)
    print(
        f"Loaded checkpoint={checkpoint_path} "
        f"loaded={len(compatible_state)} skipped={len(skipped)}"
    )
    if skipped:
        preview = ", ".join(f"{name} ({reason})" for name, reason in skipped[:8])
        print(f"Skipped checkpoint params: {preview}")


def _to_overlay(base_rgb: np.ndarray, binary_mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    overlay = base_rgb.copy().astype(np.float32)
    mask = binary_mask.astype(bool)
    if not mask.any():
        return overlay.astype(np.uint8)
    tint = np.zeros_like(overlay)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    overlay[mask] = overlay[mask] * (1.0 - alpha) + tint[mask] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _save_visualizations(
    rows: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    output_dir: Path,
    max_visualizations: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ManifestEvalDataset(rows.head(max_visualizations).reset_index(drop=True), image_size=image_size)
    saved = 0
    metadata = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            outputs = model(image_tensor)
            lesion_logits = _infer_segmentation_logits(outputs)
            if lesion_logits is None:
                continue
            cls_logits = _infer_classification_logits(outputs)
            severity_logits = _infer_severity_logits(outputs, cls_logits=cls_logits)
            concept_logits = _infer_concept_logits(outputs, cls_logits=cls_logits, severity_logits=severity_logits)

            pred_mask = (torch.sigmoid(lesion_logits)[0, 0].cpu() > 0.5).numpy().astype(np.uint8)
            gt_mask = sample["lesion_mask"][0].cpu().numpy().astype(np.uint8)

            image = Image.open(sample["image_path"]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            base = np.asarray(image, dtype=np.uint8)
            gt_overlay = _to_overlay(base, gt_mask, (255, 0, 0))
            pred_overlay = _to_overlay(base, pred_mask, (0, 255, 0))

            panel = Image.new("RGB", (image_size * 3, image_size))
            panel.paste(Image.fromarray(base), (0, 0))
            panel.paste(Image.fromarray(gt_overlay), (image_size, 0))
            panel.paste(Image.fromarray(pred_overlay), (image_size * 2, 0))
            filename = f"sample_{index:03d}.png"
            panel.save(output_dir / filename)
            record = {
                "file": filename,
                "image_path": sample["image_path"],
                "class_name": sample["class_name"],
            }
            if cls_logits is not None:
                record["pred_class_id"] = int(cls_logits.argmax(dim=1)[0].item())
            if severity_logits is not None:
                if severity_logits.ndim > 1 and severity_logits.shape[1] > 1:
                    record["pred_severity_id"] = int(severity_logits.argmax(dim=1)[0].item())
                else:
                    record["pred_severity_id"] = int(severity_logits.view(-1)[0].round().item())
            if concept_logits is not None:
                concept_probs = torch.sigmoid(concept_logits)[0].cpu().tolist()
                concept_target = sample["concept_targets"].cpu().tolist()
                concept_valid = sample["concept_valid_mask"].cpu().tolist()
                record["concept_probs"] = {name: round(float(value), 4) for name, value in zip(CONCEPT_NAMES, concept_probs)}
                record["concept_targets"] = {name: round(float(value), 4) for name, value in zip(CONCEPT_NAMES, concept_target)}
                record["concept_valid_mask"] = {name: round(float(value), 4) for name, value in zip(CONCEPT_NAMES, concept_valid)}
            metadata.append(record)
            saved += 1
    if metadata:
        (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return saved


def evaluate(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    source_dataset: str | None,
    device_override: str | None,
    output_json: Path | None,
    visualize_dir: Path | None,
    max_visualizations: int,
) -> dict[str, Any]:
    manifest_path = _resolve_repo_path(str(config["train_manifest"]))
    if manifest_path is None:
        raise ValueError("Config must include train_manifest")

    dataframe = pd.read_csv(manifest_path)
    split_column = _find_column(list(dataframe.columns), ["split"])
    split_rows = dataframe[dataframe[split_column].astype(str).str.lower() == split.lower()].copy()

    if source_dataset:
        source_column = _find_column(list(split_rows.columns), ["source_dataset", "dataset"], required=False)
        if source_column is not None:
            split_rows = split_rows[split_rows[source_column].astype(str).str.lower() == source_dataset.lower()].copy()

    if split_rows.empty:
        raise ValueError(f"No rows found for split={split!r} source_dataset={source_dataset!r}")

    class_column = _find_column(list(dataframe.columns), ["class_id", "label", "target"])
    num_classes = int(dataframe[class_column].max()) + 1
    severity_column = _find_column(
        list(dataframe.columns),
        ["severity_label", "severity", "severity_score", "severity_id"],
        required=False,
    )
    if "num_severity_grades" not in config and severity_column is not None:
        valid_severity = dataframe[dataframe[severity_column] >= 0][severity_column]
        if not valid_severity.empty:
            config = dict(config)
            config["num_severity_grades"] = int(valid_severity.max()) + 1
    image_size = int(config.get("image_size", 256))
    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    device = torch.device(device_override or config.get("device", "cpu"))

    dataset = ManifestEvalDataset(split_rows, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = _build_model(config, num_classes=num_classes).to(device)
    _load_checkpoint(model, checkpoint_path)
    model.eval()

    cls_true: list[int] = []
    cls_pred: list[int] = []
    seg_dice: list[float] = []
    seg_iou: list[float] = []
    sev_true: list[int] = []
    sev_pred: list[int] = []
    concept_acc: list[float] = []
    concept_mae: list[float] = []
    rule_scores: list[float] = []
    class_names = [str(name) for name in dataframe.sort_values(class_column)["class_name"].drop_duplicates().tolist()] if "class_name" in dataframe.columns else []
    rule_metadata = build_rule_metadata(class_names)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["class_id"].to(device)
            outputs = model(images)

            cls_logits = _infer_classification_logits(outputs)
            if cls_logits is None:
                raise KeyError("Model outputs do not contain classification logits")

            predictions = cls_logits.argmax(dim=1)
            cls_true.extend(targets.cpu().tolist())
            cls_pred.extend(predictions.cpu().tolist())

            lesion_logits = _infer_segmentation_logits(outputs)
            if lesion_logits is not None:
                pred_masks = (torch.sigmoid(lesion_logits).cpu() > 0.5).float()
                gt_masks = batch["lesion_mask"].float()
                has_masks = batch["has_lesion_mask"].bool()
                for index in range(pred_masks.shape[0]):
                    if not bool(has_masks[index].item()):
                        continue
                    dice, iou = binary_segmentation_stats(pred_masks[index], gt_masks[index])
                    seg_dice.append(dice)
                    seg_iou.append(iou)

            severity_logits = _infer_severity_logits(outputs, cls_logits=cls_logits)
            if severity_logits is not None:
                severity_targets = batch["severity"]
                valid = severity_targets >= 0
                if valid.any():
                    sev_true.extend(severity_targets[valid].cpu().tolist())
                    if severity_logits.ndim > 1 and severity_logits.shape[1] > 1:
                        sev_predictions = severity_logits.argmax(dim=1)
                        sev_pred.extend(sev_predictions[valid.to(device)].cpu().tolist())
                    else:
                        sev_predictions = severity_logits.view(-1).round().long().cpu()
                        sev_pred.extend(sev_predictions[valid].tolist())

            concept_logits = _infer_concept_logits(outputs, cls_logits=cls_logits, severity_logits=severity_logits)
            if concept_logits is not None and "concept_targets" in batch:
                concept_probs = torch.sigmoid(concept_logits).cpu()
                concept_targets = batch["concept_targets"].float().cpu()
                concept_valid_mask = batch.get("concept_valid_mask")
                if concept_valid_mask is not None:
                    concept_valid_mask = concept_valid_mask.float().cpu()
                concept_acc.append(multilabel_binary_accuracy(concept_targets, concept_probs, valid_mask=concept_valid_mask))
                concept_mae.append(mean_absolute_error(concept_targets, concept_probs, valid_mask=concept_valid_mask))
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
                    rule_scores.append(rule_loss)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "split": split,
        "source_dataset": source_dataset,
        "num_samples": int(len(split_rows)),
        "classification_accuracy": safe_mean([1.0 if t == p else 0.0 for t, p in zip(cls_true, cls_pred)]),
        "classification_macro_f1": macro_f1_score(cls_true, cls_pred),
        "segmentation_dice": safe_mean(seg_dice),
        "segmentation_miou": safe_mean(seg_iou),
        "segmentation_samples": len(seg_dice),
    }

    if sev_true:
        metrics["severity_accuracy"] = safe_mean([1.0 if t == p else 0.0 for t, p in zip(sev_true, sev_pred)])
        metrics["severity_samples"] = len(sev_true)

    if concept_acc:
        metrics["concept_binary_accuracy"] = safe_mean(concept_acc)
        metrics["concept_mae"] = safe_mean(concept_mae)
        metrics["concept_count"] = len(CONCEPT_NAMES)
        metrics["concept_names"] = list(CONCEPT_NAMES)

    if rule_scores:
        metrics["rule_consistency_loss"] = safe_mean(rule_scores)

    if visualize_dir is not None:
        saved = _save_visualizations(
            rows=split_rows,
            model=model,
            device=device,
            image_size=image_size,
            output_dir=visualize_dir,
            max_visualizations=max_visualizations,
        )
        metrics["saved_visualizations"] = saved
        metrics["visualize_dir"] = str(visualize_dir)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the training config yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--split", default="test", help="Manifest split to evaluate, e.g. val or test")
    parser.add_argument("--source-dataset", default=None, help="Optional dataset filter, e.g. PlantSeg")
    parser.add_argument("--device", default=None, help="Override device such as cpu or cuda")
    parser.add_argument("--output-json", default=None, help="Optional path to save metrics json")
    parser.add_argument("--visualize-dir", default=None, help="Optional directory to save prediction panels")
    parser.add_argument("--max-visualizations", type=int, default=12, help="Maximum number of visualization panels to save")
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    output_json = _resolve_repo_path(args.output_json)
    visualize_dir = _resolve_repo_path(args.visualize_dir)

    if config_path is None or checkpoint_path is None:
        raise ValueError("Config and checkpoint paths are required")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    metrics = evaluate(
        config=config,
        checkpoint_path=checkpoint_path,
        split=args.split,
        source_dataset=args.source_dataset,
        device_override=args.device,
        output_json=output_json,
        visualize_dir=visualize_dir,
        max_visualizations=args.max_visualizations,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
