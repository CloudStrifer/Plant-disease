from __future__ import annotations

from collections import Counter

import torch


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def binary_segmentation_stats(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred_mask.float().reshape(-1)
    target = target_mask.float().reshape(-1)
    intersection = float((pred * target).sum().item())
    pred_sum = float(pred.sum().item())
    target_sum = float(target.sum().item())
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def macro_f1_score(y_true: list[int], y_pred: list[int], eps: float = 1e-6) -> float:
    if not y_true:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    tp_counts = Counter()
    for truth, pred in zip(y_true, y_pred):
        if truth == pred:
            tp_counts[truth] += 1

    f1_values = []
    for label in labels:
        precision = tp_counts[label] / max(pred_counts[label], eps)
        recall = tp_counts[label] / max(true_counts[label], eps)
        if precision + recall == 0:
            f1_values.append(0.0)
            continue
        f1_values.append((2.0 * precision * recall) / (precision + recall))
    return safe_mean(f1_values)


def multilabel_binary_accuracy(
    target: torch.Tensor,
    prediction_probs: torch.Tensor,
    threshold: float = 0.5,
    valid_mask: torch.Tensor | None = None,
) -> float:
    if target.numel() == 0:
        return 0.0
    pred = (prediction_probs >= threshold).float()
    truth = target.float()
    correct = (pred == truth).float()
    if valid_mask is not None:
        valid = valid_mask.float()
        denom = float(valid.sum().item())
        if denom <= 0:
            return 0.0
        return float((correct * valid).sum().item() / denom)
    return float(correct.mean().item())


def mean_absolute_error(target: torch.Tensor, prediction: torch.Tensor, valid_mask: torch.Tensor | None = None) -> float:
    if target.numel() == 0:
        return 0.0
    error = (target.float() - prediction.float()).abs()
    if valid_mask is not None:
        valid = valid_mask.float()
        denom = float(valid.sum().item())
        if denom <= 0:
            return 0.0
        return float((error * valid).sum().item() / denom)
    return float(error.mean().item())
