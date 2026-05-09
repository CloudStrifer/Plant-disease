import torch

from plant_disease.eval.reporting import (
    binary_segmentation_stats,
    macro_f1_score,
    mean_absolute_error,
    multilabel_binary_accuracy,
    safe_mean,
)


def test_safe_mean_handles_empty() -> None:
    assert safe_mean([]) == 0.0


def test_binary_segmentation_stats_perfect_overlap() -> None:
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dice, iou = binary_segmentation_stats(pred, target)
    assert dice == 1.0
    assert iou == 1.0


def test_macro_f1_score_returns_average_over_labels() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    score = macro_f1_score(y_true, y_pred)
    assert 0.70 < score < 0.85


def test_multilabel_binary_accuracy_thresholds_predictions() -> None:
    target = torch.tensor([[1.0, 0.0, 1.0]])
    probs = torch.tensor([[0.9, 0.2, 0.6]])
    assert multilabel_binary_accuracy(target, probs) == 1.0


def test_multilabel_binary_accuracy_respects_valid_mask() -> None:
    target = torch.tensor([[1.0, 0.0, 1.0]])
    probs = torch.tensor([[0.9, 0.9, 0.1]])
    valid_mask = torch.tensor([[1.0, 0.0, 0.0]])
    assert multilabel_binary_accuracy(target, probs, valid_mask=valid_mask) == 1.0


def test_mean_absolute_error_returns_scalar() -> None:
    target = torch.tensor([[1.0, 0.0]])
    prediction = torch.tensor([[0.5, 0.25]])
    assert mean_absolute_error(target, prediction) == 0.375


def test_mean_absolute_error_respects_valid_mask() -> None:
    target = torch.tensor([[1.0, 0.0]])
    prediction = torch.tensor([[0.5, 1.0]])
    valid_mask = torch.tensor([[1.0, 0.0]])
    assert mean_absolute_error(target, prediction, valid_mask=valid_mask) == 0.5
