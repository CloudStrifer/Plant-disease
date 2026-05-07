import torch

from plant_disease.training.losses import compute_multitask_loss


def test_multitask_loss_returns_named_components():
    outputs = {
        "segmentation_logits": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "classification_logits": torch.zeros((2, 3), dtype=torch.float32),
        "severity_logits": torch.zeros((2, 2), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "class_id": torch.zeros(2, dtype=torch.long),
        "severity_label": torch.zeros(2, dtype=torch.long),
    }
    losses = compute_multitask_loss(outputs, batch)

    assert set(losses) == {"total", "segmentation", "classification", "severity"}


def test_multitask_loss_skips_segmentation_when_mask_is_missing():
    outputs = {
        "segmentation_logits": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "classification_logits": torch.zeros((1, 2), dtype=torch.float32),
        "severity_logits": torch.zeros((1, 2), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "class_id": torch.zeros(1, dtype=torch.long),
        "severity_label": torch.zeros(1, dtype=torch.long),
        "has_lesion_mask": torch.tensor([False]),
    }
    losses = compute_multitask_loss(outputs, batch)

    assert float(losses["segmentation"]) == 0.0
