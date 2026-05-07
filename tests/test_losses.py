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
