import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid().reshape(logits.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def compute_multitask_loss(outputs: dict, batch: dict, seg_weight: float = 1.0, sev_weight: float = 1.0) -> dict:
    seg_bce = F.binary_cross_entropy_with_logits(outputs["segmentation_logits"], batch["lesion_mask"].float())
    seg_dice = dice_loss(outputs["segmentation_logits"], batch["lesion_mask"].float())
    segmentation = seg_bce + seg_dice
    classification = F.cross_entropy(outputs["classification_logits"], batch["class_id"])
    severity = F.cross_entropy(outputs["severity_logits"], batch["severity_label"])
    total = classification + seg_weight * segmentation + sev_weight * severity
    return {
        "total": total,
        "segmentation": segmentation,
        "classification": classification,
        "severity": severity,
    }
