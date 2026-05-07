import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (union + eps)


def severity_ratio(lesion_mask: torch.Tensor, leaf_mask: torch.Tensor) -> float:
    lesion = float((lesion_mask > 0).sum())
    leaf = float((leaf_mask > 0).sum())
    if leaf == 0:
        return 0.0
    return lesion / leaf
