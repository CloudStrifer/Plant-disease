import torch


def lesion_guided_pool(features: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    mask = mask_logits.sigmoid()
    lesion_features = features * mask
    global_pool = features.mean(dim=(2, 3))
    lesion_pool = lesion_features.mean(dim=(2, 3))
    return torch.cat([global_pool, lesion_pool], dim=1)
