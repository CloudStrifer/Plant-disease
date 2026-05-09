from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from plant_disease.models.lesion_guided import lesion_guided_pool
from plant_disease_mmseg.predictions import resolve_prediction_record


def _lesion_only_pool(features: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    mask = mask_logits.sigmoid()
    lesion_features = features * mask
    return lesion_features.mean(dim=(2, 3))


def _mask_to_logits(mask_tensor: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    probs = mask_tensor.float().clamp(eps, 1.0 - eps)
    return torch.log(probs / (1.0 - probs))


def forward_from_features_with_external_masks(
    model,
    features: torch.Tensor,
    external_masks: torch.Tensor,
    input_size: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    if external_masks.ndim == 3:
        external_masks = external_masks.unsqueeze(1)
    resized_masks = F.interpolate(
        external_masks.float(),
        size=features.shape[-2:],
        mode="nearest",
    )
    segmentation_logits = _mask_to_logits(resized_masks)
    global_pool = model.pool(features).flatten(1)
    if getattr(model, "fusion_mode", "global") == "lesion_guided":
        pooled = lesion_guided_pool(features, segmentation_logits)
    else:
        pooled = global_pool

    outputs = {
        "segmentation_logits": segmentation_logits,
        "classification_logits": model.cls_head(pooled),
        "severity_logits": model.sev_head(global_pool),
    }
    if getattr(model, "concept_head", None) is not None:
        concept_pool = _lesion_only_pool(features, segmentation_logits)
        outputs["concept_logits"] = model.concept_head(concept_pool)
    if input_size is not None:
        outputs["segmentation_logits"] = F.interpolate(
            outputs["segmentation_logits"],
            size=input_size,
            mode="nearest",
        )
    return outputs


def load_external_mask_batch(
    image_paths: list[str],
    prediction_index: dict[str, dict],
    image_size: tuple[int, int],
) -> torch.Tensor:
    masks = []
    width = int(image_size[1])
    height = int(image_size[0])
    for image_path in image_paths:
        record = resolve_prediction_record(image_path, prediction_index)
        mask_path = Path(record["pred_mask_path"])
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing predicted mask: {mask_path}")
        mask_image = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
        mask_array = (np.asarray(mask_image, dtype=np.float32) > 0).astype(np.float32)
        masks.append(torch.from_numpy(mask_array))
    return torch.stack(masks, dim=0).unsqueeze(1)

