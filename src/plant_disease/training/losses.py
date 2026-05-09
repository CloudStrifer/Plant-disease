import torch
import torch.nn.functional as F

from plant_disease.concepts.rules import build_rule_metadata, rule_consistency_loss


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, reduction: str = "mean") -> torch.Tensor:
    probs = logits.sigmoid().reshape(logits.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    losses = 1.0 - ((2.0 * intersection + eps) / (union + eps))
    if reduction == "none":
        return losses
    return losses.mean()


def _weighted_segmentation_loss(
    seg_logits: torch.Tensor,
    seg_target: torch.Tensor,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    if seg_logits.shape[0] == 0:
        return torch.tensor(0.0, device=seg_logits.device)

    bce_map = F.binary_cross_entropy_with_logits(seg_logits, seg_target, reduction="none")
    bce_per_sample = bce_map.reshape(seg_logits.shape[0], -1).mean(dim=1)
    dice_per_sample = dice_loss(seg_logits, seg_target, reduction="none")
    per_sample = bce_per_sample + dice_per_sample

    if sample_weights is None:
        return per_sample.mean()

    weights = sample_weights.to(seg_logits.device).float().reshape(-1)
    if weights.numel() != per_sample.numel():
        weights = torch.ones_like(per_sample)
    weight_sum = weights.sum()
    if float(weight_sum.item()) <= 0.0:
        return torch.tensor(0.0, device=seg_logits.device)
    return (per_sample * weights).sum() / weight_sum


def compute_multitask_loss(
    outputs: dict,
    batch: dict,
    seg_weight: float = 1.0,
    sev_weight: float = 1.0,
    concept_weight: float = 0.0,
    rule_weight: float = 0.0,
    concept_real_only: bool = False,
    class_names: list[str] | None = None,
) -> dict:
    has_mask = batch.get("has_lesion_mask")
    seg_loss_weight = batch.get("seg_loss_weight")
    if has_mask is not None:
        valid = has_mask.bool()
        if valid.any():
            seg_logits = outputs["segmentation_logits"][valid]
            seg_target = batch["lesion_mask"][valid].float()
            valid_weights = seg_loss_weight[valid] if seg_loss_weight is not None else None
            segmentation = _weighted_segmentation_loss(seg_logits, seg_target, valid_weights)
        else:
            segmentation = torch.tensor(0.0, device=outputs["segmentation_logits"].device)
    else:
        segmentation = _weighted_segmentation_loss(
            outputs["segmentation_logits"],
            batch["lesion_mask"].float(),
            seg_loss_weight,
        )
    classification = F.cross_entropy(outputs["classification_logits"], batch["class_id"])
    severity_targets = batch["severity_label"]
    valid_severity = severity_targets >= 0
    if valid_severity.any():
        severity = F.cross_entropy(outputs["severity_logits"][valid_severity], severity_targets[valid_severity])
    else:
        severity = torch.tensor(0.0, device=classification.device)
    concept = torch.tensor(0.0, device=classification.device)
    rule = torch.tensor(0.0, device=classification.device)

    if concept_weight > 0.0 and "concept_logits" in outputs and "concept_targets" in batch:
        concept_target = batch["concept_targets"].float().to(classification.device)
        concept_valid_mask = batch.get("concept_valid_mask")
        concept_weights = batch.get("concept_weights")
        concept_map = F.binary_cross_entropy_with_logits(outputs["concept_logits"], concept_target, reduction="none")
        if concept_valid_mask is not None:
            valid_mask = concept_valid_mask.float().to(classification.device)
            if concept_real_only and "pseudo_label" in batch:
                pseudo_mask = batch["pseudo_label"].bool().to(classification.device).reshape(-1, 1)
                valid_mask = valid_mask * (~pseudo_mask).float()
            weight_mask = concept_weights.float().to(classification.device) if concept_weights is not None else valid_mask
            combined_weight = valid_mask * weight_mask
            denom = combined_weight.sum()
            if float(denom.item()) > 0.0:
                concept = (concept_map * combined_weight).sum() / denom
            else:
                concept = torch.tensor(0.0, device=classification.device)
        else:
            concept = concept_map.mean()

    if rule_weight > 0.0 and "concept_logits" in outputs:
        rule_metadata = build_rule_metadata(class_names or [])
        class_probs = torch.softmax(outputs["classification_logits"], dim=1)
        severity_probs = torch.softmax(outputs["severity_logits"], dim=1)
        rule = rule_consistency_loss(outputs["concept_logits"], class_probs, severity_probs, rule_metadata)

    total = classification + seg_weight * segmentation + sev_weight * severity + concept_weight * concept + rule_weight * rule
    losses = {
        "total": total,
        "segmentation": segmentation,
        "classification": classification,
        "severity": severity,
    }
    if "concept_logits" in outputs or concept_weight > 0.0:
        losses["concept"] = concept
    if "concept_logits" in outputs or rule_weight > 0.0:
        losses["rule"] = rule
    return losses
