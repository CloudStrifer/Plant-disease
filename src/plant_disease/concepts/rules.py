from __future__ import annotations

import torch
import torch.nn.functional as F

CONCEPT_NAMES = [
    "yellowing",
    "necrosis",
    "mildew_texture",
    "lesion_area_ratio",
]

MILDEW_KEYWORDS = ["powdery mildew", "downy mildew", "leaf mold", "mildew"]
NECROSIS_KEYWORDS = ["blight", "spot", "scab", "rust", "rot", "canker"]
YELLOWING_KEYWORDS = ["yellow", "curl", "virus", "mosaic", "chlorosis"]
STRONG_NON_MILDEW_KEYWORDS = ["blight", "spot", "scab", "rust", "rot", "virus", "curl", "mosaic"]


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("___", " ")
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = value.replace("(", " ")
    value = value.replace(")", " ")
    value = " ".join(value.split())
    return value


def _contains_any_keyword(class_name: str, keywords: list[str]) -> bool:
    normalized_name = normalize_text(class_name)
    normalized_keywords = [normalize_text(keyword) for keyword in keywords]
    return any(keyword in normalized_name for keyword in normalized_keywords)


def build_rule_metadata(class_names: list[str]) -> dict[str, list[int]]:
    mildew_indices = []
    necrosis_indices = []
    yellowing_indices = []
    for index, class_name in enumerate(class_names):
        if _contains_any_keyword(class_name, MILDEW_KEYWORDS):
            mildew_indices.append(index)
        if _contains_any_keyword(class_name, NECROSIS_KEYWORDS):
            necrosis_indices.append(index)
        if _contains_any_keyword(class_name, YELLOWING_KEYWORDS):
            yellowing_indices.append(index)
    return {
        "mildew_class_indices": mildew_indices,
        "necrosis_class_indices": necrosis_indices,
        "yellowing_class_indices": yellowing_indices,
    }


def _derive_continuous_concepts(image: torch.Tensor, lesion_mask: torch.Tensor) -> dict[str, float]:
    if image.ndim != 3:
        raise ValueError("image must be CHW tensor")
    if lesion_mask.ndim == 3:
        lesion_mask = lesion_mask[0]
    lesion_mask = lesion_mask.float()
    image = image.float().clamp(0.0, 1.0)

    mask_pixels = lesion_mask > 0.5
    lesion_area_ratio = float(torch.clamp(mask_pixels.float().mean(), 0.0, 1.0).item())
    if not mask_pixels.any():
        return {
            "yellowing": 0.0,
            "necrosis": 0.0,
            "mildew_texture": 0.0,
            "lesion_area_ratio": 0.0,
        }

    lesion_pixels = image[:, mask_pixels]
    r = lesion_pixels[0]
    g = lesion_pixels[1]
    b = lesion_pixels[2]
    yellowing = float(torch.clamp((((r + g) * 0.5) - b).mean() * 2.0, 0.0, 1.0).item())
    brightness = lesion_pixels.mean(dim=0)
    necrosis = float(torch.clamp(1.0 - brightness.mean(), 0.0, 1.0).item())

    return {
        "yellowing": yellowing,
        "necrosis": necrosis,
        "mildew_texture": 0.0,
        "lesion_area_ratio": lesion_area_ratio,
    }


def derive_concept_supervision(
    image: torch.Tensor,
    lesion_mask: torch.Tensor,
    class_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    concepts = _derive_continuous_concepts(image, lesion_mask)
    has_mask = float(concepts["lesion_area_ratio"]) > 0.0

    targets = torch.zeros(len(CONCEPT_NAMES), dtype=torch.float32)
    valid_mask = torch.zeros(len(CONCEPT_NAMES), dtype=torch.float32)
    weights = torch.zeros(len(CONCEPT_NAMES), dtype=torch.float32)
    concept_index = {name: idx for idx, name in enumerate(CONCEPT_NAMES)}

    yellowing_keyword = _contains_any_keyword(class_name, YELLOWING_KEYWORDS)
    necrosis_keyword = _contains_any_keyword(class_name, NECROSIS_KEYWORDS)
    mildew_keyword = _contains_any_keyword(class_name, MILDEW_KEYWORDS)
    strong_non_mildew = _contains_any_keyword(class_name, STRONG_NON_MILDEW_KEYWORDS)

    if has_mask:
        targets[concept_index["lesion_area_ratio"]] = concepts["lesion_area_ratio"]
        valid_mask[concept_index["lesion_area_ratio"]] = 1.0
        weights[concept_index["lesion_area_ratio"]] = 1.0

        if concepts["yellowing"] >= 0.35 or yellowing_keyword:
            targets[concept_index["yellowing"]] = concepts["yellowing"]
            valid_mask[concept_index["yellowing"]] = 1.0
            weights[concept_index["yellowing"]] = 1.0 if yellowing_keyword else 0.5

        if concepts["necrosis"] >= 0.45 or necrosis_keyword:
            targets[concept_index["necrosis"]] = concepts["necrosis"]
            valid_mask[concept_index["necrosis"]] = 1.0
            weights[concept_index["necrosis"]] = 1.0 if necrosis_keyword else 0.5

    if mildew_keyword:
        targets[concept_index["mildew_texture"]] = 1.0
        valid_mask[concept_index["mildew_texture"]] = 1.0
        weights[concept_index["mildew_texture"]] = 1.0
    elif strong_non_mildew:
        targets[concept_index["mildew_texture"]] = 0.0
        valid_mask[concept_index["mildew_texture"]] = 1.0
        weights[concept_index["mildew_texture"]] = 0.75

    return targets, valid_mask, weights


def derive_concept_targets(image: torch.Tensor, lesion_mask: torch.Tensor, class_name: str) -> torch.Tensor:
    targets, _, _ = derive_concept_supervision(image, lesion_mask, class_name)
    return targets


def rule_consistency_loss(
    concept_logits: torch.Tensor,
    class_probs: torch.Tensor,
    severity_probs: torch.Tensor,
    rule_metadata: dict[str, list[int]] | None,
) -> torch.Tensor:
    if concept_logits.numel() == 0:
        return torch.tensor(0.0, device=class_probs.device)

    concept_probs = torch.sigmoid(concept_logits)
    concept_index = {name: idx for idx, name in enumerate(CONCEPT_NAMES)}
    losses = []

    if rule_metadata and rule_metadata.get("mildew_class_indices"):
        mildew_mass = class_probs[:, rule_metadata["mildew_class_indices"]].max(dim=1).values
        mildew_concept = concept_probs[:, concept_index["mildew_texture"]]
        losses.append(F.relu(mildew_concept - mildew_mass).mean())

    if rule_metadata and rule_metadata.get("necrosis_class_indices"):
        necrosis_mass = class_probs[:, rule_metadata["necrosis_class_indices"]].max(dim=1).values
        necrosis_concept = concept_probs[:, concept_index["necrosis"]]
        losses.append(F.relu(necrosis_concept - necrosis_mass).mean())

    if rule_metadata and rule_metadata.get("yellowing_class_indices"):
        yellowing_mass = class_probs[:, rule_metadata["yellowing_class_indices"]].max(dim=1).values
        yellowing_concept = concept_probs[:, concept_index["yellowing"]]
        losses.append(F.relu(yellowing_concept - yellowing_mass).mean())

    lesion_area_ratio = concept_probs[:, concept_index["lesion_area_ratio"]]
    severe_mass = severity_probs[:, -1]
    losses.append(F.relu(lesion_area_ratio - severe_mass).mean())

    if not losses:
        return torch.tensor(0.0, device=class_probs.device)
    return sum(losses)
