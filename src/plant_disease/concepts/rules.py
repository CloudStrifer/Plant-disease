import torch

CONCEPT_NAMES = [
    "yellowing",
    "necrosis",
    "spot_density",
    "mildew_texture",
    "lesion_irregularity",
    "color_variance",
    "lesion_area_ratio",
]


def rule_consistency_loss(concepts: dict, class_probs: torch.Tensor, severity_probs: torch.Tensor) -> torch.Tensor:
    mildew_penalty = torch.relu(concepts["mildew_texture"] - class_probs[:, 1]).mean()
    severity_penalty = torch.relu(concepts["lesion_area_ratio"] - severity_probs[:, -1]).mean()
    return mildew_penalty + severity_penalty
