import torch

from plant_disease.concepts.rules import (
    CONCEPT_NAMES,
    build_rule_metadata,
    derive_concept_supervision,
    derive_concept_targets,
    rule_consistency_loss,
)


def test_rule_consistency_loss_is_non_negative():
    concept_logits = torch.zeros((1, len(CONCEPT_NAMES)))
    concept_logits[0, CONCEPT_NAMES.index("mildew_texture")] = 2.0
    concept_logits[0, CONCEPT_NAMES.index("lesion_area_ratio")] = 1.0
    class_probs = torch.tensor([[0.1, 0.8, 0.1]])
    severity_probs = torch.tensor([[0.2, 0.3, 0.5]])
    metadata = {"mildew_class_indices": [1], "necrosis_class_indices": [], "yellowing_class_indices": []}

    loss = rule_consistency_loss(concept_logits, class_probs, severity_probs, metadata)

    assert float(loss) >= 0.0


def test_build_rule_metadata_finds_keyword_based_class_groups():
    metadata = build_rule_metadata(
        [
            "Apple___healthy",
            "Tomato___powdery mildew",
            "Tomato___tomato mosaic virus",
            "Potato___late blight",
        ]
    )

    assert metadata["mildew_class_indices"] == [1]
    assert metadata["yellowing_class_indices"] == [2]
    assert metadata["necrosis_class_indices"] == [3]


def test_derive_concept_targets_returns_expected_shape():
    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    image[0] = 0.7
    image[1] = 0.6
    image[2] = 0.2
    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mask[:, 2:6, 2:6] = 1.0

    targets = derive_concept_targets(image, mask, "Tomato___powdery mildew")

    assert targets.shape == (len(CONCEPT_NAMES),)
    assert float(targets[CONCEPT_NAMES.index("mildew_texture")]) == 1.0


def test_derive_concept_supervision_returns_masked_four_concept_targets():
    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    image[0] = 0.7
    image[1] = 0.6
    image[2] = 0.2
    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mask[:, 2:6, 2:6] = 1.0

    targets, valid_mask, weights = derive_concept_supervision(image, mask, "Tomato___powdery mildew")

    assert tuple(CONCEPT_NAMES) == ("yellowing", "necrosis", "mildew_texture", "lesion_area_ratio")
    assert targets.shape == (4,)
    assert valid_mask.shape == (4,)
    assert weights.shape == (4,)
    assert float(targets[CONCEPT_NAMES.index("mildew_texture")]) == 1.0
    assert float(valid_mask[CONCEPT_NAMES.index("mildew_texture")]) == 1.0
