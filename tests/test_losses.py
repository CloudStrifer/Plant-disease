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


def test_multitask_loss_applies_sample_segmentation_weights():
    outputs = {
        "segmentation_logits": torch.tensor(
            [
                [[[4.0, 4.0], [4.0, 4.0]]],
                [[[-4.0, -4.0], [-4.0, -4.0]]],
            ],
            dtype=torch.float32,
        ),
        "classification_logits": torch.zeros((2, 2), dtype=torch.float32),
        "severity_logits": torch.zeros((2, 2), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ],
            dtype=torch.float32,
        ),
        "class_id": torch.zeros(2, dtype=torch.long),
        "severity_label": torch.zeros(2, dtype=torch.long),
        "has_lesion_mask": torch.tensor([True, True]),
        "seg_loss_weight": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }
    losses = compute_multitask_loss(outputs, batch)

    assert float(losses["segmentation"]) < 0.1


def test_multitask_loss_returns_concept_and_rule_terms_when_enabled():
    outputs = {
        "segmentation_logits": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "classification_logits": torch.zeros((1, 3), dtype=torch.float32),
        "severity_logits": torch.zeros((1, 2), dtype=torch.float32),
        "concept_logits": torch.zeros((1, 4), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "class_id": torch.zeros(1, dtype=torch.long),
        "severity_label": torch.zeros(1, dtype=torch.long),
        "concept_targets": torch.zeros((1, 4), dtype=torch.float32),
        "concept_valid_mask": torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        "concept_weights": torch.tensor([[1.0, 0.5, 0.0, 1.0]], dtype=torch.float32),
        "class_name": ["Tomato___powdery mildew"],
    }
    losses = compute_multitask_loss(
        outputs,
        batch,
        concept_weight=0.5,
        rule_weight=0.2,
        class_names=["Apple___healthy", "Tomato___powdery mildew", "Tomato___late blight"],
    )

    assert "concept" in losses
    assert "rule" in losses


def test_multitask_loss_masks_invalid_concept_targets():
    outputs = {
        "segmentation_logits": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "classification_logits": torch.zeros((1, 2), dtype=torch.float32),
        "severity_logits": torch.zeros((1, 2), dtype=torch.float32),
        "concept_logits": torch.zeros((1, 4), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "class_id": torch.zeros(1, dtype=torch.long),
        "severity_label": torch.zeros(1, dtype=torch.long),
        "concept_targets": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        "concept_valid_mask": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        "concept_weights": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    }

    losses = compute_multitask_loss(outputs, batch, concept_weight=1.0)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros((1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.float32),
    )
    assert torch.isclose(losses["concept"], expected)


def test_multitask_loss_can_skip_concept_supervision_for_pseudo_rows():
    outputs = {
        "segmentation_logits": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "classification_logits": torch.zeros((1, 2), dtype=torch.float32),
        "severity_logits": torch.zeros((1, 2), dtype=torch.float32),
        "concept_logits": torch.zeros((1, 4), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        "class_id": torch.zeros(1, dtype=torch.long),
        "severity_label": torch.zeros(1, dtype=torch.long),
        "pseudo_label": torch.tensor([True]),
        "concept_targets": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        "concept_valid_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        "concept_weights": torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
    }

    losses = compute_multitask_loss(outputs, batch, concept_weight=1.0, concept_real_only=True)

    assert float(losses["concept"]) == 0.0


def test_multitask_loss_skips_severity_when_labels_are_invalid():
    outputs = {
        "segmentation_logits": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "classification_logits": torch.zeros((2, 2), dtype=torch.float32),
        "severity_logits": torch.zeros((2, 4), dtype=torch.float32),
    }
    batch = {
        "lesion_mask": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "class_id": torch.zeros(2, dtype=torch.long),
        "severity_label": torch.full((2,), -1, dtype=torch.long),
    }

    losses = compute_multitask_loss(outputs, batch)

    assert float(losses["severity"]) == 0.0
