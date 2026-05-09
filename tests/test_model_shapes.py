import torch

from plant_disease.models.lesion_guided import lesion_guided_pool
from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel


def test_multitask_model_outputs_expected_shapes():
    model = MultiTaskPlantDiseaseModel(
        in_channels=32,
        num_classes=4,
        num_severity_grades=3,
    )
    x = torch.randn(2, 32, 16, 16)
    outputs = model.forward_from_features(x)

    assert outputs["segmentation_logits"].shape == (2, 1, 16, 16)
    assert outputs["classification_logits"].shape == (2, 4)
    assert outputs["severity_logits"].shape == (2, 3)


def test_lesion_guided_pool_reduces_spatial_dimensions():
    features = torch.ones((2, 4, 8, 8))
    mask = torch.ones((2, 1, 8, 8))
    fused = lesion_guided_pool(features, mask)

    assert fused.shape == (2, 8)


def test_multitask_model_forward_accepts_images():
    model = MultiTaskPlantDiseaseModel(
        in_channels=512,
        num_classes=4,
        num_severity_grades=3,
        backbone_name="resnet18",
    )
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert outputs["segmentation_logits"].shape == (2, 1, 64, 64)
    assert outputs["classification_logits"].shape == (2, 4)
    assert outputs["severity_logits"].shape == (2, 3)


def test_multitask_model_forward_supports_segformer_b0():
    model = MultiTaskPlantDiseaseModel(
        in_channels=256,
        num_classes=4,
        num_severity_grades=3,
        backbone_name="segformer_b0",
    )
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert outputs["segmentation_logits"].shape == (2, 1, 64, 64)
    assert outputs["classification_logits"].shape == (2, 4)
    assert outputs["severity_logits"].shape == (2, 3)


def test_multitask_model_segformer_b0_can_emit_concept_logits():
    model = MultiTaskPlantDiseaseModel(
        in_channels=256,
        num_classes=4,
        num_severity_grades=3,
        backbone_name="segformer_b0",
        use_concept_head=True,
        num_concepts=4,
    )
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x)

    assert outputs["concept_logits"].shape == (2, 4)
