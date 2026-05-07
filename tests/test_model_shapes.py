import torch

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
