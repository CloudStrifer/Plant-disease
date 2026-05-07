import torch

from plant_disease.concepts.rules import rule_consistency_loss


def test_rule_consistency_loss_is_non_negative():
    concepts = {
        "mildew_texture": torch.tensor([0.9]),
        "lesion_area_ratio": torch.tensor([0.7]),
    }
    class_probs = torch.tensor([[0.1, 0.8, 0.1]])
    severity_probs = torch.tensor([[0.2, 0.3, 0.5]])

    loss = rule_consistency_loss(concepts, class_probs, severity_probs)

    assert float(loss) >= 0.0
