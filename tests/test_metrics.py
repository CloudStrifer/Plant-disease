import torch

from plant_disease.eval.metrics import dice_score, mask_to_box, severity_ratio


def test_dice_score_matches_identical_masks():
    mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert dice_score(mask, mask) == 1.0


def test_severity_ratio_uses_lesion_over_leaf_area():
    lesion = torch.tensor([[1, 1], [0, 0]])
    leaf = torch.tensor([[1, 1], [1, 1]])
    assert severity_ratio(lesion, leaf) == 0.5


def test_mask_to_box_returns_xyxy_bounds():
    mask = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    assert mask_to_box(mask) == (1, 0, 2, 1)
