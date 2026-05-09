from pathlib import Path

import cv2
import pandas as pd
import pytest
import torch

from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel
from plant_disease_mmseg.dataset import materialize_binary_mask
from plant_disease_mmseg.predictions import load_mmseg_predictions
from plant_disease_mmseg.replacement import forward_from_features_with_external_masks
from scripts.infer_mmseg import _prepare_torch_load_compatibility


def test_load_mmseg_predictions_validates_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "image_path": "leaf.png",
                "threshold": 0.5,
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_mmseg_predictions(csv_path)


def test_forward_from_features_with_external_masks_updates_mask_conditioned_heads() -> None:
    model = MultiTaskPlantDiseaseModel(
        in_channels=1,
        num_classes=2,
        num_severity_grades=2,
        fusion_mode="lesion_guided",
        use_concept_head=True,
        num_concepts=4,
    )
    with torch.no_grad():
        model.seg_head.weight.zero_()
        model.seg_head.bias.zero_()
        model.cls_head.fc.weight.copy_(torch.tensor([[0.0, -1.0], [0.0, 1.0]]))
        model.cls_head.fc.bias.zero_()
        model.sev_head.fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        model.sev_head.fc.bias.zero_()
        model.concept_head.fc.weight.fill_(1.0)
        model.concept_head.fc.bias.zero_()

    features = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    native_outputs = model.forward_from_features(features)
    external_masks = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)

    replaced_outputs = forward_from_features_with_external_masks(
        model,
        features,
        external_masks=external_masks,
        input_size=(2, 2),
    )

    assert not torch.allclose(
        native_outputs["classification_logits"],
        replaced_outputs["classification_logits"],
    )
    assert not torch.allclose(
        native_outputs["concept_logits"],
        replaced_outputs["concept_logits"],
    )
    assert torch.allclose(
        native_outputs["severity_logits"],
        replaced_outputs["severity_logits"],
    )


def test_materialize_binary_mask_converts_multivalue_labels(tmp_path: Path) -> None:
    raw_mask = tmp_path / "raw_mask.png"
    cache_dir = tmp_path / "cache"
    image = (
        torch.tensor(
            [
                [0, 1, 2],
                [3, 0, 4],
                [0, 0, 8],
            ],
            dtype=torch.uint8,
        )
        .cpu()
        .numpy()
    )
    cv2.imwrite(str(raw_mask), image)

    binary_mask_path = materialize_binary_mask(raw_mask, cache_root=cache_dir)
    binary_mask = cv2.imread(str(binary_mask_path), cv2.IMREAD_GRAYSCALE)

    assert binary_mask_path.exists()
    assert sorted(set(binary_mask.reshape(-1).tolist())) == [0, 1]


def test_prepare_torch_load_compatibility_sets_force_no_weights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    _prepare_torch_load_compatibility()

    assert __import__("os").environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
