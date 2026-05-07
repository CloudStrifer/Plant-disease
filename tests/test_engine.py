import pandas as pd
import torch
from torch.utils.data import DataLoader

from plant_disease.data.dataset import PlantDiseaseDataset
from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel
from plant_disease.training.engine import train_one_epoch
from plant_disease.training.losses import compute_multitask_loss


def test_train_one_epoch_runs_on_toy_batch(tmp_path):
    image = torch.zeros((64, 64, 3), dtype=torch.uint8).numpy()
    import cv2

    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), image)

    manifest = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "class_id": 0,
                "has_lesion_mask": False,
                "has_leaf_mask": False,
                "source_dataset": "unit",
                "severity_label": 0,
            }
        ]
    )

    dataset = PlantDiseaseDataset(manifest, transform=None)
    loader = DataLoader(dataset, batch_size=1)
    model = MultiTaskPlantDiseaseModel(
        in_channels=512,
        num_classes=1,
        num_severity_grades=1,
        backbone_name="resnet18",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_value = train_one_epoch(model, loader, optimizer, compute_multitask_loss, device="cpu")

    assert loss_value >= 0.0


def test_train_one_epoch_emits_progress_logs(tmp_path, capsys):
    image = torch.zeros((64, 64, 3), dtype=torch.uint8).numpy()
    import cv2

    rows = []
    for idx in range(2):
        image_path = tmp_path / f"sample_{idx}.png"
        cv2.imwrite(str(image_path), image)
        rows.append(
            {
                "image_path": str(image_path),
                "class_id": 0,
                "has_lesion_mask": False,
                "has_leaf_mask": False,
                "source_dataset": "unit",
                "severity_label": 0,
            }
        )

    manifest = pd.DataFrame(rows)
    dataset = PlantDiseaseDataset(manifest, transform=None)
    loader = DataLoader(dataset, batch_size=1)
    model = MultiTaskPlantDiseaseModel(
        in_channels=512,
        num_classes=1,
        num_severity_grades=1,
        backbone_name="resnet18",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_one_epoch(
        model,
        loader,
        optimizer,
        compute_multitask_loss,
        device="cpu",
        log_interval=1,
        log_prefix="train",
    )
    captured = capsys.readouterr()

    assert "train step=1/2" in captured.out
