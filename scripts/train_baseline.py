import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.data.dataset import PlantDiseaseDataset
from plant_disease.data.manifest import load_manifest
from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel
from plant_disease.training.engine import evaluate_one_epoch, train_one_epoch
from plant_disease.training.losses import compute_multitask_loss


def select_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "split" not in df.columns:
        return df.copy()
    filtered = df[df["split"] == split_name].copy()
    return filtered if not filtered.empty else df.copy()


def build_model(df: pd.DataFrame, config: dict) -> MultiTaskPlantDiseaseModel:
    num_classes = int(df["class_id"].max()) + 1
    num_severity_grades = max(int(df["severity_label"].max()) + 1, 1)
    return MultiTaskPlantDiseaseModel(
        in_channels=512,
        num_classes=num_classes,
        num_severity_grades=num_severity_grades,
        backbone_name=config.get("backbone_name", "resnet18"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_segformer_b0.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    manifest_path = config["train_manifest"]
    df = load_manifest(manifest_path)
    train_df = select_split(df, "train")
    val_df = select_split(df, "val")

    train_ds = PlantDiseaseDataset(train_df, transform=None)
    val_ds = PlantDiseaseDataset(val_df, transform=None)
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 4), shuffle=True, num_workers=config.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 4), shuffle=False, num_workers=config.get("num_workers", 0))

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(df, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))

    output_dir = Path(config.get("save_dir", "artifacts/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(config.get("epochs", 1))
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, compute_multitask_loss, device=device)
        val_loss = evaluate_one_epoch(model, val_loader, compute_multitask_loss, device=device)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), output_dir / "baseline_last.pt")
    print(f"Saved checkpoint to {output_dir / 'baseline_last.pt'}")


if __name__ == "__main__":
    main()
