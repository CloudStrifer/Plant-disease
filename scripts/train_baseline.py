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
from plant_disease.data.transforms import build_eval_transform, build_train_transform
from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel
from plant_disease.training.engine import evaluate_one_epoch, train_one_epoch
from plant_disease.training.losses import compute_multitask_loss


def resolve_runtime_path(value: str, config_path: Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    repo_candidate = repo_root / path
    if repo_candidate.exists() or len(path.parts) > 1:
        return repo_candidate

    return (config_path.parent / path).resolve()


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

    config_path = resolve_runtime_path(args.config, config_path=ROOT / "configs" / "baseline_segformer_b0.yaml", repo_root=ROOT)

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    manifest_path = resolve_runtime_path(config["train_manifest"], config_path=config_path, repo_root=ROOT)
    df = load_manifest(manifest_path)
    train_df = select_split(df, "train")
    val_df = select_split(df, "val")

    image_size = int(config.get("image_size", 256))
    train_ds = PlantDiseaseDataset(train_df, transform=build_train_transform(image_size=image_size))
    val_ds = PlantDiseaseDataset(val_df, transform=build_eval_transform(image_size=image_size))
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 4), shuffle=True, num_workers=config.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 4), shuffle=False, num_workers=config.get("num_workers", 0))

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(df, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))
    log_interval = int(config.get("log_interval", 10))

    output_dir = resolve_runtime_path(config.get("save_dir", "artifacts/baseline"), config_path=config_path, repo_root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"experiment={config.get('experiment_name', 'baseline')}", flush=True)
    print(f"config={config_path}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    print(f"device={device} image_size={image_size} batch_size={config.get('batch_size', 4)}", flush=True)
    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    epochs = int(config.get("epochs", 1))
    for epoch in range(epochs):
        print(f"epoch={epoch + 1}/{epochs} start", flush=True)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            compute_multitask_loss,
            device=device,
            log_interval=log_interval,
            log_prefix="train",
        )
        val_loss = evaluate_one_epoch(
            model,
            val_loader,
            compute_multitask_loss,
            device=device,
            log_interval=log_interval,
            log_prefix="val",
        )
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

    torch.save(model.state_dict(), output_dir / "baseline_last.pt")
    print(f"Saved checkpoint to {output_dir / 'baseline_last.pt'}")


if __name__ == "__main__":
    main()
