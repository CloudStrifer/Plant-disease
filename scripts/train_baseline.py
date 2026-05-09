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
from plant_disease.concepts.rules import CONCEPT_NAMES
from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel
from plant_disease.training.engine import evaluate_one_epoch, train_one_epoch
from plant_disease.training.losses import compute_multitask_loss

BACKBONE_DEFAULT_IN_CHANNELS = {
    "resnet18": 512,
    "segformer_b0": 256,
}


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


def resolve_model_in_channels(config: dict) -> int:
    if "in_channels" in config and config["in_channels"] is not None:
        return int(config["in_channels"])
    backbone_name = config.get("backbone_name", "resnet18")
    if backbone_name not in BACKBONE_DEFAULT_IN_CHANNELS:
        supported = ", ".join(sorted(BACKBONE_DEFAULT_IN_CHANNELS))
        raise ValueError(f"Unsupported backbone_name {backbone_name!r}. Supported backbones: {supported}")
    return BACKBONE_DEFAULT_IN_CHANNELS[backbone_name]


def build_model(df: pd.DataFrame, config: dict) -> MultiTaskPlantDiseaseModel:
    num_classes = int(df["class_id"].max()) + 1
    if "num_severity_grades" in config:
        num_severity_grades = max(int(config["num_severity_grades"]), 1)
    else:
        valid_severity = df[df["severity_label"] >= 0]["severity_label"]
        num_severity_grades = max(int(valid_severity.max()) + 1, 1) if not valid_severity.empty else 1
    return MultiTaskPlantDiseaseModel(
        in_channels=resolve_model_in_channels(config),
        num_classes=num_classes,
        num_severity_grades=num_severity_grades,
        backbone_name=config.get("backbone_name", "resnet18"),
        fusion_mode=config.get("fusion_mode", "lesion_guided"),
        use_concept_head=bool(config.get("use_concept_head", False)),
        num_concepts=int(config.get("num_concepts", len(CONCEPT_NAMES))),
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
    init_checkpoint = config.get("init_checkpoint")
    if init_checkpoint:
        init_checkpoint_path = resolve_runtime_path(init_checkpoint, config_path=config_path, repo_root=ROOT)
        state = torch.load(init_checkpoint_path, map_location=device)
        model_state = model.state_dict()
        compatible_state = {}
        skipped = []
        for key, value in state.items():
            if key not in model_state:
                skipped.append((key, "missing_in_model"))
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                skipped.append(
                    (key, f"shape_mismatch ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}")
                )
                continue
            compatible_state[key] = value

        model.load_state_dict(compatible_state, strict=False)
        print(
            f"Initialized from checkpoint={init_checkpoint} "
            f"loaded={len(compatible_state)} skipped={len(skipped)}"
        )
        if skipped:
            preview = ", ".join(f"{name} ({reason})" for name, reason in skipped[:8])
            print(f"Skipped checkpoint params: {preview}")
        print(f"loaded_checkpoint={init_checkpoint_path}", flush=True)
    learning_rate = float(config.get("learning_rate", config.get("lr", 1e-4)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log_interval = int(config.get("log_interval", 10))
    seg_weight = float(config.get("seg_weight", 1.0))
    sev_weight = float(config.get("sev_weight", 1.0))
    concept_weight = float(config.get("concept_weight", 0.0))
    rule_weight = float(config.get("rule_weight", 0.0))
    concept_real_only = bool(config.get("concept_real_only", False))
    class_names = [str(name) for name in df.sort_values("class_id")["class_name"].drop_duplicates().tolist()] if "class_name" in df.columns else []

    def configured_loss_fn(outputs, batch):
        return compute_multitask_loss(
            outputs,
            batch,
            seg_weight=seg_weight,
            sev_weight=sev_weight,
            concept_weight=concept_weight,
            rule_weight=rule_weight,
            concept_real_only=concept_real_only,
            class_names=class_names,
        )

    output_dir = resolve_runtime_path(config.get("save_dir", "artifacts/baseline"), config_path=config_path, repo_root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"experiment={config.get('experiment_name', 'baseline')}", flush=True)
    print(f"config={config_path}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    print(f"device={device} image_size={image_size} batch_size={config.get('batch_size', 4)}", flush=True)
    print(f"learning_rate={learning_rate}", flush=True)
    valid_severity_count = int((df["severity_label"] >= 0).sum()) if "severity_label" in df.columns else 0
    print(
        f"num_severity_grades={model.sev_head.fc.out_features} valid_severity_samples={valid_severity_count}",
        flush=True,
    )
    print(
        f"fusion_mode={config.get('fusion_mode', 'lesion_guided')} "
        f"seg_weight={seg_weight} sev_weight={sev_weight} "
        f"concept_weight={concept_weight} rule_weight={rule_weight} "
        f"concept_real_only={concept_real_only} "
        f"use_concept_head={bool(config.get('use_concept_head', False))}",
        flush=True,
    )
    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    epochs = int(config.get("epochs", 1))
    for epoch in range(epochs):
        print(f"epoch={epoch + 1}/{epochs} start", flush=True)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            configured_loss_fn,
            device=device,
            log_interval=log_interval,
            log_prefix="train",
        )
        val_loss = evaluate_one_epoch(
            model,
            val_loader,
            configured_loss_fn,
            device=device,
            log_interval=log_interval,
            log_prefix="val",
        )
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

    torch.save(model.state_dict(), output_dir / "baseline_last.pt")
    print(f"Saved checkpoint to {output_dir / 'baseline_last.pt'}")


if __name__ == "__main__":
    main()
