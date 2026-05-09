from pathlib import Path, PureWindowsPath
import sys
import argparse

import numpy as np
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
from plant_disease.data.transforms import build_eval_transform
from plant_disease.models.multitask_model import MultiTaskPlantDiseaseModel
from plant_disease.pseudo_labels.cam_pipeline import heatmap_to_mask


def portable_stem(path_text: str) -> str:
    text = str(path_text).strip()
    if "\\" in text and "/" not in text:
        return PureWindowsPath(text).stem
    normalized = text.replace("\\", "/")
    return Path(normalized).stem


def save_demo_mask(output_dir: str = "artifacts/pseudo_masks"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap = np.array([[0.0, 0.9], [0.2, 0.7]], dtype=np.float32)
    mask = heatmap_to_mask(heatmap, threshold=0.5)
    np.save(out_dir / "demo_mask.npy", mask)


def save_mask_png(mask: np.ndarray, output_path: Path):
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), (mask * 255).astype(np.uint8))


def generate_pseudo_masks(
    manifest_csv: str,
    checkpoint_path: str,
    output_dir: str,
    image_size: int = 256,
    batch_size: int = 8,
    device: str | None = None,
    threshold: float = 0.5,
):
    df = load_manifest(manifest_csv)
    dataset = PlantDiseaseDataset(df, transform=build_eval_transform(image_size=image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = int(df["class_id"].max()) + 1
    num_severity_grades = max(int(df["severity_label"].max()) + 1, 1)
    model = MultiTaskPlantDiseaseModel(
        in_channels=512,
        num_classes=num_classes,
        num_severity_grades=num_severity_grades,
        backbone_name="resnet18",
        fusion_mode="global",
    )
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=run_device)
    model.load_state_dict(state, strict=False)
    model.to(run_device)
    model.eval()

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    records = []
    sample_index = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(run_device)
            features = model.extract_features(images)
            class_indices = batch["class_id"].to(run_device)
            cams = model.classification_cam(features, class_indices)
            for row_index in range(images.shape[0]):
                heatmap = cams[row_index].detach().cpu().numpy()
                mask = heatmap_to_mask(heatmap, threshold=threshold)
                row = df.iloc[sample_index]
                mask_name = f"{portable_stem(row['image_path'])}.png"
                mask_path = output_root / mask_name
                save_mask_png(mask, mask_path)
                record = {key: value for key, value in row.to_dict().items() if not str(key).startswith("__")}
                record["has_lesion_mask"] = True
                record["lesion_mask_path"] = str(mask_path)
                record["source_dataset"] = f"{row['source_dataset']}_pseudo"
                record["pseudo_label"] = True
                records.append(record)
                sample_index += 1

    pseudo_manifest = pd.DataFrame(records)
    manifest_out = output_root / "pseudo_manifest.csv"
    pseudo_manifest.to_csv(manifest_out, index=False)
    return manifest_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", help="Classification manifest CSV to pseudo-label")
    parser.add_argument("--checkpoint", help="Classification checkpoint path")
    parser.add_argument("--output-dir", default="artifacts/pseudo_masks")
    parser.add_argument("--config", help="Optional YAML config for image_size/device/batch_size/threshold")
    args = parser.parse_args()

    if not args.manifest or not args.checkpoint:
        save_demo_mask()
    else:
        image_size = 256
        batch_size = 8
        device = None
        threshold = 0.5
        if args.config:
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            image_size = int(cfg.get("image_size", image_size))
            batch_size = int(cfg.get("batch_size", batch_size))
            device = cfg.get("device", device)
            threshold = float(cfg.get("pseudo_label_threshold", threshold))
        manifest_out = generate_pseudo_masks(
            manifest_csv=args.manifest,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
        )
        print(f"saved_pseudo_manifest={manifest_out}")
