from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.data.dataset import _resolve_data_path


def _prepare_torch_load_compatibility() -> None:
    """Work around PyTorch 2.6+ weights_only default for trusted local mmengine checkpoints."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import torch
        from mmengine.logging.history_buffer import HistoryBuffer
    except ModuleNotFoundError:
        return
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([HistoryBuffer])


def _require_mmseg() -> tuple[object, object]:
    try:
        from mmseg.apis import inference_model, init_model
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "MMSegmentation dependencies are not installed. "
            "Install mmsegmentation, mmengine, and mmcv before running infer_mmseg.py."
        ) from exc
    return init_model, inference_model


def _portable_stem(raw_path: str) -> str:
    text = str(raw_path).replace("\\", "/").strip()
    parts = [part for part in text.split("/") if part]
    if not parts:
        return "sample"
    stem = "__".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return stem.replace(".", "_").replace(" ", "_")


def _filter_rows(dataframe: pd.DataFrame, split: str | None, source_dataset: str | None) -> pd.DataFrame:
    rows = dataframe.copy()
    if split and "split" in rows.columns:
        rows = rows[rows["split"].astype(str).str.lower() == split.lower()].copy()
    if source_dataset and "source_dataset" in rows.columns:
        rows = rows[rows["source_dataset"].astype(str).str.lower() == source_dataset.lower()].copy()
    return rows


def generate_predictions(
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    split: str | None = None,
    source_dataset: str | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Path:
    _prepare_torch_load_compatibility()
    init_model, inference_model = _require_mmseg()
    import plant_disease_mmseg.dataset  # noqa: F401

    dataframe = pd.read_csv(manifest_path)
    rows = _filter_rows(dataframe, split=split, source_dataset=source_dataset)
    if rows.empty:
        raise ValueError(f"No rows found for split={split!r} source_dataset={source_dataset!r}")

    model = init_model(str(config_path), str(checkpoint_path), device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    records = []
    manifest_dir = str(manifest_path.parent)
    for row in rows.to_dict(orient="records"):
        image_path = _resolve_data_path(row["image_path"], manifest_dir=manifest_dir)
        result = inference_model(model, image_path)
        pred = result.pred_sem_seg.data.squeeze().detach().cpu().numpy()
        binary_mask = (pred > 0).astype(np.uint8)
        if threshold > 0.5:
            binary_mask = (pred >= threshold).astype(np.uint8)
        mask_name = f"{_portable_stem(str(row['image_path']))}.png"
        mask_path = mask_dir / mask_name
        cv2.imwrite(str(mask_path), binary_mask * 255)
        records.append(
            {
                "image_path": image_path,
                "pred_mask_path": str(mask_path.resolve()),
                "split": str(row["split"]) if "split" in row else "",
                "source_dataset": str(row["source_dataset"]) if "source_dataset" in row else "",
                "threshold": float(threshold),
                "mask_area": int(binary_mask.sum()),
                "image_height": int(binary_mask.shape[0]),
                "image_width": int(binary_mask.shape[1]),
            }
        )

    predictions_path = output_dir / "predictions.csv"
    pd.DataFrame(records).to_csv(predictions_path, index=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": str(config_path),
                "checkpoint": str(checkpoint_path),
                "manifest": str(manifest_path),
                "split": split,
                "source_dataset": source_dataset,
                "num_predictions": len(records),
                "predictions_csv": str(predictions_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMSegmentation inference and export masks plus predictions.csv.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictions_path = generate_predictions(
        config_path=(ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config),
        checkpoint_path=(ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint),
        manifest_path=(ROOT / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest),
        output_dir=(ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir),
        split=args.split,
        source_dataset=args.source_dataset,
        device=args.device,
        threshold=args.threshold,
    )
    print(json.dumps({"predictions_csv": str(predictions_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
