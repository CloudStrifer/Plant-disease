from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import pandas as pd

from plant_disease.data.dataset import _resolve_data_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def materialize_binary_mask(mask_path: str | Path, cache_root: str | Path | None = None) -> Path:
    source = Path(mask_path)
    if cache_root is None:
        cache_root = REPO_ROOT / "artifacts" / "mmseg" / "binary_masks"
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.md5(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    target = cache_dir / f"{source.stem}_{digest}.png"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read segmentation mask: {source}")
    binary = (mask > 0).astype("uint8")
    cv2.imwrite(str(target), binary)
    return target

try:
    from mmseg.datasets import BaseSegDataset
    from mmseg.registry import DATASETS
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in mmseg-enabled envs
    BaseSegDataset = object  # type: ignore[assignment]
    DATASETS = None
    MMSEG_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised only in mmseg-enabled envs
    MMSEG_IMPORT_ERROR = None


if DATASETS is not None:  # pragma: no cover - exercised only in mmseg-enabled envs
    @DATASETS.register_module()
    class PlantDiseaseBinarySegDataset(BaseSegDataset):
        METAINFO = dict(
            classes=("background", "lesion"),
            palette=[[0, 0, 0], [255, 255, 255]],
        )

        def __init__(
            self,
            manifest_path: str | Path,
            split: str | None = None,
            source_dataset: str | None = None,
            **kwargs,
        ) -> None:
            self.manifest_path = Path(manifest_path)
            self.split = split
            self.source_dataset = source_dataset
            super().__init__(
                img_suffix="",
                seg_map_suffix="",
                reduce_zero_label=False,
                **kwargs,
            )

        def load_data_list(self) -> list[dict]:
            dataframe = pd.read_csv(self.manifest_path)
            if self.split and "split" in dataframe.columns:
                dataframe = dataframe[dataframe["split"].astype(str).str.lower() == self.split.lower()].copy()
            if self.source_dataset and "source_dataset" in dataframe.columns:
                dataframe = dataframe[
                    dataframe["source_dataset"].astype(str).str.lower() == self.source_dataset.lower()
                ].copy()

            if "has_lesion_mask" not in dataframe.columns:
                raise KeyError("Expected has_lesion_mask column in manifest")
            dataframe = dataframe[dataframe["has_lesion_mask"].astype(bool)].copy()
            if "lesion_mask_path" not in dataframe.columns:
                raise KeyError("Expected lesion_mask_path column in manifest")

            manifest_dir = str(self.manifest_path.parent)
            data_list: list[dict] = []
            for row in dataframe.to_dict(orient="records"):
                img_path = _resolve_data_path(row["image_path"], manifest_dir=manifest_dir)
                raw_seg_map_path = _resolve_data_path(row["lesion_mask_path"], manifest_dir=manifest_dir)
                seg_map_path = str(materialize_binary_mask(raw_seg_map_path))
                data_list.append(
                    {
                        "img_path": img_path,
                        "seg_map_path": seg_map_path,
                        "label_map": {0: 0, 255: 1},
                        "reduce_zero_label": False,
                        "seg_fields": [],
                    }
                )
            return data_list

else:
    class PlantDiseaseBinarySegDataset:  # pragma: no cover - import guard only
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "MMSegmentation is required to use PlantDiseaseBinarySegDataset"
            ) from MMSEG_IMPORT_ERROR
