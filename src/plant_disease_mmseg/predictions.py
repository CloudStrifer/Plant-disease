from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_PREDICTION_COLUMNS = {
    "image_path",
    "pred_mask_path",
}


def normalize_prediction_key(path: str | Path) -> str:
    value = str(path).strip().replace("\\", "/")
    candidate = Path(value)
    if candidate.exists():
        candidate = candidate.resolve()
        value = candidate.as_posix()
    return value.lower()


def load_mmseg_predictions(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    missing = REQUIRED_PREDICTION_COLUMNS.difference(dataframe.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_text}")
    return dataframe.copy()


def build_prediction_index(dataframe: pd.DataFrame) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for record in dataframe.to_dict(orient="records"):
        image_key = normalize_prediction_key(record["image_path"])
        index[image_key] = record
    return index


def resolve_prediction_record(image_path: str | Path, prediction_index: dict[str, dict]) -> dict:
    key = normalize_prediction_key(image_path)
    if key not in prediction_index:
        raise KeyError(f"Missing mmseg prediction for image_path={image_path}")
    return prediction_index[key]

