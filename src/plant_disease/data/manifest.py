from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "image_path",
    "class_id",
    "has_lesion_mask",
    "has_leaf_mask",
    "source_dataset",
    "severity_label",
}


def load_manifest(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"missing required columns: {missing_text}")
    df = df.copy()
    df["__manifest_dir"] = str(csv_path.resolve().parent)
    return df
