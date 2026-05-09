import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from plant_disease.data.builders import annotate_severity_columns
from plant_disease.data.dataset import _resolve_data_path
from plant_disease.data.manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute severity labels from lesion masks for an existing manifest.")
    parser.add_argument("--input", required=True, help="Input manifest CSV")
    parser.add_argument("--output", required=True, help="Output manifest CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    dataframe = load_manifest(input_path).copy()

    if "lesion_mask_path" in dataframe.columns:
        dataframe["lesion_mask_path"] = dataframe.apply(
            lambda row: _resolve_data_path(row["lesion_mask_path"], manifest_dir=row["__manifest_dir"])
            if pd.notna(row["lesion_mask_path"])
            else row["lesion_mask_path"],
            axis=1,
        )

    recomputed = annotate_severity_columns(dataframe.drop(columns=["__manifest_dir"], errors="ignore"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recomputed.to_csv(output_path, index=False)
    valid_count = int(recomputed["has_valid_severity"].sum()) if "has_valid_severity" in recomputed.columns else 0
    print(f"Saved {len(recomputed)} rows with {valid_count} valid severity labels to {output_path}")


if __name__ == "__main__":
    main()
