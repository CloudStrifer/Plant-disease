import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from src.plant_disease.data.builders import build_plantseg_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="PlantSeg root containing images/, annotations/, and Metadatav2.csv")
    parser.add_argument("--output", required=True, help="Output manifest CSV path")
    args = parser.parse_args()

    df, class_to_idx = build_plantseg_manifest(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    pd.DataFrame({"class_name": list(class_to_idx.keys()), "class_id": list(class_to_idx.values())}).to_csv(
        output.with_suffix(".classes.csv"),
        index=False,
    )
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
