"""Create a mixed manifest with reduced segmentation supervision on pseudo labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input mixed manifest csv")
    parser.add_argument("--output", required=True, help="Output manifest csv")
    parser.add_argument(
        "--pseudo-source",
        default="PlantVillage_pseudo",
        help="source_dataset value to treat as pseudo-label rows",
    )
    parser.add_argument(
        "--pseudo-seg-weight",
        type=float,
        required=True,
        help="Segmentation loss weight to assign to pseudo-label rows, e.g. 0.1 or 0.3",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_csv(input_path)

    if "source_dataset" not in df.columns:
        raise KeyError("Expected 'source_dataset' column in mixed manifest")

    pseudo_mask = df["source_dataset"].astype(str).str.lower() == args.pseudo_source.lower()
    df["seg_loss_weight"] = 1.0
    df.loc[pseudo_mask, "seg_loss_weight"] = float(args.pseudo_seg_weight)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"Saved manifest to {output_path} total_rows={len(df)} "
        f"pseudo_rows={int(pseudo_mask.sum())} pseudo_seg_weight={args.pseudo_seg_weight}"
    )


if __name__ == "__main__":
    main()
