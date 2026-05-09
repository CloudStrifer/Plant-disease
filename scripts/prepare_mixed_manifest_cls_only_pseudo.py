"""Create a mixed manifest where pseudo-labeled samples keep classification supervision only.

This script is a conservative follow-up experiment for weak supervision:
- Real PlantSeg rows keep lesion-mask supervision.
- PlantVillage pseudo rows keep class labels but disable lesion-mask supervision.
"""

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
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_csv(input_path)

    if "source_dataset" not in df.columns:
        raise KeyError("Expected 'source_dataset' column in mixed manifest")
    if "has_lesion_mask" not in df.columns:
        raise KeyError("Expected 'has_lesion_mask' column in mixed manifest")

    pseudo_mask = df["source_dataset"].astype(str).str.lower() == args.pseudo_source.lower()
    df.loc[pseudo_mask, "has_lesion_mask"] = False
    if "lesion_mask_path" in df.columns:
        df.loc[pseudo_mask, "lesion_mask_path"] = ""
    if "mask_path" in df.columns:
        df.loc[pseudo_mask, "mask_path"] = ""
    df["seg_loss_weight"] = 1.0
    df.loc[pseudo_mask, "seg_loss_weight"] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"Saved manifest to {output_path} total_rows={len(df)} "
        f"pseudo_rows={int(pseudo_mask.sum())} pseudo_seg_weight=0.0"
    )


if __name__ == "__main__":
    main()
