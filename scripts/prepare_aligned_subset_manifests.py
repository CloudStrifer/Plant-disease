"""Create aligned subset manifests from taxonomy audit canonical matches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.data.taxonomy_audit import build_aligned_manifests, save_aligned_manifests


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-manifest", required=True, help="Left manifest csv, e.g. PlantSeg")
    parser.add_argument("--right-manifest", required=True, help="Right manifest csv, e.g. PlantVillage pseudo")
    parser.add_argument("--matches-csv", required=True, help="canonical_matches.csv from taxonomy audit")
    parser.add_argument("--output-dir", required=True, help="Directory for aligned outputs")
    args = parser.parse_args()

    left_manifest = pd.read_csv(_resolve_path(args.left_manifest))
    right_manifest = pd.read_csv(_resolve_path(args.right_manifest))
    matches_df = pd.read_csv(_resolve_path(args.matches_csv))

    outputs = build_aligned_manifests(left_manifest, right_manifest, matches_df)
    output_dir = _resolve_path(args.output_dir)
    save_aligned_manifests(outputs, output_dir)

    print(
        f"Saved aligned manifests to {output_dir} "
        f"class_count={len(outputs['class_map'])} "
        f"left_rows={len(outputs['left_aligned'])} "
        f"right_rows={len(outputs['right_aligned'])} "
        f"mixed_rows={len(outputs['mixed_aligned'])}"
    )


if __name__ == "__main__":
    main()
