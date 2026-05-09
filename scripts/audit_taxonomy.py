"""Audit cross-dataset class taxonomy overlap and mismatches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.data.taxonomy_audit import build_class_inventory, compare_taxonomies, save_audit_report


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left manifest csv")
    parser.add_argument("--right", required=True, help="Right manifest csv")
    parser.add_argument("--left-source", default=None, help="Optional left source_dataset filter")
    parser.add_argument("--right-source", default=None, help="Optional right source_dataset filter")
    parser.add_argument("--output-dir", required=True, help="Directory to save audit report")
    parser.add_argument("--min-ambiguous-score", type=float, default=0.5, help="Token Jaccard threshold for possible synonyms")
    args = parser.parse_args()

    left_df = pd.read_csv(_resolve_path(args.left))
    right_df = pd.read_csv(_resolve_path(args.right))

    if args.left_source and "source_dataset" in left_df.columns:
        left_df = left_df[left_df["source_dataset"].astype(str).str.lower() == args.left_source.lower()].copy()
    if args.right_source and "source_dataset" in right_df.columns:
        right_df = right_df[right_df["source_dataset"].astype(str).str.lower() == args.right_source.lower()].copy()

    left_inventory = build_class_inventory(left_df)
    right_inventory = build_class_inventory(right_df)
    report = compare_taxonomies(left_inventory, right_inventory, min_ambiguous_score=args.min_ambiguous_score)
    save_audit_report(report, _resolve_path(args.output_dir))

    summary = report["summary"]
    print(summary.to_json())
    print(f"Saved taxonomy audit to {_resolve_path(args.output_dir)}")


if __name__ == "__main__":
    main()
