import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.data.builders import build_classification_manifest


def add_stratified_splits(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int) -> pd.DataFrame:
    rng = pd.Series(range(len(df)), index=df.index).sample(frac=1.0, random_state=seed)
    shuffled = df.loc[rng.index].reset_index(drop=True)
    parts = []
    for _, group in shuffled.groupby("class_id", sort=False):
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n - n_train > 1 else 0
        group = group.copy()
        group["split"] = "test"
        group.iloc[:n_train, group.columns.get_loc("split")] = "train"
        if n_val:
            group.iloc[n_train:n_train + n_val, group.columns.get_loc("split")] = "val"
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="PlantVillage class-directory root, e.g. raw/color")
    parser.add_argument("--output", required=True, help="Output manifest CSV path")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, class_to_idx = build_classification_manifest(Path(args.root), source_dataset="PlantVillage", split="train")
    df = add_stratified_splits(df, args.train_ratio, args.val_ratio, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    classes_path = output.with_suffix(".classes.csv")
    pd.DataFrame({"class_name": list(class_to_idx.keys()), "class_id": list(class_to_idx.values())}).to_csv(classes_path, index=False)
    print(f"Saved {len(df)} rows to {output}")
    print(f"Saved class map to {classes_path}")


if __name__ == "__main__":
    main()
