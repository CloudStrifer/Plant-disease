import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.data.builders import build_segmentation_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--class-id", required=True, type=int)
    parser.add_argument("--source-dataset", default="PlantSeg")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = build_segmentation_manifest(
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        source_dataset=args.source_dataset,
        class_name=args.class_name,
        class_id=args.class_id,
        split=args.split,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
