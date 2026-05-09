import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.data.builders import build_mixed_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--pseudo-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = build_mixed_manifest(args.real_manifest, args.pseudo_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
