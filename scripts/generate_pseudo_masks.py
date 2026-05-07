from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plant_disease.pseudo_labels.cam_pipeline import heatmap_to_mask


def save_demo_mask(output_dir: str = "artifacts/pseudo_masks"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap = np.array([[0.0, 0.9], [0.2, 0.7]], dtype=np.float32)
    mask = heatmap_to_mask(heatmap, threshold=0.5)
    np.save(out_dir / "demo_mask.npy", mask)


if __name__ == "__main__":
    save_demo_mask()
