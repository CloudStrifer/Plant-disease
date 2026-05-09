import numpy as np
import pandas as pd

from scripts.generate_pseudo_masks import portable_stem, save_mask_png
from plant_disease.pseudo_labels.cam_pipeline import heatmap_to_mask


def test_heatmap_to_mask_thresholds_activation():
    heatmap = np.array([[0.2, 0.9], [0.1, 0.8]], dtype=np.float32)
    mask = heatmap_to_mask(heatmap, threshold=0.5)

    assert mask.tolist() == [[0, 1], [0, 1]]


def test_save_mask_png_writes_file(tmp_path):
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    output = tmp_path / "mask.png"

    save_mask_png(mask, output)

    assert output.exists()


def test_portable_stem_supports_windows_style_paths():
    path_text = r"E:\Multi_modal_Code\Plant disease\src\plant_disease\dataset\plantvillage\color\Tomato___Leaf_Mold\abc123.JPG"
    assert portable_stem(path_text) == "abc123"
