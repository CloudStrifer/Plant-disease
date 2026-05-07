import numpy as np

from plant_disease.pseudo_labels.cam_pipeline import heatmap_to_mask


def test_heatmap_to_mask_thresholds_activation():
    heatmap = np.array([[0.2, 0.9], [0.1, 0.8]], dtype=np.float32)
    mask = heatmap_to_mask(heatmap, threshold=0.5)

    assert mask.tolist() == [[0, 1], [0, 1]]
