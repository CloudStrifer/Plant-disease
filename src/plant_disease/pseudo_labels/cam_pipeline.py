import numpy as np


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value == min_value:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - min_value) / (max_value - min_value)).astype(np.float32)


def heatmap_to_mask(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    norm = normalize_heatmap(heatmap)
    return (norm >= threshold).astype(np.uint8)
