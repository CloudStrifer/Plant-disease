from __future__ import annotations

import cv2
import numpy as np


def _build_resize_transform(image_size: int):
    def apply(image: np.ndarray, mask: np.ndarray | None = None):
        resized_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        resized_mask = None
        if mask is not None:
            resized_mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        return resized_image, resized_mask

    return apply


def build_train_transform(image_size: int = 256):
    return _build_resize_transform(image_size)


def build_eval_transform(image_size: int = 256):
    return _build_resize_transform(image_size)
