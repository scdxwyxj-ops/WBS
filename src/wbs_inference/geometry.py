"""Image/bounding-box coordinate helpers."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def validate_bbox(bbox_xyxy: Sequence[float], image_shape: Sequence[int]) -> BBox:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain exactly four values")
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
    x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox after clipping: {(x1, y1, x2, y2)} for image {(height, width)}")
    return x1, y1, x2, y2


def crop_image(image: np.ndarray, bbox_xyxy: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    if crop.size == 0:
        raise ValueError("Bounding-box crop is empty")
    return crop


def paste_crop_mask(mask: np.ndarray, original_shape: Sequence[int], bbox_xyxy: BBox) -> np.ndarray:
    height, width = int(original_shape[0]), int(original_shape[1])
    x1, y1, x2, y2 = bbox_xyxy
    crop_h, crop_w = y2 - y1, x2 - x1
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != (crop_h, crop_w):
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST) > 0
    full = np.zeros((height, width), dtype=bool)
    full[y1:y2, x1:x2] = mask_bool
    return full
