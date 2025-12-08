"""Utilities for scoring and selecting candidate masks from the pool."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage import color, filters, measure, segmentation, util
from sklearn.cluster import KMeans

__all__ = ["pick_obj", "score_mask"]


def _extract_features(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    conv_quality: float,
    target_area_ratio: float,
) -> Dict[str, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[:2] != img_rgb.shape[:2]:
        if mask_bool.size == np.prod(img_rgb.shape[:2]):
            mask_bool = mask_bool.reshape(img_rgb.shape[:2])
        else:
            mask_bool = np.resize(mask_bool, img_rgb.shape[:2])

    area_pixels = float(mask_bool.sum())
    total_pixels = float(mask_bool.size) if mask_bool.size else 1.0
    area_ratio = area_pixels / total_pixels

    target = float(target_area_ratio)
    area_score = -abs(area_ratio - target)

    perimeter = float(measure.perimeter(mask_bool.astype(np.uint8), neighborhood=8)) if area_pixels > 0 else 0.0
    if perimeter <= 0.0:
        circularity = 0.0
    else:
        circularity = float(np.clip((4.0 * np.pi * area_pixels) / (perimeter ** 2), 0.0, 1.0))

    image_float = util.img_as_float(img_rgb)
    gray = color.rgb2gray(image_float)
    gradient = filters.sobel(gray)
    boundary = segmentation.find_boundaries(mask_bool, mode="inner")
    boundary_values = gradient[boundary]
    edge_quality = float(boundary_values.mean()) if boundary_values.size else 0.0

    lab_image = color.rgb2lab(image_float)
    lab_pixels = lab_image[mask_bool]
    if lab_pixels.shape[0] < 2:
        bcs = 0.0
    else:
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=0)
        labels = kmeans.fit_predict(lab_pixels)
        centers = kmeans.cluster_centers_
        diff = np.linalg.norm(centers[0] - centers[1])

        def _cluster_trace(values: np.ndarray) -> float:
            if values.shape[0] <= 1:
                return 0.0
            cov = np.cov(values, rowvar=False)
            if np.isscalar(cov):
                return float(cov)
            return float(np.trace(cov))

        trace0 = _cluster_trace(lab_pixels[labels == 0])
        trace1 = _cluster_trace(lab_pixels[labels == 1])
        denom = np.sqrt(trace0) + np.sqrt(trace1) + 1e-6
        bcs = float(diff / denom)

    return {
        "area": area_score,
        "area_ratio": area_ratio,
        "edge": edge_quality,
        "circularity": circularity,
        "conv_quality": float(conv_quality),
        "bcs": bcs,
    }


def _build_pool_stats(
    feature_list: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    if not feature_list:
        return {}

    keys = ["area", "edge", "circularity", "conv_quality", "bcs"]
    stats: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [feat[key] for feat in feature_list]
        stats[key] = {"min": float(np.min(values)), "max": float(np.max(values))}
    return stats


def _min_max_normalise(value: float, bounds: Dict[str, float]) -> float:
    minimum = bounds.get("min", 0.0)
    maximum = bounds.get("max", 0.0)
    if maximum - minimum < 1e-6:
        return 1.0
    return (value - minimum) / (maximum - minimum)


def score_mask(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    conv_quality: float,
    pool_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    target_area_ratio = float(pool_stats.get("target_area_ratio", 0.65))
    features = _extract_features(img_rgb, mask, conv_quality, target_area_ratio)

    area_n = _min_max_normalise(features["area"], pool_stats.get("area", {"min": 0.0, "max": 0.0}))
    edge_n = _min_max_normalise(features["edge"], pool_stats.get("edge", {"min": 0.0, "max": 0.0}))
    circ_n = _min_max_normalise(features["circularity"], pool_stats.get("circularity", {"min": 0.0, "max": 0.0}))
    conv_n = _min_max_normalise(features["conv_quality"], pool_stats.get("conv_quality", {"min": 0.0, "max": 0.0}))
    bcs_n = _min_max_normalise(features["bcs"], pool_stats.get("bcs", {"min": 0.0, "max": 0.0}))

    score = float(area_n + edge_n + circ_n + conv_n + bcs_n)

    return {
        "score": score,
        "area_n": area_n,
        "edge_n": edge_n,
        "circ_n": circ_n,
        "conv_quality_n": conv_n,
        "bcs_n": bcs_n,
        "raw": features,
    }


def pick_obj(
    img_rgb: np.ndarray,
    pool: List[Dict[str, object]],
    *,
    target_area_ratio: float = 0.05,
) -> Tuple[Optional[Dict[str, object]], Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    if not pool:
        return None, {}, []

    feature_list = [
        _extract_features(
            img_rgb,
            entry["mask"],
            float(entry.get("conv_quality", entry.get("score", 0.0))),
            target_area_ratio,
        )
        for entry in pool
    ]
    pool_stats = _build_pool_stats(feature_list)
    pool_stats["target_area_ratio"] = target_area_ratio
    pool_stats["raw_features"] = feature_list

    best_entry: Optional[Dict[str, object]] = None
    best_score = float("-inf")
    scored_details: List[Dict[str, float]] = []

    for entry in pool:
        entry_score = score_mask(
            img_rgb,
            entry["mask"],
            float(entry.get("conv_quality", entry.get("score", 0.0))),
            pool_stats,
        )
        entry["score_details"] = entry_score
        scored_details.append(entry_score)
        if entry_score["score"] > best_score:
            best_score = entry_score["score"]
            best_entry = entry

    return best_entry, pool_stats, scored_details
