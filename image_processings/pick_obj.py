"""Mask selection utilities: heuristic, entropy (logits), edge-gradient scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage import color, filters, measure, segmentation, util
from sklearn.cluster import KMeans

MaskEntry = Dict[str, Any]

__all__ = [
    "pick_obj_using_heuristic",
    "pick_obj_using_entropy",
    "pick_obj_using_edge_gradient",
    "score_mask",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Heuristic scoring (existing)
# --------------------------------------------------------------------------- #
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


def pick_obj_using_heuristic(
    img_rgb: np.ndarray,
    pool: List[MaskEntry],
    *,
    target_area_ratio: float = 0.05,
) -> Tuple[Optional[MaskEntry], Dict[str, Dict[str, float]], List[Dict[str, float]]]:
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


# --------------------------------------------------------------------------- #
# Entropy scoring (logits -> softmax/sigmoid)
# --------------------------------------------------------------------------- #
def _foreground_probabilities(logits: np.ndarray) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=np.float32)
    if logits_arr.ndim == 2:
        return 1.0 / (1.0 + np.exp(-logits_arr))

    if logits_arr.ndim == 3:
        # If channel-first (C,H,W)
        if logits_arr.shape[0] > 1:
            max_logits = np.max(logits_arr, axis=0, keepdims=True)
            exp_logits = np.exp(logits_arr - max_logits)
            probs = exp_logits / (np.sum(exp_logits, axis=0, keepdims=True) + 1e-9)
            # take foreground channel if present else max
            if probs.shape[0] > 1:
                return probs[1]
            return probs[0]
        # If channel-last (H,W,C)
        if logits_arr.shape[-1] > 1:
            max_logits = np.max(logits_arr, axis=-1, keepdims=True)
            exp_logits = np.exp(logits_arr - max_logits)
            probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
            if probs.shape[-1] > 1:
                return probs[..., 1]
            return probs[..., 0]
        # single channel
        return 1.0 / (1.0 + np.exp(-logits_arr.squeeze()))

    return 1.0 / (1.0 + np.exp(-logits_arr))


def pick_obj_using_entropy(
    img_rgb: np.ndarray,  # unused but kept for parity
    pool: List[MaskEntry],
    *,
    target_area_ratio: float = 0.0,  # unused
) -> Tuple[Optional[MaskEntry], Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    if not pool:
        return None, {}, []

    scored_details: List[Dict[str, float]] = []
    best_entry: Optional[MaskEntry] = None
    best_entropy = float("inf")

    entropies: List[float] = []
    ratios: List[float] = []

    for entry in pool:
        logits = entry.get("logits")
        if logits is None:
            entropy = float("inf")
            ratio = float("nan")
        else:
            probs = _foreground_probabilities(logits)
            probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
            entropy_map = -probs * np.log2(probs) - (1.0 - probs) * np.log2(1.0 - probs)
            entropy = float(np.mean(entropy_map))
            ratio = float(np.mean(probs))

        entropies.append(entropy)
        ratios.append(ratio)

        score_details = {"entropy": entropy, "foreground_ratio": ratio}
        entry["score_details"] = score_details
        scored_details.append(score_details)

        if entropy < best_entropy:
            best_entropy = entropy
            best_entry = entry

    pool_stats = {
        "entropy": {"min": float(np.min(entropies)), "max": float(np.max(entropies))},
        "foreground_ratio": {"min": float(np.nanmin(ratios)), "max": float(np.nanmax(ratios))},
    }

    return best_entry, pool_stats, scored_details


# --------------------------------------------------------------------------- #
# Edge gradient scoring on probability map
# --------------------------------------------------------------------------- #
def _boundary_mean_gradient(probs: np.ndarray, mask: np.ndarray) -> float:
    prob_map = np.asarray(probs, dtype=np.float32)
    if prob_map.ndim != 2:
        prob_map = np.squeeze(prob_map)
    prob_map = np.clip(prob_map, 0.0, 1.0)
    grad = filters.sobel(prob_map)
    boundary = segmentation.find_boundaries(np.asarray(mask, dtype=bool), mode="inner")
    if not np.any(boundary):
        return 0.0
    return float(grad[boundary].mean())


def pick_obj_using_edge_gradient(
    img_rgb: np.ndarray,  # unused
    pool: List[MaskEntry],
    *,
    target_area_ratio: float = 0.0,  # unused
) -> Tuple[Optional[MaskEntry], Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    if not pool:
        return None, {}, []

    scored_details: List[Dict[str, float]] = []
    best_entry: Optional[MaskEntry] = None
    best_score = float("-inf")
    gradients: List[float] = []

    for entry in pool:
        logits = entry.get("logits")
        if logits is None:
            grad_score = float("-inf")
            ratio = float("nan")
        else:
            probs = _foreground_probabilities(logits)
            grad_score = _boundary_mean_gradient(probs, entry["mask"])
            ratio = float(np.mean(probs))

        gradients.append(grad_score)
        score_details = {"edge_score": grad_score, "foreground_ratio": ratio}
        entry["score_details"] = score_details
        scored_details.append(score_details)

        if grad_score > best_score:
            best_score = grad_score
            best_entry = entry

    pool_stats = {
        "edge_score": {"min": float(np.min(gradients)), "max": float(np.max(gradients))},
    }
    return best_entry, pool_stats, scored_details
