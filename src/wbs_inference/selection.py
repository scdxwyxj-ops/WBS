"""Configurable proposal scoring and area-cluster selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage import color, filters, measure, segmentation, util
from sklearn.cluster import KMeans

from .config import SelectionConfig

EPSILON = 1e-6


@dataclass
class Proposal:
    mask: np.ndarray
    sam_score: float
    logits: np.ndarray | None
    low_res_mask: np.ndarray | None
    iteration: int
    prompts: Any
    candidate_id: int | None
    heuristic_score: float = 0.0
    heuristic_features: dict[str, float] = field(default_factory=dict)

    @property
    def area_ratio(self) -> float:
        return float(np.asarray(self.mask, dtype=bool).mean()) if self.mask.size else 0.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def deduplicate(proposals: list[Proposal], threshold: float) -> list[Proposal]:
    retained: list[Proposal] = []
    for proposal in sorted(proposals, key=lambda item: item.sam_score, reverse=True):
        if all(mask_iou(proposal.mask, other.mask) < threshold for other in retained):
            retained.append(proposal)
    return retained


def _sample_pixels(values: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(values) <= limit:
        return values
    return values[np.linspace(0, len(values) - 1, limit, dtype=np.int64)]


def _cluster_trace(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    covariance = np.cov(values, rowvar=False)
    return float(covariance if np.isscalar(covariance) else np.trace(covariance))


def _raw_features(
    image: np.ndarray, proposal: Proposal, config: SelectionConfig, image_cache: dict[str, np.ndarray]
) -> dict[str, float]:
    mask = np.asarray(proposal.mask, dtype=bool)
    area_pixels = float(mask.sum())
    area_ratio = area_pixels / float(mask.size or 1)
    perimeter = float(measure.perimeter(mask.astype(np.uint8), neighborhood=8)) if area_pixels else 0.0
    circularity = float(np.clip(4.0 * np.pi * area_pixels / perimeter**2, 0.0, 1.0)) if perimeter > 0 else 0.0
    boundary = segmentation.find_boundaries(mask, mode="inner")
    boundary_values = image_cache["gradient"][boundary]
    edge = float(boundary_values.mean()) if boundary_values.size else 0.0
    lab_pixels = image_cache["lab"][mask]
    if len(lab_pixels) < config.color_clusters:
        color_separation = 0.0
    else:
        sampled = _sample_pixels(lab_pixels, config.color_sample_max_pixels)
        if len(np.unique(sampled, axis=0)) < config.color_clusters:
            return {
                "target_area": -abs(area_ratio - config.target_area_ratio),
                "boundary_edge": edge,
                "circularity": circularity,
                "sam_score": float(proposal.sam_score),
                "color_separation": 0.0,
                "area_ratio": area_ratio,
            }
        kmeans = KMeans(
            n_clusters=config.color_clusters,
            n_init=config.color_kmeans_n_init,
            random_state=config.kmeans_random_state,
        )
        labels = kmeans.fit_predict(sampled)
        if config.color_clusters == 2:
            center_distance = float(np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]))
            denominator = (
                _cluster_trace(sampled[labels == 0]) ** 0.5 + _cluster_trace(sampled[labels == 1]) ** 0.5 + EPSILON
            )
            color_separation = center_distance / denominator
        else:
            distances = []
            for left in range(config.color_clusters):
                for right in range(left + 1, config.color_clusters):
                    distances.append(
                        float(np.linalg.norm(kmeans.cluster_centers_[left] - kmeans.cluster_centers_[right]))
                    )
            color_separation = float(np.mean(distances)) if distances else 0.0
    return {
        "target_area": -abs(area_ratio - config.target_area_ratio),
        "boundary_edge": edge,
        "circularity": circularity,
        "sam_score": float(proposal.sam_score),
        "color_separation": color_separation,
        "area_ratio": area_ratio,
    }


def _normalise(value: float, values: list[float], epsilon: float) -> float:
    low, high = min(values), max(values)
    return 1.0 if high - low < epsilon else float((value - low) / (high - low))


def score_heuristics(image: np.ndarray, proposals: list[Proposal], config: SelectionConfig) -> None:
    image_float = util.img_as_float(image)
    cache = {"gradient": filters.sobel(color.rgb2gray(image_float)), "lab": color.rgb2lab(image_float)}
    feature_rows = [_raw_features(image, proposal, config, cache) for proposal in proposals]
    keys = ("target_area", "boundary_edge", "circularity", "sam_score", "color_separation")
    weights = config.heuristic_weights
    for proposal, features in zip(proposals, feature_rows, strict=True):
        normalised = {key: _normalise(features[key], [row[key] for row in feature_rows], EPSILON) for key in keys}
        proposal.heuristic_score = float(sum(normalised[key] * float(getattr(weights, key)) for key in keys))
        proposal.heuristic_features = {**features, **{f"{key}_normalised": value for key, value in normalised.items()}}


def _initial_centers(areas: np.ndarray, count: int) -> np.ndarray:
    ordered = np.sort(areas)
    if count == 1:
        centers = [ordered[len(ordered) // 2]]
    elif count == 2:
        centers = [ordered[0], ordered[-1]]
    elif count == 3:
        centers = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    else:
        indexes = np.linspace(0, len(ordered) - 1, count, dtype=int)
        centers = [ordered[index] for index in indexes]
    return np.asarray(centers, dtype=np.float32).reshape(-1, 1)


def select_proposal(
    image: np.ndarray, proposals: list[Proposal], config: SelectionConfig
) -> tuple[Proposal, dict[str, Any]]:
    if not proposals:
        raise ValueError("Cannot select from an empty proposal pool")
    pool = deduplicate(proposals, config.deduplicate_iou_threshold)
    score_heuristics(image, pool, config)
    ranked = sorted(enumerate(pool), key=lambda item: (-item[1].heuristic_score, item[0]))
    retain_count = min(len(ranked), config.retain_n)
    retained = [proposal for _, proposal in ranked[:retain_count]]
    distinct_areas = len(np.unique(np.asarray([proposal.area_ratio for proposal in retained], dtype=np.float32)))
    possible_clusters = min(len(retained), distinct_areas)
    effective_clusters = max(1, min(config.area_clusters, possible_clusters))
    areas = np.asarray([proposal.area_ratio for proposal in retained], dtype=np.float32)
    if effective_clusters == 1:
        labels = np.zeros(len(retained), dtype=np.int32)
        centers = np.asarray([float(areas.mean())], dtype=np.float32)
    else:
        kmeans = KMeans(
            n_clusters=effective_clusters,
            init=_initial_centers(areas, effective_clusters),
            n_init=config.area_kmeans_n_init,
            random_state=config.kmeans_random_state,
        )
        labels = kmeans.fit_predict(areas.reshape(-1, 1))
        centers = kmeans.cluster_centers_.reshape(-1)
    center_order = np.argsort(centers)
    middle_index = effective_clusters // 2
    selected_label = int(center_order[middle_index])
    selected_cluster = [
        proposal for proposal, label in zip(retained, labels, strict=True) if int(label) == selected_label
    ]
    selected = max(enumerate(selected_cluster), key=lambda item: (item[1].sam_score, -item[0]))[1]
    metadata = {
        "proposal_pool_size": len(proposals),
        "ranked_pool_size": len(pool),
        "retain_n": config.retain_n,
        "retained_pool_size": len(retained),
        "selected_iteration": selected.iteration,
        "selected_sam_score": selected.sam_score,
        "selected_heuristic_score": selected.heuristic_score,
    }
    return selected, metadata
