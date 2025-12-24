"""Prompt management and graph utilities for the unsupervised SAM2 pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from configs.pipeline_config import AlgorithmConfig as PipelineAlgorithmConfig
from image_processings.node import Node
from image_processings.pick_obj import (
    pick_obj_using_entropy,
    pick_obj_using_heuristic,
    pick_obj_using_edge_gradient,
)


@dataclass(frozen=True)
class AlgorithmSettings:
    negative_pct: float = 0.2
    score_lower_bound: float = -10.0
    threshold_mode: str = "constant"
    threshold_value: float = 0.5
    candidate_top_k: int = 3
    augment_positive_points: bool = True
    use_subset_points: bool = True
    center_range: Tuple[float, float] = (0.1, 0.9)
    min_point_distance: float = 10.0
    use_convex_hull: bool = True
    convex_hull_threshold: float = 0.85
    deduplicate_mask_pool: bool = True
    produce_low_res_mask: bool = True
    mask_pool_iou_threshold: float = 0.9
    target_area_ratio: float = 0.05
    initial_color_mode: str = "dark"
    initial_positive_count: int = 1
    selection_strategy: str = "heuristic"

    @classmethod
    def from_pipeline_config(
        cls,
        config: Optional[PipelineAlgorithmConfig],
        *,
        mask_prompt_source: Optional[str] = None,
    ) -> "AlgorithmSettings":
        if config is None:
            return cls()
        strategy = (mask_prompt_source or "").lower()
        produce_low_res = strategy in {"slic", "foreground", "algorithm", "slic_foreground"}
        return cls(
            negative_pct=config.negative_pct,
            score_lower_bound=config.score_lower_bound,
            threshold_mode=config.threshold.mode,
            threshold_value=config.threshold.value,
            candidate_top_k=config.candidate_top_k,
            augment_positive_points=config.augment_positive_points,
            use_subset_points=config.use_subset_points,
            center_range=config.center_range,
            min_point_distance=config.min_point_distance,
            use_convex_hull=config.use_convex_hull,
            convex_hull_threshold=config.convex_hull_threshold,
            deduplicate_mask_pool=getattr(config, "deduplicate_mask_pool", True),
            produce_low_res_mask=produce_low_res,
            mask_pool_iou_threshold=config.mask_pool_iou_threshold,
            target_area_ratio=config.target_area_ratio,
            initial_color_mode=config.initial_color_mode,
            initial_positive_count=config.initial_positive_count,
            selection_strategy=getattr(config, "selection_strategy", "heuristic"),
        )


@dataclass
class Candidate:
    node_id: int
    score: float
    center: Tuple[float, float]


@dataclass
class PromptBundle:
    points: np.ndarray
    labels: np.ndarray
    mask_prompt: Optional[np.ndarray]
    low_res_mask: Optional[np.ndarray]
    candidate_id: Optional[int] = None
    all_positive_points: Optional[List[Tuple[float, float]]] = None


class Info:
    """Maintain per-image state for iterative prompt selection."""

    def __init__(
        self,
        segment: np.ndarray,
        logits: Optional[np.ndarray],
        image: np.ndarray,
        graph,
        *,
        settings: Optional[PipelineAlgorithmConfig] = None,
        debug_mode: bool = True,
        mask_prompt_source: Optional[str] = None,
        **legacy_kwargs: Any,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG if debug_mode else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
            self.logger.propagate = False

        self.settings = AlgorithmSettings.from_pipeline_config(settings, mask_prompt_source=mask_prompt_source)
        legacy_negative = legacy_kwargs.pop("negative_pct", None)
        if legacy_negative is not None:
            self.settings.negative_pct = float(legacy_negative)
        self.segment_indices, self.segment_ids = self._normalise_segments(segment)
        self.image = image
        self.graph = graph
        self.num_segments = len(self.segment_ids)

        self.node_list: List[Node] = self._build_nodes(image)
        self.labels = np.full(self.num_segments, -1, dtype=int)
        self.prompt_mask = np.zeros_like(self.segment_indices, dtype=bool)
        self.logits = logits
        self.mask = logits > 0 if logits is not None else None

        self.positive_point_coords: List[Tuple[float, float]] = []
        self.negative_point_coords: List[Tuple[float, float]] = []
        self.next_iter_points: List[Tuple[float, float]] = []

        self.score_lower_bound = self.settings.score_lower_bound
        self.threshold = self.settings.threshold_value
        self._aug_point_added = False
        self.last_low_res_mask: Optional[np.ndarray] = None

        self.mask_pool: List[Dict[str, Any]] = []
        self.mask_pool_full: List[Dict[str, Any]] = []
        self.pool_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.selected_entry: Optional[Dict[str, Any]] = None
        self.selection_metadata: Dict[str, Any] = {}

        self._initialise_prompt_points()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_segments(segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seg = np.asarray(segment, dtype=int)
        unique_ids = np.unique(seg)
        sorted_ids = np.sort(unique_ids)
        index_map = {seg_id: idx for idx, seg_id in enumerate(sorted_ids)}
        normalised = np.vectorize(index_map.get)(seg)
        return normalised, sorted_ids

    def _build_nodes(self, image: np.ndarray) -> List[Node]:
        nodes: List[Node] = []
        for idx in range(self.num_segments):
            mask = (self.segment_indices == idx)
            nodes.append(Node(idx, score=-1, mask=mask, image=image, color_mode=self.settings.initial_color_mode))
        return nodes

    def _initialise_prompt_points(self) -> None:
        nodes_by_color = sorted(self.node_list, key=lambda n: n.color, reverse=True)

        negative_quota = max(1, int(round(self.settings.negative_pct * self.num_segments)))
        for node in nodes_by_color:
            if len(self.negative_point_coords) >= negative_quota:
                break
            if node.is_edge and self.labels[node.index] != 0:
                self.labels[node.index] = 0
                node.label = 0
                self.negative_point_coords.append(self._round_point(node.center))

        positive_quota = max(1, int(self.settings.initial_positive_count))
        promoted = 0
        for node in nodes_by_color:
            if node.is_center and self.labels[node.index] != 1:
                self.labels[node.index] = 1
                node.label = 1
                self.positive_point_coords.append(self._round_point(node.center))
                promoted += 1
                if promoted >= positive_quota:
                    break

        if promoted == 0:
            # fallback: choose the highest-ranked node even if not marked center
            for node in nodes_by_color:
                if self.labels[node.index] != 1:
                    self.labels[node.index] = 1
                    node.label = 1
                    self.positive_point_coords.append(self._round_point(node.center))
                    break

        self._update_prompt_mask()

    @staticmethod
    def _round_point(point: Tuple[float, float]) -> Tuple[float, float]:
        return (float(point[0]), float(point[1]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_initial_prompts(self) -> PromptBundle:
        points = np.array(self.positive_point_coords + self.negative_point_coords, dtype=np.float32)
        labels = np.array([1] * len(self.positive_point_coords) + [0] * len(self.negative_point_coords), dtype=np.int64)
        return PromptBundle(points=points, labels=labels, mask_prompt=None, low_res_mask=None)

    def update_from_logits(self, logits: np.ndarray) -> None:
        self.logits = logits
        self.mask = logits > 0
        self.next_iter_points = []
        self._aug_point_added = False
        self.threshold = self._compute_threshold()
        self._update_foreground_labels()

    def get_candidates(self) -> List[Candidate]:
        foreground_ids = np.flatnonzero(self.labels == 1)
        if foreground_ids.size == 0:
            return []

        candidate_ids = set()
        for idx in foreground_ids:
            for neighbor in self._get_neighbors(idx):
                if neighbor >= self.num_segments:
                    continue
                if self.labels[neighbor] != -1:
                    continue
                if self.node_list[neighbor].is_edge:
                    continue
                candidate_ids.add(neighbor)

        scored: List[Candidate] = []
        for node_id in candidate_ids:
            score = self._node_score(node_id)
            if score < self.score_lower_bound:
                continue
            scored.append(Candidate(node_id=node_id, score=score, center=self._round_point(self.node_list[node_id].center)))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: self.settings.candidate_top_k]

    def commit_candidate(self, node_id: int) -> None:
        if self.labels[node_id] == 0:
            raise ValueError("Cannot promote a negative node to foreground.")

        if self.labels[node_id] != 1:
            self.labels[node_id] = 1
            node = self.node_list[node_id]
            node.label = 1
            point = self._round_point(node.center)
            if point not in self.positive_point_coords:
                self.positive_point_coords.append(point)
            self.prompt_mask = np.logical_or(self.prompt_mask, node.mask)
            self.next_iter_points.append(point)

    def build_prompts(
        self,
        *,
        candidate_id: Optional[int] = None,
        include_augmented_point: bool = True,
        best_point: Optional[Tuple[float, float]] = None,
    ) -> PromptBundle:
        pos_points = list(self.positive_point_coords)

        if candidate_id is not None:
            pos_points.append(self._round_point(self.node_list[candidate_id].center))

        if best_point is not None:
            best_point = self._round_point(best_point)
            if best_point not in pos_points:
                pos_points.append(best_point)

        mask_prompt = self._compose_foreground_mask(additional_indices=[candidate_id] if candidate_id is not None else None)
        low_res_mask = None

        if mask_prompt is not None and self.settings.produce_low_res_mask:
            low_res_mask = self.reshape_mask(mask_prompt)

        if include_augmented_point and self.settings.augment_positive_points and self.mask is not None:
            aug_point = self._maybe_add_aug_point(mask_prompt)
            if aug_point is not None and aug_point not in pos_points:
                pos_points.append(aug_point)

        if self.settings.use_subset_points and len(pos_points) >= 3:
            filtered_pos = self.points_filter(pos_points)
        else:
            filtered_pos = pos_points

        neg_points = list(self.negative_point_coords)
        points = np.array(filtered_pos + neg_points, dtype=np.float32)
        labels = np.array([1] * len(filtered_pos) + [0] * len(neg_points), dtype=np.int64)

        return PromptBundle(
            points=points,
            labels=labels,
            mask_prompt=mask_prompt,
            low_res_mask=low_res_mask,
            candidate_id=candidate_id,
            all_positive_points=pos_points,
        )

    def record_low_res_mask(self, mask: Optional[np.ndarray]) -> None:
        self.last_low_res_mask = mask

    def add_pool_entry(
        self,
        *,
        mask: np.ndarray,
        logits: Optional[np.ndarray],
        score: Optional[float],
        iteration: int,
        prompts: PromptBundle,
        candidate_id: Optional[int],
        positive_mask: Optional[np.ndarray],
    ) -> None:
        mask_bool = np.asarray(mask, dtype=bool)
        entry: Dict[str, Any] = {
            "mask": mask_bool.copy(),
            "logits": np.array(logits) if logits is not None else None,
            "score": float(score) if score is not None else 0.0,
            "conv_quality": float(score) if score is not None else 0.0,
            "iteration": int(iteration),
            "prompts": prompts,
            "candidate_id": int(candidate_id) if candidate_id is not None else None,
            "positive_mask": np.asarray(positive_mask, dtype=bool).copy() if positive_mask is not None else None,
            "score_details": {"score": float(score) if score is not None else 0.0},
        }
        self.mask_pool.append(entry)
        self.mask_pool_full.append(entry)

    def deduplicate_mask_pool(self, threshold: float) -> None:
        if not self.mask_pool:
            return
        sorted_pool = sorted(self.mask_pool, key=lambda e: e.get("score", 0.0), reverse=True)
        deduped: List[Dict[str, Any]] = []
        for entry in sorted_pool:
            mask = entry["mask"]
            if all(self._mask_iou(mask, kept["mask"]) < threshold for kept in deduped):
                deduped.append(entry)
        self.mask_pool = deduped

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = np.asarray(mask_a, dtype=bool)
        b = np.asarray(mask_b, dtype=bool)
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def get_mask_pool(self, *, full: bool = False) -> List[Dict[str, Any]]:
        pool = self.mask_pool_full if full else self.mask_pool
        return list(pool)

    def set_pool_stats(self, stats: Dict[str, Dict[str, float]]) -> None:
        self.pool_stats = stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_threshold(self) -> float:
        if self.mask is None:
            return self.settings.threshold_value

        mean_val = float(np.mean(self.mask))
        mode = self.settings.threshold_mode.lower()
        if mode == "constant":
            return self.settings.threshold_value
        if mode == "mean":
            return mean_val
        if mode == "scaled_mean":
            return self.settings.threshold_value * mean_val
        self.logger.warning("Unknown threshold mode %s, defaulting to constant.", self.settings.threshold_mode)
        return self.settings.threshold_value

    def _update_foreground_labels(self) -> None:
        if self.mask is None:
            return

        flat_segments = self.segment_indices.ravel()
        segment_area = np.bincount(flat_segments, minlength=self.num_segments)
        mask_area = np.bincount(flat_segments[self.mask.ravel()], minlength=self.num_segments)

        valid = segment_area > 0
        ratios = np.zeros_like(segment_area, dtype=float)
        ratios[valid] = mask_area[valid] / segment_area[valid]

        updated = np.flatnonzero(ratios >= self.threshold)
        if updated.size == 0:
            return

        for idx in updated:
            if self.labels[idx] == -1:
                self.labels[idx] = 1
                self.node_list[idx].label = 1
                self.positive_point_coords.append(self._round_point(self.node_list[idx].center))

        self._update_prompt_mask()

    def _update_prompt_mask(self, additional_indices: Optional[Iterable[int]] = None) -> None:
        mask = np.zeros_like(self.segment_indices, dtype=bool)
        indices = set(np.flatnonzero(self.labels == 1))
        if additional_indices:
            indices.update(i for i in additional_indices if i is not None)

        for idx in indices:
            mask = np.logical_or(mask, self.node_list[idx].mask)

        self.prompt_mask = mask

    def _compose_foreground_mask(self, additional_indices: Optional[Iterable[int]] = None) -> Optional[np.ndarray]:
        indices = list(np.flatnonzero(self.labels == 1))
        if additional_indices:
            indices.extend(i for i in additional_indices if i is not None)

        if not indices:
            return None

        mask = np.zeros_like(self.segment_indices, dtype=np.uint8)
        for idx in indices:
            mask = np.logical_or(mask, self.node_list[idx].mask)

        mask = mask.astype(np.uint8)

        if self.settings.use_convex_hull:
            hull = self.apply_selective_convex_hull(mask, threshold=self.settings.convex_hull_threshold)
            return (hull > 0).astype(np.uint8)
        return mask

    def _maybe_add_aug_point(self, mask_aug: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if mask_aug is None or self.mask is None:
            return None

        if not self._aug_point_added:
            mask_origin = self.mask.astype(bool)
            aug_bool = mask_aug.astype(bool)
            existing = np.logical_or(mask_origin, self.prompt_mask)
            new_point = self.add_more_pos_points(mask_origin, aug_bool, existing)
            if new_point is not None:
                self.logger.info("add a new pos point, %s", new_point)
                self._aug_point_added = True
                return new_point
            self._aug_point_added = True
        return None

    def _node_score(self, node_id: int) -> float:
        if self.logits is None:
            return float("-inf")
        mask = self.node_list[node_id].mask
        masked_logits = self.logits[mask]
        if masked_logits.size == 0:
            return float("-inf")
        return float(masked_logits.mean())

    def _get_neighbors(self, index: int) -> Iterable[int]:
        if self.graph is None or not self.graph.has_node(index):
            return []
        if self.graph.is_directed():
            successors = set(self.graph.successors(index))
            predecessors = set(self.graph.predecessors(index))
            return successors.union(predecessors)
        return set(self.graph.neighbors(index))

    # ------------------------------------------------------------------
    # Resampling helpers and geometry utils
    # ------------------------------------------------------------------
    @staticmethod
    def apply_selective_convex_hull(binary_mask: np.ndarray, threshold: float = 0.85) -> np.ndarray:
        mask = binary_mask
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_mask = np.zeros_like(mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            if hull_area == 0:
                continue

            ratio = area / hull_area
            contour_to_draw = hull if ratio < threshold else cnt
            cv2.drawContours(result_mask, [contour_to_draw], -1, 255, thickness=-1)

        filled_mask = binary_fill_holes(result_mask > 0)
        return (filled_mask.astype(np.uint8)) * 255

    @staticmethod
    def get_largest_component(mask: np.ndarray) -> np.ndarray:
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        mask_uint8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        if num_labels <= 1:
            return np.zeros_like(mask, dtype=bool)

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return labels == largest_label

    @staticmethod
    def reshape_mask(mask_input: np.ndarray) -> np.ndarray:
        mask_uint8 = (mask_input > 0).astype(np.uint8)
        resized_mask = cv2.resize(mask_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)
        return resized_mask[None, :, :]

    @staticmethod
    def add_more_pos_points(
        mask_origin: np.ndarray,
        mask_aug: np.ndarray,
        existing_mask: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        new_mask = np.logical_and(mask_aug, np.logical_not(existing_mask))

        if not np.any(new_mask):
            return None

        largest_component = Info.get_largest_component(new_mask)
        coords = np.argwhere(largest_component)
        if coords.size == 0:
            return None

        centroid = coords.mean(axis=0)
        sampled_coord = tuple(np.round(centroid)[::-1].astype(float))
        return sampled_coord

    def points_filter(self, point_coords_list: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        filtered: List[Tuple[float, float]] = []
        height, width = self.image.shape[:2]
        low, high = self.settings.center_range

        x_low, x_high = width * low, width * high
        y_low, y_high = height * low, height * high

        center_points = [
            p
            for p in point_coords_list
            if x_low <= p[0] <= x_high and y_low <= p[1] <= y_high
        ]

        if not center_points:
            return list(point_coords_list)

        rng = np.random.default_rng(0)
        order = rng.permutation(len(center_points))

        for idx in order:
            p = center_points[idx]
            if all(
                np.hypot(p[0] - fp[0], p[1] - fp[1]) >= self.settings.min_point_distance
                for fp in filtered
            ):
                filtered.append(p)

        return filtered if filtered else list(center_points)
