"""Utilities for loading and validating the segmentation pipeline configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    target_long_edge: Optional[int] = None


@dataclass(frozen=True)
class SLICConfig:
    compactness: float
    sigma: float
    min_size_factor: float
    max_size_factor: float


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int
    num_graph_nodes: int
    slic: SLICConfig


@dataclass(frozen=True)
class ThresholdConfig:
    mode: str
    value: float


@dataclass(frozen=True)
class AlgorithmConfig:
    negative_pct: float
    score_lower_bound: float
    threshold: ThresholdConfig
    candidate_top_k: int
    max_iterations: int
    augment_positive_points: bool
    use_subset_points: bool
    center_range: Tuple[float, float]
    min_point_distance: float
    use_convex_hull: bool
    convex_hull_threshold: float
    mask_pool_iou_threshold: float
    target_area_ratio: float
    initial_color_mode: str
    initial_positive_count: int
    selection_strategy: str = "pick_obj"


@dataclass(frozen=True)
class SAMConfig:
    multimask_output: bool
    mask_prompt_source: str
    refine_with_previous_low_res: bool
    refine_rounds: int


@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessConfig
    algorithm: AlgorithmConfig
    sam: SAMConfig


def _as_tuple(value: Any, *, length: int, name: str) -> Tuple[float, ...]:
    if isinstance(value, (list, tuple)) and len(value) == length:
        return tuple(float(v) for v in value)
    raise ValueError(f"`{name}` must be a sequence of length {length}.")


def load_pipeline_config(path: Path) -> PipelineConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    dataset_cfg = DatasetConfig(
        name=data["dataset"]["name"],
        target_long_edge=data["dataset"].get("target_long_edge"),
    )

    slic_cfg = SLICConfig(
        compactness=float(data["preprocessing"]["slic"]["compactness"]),
        sigma=float(data["preprocessing"]["slic"]["sigma"]),
        min_size_factor=float(data["preprocessing"]["slic"]["min_size_factor"]),
        max_size_factor=float(data["preprocessing"]["slic"]["max_size_factor"]),
    )

    preprocess_cfg = PreprocessConfig(
        image_size=int(data["preprocessing"]["image_size"]),
        num_graph_nodes=int(data["preprocessing"]["num_graph_nodes"]),
        slic=slic_cfg,
    )

    threshold_cfg = ThresholdConfig(
        mode=str(data["algorithm"]["threshold"]["mode"]),
        value=float(data["algorithm"]["threshold"]["value"]),
    )

    algorithm_cfg = AlgorithmConfig(
        negative_pct=float(data["algorithm"]["negative_pct"]),
        score_lower_bound=float(data["algorithm"]["score_lower_bound"]),
        threshold=threshold_cfg,
        candidate_top_k=int(data["algorithm"]["candidate_top_k"]),
        max_iterations=int(data["algorithm"]["max_iterations"]),
        augment_positive_points=bool(data["algorithm"]["augment_positive_points"]),
        use_subset_points=bool(data["algorithm"]["use_subset_points"]),
        center_range=_as_tuple(data["algorithm"]["center_range"], length=2, name="center_range"),
        min_point_distance=float(data["algorithm"]["min_point_distance"]),
        use_convex_hull=bool(data["algorithm"]["use_convex_hull"]),
        convex_hull_threshold=float(data["algorithm"]["convex_hull_threshold"]),
        mask_pool_iou_threshold=float(data["algorithm"]["mask_pool_iou_threshold"]),
        target_area_ratio=float(data["algorithm"]["target_area_ratio"]),
        initial_color_mode=str(data["algorithm"].get("initial_color_mode", "dark")),
        initial_positive_count=int(data["algorithm"].get("initial_positive_count", 1)),
        selection_strategy=str(data["algorithm"].get("selection_strategy", "pick_obj")),
    )

    sam_cfg = SAMConfig(
        multimask_output=bool(data["sam"]["multimask_output"]),
        mask_prompt_source=str(data["sam"].get("mask_prompt_source", data["sam"].get("mask_prompt_strategy", "none"))),
        refine_with_previous_low_res=bool(data["sam"]["refine_with_previous_low_res"]),
        refine_rounds=int(data["sam"]["refine_rounds"]),
    )

    return PipelineConfig(
        dataset=dataset_cfg,
        preprocessing=preprocess_cfg,
        algorithm=algorithm_cfg,
        sam=sam_cfg,
    )
