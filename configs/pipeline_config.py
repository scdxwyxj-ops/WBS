"""Utilities for loading and validating the segmentation pipeline configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
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
    seed: Optional[int]
    candidate_top_k: int
    max_iterations: int
    augment_positive_points: bool
    use_subset_points: bool
    center_range: Tuple[float, float]
    min_point_distance: float
    use_convex_hull: bool
    convex_hull_threshold: float
    deduplicate_mask_pool: bool
    mask_pool_iou_threshold: float
    target_area_ratio: float
    initial_color_mode: str
    initial_positive_count: int
    selection_strategy: str = "heuristic"


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
        seed=data["algorithm"].get("seed"),
        candidate_top_k=int(data["algorithm"]["candidate_top_k"]),
        max_iterations=int(data["algorithm"]["max_iterations"]),
        augment_positive_points=bool(data["algorithm"]["augment_positive_points"]),
        use_subset_points=bool(data["algorithm"]["use_subset_points"]),
        center_range=_as_tuple(data["algorithm"]["center_range"], length=2, name="center_range"),
        min_point_distance=float(data["algorithm"]["min_point_distance"]),
        use_convex_hull=bool(data["algorithm"]["use_convex_hull"]),
        convex_hull_threshold=float(data["algorithm"]["convex_hull_threshold"]),
        deduplicate_mask_pool=bool(data["algorithm"].get("deduplicate_mask_pool", True)),
        mask_pool_iou_threshold=float(data["algorithm"]["mask_pool_iou_threshold"]),
        target_area_ratio=float(data["algorithm"]["target_area_ratio"]),
        initial_color_mode=str(data["algorithm"].get("initial_color_mode", "dark")),
        initial_positive_count=int(data["algorithm"].get("initial_positive_count", 1)),
        selection_strategy=str(data["algorithm"].get("selection_strategy", "heuristic")),
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


def apply_pipeline_overrides(cfg: PipelineConfig, overrides: Dict[str, Any]) -> PipelineConfig:
    """Apply a flat override dict (dot-separated keys) to a PipelineConfig."""
    dataset = cfg.dataset
    preprocessing = cfg.preprocessing
    slic = preprocessing.slic
    algorithm = cfg.algorithm
    threshold = algorithm.threshold
    sam_cfg = cfg.sam

    for key, value in overrides.items():
        if key == "dataset.name":
            dataset = replace(dataset, name=str(value))
        elif key == "dataset.target_long_edge":
            dataset = replace(dataset, target_long_edge=int(value) if value is not None else None)
        elif key == "preprocessing.image_size":
            preprocessing = replace(preprocessing, image_size=int(value))
        elif key == "preprocessing.num_graph_nodes":
            preprocessing = replace(preprocessing, num_graph_nodes=int(value))
        elif key == "preprocessing.slic.compactness":
            slic = replace(slic, compactness=float(value))
        elif key == "preprocessing.slic.sigma":
            slic = replace(slic, sigma=float(value))
        elif key == "preprocessing.slic.min_size_factor":
            slic = replace(slic, min_size_factor=float(value))
        elif key == "preprocessing.slic.max_size_factor":
            slic = replace(slic, max_size_factor=float(value))
        elif key == "algorithm.threshold.value":
            threshold = replace(threshold, value=float(value))
        elif key == "algorithm.threshold.mode":
            threshold = replace(threshold, mode=str(value))
        elif key == "algorithm.negative_pct":
            algorithm = replace(algorithm, negative_pct=float(value))
        elif key == "algorithm.score_lower_bound":
            algorithm = replace(algorithm, score_lower_bound=float(value))
        elif key == "algorithm.seed":
            algorithm = replace(algorithm, seed=None if value is None else int(value))
        elif key == "algorithm.candidate_top_k":
            algorithm = replace(algorithm, candidate_top_k=int(value))
        elif key == "algorithm.max_iterations":
            algorithm = replace(algorithm, max_iterations=int(value))
        elif key == "algorithm.augment_positive_points":
            algorithm = replace(algorithm, augment_positive_points=bool(value))
        elif key == "algorithm.use_subset_points":
            algorithm = replace(algorithm, use_subset_points=bool(value))
        elif key == "algorithm.center_range":
            algorithm = replace(algorithm, center_range=_as_tuple(value, length=2, name="center_range"))
        elif key == "algorithm.min_point_distance":
            algorithm = replace(algorithm, min_point_distance=float(value))
        elif key == "algorithm.use_convex_hull":
            algorithm = replace(algorithm, use_convex_hull=bool(value))
        elif key == "algorithm.convex_hull_threshold":
            algorithm = replace(algorithm, convex_hull_threshold=float(value))
        elif key == "algorithm.deduplicate_mask_pool":
            algorithm = replace(algorithm, deduplicate_mask_pool=bool(value))
        elif key == "algorithm.mask_pool_iou_threshold":
            algorithm = replace(algorithm, mask_pool_iou_threshold=float(value))
        elif key == "algorithm.target_area_ratio":
            algorithm = replace(algorithm, target_area_ratio=float(value))
        elif key == "algorithm.initial_color_mode":
            algorithm = replace(algorithm, initial_color_mode=str(value))
        elif key == "algorithm.initial_positive_count":
            algorithm = replace(algorithm, initial_positive_count=int(value))
        elif key == "algorithm.selection_strategy":
            algorithm = replace(algorithm, selection_strategy=str(value))
        elif key == "sam.multimask_output":
            sam_cfg = replace(sam_cfg, multimask_output=bool(value))
        elif key == "sam.mask_prompt_source":
            sam_cfg = replace(sam_cfg, mask_prompt_source=str(value))
        elif key == "sam.refine_with_previous_low_res":
            sam_cfg = replace(sam_cfg, refine_with_previous_low_res=bool(value))
        elif key == "sam.refine_rounds":
            sam_cfg = replace(sam_cfg, refine_rounds=int(value))
        else:
            raise ValueError(f"Unsupported override key: {key}")

    preprocessing = replace(preprocessing, slic=slic)
    algorithm = replace(algorithm, threshold=threshold)
    return replace(cfg, dataset=dataset, preprocessing=preprocessing, algorithm=algorithm, sam=sam_cfg)
