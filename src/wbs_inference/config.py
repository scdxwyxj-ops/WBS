"""Inference configuration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints


@dataclass(frozen=True)
class ModelConfig:
    sam2_model_cfg: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    seed: int
    amp_dtype: str


@dataclass(frozen=True)
class PreprocessingConfig:
    long_edge: int
    num_superpixels: int
    compactness: float
    sigma: float
    min_size_factor: float
    max_size_factor: float


@dataclass(frozen=True)
class PromptConfig:
    initial_positive_count: int
    initial_positive_center_range: tuple[float, float]
    initial_negative_fraction: float
    initial_negative_min_count: int
    subset_center_range: tuple[float, float]
    min_point_distance: float
    convex_hull_area_ratio_threshold: float
    mask_prompt_low_res_size: int


@dataclass(frozen=True)
class GrowingConfig:
    sam_logit_threshold: float
    occupancy_threshold: float
    region_score_lower_bound: float
    candidate_top_n: int
    max_iterations: int
    refine_rounds: int


@dataclass(frozen=True)
class HeuristicWeights:
    target_area: float
    boundary_edge: float
    circularity: float
    sam_score: float
    color_separation: float


@dataclass(frozen=True)
class SelectionConfig:
    deduplicate_iou_threshold: float
    target_area_ratio: float
    color_clusters: int
    color_kmeans_n_init: int
    color_sample_max_pixels: int
    kmeans_random_state: int
    heuristic_weights: HeuristicWeights
    retain_n: int
    area_clusters: int
    area_kmeans_n_init: int


@dataclass(frozen=True)
class OutputConfig:
    mask_value: int
    save_metadata: bool
    include_proposal_summary: bool


@dataclass(frozen=True)
class InferenceConfig:
    model: ModelConfig
    runtime: RuntimeConfig
    preprocessing: PreprocessingConfig
    prompts: PromptConfig
    growing: GrowingConfig
    selection: SelectionConfig
    output: OutputConfig
    sha256: str


T = TypeVar("T")


def _construct(cls: type[T], raw: Any, path: str) -> T:
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must be an object")
    fields = {field.name: field for field in dataclasses.fields(cls)}
    missing = sorted(set(fields) - set(raw))
    unknown = sorted(set(raw) - set(fields))
    if missing or unknown:
        raise ValueError(f"{path}: missing={missing}, unknown={unknown}")

    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for name, value in raw.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            values[name] = _construct(hint, value, f"{path}.{name}")
        elif getattr(hint, "__origin__", None) is tuple:
            if not isinstance(value, list) or len(value) != 2:
                raise TypeError(f"{path}.{name} must be a two-item array")
            if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
                raise TypeError(f"{path}.{name} must contain numbers")
            values[name] = (float(value[0]), float(value[1]))
        else:
            valid = type(value) is hint or (
                hint is float and not isinstance(value, bool) and isinstance(value, (int, float))
            )
            if not valid:
                raise TypeError(f"{path}.{name} must be {hint.__name__}, got {type(value).__name__}")
            values[name] = value
    return cls(**values)


def _validate_range(value: tuple[float, float], name: str) -> None:
    if not (0.0 <= value[0] < value[1] <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= low < high <= 1")


def validate_config(config: InferenceConfig) -> None:
    if config.runtime.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device must be auto, cpu, or cuda")
    if config.runtime.amp_dtype not in {"none", "float16", "bfloat16"}:
        raise ValueError("runtime.amp_dtype must be none, float16, or bfloat16")
    if config.runtime.seed < 0:
        raise ValueError("runtime.seed must be non-negative")
    if not re.fullmatch(r"[0-9a-f]{64}", config.model.checkpoint_sha256):
        raise ValueError("model.checkpoint_sha256 must be a lowercase SHA256 digest")

    preprocessing = config.preprocessing
    if (
        preprocessing.long_edge < 1
        or preprocessing.num_superpixels < 2
        or preprocessing.compactness <= 0
        or preprocessing.sigma < 0
    ):
        raise ValueError("Invalid preprocessing values")
    if preprocessing.min_size_factor <= 0 or preprocessing.max_size_factor < preprocessing.min_size_factor:
        raise ValueError("SLIC size factors must satisfy 0 < min_size_factor <= max_size_factor")

    prompts = config.prompts
    _validate_range(prompts.initial_positive_center_range, "initial_positive_center_range")
    _validate_range(prompts.subset_center_range, "subset_center_range")
    if prompts.initial_positive_count < 1 or prompts.initial_negative_min_count < 0:
        raise ValueError("Invalid prompt counts")
    if not 0 <= prompts.initial_negative_fraction <= 1:
        raise ValueError("initial_negative_fraction must be in [0, 1]")
    if prompts.min_point_distance < 0 or prompts.mask_prompt_low_res_size < 1:
        raise ValueError("Invalid prompt geometry")
    if not 0 <= prompts.convex_hull_area_ratio_threshold <= 1:
        raise ValueError("convex_hull_area_ratio_threshold must be in [0, 1]")

    growing = config.growing
    if growing.candidate_top_n < 1 or growing.max_iterations < 0 or growing.refine_rounds < 0:
        raise ValueError("Invalid growing counts")
    if not all(
        math.isfinite(value)
        for value in (growing.sam_logit_threshold, growing.occupancy_threshold, growing.region_score_lower_bound)
    ):
        raise ValueError("Growing thresholds must be finite")

    selection = config.selection
    if not 0 <= selection.deduplicate_iou_threshold <= 1 or not 0 <= selection.target_area_ratio <= 1:
        raise ValueError("Selection ratios must be in [0, 1]")
    if selection.color_clusters < 1 or selection.color_sample_max_pixels < 0:
        raise ValueError("Invalid color-clustering values")
    if selection.color_kmeans_n_init < 1 or selection.area_kmeans_n_init < 1:
        raise ValueError("KMeans n_init values must be positive")
    if selection.retain_n < 1 or selection.area_clusters < 1:
        raise ValueError("Selection counts must be positive")
    weights = dataclasses.astuple(selection.heuristic_weights)
    if any(not math.isfinite(value) or value < 0 for value in weights) or not any(value > 0 for value in weights):
        raise ValueError("Heuristic weights must be finite, non-negative, and not all zero")

    if not 1 <= config.output.mask_value <= 255:
        raise ValueError("output.mask_value must be in [1, 255]")


def load_config(path: str | Path) -> InferenceConfig:
    source = Path(path).expanduser().resolve()
    payload_bytes = source.read_bytes()
    raw = json.loads(payload_bytes)
    expected = {"model", "runtime", "preprocessing", "prompts", "growing", "selection", "output"}
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be an object")
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"config: missing={missing}, unknown={unknown}")

    config = InferenceConfig(
        model=_construct(ModelConfig, raw["model"], "model"),
        runtime=_construct(RuntimeConfig, raw["runtime"], "runtime"),
        preprocessing=_construct(PreprocessingConfig, raw["preprocessing"], "preprocessing"),
        prompts=_construct(PromptConfig, raw["prompts"], "prompts"),
        growing=_construct(GrowingConfig, raw["growing"], "growing"),
        selection=_construct(SelectionConfig, raw["selection"], "selection"),
        output=_construct(OutputConfig, raw["output"], "output"),
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )
    validate_config(config)
    return config


def resolved_config_dict(config: InferenceConfig) -> dict[str, Any]:
    payload = dataclasses.asdict(config)
    payload.pop("sha256")
    return payload
