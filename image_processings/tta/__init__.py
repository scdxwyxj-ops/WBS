"""Test-time adaptation utilities for SAM2-based segmentation."""

from .tta_core import (
    TTAPipeline,
    TTALossWeights,
    TTAPartition,
    TTAStepOutputs,
    build_soft_teacher,
    partition_regions,
    compute_anchor_loss,
    compute_entropy_loss,
    compute_consistency_loss,
    default_multi_view_augment,
)
from .tta_runner import run_tta_from_pool

__all__ = [
    "TTAPipeline",
    "TTALossWeights",
    "TTAPartition",
    "TTAStepOutputs",
    "build_soft_teacher",
    "partition_regions",
    "compute_anchor_loss",
    "compute_entropy_loss",
    "compute_consistency_loss",
    "default_multi_view_augment",
    "run_tta_from_pool",
]
