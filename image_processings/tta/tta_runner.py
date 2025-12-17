"""Glue utilities to run TTA on masks produced by the existing pipeline."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

from .tta_core import TTAPipeline, TTALossWeights, build_soft_teacher, partition_regions, default_multi_view_augment


def _build_nested_from_pool(mask_pool: Sequence[Dict[str, object]], top_k: int = 3) -> Tuple[List[np.ndarray], List[float]]:
    """Select top-k masks by score and create a nested list via cumulative union."""
    if not mask_pool:
        raise ValueError("mask_pool is empty")
    sorted_pool = sorted(mask_pool, key=lambda e: float(e.get("score", 0.0)), reverse=True)[:top_k]
    nested_masks: List[np.ndarray] = []
    scores: List[float] = []
    accum = None
    for entry in sorted_pool:
        mask = np.asarray(entry["mask"], dtype=bool)
        accum = mask if accum is None else np.logical_or(accum, mask)
        nested_masks.append(accum.copy())
        scores.append(float(entry.get("score", 0.0)))
    return nested_masks, scores


def run_tta_from_pool(
    predictor,
    image: np.ndarray,
    mask_pool: Sequence[Dict[str, object]],
    prompts: Dict,
    *,
    loss_weights: Optional[TTALossWeights] = None,
    top_k: int = 3,
    augment_fn=None,
    optimizer_step_fn=None,
) -> Dict[str, object]:
    """Run one TTA step using masks from the existing pipeline."""
    nested_masks, scores = _build_nested_from_pool(mask_pool, top_k=top_k)
    teacher = build_soft_teacher(nested_masks, scores)
    partition = partition_regions(nested_masks, teacher)

    tta = TTAPipeline(
        predictor,
        loss_weights=loss_weights or TTALossWeights(),
        augment_fn=augment_fn or default_multi_view_augment(),
        optimizer_step_fn=optimizer_step_fn,
    )
    out = tta.step(image, prompts, nested_masks, scores)

    return {
        "teacher": teacher,
        "partition": partition,
        "tta_outputs": out,
    }
