"""Glue utilities to run TTA on masks produced by the existing pipeline."""

from __future__ import annotations

from typing import Dict, List, Sequence, Optional

import numpy as np

from image_processings.mask_cluster import cluster_masks_by_area
from .tta_core import TTAPipeline, TTALossWeights, default_multi_view_augment


def _select_pseudo_masks(
    mask_pool: Sequence[Dict[str, object]],
    *,
    selection_strategy: str = "score_top_k",
    top_k: int = 3,
) -> List[np.ndarray]:
    """Select masks from the pool according to strategy."""
    if not mask_pool:
        raise ValueError("mask_pool is empty")
    strategy = selection_strategy.lower()
    if strategy == "cluster_middle":
        cluster_entries, _ = cluster_masks_by_area(mask_pool, n_clusters=3)
        if not cluster_entries:
            cluster_entries = list(mask_pool)
        return [np.asarray(entry["mask"], dtype=bool) for entry in cluster_entries]

    top_k = max(1, int(top_k))
    sorted_pool = sorted(mask_pool, key=lambda e: float(e.get("score", 0.0)), reverse=True)[:top_k]
    return [np.asarray(entry["mask"], dtype=bool) for entry in sorted_pool]


def _merge_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    merged = None
    for mask in masks:
        merged = mask if merged is None else np.logical_or(merged, mask)
    if merged is None:
        raise ValueError("No masks selected for pseudo label.")
    return np.asarray(merged, dtype=bool)


def run_tta_from_pool(
    predictor,
    image: np.ndarray,
    mask_pool: Sequence[Dict[str, object]],
    prompts: Dict,
    *,
    loss_weights: Optional[TTALossWeights] = None,
    selection_strategy: str = "score_top_k",
    top_k: int = 3,
    augment_fn=None,
    optimizer_step_fn=None,
) -> Dict[str, object]:
    """Run one TTA step using a pseudo-label selected from the pool."""
    selected_masks = _select_pseudo_masks(
        mask_pool,
        selection_strategy=selection_strategy,
        top_k=top_k,
    )
    pseudo_mask = _merge_masks(selected_masks)

    tta = TTAPipeline(
        predictor,
        loss_weights=loss_weights or TTALossWeights(),
        augment_fn=augment_fn or default_multi_view_augment(),
        optimizer_step_fn=optimizer_step_fn,
    )
    out = tta.step(image, prompts, pseudo_mask)

    return {
        "pseudo_mask": pseudo_mask,
        "selected_masks": selected_masks,
        "tta_outputs": out,
    }
