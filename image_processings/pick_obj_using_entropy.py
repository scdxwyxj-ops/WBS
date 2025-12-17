"""Select mask entry with lowest binary entropy of foreground ratio."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

MaskEntry = Dict[str, object]

__all__ = ["pick_obj_using_entropy"]


def _mask_entropy(mask: np.ndarray) -> Tuple[float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    total = mask_bool.size
    if total == 0:
        return 1.0, 0.0

    foreground_ratio = float(mask_bool.sum() / total)
    p = np.clip(foreground_ratio, 1e-9, 1.0 - 1e-9)
    entropy = float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))
    return entropy, foreground_ratio


def pick_obj_using_entropy(
    img_rgb: np.ndarray,  # unused but kept for signature parity
    pool: List[MaskEntry],
    *,
    target_area_ratio: float = 0.0,  # unused but kept for signature parity
) -> Tuple[Optional[MaskEntry], Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    if not pool:
        return None, {}, []

    scored_details: List[Dict[str, float]] = []
    best_entry: Optional[MaskEntry] = None
    best_entropy = float("inf")

    entropies: List[float] = []
    ratios: List[float] = []

    for entry in pool:
        entropy, ratio = _mask_entropy(entry["mask"])
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
        "foreground_ratio": {"min": float(np.min(ratios)), "max": float(np.max(ratios))},
    }

    return best_entry, pool_stats, scored_details
