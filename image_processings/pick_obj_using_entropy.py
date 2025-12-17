"""Select mask entry with lowest binary entropy of foreground ratio."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MaskEntry = Dict[str, Any]

__all__ = ["pick_obj_using_entropy"]


def _entry_entropy(entry: MaskEntry) -> Tuple[float, float]:
    logits = entry.get("logits")
    if logits is None:
        return float("inf"), float("nan")

    logits_arr = np.asarray(logits, dtype=np.float32)
    # sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits_arr))
    probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
    entropy = -probs * np.log2(probs) - (1.0 - probs) * np.log2(1.0 - probs)
    mean_entropy = float(np.mean(entropy))
    mean_prob = float(np.mean(probs))
    return mean_entropy, mean_prob


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
        entropy, ratio = _entry_entropy(entry)
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
