"""Mask pool clustering utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from .mask_pool import entry_area_ratio, entry_score

__all__ = ["cluster_masks_by_area", "select_middle_cluster_entry"]

MaskEntry = Dict[str, Any]


def _initial_centers(values: np.ndarray, n_clusters: int) -> np.ndarray:
    """Select min/median/max values as initial centres."""
    sorted_vals = np.sort(values)
    centers: List[List[float]] = []
    if n_clusters >= 1:
        centers.append([sorted_vals[0]])
    if n_clusters >= 2:
        centers.append([sorted_vals[len(sorted_vals) // 2]])
    if n_clusters >= 3:
        centers.append([sorted_vals[-1]])

    # If duplicates reduce diversity, repeat median to fill remaining slots.
    while len(centers) < n_clusters:
        centers.append([sorted_vals[len(sorted_vals) // 2]])

    return np.asarray(centers, dtype=np.float32)


def cluster_masks_by_area(
    entries: Sequence[MaskEntry],
    *,
    n_clusters: int = 3,
    random_state: Optional[int] = 0,
) -> Tuple[List[MaskEntry], Dict[str, Any]]:
    """Cluster mask entries by foreground area and return the middle cluster."""
    if not entries:
        return [], {"centers": [], "selected_label": None}

    n_clusters = max(1, min(n_clusters, len(entries)))
    areas = np.array([entry_area_ratio(entry) for entry in entries], dtype=np.float32)
    features = areas.reshape(-1, 1)

    if n_clusters == 1:
        return list(entries), {"centers": [float(areas.mean())], "selected_label": 0}

    init_centers = _initial_centers(areas, n_clusters)
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        random_state=0 if random_state is None else int(random_state),
    )
    labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_.reshape(-1)

    order = np.argsort(centers)
    selected_label = int(order[len(order) // 2])

    selected_entries = [
        entry for entry, label in zip(entries, labels) if int(label) == selected_label
    ]
    meta = {
        "centers": centers.tolist(),
        "selected_label": selected_label,
        "areas": areas.tolist(),
    }
    return selected_entries, meta


def select_middle_cluster_entry(
    entries: Sequence[MaskEntry],
    *,
    n_clusters: int = 3,
    random_state: Optional[int] = 0,
) -> Tuple[Optional[MaskEntry], Dict[str, Any]]:
    """Return the highest-scoring entry from the middle-area cluster."""
    cluster_entries, meta = cluster_masks_by_area(entries, n_clusters=n_clusters, random_state=random_state)
    if not cluster_entries:
        return None, meta

    best_entry = max(cluster_entries, key=entry_score)
    meta = dict(meta)
    meta["selected_size"] = len(cluster_entries)
    return best_entry, meta
