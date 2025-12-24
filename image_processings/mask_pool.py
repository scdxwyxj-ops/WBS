"""Mask pool schema helpers to keep selection strategies consistent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

MaskEntry = Dict[str, Any]


@dataclass(frozen=True)
class MaskPoolEntry:
    mask: np.ndarray
    score: float
    logits: Optional[np.ndarray]
    score_details: Optional[Dict[str, Any]]
    candidate_id: Optional[int]
    iteration: Optional[int]
    prompts: Any
    positive_mask: Optional[np.ndarray]
    raw: MaskEntry

    @classmethod
    def from_entry(cls, entry: MaskEntry) -> "MaskPoolEntry":
        mask = np.asarray(entry.get("mask"), dtype=bool)
        score_details = entry.get("score_details") or {}
        score = float(score_details.get("score", entry.get("score", 0.0) or 0.0))
        return cls(
            mask=mask,
            score=score,
            logits=entry.get("logits"),
            score_details=score_details or None,
            candidate_id=entry.get("candidate_id"),
            iteration=entry.get("iteration"),
            prompts=entry.get("prompts"),
            positive_mask=entry.get("positive_mask"),
            raw=entry,
        )


def entry_mask(entry: MaskEntry) -> np.ndarray:
    return MaskPoolEntry.from_entry(entry).mask


def entry_score(entry: MaskEntry) -> float:
    return MaskPoolEntry.from_entry(entry).score


def entry_area_ratio(entry: MaskEntry) -> float:
    mask_bool = entry_mask(entry)
    total = mask_bool.size
    if total == 0:
        return 0.0
    return float(mask_bool.sum() / total)
