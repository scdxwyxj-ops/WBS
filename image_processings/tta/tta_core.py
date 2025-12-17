"""TTA framework: pseudo-label anchoring + entropy + multi-view consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from skimage import transform, segmentation


Array = np.ndarray


@dataclass(frozen=True)
class TTALossWeights:
    anchor: float = 1.0
    entropy: float = 0.1
    consistency: float = 0.5
    regularization: float = 0.0


@dataclass(frozen=True)
class TTAPartition:
    sure_fg: Array
    unsure: Array
    sure_bg: Array


@dataclass
class TTAStepOutputs:
    teacher: Array
    student_logits: Array
    student_probs: Array
    student_logits_aug: Optional[Array]
    student_probs_aug: Optional[Array]
    partition: TTAPartition
    losses: Dict[str, float]


def build_soft_teacher(masks: Iterable[Array], scores: Iterable[float]) -> Array:
    masks_list = [np.asarray(m, dtype=bool) for m in masks]
    scores_arr = np.asarray(list(scores), dtype=float)
    if not masks_list or masks_list[0].size == 0:
        raise ValueError("No masks provided for teacher construction.")

    scores_norm = scores_arr - scores_arr.min()
    scores_norm = scores_norm / (scores_norm.max() + 1e-6)
    weights = scores_norm + 1.0  # avoid zero weights

    teacher = np.zeros_like(masks_list[0], dtype=float)
    total_w = np.zeros_like(teacher, dtype=float)
    for m, w in zip(masks_list, weights):
        teacher += m.astype(float) * w
        total_w += w
    teacher = teacher / np.clip(total_w, 1e-6, None)
    return np.clip(teacher, 0.0, 1.0)


def partition_regions(
    nested_masks: List[Array],
    teacher: Array,
    inner_idx: int = 0,
    outer_idx: Optional[int] = None,
) -> TTAPartition:
    if not nested_masks:
        raise ValueError("nested_masks cannot be empty.")
    outer_idx = len(nested_masks) - 1 if outer_idx is None else outer_idx
    inner = np.asarray(nested_masks[max(0, inner_idx)], dtype=bool)
    outer = np.asarray(nested_masks[min(len(nested_masks) - 1, outer_idx)], dtype=bool)

    sure_fg = inner
    unsure = np.logical_and(outer, np.logical_not(inner))
    sure_bg = np.logical_not(outer)
    return TTAPartition(sure_fg=sure_fg, unsure=unsure, sure_bg=sure_bg)


def _binary_cross_entropy(pred: Array, target: Array, mask: Array) -> float:
    pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
    target = np.clip(target, 0.0, 1.0)
    m = mask.astype(float)
    if m.sum() == 0:
        return 0.0
    loss = -(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))
    return float((loss * m).sum() / m.sum())


def compute_anchor_loss(probs: Array, teacher: Array, partition: TTAPartition) -> float:
    mask_anchor = np.logical_or(partition.sure_fg, partition.sure_bg)
    target = np.where(partition.sure_fg, 1.0, 0.0)
    return _binary_cross_entropy(probs, target, mask_anchor)


def compute_entropy_loss(probs: Array, partition: TTAPartition) -> float:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    m = partition.unsure.astype(float)
    if m.sum() == 0:
        return 0.0
    return float((entropy * m).sum() / m.sum())


def compute_consistency_loss(
    probs: Array,
    probs_aug: Array,
    partition: TTAPartition,
    align_fn: Callable[[Array], Array],
) -> float:
    if probs_aug is None:
        return 0.0
    aligned = align_fn(probs_aug)
    diff = np.abs(aligned - probs)
    weight = np.where(partition.unsure, 1.0, 0.5)
    norm = weight.sum() + 1e-6
    return float((diff * weight).sum() / norm)


def _sigmoid(logits: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-logits))


def _soft_dice(a: Array, b: Array, weight: Optional[Array] = None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if weight is None:
        weight = np.ones_like(a, dtype=float)
    intersect = np.sum(weight * a * b)
    denom = np.sum(weight * a) + np.sum(weight * b) + 1e-6
    return float(2.0 * intersect / denom)


class TTAPipeline:
    """Lightweight, model-agnostic TTA loop skeleton."""

    def __init__(
        self,
        predictor,
        *,
        loss_weights: Optional[TTALossWeights] = None,
        augment_fn: Optional[Callable[[Array], Sequence[Tuple[Array, Callable[[Array], Array]]]]] = None,
        optimizer_step_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        self.predictor = predictor
        self.loss_weights = loss_weights or TTALossWeights()
        self.augment_fn = augment_fn
        self.optimizer_step_fn = optimizer_step_fn

    def _predict_probs(self, image: Array, prompts: Dict) -> Tuple[Array, Array]:
        logits, _, _ = self.predictor.predict(**prompts, return_logits=True)
        logits = np.asarray(logits[0])
        return logits, _sigmoid(logits)

    def step(
        self,
        image: Array,
        prompts: Dict,
        nested_masks: List[Array],
        scores: List[float],
        *,
        inner_idx: int = 0,
        outer_idx: Optional[int] = None,
    ) -> TTAStepOutputs:
        teacher = build_soft_teacher(nested_masks, scores)
        partition = partition_regions(nested_masks, teacher, inner_idx=inner_idx, outer_idx=outer_idx)

        logits, probs = self._predict_probs(image, prompts)

        loss_anchor = compute_anchor_loss(probs, teacher, partition) * self.loss_weights.anchor
        loss_entropy = compute_entropy_loss(probs, partition) * self.loss_weights.entropy
        loss_cons = 0.0
        logits_aug = None
        probs_aug = None

        if self.augment_fn is not None:
            views = self.augment_fn(image)
            view_probs: List[Array] = []
            view_logits: List[Array] = []
            for img_aug, _ in views:
                lg, pr = self._predict_probs(img_aug, prompts)
                view_logits.append(lg)
                view_probs.append(pr)
            if view_probs:
                logits_aug = view_logits[0]
                probs_aug = view_probs[0]
                # Consistency: dice between base probs and aligned aug probs
                cons_losses = []
                base_prob = probs
                for (_, align_back), pr_aug in zip(views, view_probs):
                    aligned = align_back(pr_aug)
                    cons_losses.append(1.0 - _soft_dice(base_prob, aligned, weight=np.where(partition.unsure, 1.0, 0.5)))
                if cons_losses:
                    loss_cons = float(np.mean(cons_losses)) * self.loss_weights.consistency

        total_loss = loss_anchor + loss_entropy + loss_cons
        losses = {
            "anchor": loss_anchor,
            "entropy": loss_entropy,
            "consistency": loss_cons,
            "total": total_loss,
        }

        if self.optimizer_step_fn is not None:
            self.optimizer_step_fn(losses)

        return TTAStepOutputs(
            teacher=teacher,
            student_logits=logits,
            student_probs=probs,
            student_logits_aug=logits_aug,
            student_probs_aug=probs_aug,
            partition=partition,
            losses=losses,
        )


def default_multi_view_augment(
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    do_flip: bool = True,
) -> Callable[[Array], List[Tuple[Array, Callable[[Array], Array]]]]:
    def _aug(image: Array) -> List[Tuple[Array, Callable[[Array], Array]]]:
        h, w = image.shape[:2]
        candidates: List[Tuple[Array, Callable[[Array], Array]]] = []
        for s in scales:
            new_h, new_w = int(h * s), int(w * s)
            img_resized = transform.resize(image, (new_h, new_w), preserve_range=True, anti_aliasing=True)

            def _align_back_factory(target_shape):
                def _align(arr: Array) -> Array:
                    return transform.resize(arr, target_shape, preserve_range=True, anti_aliasing=True)
                return _align

            candidates.append((img_resized.astype(image.dtype), _align_back_factory((h, w))))

            if do_flip:
                flipped = np.flip(img_resized, axis=1)

                def _align_back_flip(target_shape):
                    def _align(arr: Array) -> Array:
                        arr = np.flip(arr, axis=1)
                        return transform.resize(arr, target_shape, preserve_range=True, anti_aliasing=True)
                    return _align

                candidates.append((flipped.astype(image.dtype), _align_back_flip((h, w))))

        # sample up to 2 views per step (deterministic order)
        return candidates[:2]

    return _aug
