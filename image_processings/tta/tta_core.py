"""TTA framework: pseudo-label supervision + entropy + multi-view consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from skimage import transform, segmentation
import torch.nn.functional as F


Array = np.ndarray

try:  # optional torch support
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class TTALossWeights:
    anchor: float = 1.0
    entropy: float = 0.1
    consistency: float = 0.5
    regularization: float = 0.0


@dataclass
class TTAStepOutputs:
    pseudo_mask: Array
    student_logits: Array
    student_probs: Array
    student_logits_aug: Optional[Array]
    student_probs_aug: Optional[Array]
    losses: Dict[str, float]
    total_loss_tensor: Optional[Any]


def _is_torch(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _binary_cross_entropy(pred: Array, target: Array):
    if _is_torch(pred):
        pred_t = pred.clamp(1e-6, 1.0 - 1e-6)
        target_t = target
        loss = -(target_t * torch.log(pred_t) + (1.0 - target_t) * torch.log(1.0 - pred_t))
        return loss.mean()
    pred_np = np.clip(pred, 1e-6, 1.0 - 1e-6)
    target_np = np.clip(target, 0.0, 1.0)
    loss = -(target_np * np.log(pred_np) + (1.0 - target_np) * np.log(1.0 - pred_np))
    return float(np.mean(loss))


def compute_supervision_loss(probs: Array, pseudo_mask: Array) -> float:
    if _is_torch(probs):
        target = torch.as_tensor(pseudo_mask, device=probs.device, dtype=probs.dtype)
        if target.shape != probs.shape:
            target = target.unsqueeze(0).unsqueeze(0) if target.ndim == 2 else target.unsqueeze(0)
            target = torch.nn.functional.interpolate(
                target,
                size=probs.shape[-2:],
                mode="nearest",
            )
            if probs.ndim == 2:
                target = target.squeeze(0).squeeze(0)
            else:
                target = target.squeeze(0)
        return _binary_cross_entropy(probs, target)
    target = np.asarray(pseudo_mask, dtype=float)
    if target.shape != probs.shape:
        target = transform.resize(target, probs.shape, preserve_range=True, order=0, anti_aliasing=False)
    return _binary_cross_entropy(probs, target)


def compute_entropy_loss(probs: Array) -> float:
    if _is_torch(probs):
        p = probs.clamp(1e-6, 1.0 - 1e-6)
        entropy = -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)
        return entropy.mean()
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    return float(np.mean(entropy))


def _sigmoid(logits: Array) -> Array:
    if _is_torch(logits):
        return torch.sigmoid(logits)
    return 1.0 / (1.0 + np.exp(-logits))


def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        return t.unsqueeze(1) if t.shape[0] != 1 else t.unsqueeze(0)
    return t


def _soft_dice(a: Array, b: Array, weight: Optional[Array] = None) -> float:
    if _is_torch(a) or _is_torch(b):
        a_t = a if _is_torch(a) else torch.as_tensor(a)
        b_t = b if _is_torch(b) else torch.as_tensor(b, device=a_t.device)
        if a_t.shape != b_t.shape:
            b_t = F.interpolate(_ensure_4d(b_t), size=a_t.shape[-2:], mode="bilinear", align_corners=False)
            b_t = b_t.squeeze(0).squeeze(0) if a_t.ndim == 2 else b_t.squeeze(0)
        if weight is None:
            weight_t = torch.ones_like(a_t)
        else:
            weight_t = weight if _is_torch(weight) else torch.as_tensor(weight, device=a_t.device)
        intersect = torch.sum(weight_t * a_t * b_t)
        denom = torch.sum(weight_t * a_t) + torch.sum(weight_t * b_t) + 1e-6
        return 2.0 * intersect / denom
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
        optimizer_step_fn: Optional[Callable[[Any, Dict[str, float]], None]] = None,
    ) -> None:
        self.predictor = predictor
        self.loss_weights = loss_weights or TTALossWeights()
        self.augment_fn = augment_fn
        self.optimizer_step_fn = optimizer_step_fn
        self._train_with_grad = optimizer_step_fn is not None and torch is not None

    def _predict_probs(self, image: Array, prompts: Dict) -> Tuple[Array, Array]:
        if self._train_with_grad and hasattr(self.predictor, "model"):
            return self._predict_probs_with_grad(image, prompts)
        logits, _, _ = self.predictor.predict(**prompts, return_logits=True)
        logits = logits[0]
        if not _is_torch(logits):
            logits = np.asarray(logits)
        return logits, _sigmoid(logits)

    def _predict_probs_with_grad(self, image: Array, prompts: Dict) -> Tuple[Array, Array]:
        """Run SAM2 mask decoder forward with gradients (decoder-only training)."""
        self.predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
            prompts.get("point_coords"),
            prompts.get("point_labels"),
            prompts.get("box"),
            prompts.get("mask_input"),
            normalize_coords=True,
        )
        point_inputs = None
        if unnorm_coords is not None and labels is not None:
            point_inputs = {
                "point_coords": unnorm_coords,
                "point_labels": labels,
            }
        features = self.predictor._features["image_embed"]
        high_res_feats = self.predictor._features["high_res_feats"]
        low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, _ = (
            self.predictor.model._forward_sam_heads(
                backbone_features=features,
                point_inputs=point_inputs,
                mask_inputs=mask_input,
                high_res_features=high_res_feats,
                multimask_output=prompts.get("multimask_output", False),
            )
        )
        logits = high_res_masks[:, 0]
        return logits, _sigmoid(logits)

    def step(
        self,
        image: Array,
        prompts: Dict,
        pseudo_mask: Array,
    ) -> TTAStepOutputs:
        logits, probs = self._predict_probs(image, prompts)
        loss_sup = compute_supervision_loss(probs, pseudo_mask) * self.loss_weights.anchor
        loss_entropy = compute_entropy_loss(probs) * self.loss_weights.entropy
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
                cons_losses = []
                base_prob = probs
                for (_, align_back), pr_aug in zip(views, view_probs):
                    aligned = align_back(pr_aug)
                    cons_losses.append(1.0 - _soft_dice(base_prob, aligned))
                if cons_losses:
                    if _is_torch(cons_losses[0]):
                        loss_cons = torch.stack(cons_losses).mean() * self.loss_weights.consistency
                    else:
                        loss_cons = float(np.mean(cons_losses)) * self.loss_weights.consistency

        total_loss = loss_sup + loss_entropy + loss_cons
        losses = {
            "supervision": float(loss_sup.detach().cpu()) if _is_torch(loss_sup) else float(loss_sup),
            "entropy": float(loss_entropy.detach().cpu()) if _is_torch(loss_entropy) else float(loss_entropy),
            "consistency": float(loss_cons.detach().cpu()) if _is_torch(loss_cons) else float(loss_cons),
            "total": float(total_loss.detach().cpu()) if _is_torch(total_loss) else float(total_loss),
        }

        if self.optimizer_step_fn is not None:
            self.optimizer_step_fn(total_loss, losses)

        return TTAStepOutputs(
            pseudo_mask=np.asarray(pseudo_mask, dtype=bool),
            student_logits=logits,
            student_probs=probs,
            student_logits_aug=logits_aug,
            student_probs_aug=probs_aug,
            losses=losses,
            total_loss_tensor=total_loss if _is_torch(total_loss) else None,
        )


def default_multi_view_augment(
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    do_flip: bool = True,
    views_per_step: int = 2,
) -> Callable[[Array], List[Tuple[Array, Callable[[Array], Array]]]]:
    def _aug(image: Array) -> List[Tuple[Array, Callable[[Array], Array]]]:
        h, w = image.shape[:2]
        candidates: List[Tuple[Array, Callable[[Array], Array]]] = []
        for s in scales:
            new_h, new_w = int(h * s), int(w * s)
            if _is_torch(image):
                img_resized = F.interpolate(
                    image.unsqueeze(0) if image.ndim == 3 else image,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                if image.ndim == 3:
                    img_resized = img_resized.squeeze(0)
            else:
                img_resized = transform.resize(image, (new_h, new_w), preserve_range=True, anti_aliasing=True)

            def _align_back_factory(target_shape):
                def _align(arr: Array) -> Array:
                    if _is_torch(arr):
                        arr_t = _ensure_4d(arr)
                        out = F.interpolate(arr_t, size=target_shape, mode="bilinear", align_corners=False)
                        return out.squeeze(0).squeeze(0)
                    return transform.resize(arr, target_shape, preserve_range=True, anti_aliasing=True)
                return _align

            if _is_torch(img_resized):
                candidates.append((img_resized, _align_back_factory((h, w))))
            else:
                candidates.append((img_resized.astype(image.dtype), _align_back_factory((h, w))))

            if do_flip:
                flipped = torch.flip(img_resized, dims=[-1]) if _is_torch(img_resized) else np.flip(img_resized, axis=1)

                def _align_back_flip(target_shape):
                    def _align(arr: Array) -> Array:
                        if _is_torch(arr):
                            arr = torch.flip(arr, dims=[-1])
                            arr_t = _ensure_4d(arr)
                            out = F.interpolate(arr_t, size=target_shape, mode="bilinear", align_corners=False)
                            return out.squeeze(0).squeeze(0)
                        arr = np.flip(arr, axis=1)
                        return transform.resize(arr, target_shape, preserve_range=True, anti_aliasing=True)
                    return _align

                if _is_torch(img_resized):
                    candidates.append((flipped, _align_back_flip((h, w))))
                else:
                    candidates.append((flipped.astype(image.dtype), _align_back_flip((h, w))))

        # sample a fixed number of views per step (deterministic order)
        max_views = max(1, int(views_per_step))
        return candidates[:max_views]

    return _aug
