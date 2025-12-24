"""Prompt mapping utilities shared by TTA core and notebooks."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from skimage import transform

try:  # optional torch support
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

Array = Union[np.ndarray, "torch.Tensor"]
MASK_PROMPT_SIZE: Tuple[int, int] = (256, 256)


def _is_torch(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _ensure_mask_channel(mask: Array) -> Array:
    if _is_torch(mask):
        if mask.ndim == 2:
            return mask.unsqueeze(0)
        if mask.ndim == 3:
            return mask
        if mask.ndim == 4:
            return mask.squeeze(0)
        return mask
    if mask.ndim == 2:
        return mask[None, :, :]
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask.transpose(2, 0, 1)
    return mask


def _resize_mask(mask: Array, size: Tuple[int, int]) -> Array:
    if _is_torch(mask):
        mask_t = mask
        if mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=size, mode="nearest")
        return mask_t.squeeze(0)
    base = mask[0] if mask.ndim == 3 and mask.shape[0] == 1 else mask
    resized = transform.resize(base, size, preserve_range=True, anti_aliasing=False)
    if mask.ndim == 3 and mask.shape[0] == 1:
        resized = resized[None, :, :]
    return resized


def prepare_prompts_for_model(transform_view, prompts: Dict) -> Dict:
    """Map prompts to augmented view and normalize mask_input for SAM2."""
    mapped = transform_view.apply_prompts(prompts)
    mask_input = mapped.get("mask_input")
    if mask_input is not None:
        mask_input = _ensure_mask_channel(mask_input)
        mask_input = _resize_mask(mask_input, MASK_PROMPT_SIZE)
    return {
        **mapped,
        "mask_input": mask_input,
    }


def prepare_prompts_for_vis(transform_view, prompts: Dict) -> Dict:
    """Map prompts to augmented view and make mask_input 2D for display."""
    mapped = transform_view.apply_prompts(prompts)
    mask_input = mapped.get("mask_input")
    if mask_input is not None:
        if _is_torch(mask_input):
            mask_input = mask_input.detach().cpu()
        if mask_input.ndim == 3 and mask_input.shape[0] == 1:
            mask_input = mask_input[0]
    return {
        **mapped,
        "mask_input": mask_input,
    }
