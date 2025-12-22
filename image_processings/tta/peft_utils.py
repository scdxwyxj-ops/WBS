"""PEFT/LoRA helpers for SAM2 mask decoder."""

from __future__ import annotations

from typing import Iterable, List, Optional


def _default_target_modules() -> List[str]:
    # Attention projections in sam_mask_decoder.transformer.*
    return ["q_proj", "k_proj", "v_proj", "out_proj"]


import torch.nn as nn


class _MaskDecoderPeftWrapper(nn.Module):
    """Adapter to make MaskDecoder compatible with PEFT's input_ids signature."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids=None, **kwargs):
        return self.inner(**kwargs)


def apply_lora_to_mask_decoder(
    model,
    *,
    r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    target_modules: Optional[Iterable[str]] = None,
) -> None:
    """Attach LoRA adapters to SAM2 mask decoder attention projections."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("peft is required for LoRA. Install with: pip install peft") from exc

    modules = list(target_modules) if target_modules is not None else _default_target_modules()
    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    if not hasattr(model, "sam_mask_decoder"):
        raise AttributeError("Expected model.sam_mask_decoder for SAM2 mask decoder.")

    wrapped = _MaskDecoderPeftWrapper(model.sam_mask_decoder)
    model.sam_mask_decoder = get_peft_model(wrapped, lora_cfg)
