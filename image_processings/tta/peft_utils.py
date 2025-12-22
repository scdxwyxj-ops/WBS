"""Manual LoRA helpers for SAM2 mask decoder."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


def _default_target_modules() -> List[str]:
    # Attention projections in sam_mask_decoder.transformer.*
    return ["q_proj", "k_proj", "v_proj", "out_proj"]


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation (no structural changes to caller)."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base module.")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha / max(1, r))
        self.dropout = nn.Dropout(p=float(dropout))

        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def _get_parent_module(root: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    current = root
    for part in parts[:-1]:
        current = getattr(current, part)
    return current


def apply_lora_to_mask_decoder(
    model: nn.Module,
    *,
    r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    target_modules: Optional[Iterable[str]] = None,
) -> List[str]:
    """Attach LoRA adapters to SAM2 mask decoder attention projections."""
    if not hasattr(model, "sam_mask_decoder"):
        raise AttributeError("Expected model.sam_mask_decoder for SAM2 mask decoder.")

    targets = set(target_modules or _default_target_modules())
    replaced: List[str] = []

    # Freeze all parameters; LoRA modules will be trainable.
    for p in model.parameters():
        p.requires_grad = False

    decoder = model.sam_mask_decoder
    for name, module in decoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in targets:
            continue
        parent = _get_parent_module(decoder, name)
        child_name = name.split(".")[-1]
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=lora_alpha, dropout=lora_dropout))
        replaced.append(name)

    if not replaced:
        raise ValueError("No target modules matched for LoRA injection.")

    return replaced
