"""Test-time adaptation utilities for SAM2-based segmentation."""

from .tta_core import (
    TTAPipeline,
    TTALossWeights,
    TTAStepOutputs,
    compute_supervision_loss,
    compute_entropy_loss,
    default_multi_view_augment,
)
from .prompt_utils import prepare_prompts_for_model, prepare_prompts_for_vis
from .peft_utils import apply_lora_to_mask_decoder
from .tta_runner import run_tta_from_pool

__all__ = [
    "TTAPipeline",
    "TTALossWeights",
    "TTAStepOutputs",
    "compute_supervision_loss",
    "compute_entropy_loss",
    "default_multi_view_augment",
    "prepare_prompts_for_model",
    "prepare_prompts_for_vis",
    "apply_lora_to_mask_decoder",
    "run_tta_from_pool",
]
