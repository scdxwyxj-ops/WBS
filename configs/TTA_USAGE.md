# TTA Config Usage

This document describes `configs/tta_config.json` (use `configs/tta_config.example.json` as a template).

## Key Sections

### `tta_steps`
- Number of TTA update steps per image.

### `batch_size`
- Reserved for future use. Current implementation uses per-image TTA.

### `pseudo_label`
- `strategy`: `"score_top_k"` or `"cluster_middle"`.
  - `score_top_k`: choose top‑k masks by score.
  - `cluster_middle`: cluster by mask area and use the middle cluster.
- `top_k_masks`: number of masks used when `score_top_k` is selected.

### `prompt`
- `mask_prompt_source`:
  - `"none"`: do not use mask prompt.
  - `"pipeline_mask_prompt"` / `"mask_prompt"`: use `PromptBundle.mask_prompt`.
  - `"pipeline_low_res"` / `"low_res"`: use `PromptBundle.low_res_mask`.

### `loss_weights`
- `anchor`: weight for pseudo‑label supervision.
- `entropy`: weight for entropy minimisation.
- `consistency`: weight for multi‑view consistency.
- `regularization`: reserved (0 by default).

### `augment`
- `scales`: list of scale factors for multi‑view augmentation.
- `use_flip`: whether to add horizontal flips.
- `views_per_step`: how many views are sampled per step (deterministic order).

### `optimizer`
- `lr`, `weight_decay`, `max_grad_norm`: LoRA optimizer settings.

### `lora`
- `target`: `"mask_decoder"` to inject LoRA into SAM2 mask decoder.
- `rank`, `alpha`, `dropout`: LoRA hyper‑parameters.
- `target_modules`: list of linear layers to adapt (e.g. `q_proj`, `k_proj`, `v_proj`, `out_proj`).

### `teacher_student`
- Currently unused (EMA teacher not enabled).

## Outputs
TTA runs save:
- `train.log`, `summary.json`, `per_image_metrics.json`, `per_image_metrics.csv`
- `tta_gain_histogram.json` (+ `.png` if matplotlib available)
