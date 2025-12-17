# Pipeline Configuration Guide

This document describes how to edit `configs/pipeline.json` to control the
unsupervised SAM2 segmentation pipeline.

## Overview

`pipeline.json` is loaded by `debug_tests/debug_test.py` (and any other Python
entry point that calls `configs.pipeline_config.load_pipeline_config`). Every
field is validated and converted into typed dataclasses, so missing keys or
invalid values will raise errors immediately. Update the sections below to
tune datasets, SLIC pre-processing, iterative prompt expansion, and SAM2
inference behaviour without touching code.

```text
pipeline.json
├── dataset              # Dataset source and resizing
├── preprocessing        # SLIC and graph construction
├── algorithm            # Prompt selection, scoring & iteration loop
└── sam                  # Predictor call options
```

## `dataset`

- `name` *(str)* – Name of the dataset directory under the data root (see
  `CONSTANT.json:data_path`).
- `target_long_edge` *(int | null)* – Optional long-edge resize applied when
  loading images and masks via `datasets.load_dataset`. Use this when you want
  the raw data scaled to a fixed size before SLIC.

## `preprocessing`

- `image_size` *(int)* – Target square size fed into `image_pre_seg.image_i_segment`.
  Images are resized and padded so the long edge matches this value.
- `num_graph_nodes` *(int)* – Upper bound on SLIC superpixels; also controls the
  graph node budget.
- `slic` *(object)* – Settings forwarded to `skimage.segmentation.slic`:
  - `compactness` *(float)* – Colour vs. spatial weighting. Higher → smoother, more
    regular superpixels.
  - `sigma` *(float)* – Gaussian blur strength before SLIC.
  - `min_size_factor`, `max_size_factor` *(float)* – Size bounds relative to the
    average superpixel size (requires newer `skimage`).

## `algorithm`

Controls the iterative prompt-promotion loop (`image_processings.info.Info`).

- `negative_pct` *(float)* – Fraction of superpixels on image borders marked as
  initial negative prompts.
- `score_lower_bound` *(float)* – Minimum per-node logit mean required to keep
  a candidate.
- `threshold` *(object)* – How we convert logits into foreground labels:
  - `mode` *("constant" | "mean" | "scaled_mean")* – Use the supplied value, the
    mask mean, or `value × mask_mean`.
  - `value` *(float)* – Scalar used by the chosen mode.
- `candidate_top_k` *(int)* – Number of neighbour candidates evaluated by SAM2
  each iteration (the algorithm picks the best score among them).
- `max_iterations` *(int)* – Hard stop on promotion rounds per image.
- `augment_positive_points` *(bool)* – Enable convex-hull based positive point
  augmentation after promotions.
- `use_subset_points` *(bool)* – Whether to run spatial filtering on positive
  prompts (keeps them within a central window and separated by
  `min_point_distance`).
- `center_range` *(list[float, float])* – Normalised min/max for the central window.
- `min_point_distance` *(float)* – Minimum Euclidean distance between filtered
  positive points (pixels).
- `use_convex_hull` *(bool)* – If true, build mask prompts from the convex hull
  of current foreground segments (selective hull controlled by the threshold).
- `convex_hull_threshold` *(float)* – Area ratio threshold for hull replacement.
- `initial_color_mode` *("red" | "dark")* – How to rank nodes when seeding
  positives/negatives (`red` = bright red first, `dark` = darker regions first).
- `initial_positive_count` *(int)* – Number of seed positive nodes.
- `mask_pool_iou_threshold` *(float)* – IoU threshold used to deduplicate the
  stored mask pool before final selection.
- `target_area_ratio` *(float)* – Target foreground ratio when scoring masks
  with the heuristic selector (entropy selector ignores it and only uses logits).
- `selection_strategy` *("heuristic" | "entropy" | "cluster_middle")* – How the
  final mask is chosen: heuristic scoring, entropy minimisation, or cluster-based
  re-ranking (`mask_cluster.select_middle_cluster_entry` uses the current pool).

## `sam`

Options passed directly into `SAM2ImagePredictor.predict`. This is separate
from the algorithm-level mask prompt handling: both are used.

- `multimask_output` *(bool)* – Enable multi-mask sampling if you want multiple
  hypotheses returned per call (defaults to `False` for efficiency).
- `mask_prompt_source` *("slic" | "previous_low_res" | "none")* – Single switch
  for mask prompts fed into `mask_input`:
  - `slic` – Use the foreground mask built from the promoted SLIC segments
    (convex hull optional). This also triggers creation of a low-res mask inside
    the algorithm loop.
  - `previous_low_res` – Reuse the last low-res mask returned by SAM.
  - `none` – Disable mask prompting entirely.
- `refine_with_previous_low_res` *(bool)* – After the main loop, optionally run
  extra SAM passes seeded with the last low-res mask.
- `refine_rounds` *(int)* – Number of refinement passes when the above flag is
  true.

## Workflow Tips

1. Adjust `dataset.name` and `target_long_edge` when switching dataset splits.
2. Tune `candidate_top_k` and `score_lower_bound` jointly: higher values require
   more confident logits but reduce false positives.
3. Disable `augment_positive_points` and/or `use_convex_hull` to evaluate their
   impact on new datasets.
4. When experimenting, copy `pipeline.json` and point `CONSTANT.json:pipeline_cfg`
   to the variant so you can switch configurations quickly.
5. Mask prompts are now controlled by a single switch: set `sam.mask_prompt_source`
   to `slic`, `previous_low_res`, or `none` depending on whether you want SLIC
   foreground, SAM’s last low-res output, or no mask prompt.
