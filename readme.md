# Notes for WBS

## Overview
- Unsupervised white blood cell segmentation around SAM2 prompt refinement with a scored mask pool and optional clustering for final selection.
- Main entrypoint `debug_tests/debug_test.py` and batch runner `debug_tests/run_full_experiment.py` wire SLIC preprocessing → iterative prompt promotion → mask-pool scoring/selection → optional low-res refinement → evaluation/visualisation.
- Configuration lives in `CONSTANT.json` (paths + pipeline config pointer) and `configs/pipeline.json` (dataset, preprocessing, algorithm, SAM options).

## Directory Guide
- `debug_tests/` – runnable scripts (`debug_test.py` main, `run_full_experiment.py` batch runner, `calculate_foreground_ratio.py` stats helper), notebooks, smoke tests.
- `image_processings/` – preprocessing (`image_pre_seg.py` + `simple_graph` fallback), prompt state machine (`info.py`), node definition, mask clustering/scoring (`mask_cluster.py`, `pick_obj.py`), post-processing (`image_post_process.py`), logging helpers.
- `configs/` – dataclasses + loader for `pipeline.json`, usage notes.
- `datasets/` – dataset loader utilities respecting `CONSTANT.json:data_path`.
- `metrics/` – IoU/Dice/mAP calculators and visualisation helpers.
- `assets/` – output artefacts saved by debug and experiment scripts.

## Data & Configuration
- `CONSTANT.json`
  - `data_path` – default dataset root used by `datasets.dataset.load_dataset`.
  - `checkpoint`, `model_cfg` – forwarded to `sam2.build_sam.build_sam2`.
  - `pipeline_cfg` – path to the active `pipeline.json`.
- `configs/pipeline.json` (validated via `configs/pipeline_config.load_pipeline_config`)
  - `dataset`: dataset folder name + optional `target_long_edge` resize.
  - `preprocessing`: `image_size`, graph node budget, and SLIC hyper-parameters.
  - `algorithm`: consumed by `image_processings.info.AlgorithmSettings`.
    - Includes negative/positive prompt balance, threshold mode/value, candidate_top_k, convex hull toggle, point filtering window, initial color mode (`red` or `dark`), initial positive count, mask pool IoU dedup threshold, target area ratio for scoring, and `selection_strategy` (`heuristic`, `entropy`, `edge_gradient`, or `cluster_middle`).
  - `sam`: predictor settings (`mask_prompt_source`, multimask, refine-with-previous-low-res toggle + rounds).
- Legacy `image_processings/config.py` contains older hard-coded config dictionaries; current pipeline does not import it.

## Preprocessing (`image_processings/image_pre_seg.py`)
- `change_image_type` handles conversions between numpy / torch tensor / PIL Image.
- `resize_slic_pad`
  - Resizes image to preserve aspect ratio with max edge = `new_size`, pads to square.
  - Runs `skimage.segmentation.slic` with configurable `n_segments`, `compactness`, `sigma`, `min_size_factor`, `max_size_factor`; returns padded tensors and raw segment labels.
- `segment2graph`
  - Builds an undirected graph (NetworkX if installed, else `SimpleGraph` fallback) with `num_node_for_graph` nodes.
  - Adds a global node (`num_node_for_graph-1`), connects every superpixel to it, adds self-loops, and edges between adjacent labels.
- `image_i_segment`
  - Wrapper that runs resize/SLIC/graph creation for a single image, storing resized images, padded segments, and the neighbour graph for later use.

## Superpixel Nodes (`image_processings/node.py`)
- Each `Node` tracks index, mask, heuristic score, and derived attributes:
  - `is_edge` if the mask touches any image border; defaults to label 0 when selected as negative.
  - `is_center` flagged when centroid lies within central window and node is not an edge (used for initial positive selection).
  - `color` computed from mask pixels; mode `red` uses mean red channel, `dark` favours darker regions.
  - `center` stores centroid coordinates (x, y) for prompt placement.

## Prompt State Machine (`image_processings/info.py`)
- `Info` encapsulates per-image iteration state.
  - Normalises SLIC labels, instantiates one `Node` per superpixel (respecting `initial_color_mode`), and keeps arrays for labels (`-1` unknown, `0` negative, `1` positive).
  - Initialisation
    - Sorts nodes by `color`. Marks edge nodes as negatives until reaching `negative_pct` quota.
    - Promotes up to `initial_positive_count` central nodes as positives (fallback to highest-ranked).
  - `build_initial_prompts` returns the first `PromptBundle` for SAM2.
- Update cycle
  - `update_from_logits`: stores logits/mask, recomputes adaptive threshold, promotes all unlabelled nodes whose mean mask ratio exceeds threshold.
  - `get_candidates`: gathers all `label=-1` neighbours of current positives (graph connectivity), drops edge nodes, filters by `score_lower_bound`, sorts by mean logits, and truncates to `candidate_top_k`.
  - `commit_candidate`: permanently promotes a selected node, updates prompts and foreground mask.
- `build_prompts`:
    - Generates positive point list (current positives + optional candidate + optional convex-hull augmentation point).
    - Optional `points_filter` keeps positives within configurable central window and enforces `min_point_distance` when `use_subset_points` is true.
    - Negative points remain the initial edge-based set.
    - Builds `mask_prompt` from union of positive segments, optionally wrapped by `apply_selective_convex_hull` when `use_convex_hull` is enabled.
    - When `sam.mask_prompt_source="slic"` the foreground mask is downsampled and stored as `low_res_mask`; other modes skip mask prompts.
  - `record_low_res_mask` caches the most recent SAM low-resolution mask for future rounds.
- Mask pool / selection
  - Every SAM call is stored in `mask_pool`/`mask_pool_full` with mask/logits/prompts/score; pool is deduped by IoU via `mask_pool_iou_threshold`.
  - Final selection can use heuristic scoring (area, edge strength, circularity, convex quality, LAB bimodality), logits-entropy minimisation (softmax/sigmoid → per-pixel entropy mean), or edge-gradient scoring (Sobel on probability map averaged on mask boundary). When `selection_strategy=="cluster_middle"`, the highest-scoring mask within the middle area cluster from `mask_cluster.cluster_masks_by_area` is chosen.
  - Selection metadata and pool stats are recorded for downstream inspection.
- Geometry utilities
  - `apply_selective_convex_hull`: per-connected-component convex hull drawing with ratio threshold (implements requirement §4.2 toggle).
  - `get_largest_component`, `add_more_pos_points`: extract dominant new region and place an augmented positive prompt at its centroid.
  - `points_filter`: deterministic random filtering (seed 0) to reduce clustered positives.

## Iterative Pipeline (`debug_tests/debug_test.py`)
- `_prepare_segment_data` pulls numpy arrays from `image_i_segment`.
- `_select_mask_input` maps SAM configuration strings:
  - `"none"` – no mask input.
  - `"previous_low_res"` – use last low-res mask stored in `Info`.
  - `"algorithm"/"foreground"` – use `PromptBundle.low_res_mask` (foreground mask).
- `_run_prediction` makes a single SAM2 call and packages logits, boolean mask, scores, low-res mask, and prompts into `CandidateEvaluation`.
- `_evaluate_candidates` loops over `Info.get_candidates()`, evaluates each candidate prompt, and retains the best-scoring one (implements requirement §4.1 with configurable top-n).
- `_refine_with_low_res` optionally reruns SAM2 multiple times seeded with stored low-res mask (requirement §4.3 variant when using previous predictions).
- `run_unsupervised_segmentation`
  1. Build SLIC + graph via `image_i_segment`.
  2. Instantiate `Info` with config-driven settings.
  3. Run initial SAM2 prediction, log history and add to mask pool.
  4. Iterate up to `max_iterations`: update logits → fetch candidates → evaluate SAM2 for top-k → commit the best → push to mask pool.
  5. Optional refinement loop using previous low-res masks.
  6. Deduplicate mask pool, score candidates via heuristic/entropy/edge-gradient selection, optionally re-rank with `mask_cluster.select_middle_cluster_entry`, record selection metadata, and append a selection step to history.
  7. Return final mask, iteration history, resized image, and segments.
- `main()` loads constants/config, constructs predictor, iterates over dataset images/masks, saves visualisations (`assets/unsupervised_debug/result_XXX.png`), and reports mIoU + failure cases.

## Mask Pool Scoring Helpers
- `image_processings/pick_obj.py`: contains heuristic, logits-entropy, and edge-gradient scoring utilities.
- `image_processings/mask_cluster.py`: clusters masks by area ratio using KMeans with min/median/max seeds; `select_middle_cluster_entry` picks the top-scoring entry within the middle-area cluster.
- `image_processings/image_post_process.py`: erosion → largest component → dilation filter (currently unused in main pipeline).

## Experiment & Diagnostics
- `debug_tests/run_full_experiment.py`: runs the full pipeline over predefined datasets (`cropped`, `dataset_v0`), saves config snapshots, per-image metrics, worst-case histories (prompts/logits/masks), and summary JSON under timestamped `assets/experiment_*/`.
- `debug_tests/calculate_foreground_ratio.py`: reports foreground ratios for datasets (optionally saving JSON) using pipeline resize settings.
- `debug_tests/show_worst_cases.ipynb`, `tta_debug.ipynb`, `test_slic.ipynb` etc. provide exploratory analysis.

## Dataset Utilities (`datasets/dataset.py`)
- `load_dataset` reads images/masks from `<data_root>/<dataset_name>/{images,masks}`, optionally rescales both so longest side equals `target_long_edge`, and can return image filenames (`return_paths=True`).
- `_resolve_data_root` respects explicit argument first, then `CONSTANT.json:data_path`, handling Windows drive letters in WSL.

## Metrics & Visualisation
- `metrics/metric.py`: `calculate_miou`, `calculate_map`, `calculate_dice`.
- `metrics/visualize.py`:
  - `show_combined_plots` overlays SLIC boundaries + prompts alongside the coloured mask (used by debug script).
  - Helpers to draw bounding boxes, visualise specific segments, and annotate superpixel indices.

## Additional Scripts
- `debug_tests/debug_test_refactored.py` – older refactoring that wires the same components with extensive logging and manual path configuration; useful reference but not tied to `pipeline.json`.
- `debug_tests/test_pre_seg.py` – simple SLIC smoke test.
- Log utility `image_processings/logger.py` creates rotating loggers and redirects `print`, primarily used by legacy scripts.

## Key Requirements Mapping
- Requirement §1/§2: SLIC preprocessing + initial prompt seeding implemented by `image_i_segment` + `Info._initialise_prompt_points`.
- Requirement §3: Per-superpixel label assignment driven by `Info.update_from_logits` with logits mean thresholding; negatives remain fixed.
- Requirement §4:
  - Candidate expansion (4.1) realised via `Info.get_candidates` and `_evaluate_candidates`, with `candidate_top_k` controlling how many neighbours to test.
  - Convex-hull augmentation (4.2) toggled by `algorithm.use_convex_hull`; centroid-based extra prompt added during `Info.build_prompts`.
  - Prompt/mask packaging (4.3) handled inside `Info.build_prompts` + `_select_mask_input`, aligning with `sam.mask_prompt_source` (`slic`, `previous_low_res`, `none`).
  - Final selection now ranks the deduped mask pool via heuristic/entropy/edge-gradient scoring (`selection_strategy`) and optional area clustering.
- Hyper-parameters for these behaviours live in `pipeline.json` for experimentation.

## Quickstart
- Install dependencies (requires SAM2 code in sibling `sam2/`): ensure Python env has `torch`, `opencv-python`, `networkx` (optional), `scikit-image`, `scikit-learn`, `tqdm`, `einops`, `Pillow`.
- Configure `CONSTANT.json` with your `data_path`, `checkpoint`, `model_cfg`, and the active `pipeline_cfg`.
- Run a single-pass debug loop: `python debug_tests/debug_test.py`.
- Run full evaluation across predefined datasets: `python debug_tests/run_full_experiment.py` (saves results under `assets/experiment_*/`).
- Inspect foreground ratios: `python debug_tests/calculate_foreground_ratio.py --datasets cropped dataset_v0 original`.

## Mask Prompt Modes (single switch)
- Set `sam.mask_prompt_source` in `configs/pipeline.json`:
  - `slic`: use the foreground mask built from promoted SLIC segments (convex hull optional) as `mask_input`.
  - `previous_low_res`: reuse the last low-res mask returned by SAM.
  - `none`: disable mask prompting.
  - No separate algorithm flag is needed; SLIC mask generation is implied only when `mask_prompt_source="slic"`.

## Common Tuning Knobs
- `candidate_top_k`, `score_lower_bound`: balance exploration vs. confidence for node promotion.
- `use_convex_hull`, `augment_positive_points`, `use_subset_points`: toggle geometry smoothing and prompt sparsification.
- `initial_color_mode`, `initial_positive_count`, `negative_pct`: control seed prompts.
- `target_area_ratio`, `selection_strategy`: influence final mask scoring and clustering.

## Outputs
- Debug runs: `assets/unsupervised_debug/result_XXX.png`.
- Experiments: `assets/experiment_*/overall_summary.json`, per-dataset summaries, worst-case histories (prompts/logits/masks), and config snapshots.
