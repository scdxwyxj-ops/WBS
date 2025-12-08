"""Executable pipeline for the unsupervised SAM2 segmentation workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from configs.pipeline_config import PipelineConfig, load_pipeline_config
from datasets.dataset import load_dataset
from image_processings.image_pre_seg import change_image_type, image_i_segment
from image_processings.info import Candidate, Info, PromptBundle
from image_processings.mask_cluster import select_middle_cluster_entry
from image_processings.pick_obj import pick_obj
from metrics.metric import calculate_miou
from metrics.visualize import show_combined_plots
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


MAIN_DIR = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = MAIN_DIR / "CONSTANT.json"


@dataclass
class CandidateEvaluation:
    candidate_id: int
    logits: np.ndarray
    mask: np.ndarray
    score: float
    low_res_mask: Optional[np.ndarray]
    prompts: PromptBundle


@dataclass
class StepRecord:
    iteration: int
    stage: str
    prompts: PromptBundle
    logits: np.ndarray
    mask: np.ndarray
    positive_mask: np.ndarray
    candidate_id: Optional[int] = None


def _load_constants() -> dict:
    return json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def _prepare_segment_data(pre_segment: image_i_segment) -> Tuple[np.ndarray, np.ndarray]:
    img_resized = change_image_type(pre_segment.image_resized, "np.array")
    segment_tensor = pre_segment.segment_without_padding
    if hasattr(segment_tensor, "cpu"):
        segment = segment_tensor.cpu().numpy()
    else:
        segment = np.array(segment_tensor)
    return img_resized, segment


def _select_mask_input(bundle: PromptBundle, info: Info, sam_cfg) -> Optional[np.ndarray]:
    strategy = sam_cfg.mask_prompt_strategy.lower()
    if strategy in {"none", "off"}:
        return None
    if strategy in {"previous_low_res", "previous"}:
        return info.last_low_res_mask
    if strategy in {"algorithm", "foreground"}:
        return bundle.low_res_mask
    raise ValueError(f"Unknown mask_prompt_strategy: {sam_cfg.mask_prompt_strategy}")


def _run_prediction(
    predictor: SAM2ImagePredictor,
    bundle: PromptBundle,
    info: Info,
    sam_cfg,
) -> CandidateEvaluation:
    mask_input = _select_mask_input(bundle, info, sam_cfg)
    logits, scores, low_res_mask = predictor.predict(
        point_coords=bundle.points,
        point_labels=bundle.labels,
        box=None,
        mask_input=mask_input,
        multimask_output=sam_cfg.multimask_output,
        return_logits=True,
    )

    logits = logits[0]
    mask = logits > 0
    score = float(scores[0]) if isinstance(scores, Sequence) else float(scores)
    return CandidateEvaluation(
        candidate_id=bundle.candidate_id or -1,
        logits=logits,
        mask=mask,
        score=score,
        low_res_mask=low_res_mask,
        prompts=bundle,
    )


def _evaluate_candidates(
    predictor: SAM2ImagePredictor,
    info: Info,
    sam_cfg,
    candidates: List[Candidate],
) -> Optional[CandidateEvaluation]:
    best_result: Optional[CandidateEvaluation] = None
    for candidate in candidates:
        bundle = info.build_prompts(candidate_id=candidate.node_id)
        result = _run_prediction(predictor, bundle, info, sam_cfg)
        result.candidate_id = candidate.node_id
        if best_result is None or result.score > best_result.score:
            best_result = result
    return best_result


def _refine_with_low_res(
    predictor: SAM2ImagePredictor,
    info: Info,
    sam_cfg,
    current_logits: np.ndarray,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    logits = current_logits
    mask = logits > 0
    for _ in range(iterations):
        if info.last_low_res_mask is None:
            break
        bundle = info.build_prompts(candidate_id=None)
        logits_pred, scores, low_res_mask = predictor.predict(
            point_coords=bundle.points,
            point_labels=bundle.labels,
            box=None,
            mask_input=info.last_low_res_mask,
            multimask_output=sam_cfg.multimask_output,
            return_logits=True,
        )
        logits = logits_pred[0]
        mask = logits > 0
        info.record_low_res_mask(low_res_mask)
    return logits, mask


def run_unsupervised_segmentation(
    image: np.ndarray,
    config: PipelineConfig,
    predictor: SAM2ImagePredictor,
) -> Tuple[np.ndarray, List[StepRecord], np.ndarray, np.ndarray]:
    pre_segment = image_i_segment(
        image=image,
        new_size_of_image=config.preprocessing.image_size,
        num_node_for_graph=config.preprocessing.num_graph_nodes,
        compactness_in_SLIC=config.preprocessing.slic.compactness,
        sigma_in_SLIC=config.preprocessing.slic.sigma,
        min_size_factor_in_SLIC=config.preprocessing.slic.min_size_factor,
        max_size_factor_in_SLIC=config.preprocessing.slic.max_size_factor,
    )

    img_resized, segment = _prepare_segment_data(pre_segment)
    img_resized = _ensure_uint8_image(img_resized)

    predictor.set_image(img_resized)

    info = Info(
        segment=segment,
        logits=None,
        image=img_resized,
        graph=pre_segment.graph,
        settings=config.algorithm,
        debug_mode=False,
    )

    history: List[StepRecord] = []

    initial_bundle = info.build_initial_prompts()
    initial_result = _run_prediction(predictor, initial_bundle, info, config.sam)
    info.record_low_res_mask(initial_result.low_res_mask)

    logits = initial_result.logits
    mask = initial_result.mask
    history.append(
        StepRecord(
            iteration=0,
            stage="initial",
            prompts=initial_bundle,
            logits=logits,
            mask=mask,
            positive_mask=info.prompt_mask.copy(),
            candidate_id=initial_result.candidate_id if initial_result.candidate_id != -1 else None,
        )
    )
    info.add_pool_entry(
        mask=mask,
        logits=logits,
        score=initial_result.score,
        iteration=0,
        prompts=initial_bundle,
        candidate_id=initial_result.candidate_id if initial_result.candidate_id != -1 else None,
        positive_mask=info.prompt_mask,
    )

    for iteration in range(1, config.algorithm.max_iterations + 1):
        info.update_from_logits(logits)
        candidates = info.get_candidates()
        if not candidates:
            break

        best_candidate = _evaluate_candidates(predictor, info, config.sam, candidates)
        if best_candidate is None:
            break

        info.commit_candidate(best_candidate.candidate_id)
        logits = best_candidate.logits
        mask = best_candidate.mask
        info.record_low_res_mask(best_candidate.low_res_mask)
        info.add_pool_entry(
            mask=mask,
            logits=logits,
            score=best_candidate.score,
            iteration=iteration,
            prompts=best_candidate.prompts,
            candidate_id=best_candidate.candidate_id,
            positive_mask=info.prompt_mask,
        )
        history.append(
            StepRecord(
                iteration=iteration,
                stage="promotion",
                prompts=best_candidate.prompts,
                logits=logits,
                mask=mask,
                positive_mask=info.prompt_mask.copy(),
                candidate_id=best_candidate.candidate_id,
            )
        )

    if config.sam.refine_with_previous_low_res:
        selection_prompts = info.build_prompts(candidate_id=None)
        logits, mask = _refine_with_low_res(
            predictor,
            info,
            config.sam,
            logits,
            config.sam.refine_rounds,
        )
        info.add_pool_entry(
            mask=mask,
            logits=logits,
            score=None,
            iteration=len(history),
            prompts=selection_prompts,
            candidate_id=None,
            positive_mask=info.prompt_mask,
        )
        history.append(
            StepRecord(
                iteration=len(history),
                stage="refine",
                prompts=selection_prompts,
                logits=logits,
                mask=mask,
                positive_mask=info.prompt_mask.copy(),
                candidate_id=None,
            )
        )

    info.deduplicate_mask_pool(info.settings.mask_pool_iou_threshold)
    selection_strategy = info.settings.selection_strategy.lower()
    selected_entry, pool_stats, _ = pick_obj(
        img_resized,
        info.get_mask_pool(),
        target_area_ratio=info.settings.target_area_ratio,
    )
    if selected_entry is not None:
        info.set_pool_stats(pool_stats)
        final_entry = selected_entry
        selection_meta = {"method": selection_strategy}
        if selection_strategy == "cluster_middle":
            clustered_entry, cluster_meta = select_middle_cluster_entry(info.get_mask_pool(), n_clusters=3)
            selection_meta["cluster_meta"] = cluster_meta
            if clustered_entry is not None:
                final_entry = clustered_entry
        else:
            selection_meta["cluster_meta"] = None

        info.selection_metadata = selection_meta
        info.selected_entry = final_entry
        mask = np.asarray(final_entry["mask"], dtype=bool)
        entry_logits = final_entry.get("logits")
        if entry_logits is not None:
            logits = entry_logits
        final_prompts = final_entry.get("prompts", info.build_prompts(candidate_id=None))
        final_positive_mask = final_entry.get("positive_mask", info.prompt_mask.copy())
        history.append(
            StepRecord(
                iteration=len(history),
                stage="selection",
                prompts=final_prompts,
                logits=logits,
                mask=mask,
                positive_mask=final_positive_mask,
                candidate_id=final_entry.get("candidate_id"),
            )
        )
    else:
        info.selection_metadata = {"method": selection_strategy, "cluster_meta": None}

    return mask, history, img_resized, segment


def main() -> None:
    constants = _load_constants()
    pipeline_cfg = load_pipeline_config(MAIN_DIR / constants["pipeline_cfg"])

    predictor = SAM2ImagePredictor(build_sam2(constants["model_cfg"], constants["checkpoint"]))

    images, gt_masks = load_dataset(
        pipeline_cfg.dataset.name,
        data_root=None,
        target_long_edge=pipeline_cfg.dataset.target_long_edge,
    )

    predictions: List[np.ndarray] = []

    for image, gt_mask in tqdm(
        list(zip(images, gt_masks)),
        total=len(images),
        desc="Processing images",
    ):
        pred_mask, history, vis_image, segments = run_unsupervised_segmentation(image, pipeline_cfg, predictor)
        predictions.append(pred_mask)

        output_dir = MAIN_DIR / "assets" / "unsupervised_debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        last_step = history[-1]
        show_combined_plots(
            vis_image,
            segments,
            last_step.prompts.points,
            last_step.prompts.labels,
            pred_mask,
            color=(0, 0, 255),
            save_path=str(output_dir / f"result_{len(predictions):03d}.png"),
            need_show=False,
        )

    miou, iou_list = calculate_miou(predictions, gt_masks)
    fail_cases = [idx for idx, iou in enumerate(iou_list) if iou < 0.75]

    print("\n========== 结果 ==========")
    print("Mean IoU:", miou)
    print("IOU < 0.75:", fail_cases)
    print("IoU List:", iou_list)


if __name__ == "__main__":
    main()
