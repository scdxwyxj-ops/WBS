"""Run unsupervised pipeline + TTA (anchor/entropy/consistency) using pipeline/tta configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn

import sys

# Ensure project root is on PYTHONPATH when running from subdirs.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SAM2_DIR = ROOT_DIR.parent / "sam2"
if SAM2_DIR.exists() and str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

from configs.pipeline_config import PipelineConfig, load_pipeline_config
from datasets.dataset import load_dataset
from image_processings.image_pre_seg import change_image_type, image_i_segment
from image_processings.info import Info, Candidate, PromptBundle
from image_processings.mask_cluster import select_middle_cluster_entry
from image_processings.pick_obj import (
    pick_obj_using_entropy,
    pick_obj_using_heuristic,
    pick_obj_using_edge_gradient,
)
from image_processings.tta import (
    TTALossWeights,
    run_tta_from_pool,
    default_multi_view_augment,
    apply_lora_to_mask_decoder,
)
from metrics.metric import calculate_miou, calculate_dice, calculate_map
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MAIN_DIR = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = MAIN_DIR / "CONSTANT.json"


def _load_constants() -> dict:
    return json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))


def _ensure_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = MAIN_DIR / "assets" / f"tta_experiment_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _save_config_snapshot(output_dir: Path, constants: Dict, pipeline_cfg: PipelineConfig, tta_cfg: Dict) -> None:
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "constants": constants,
        "pipeline_config": json.loads((MAIN_DIR / constants["pipeline_cfg"]).read_text(encoding="utf-8")),
        "tta_config": tta_cfg,
    }
    (output_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


# --- Segmentation pipeline (copied from debug_test, returns Info) ------------------
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


def _prepare_segment_data(pre_segment: image_i_segment) -> Tuple[np.ndarray, np.ndarray]:
    img_resized = change_image_type(pre_segment.image_resized, "np.array")
    segment_tensor = pre_segment.segment_without_padding
    if hasattr(segment_tensor, "cpu"):
        segment = segment_tensor.cpu().numpy()
    else:
        segment = np.array(segment_tensor)
    return img_resized, segment


def _select_mask_input(bundle: PromptBundle, info: Info, sam_cfg) -> Optional[np.ndarray]:
    strategy = sam_cfg.mask_prompt_source.lower()
    if strategy in {"none", "off"}:
        return None
    if strategy in {"previous_low_res", "previous"}:
        return info.last_low_res_mask
    if strategy in {"algorithm", "foreground", "slic"}:
        return bundle.low_res_mask
    raise ValueError(f"Unknown mask_prompt_source: {sam_cfg.mask_prompt_source}")


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


def run_segmentation_with_info(
    image: np.ndarray,
    config: PipelineConfig,
    predictor: SAM2ImagePredictor,
) -> Tuple[np.ndarray, List[StepRecord], np.ndarray, np.ndarray, Info]:
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
        mask_prompt_source=config.sam.mask_prompt_source,
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
    pool = info.get_mask_pool()
    if selection_strategy == "entropy":
        selected_entry, pool_stats, _ = pick_obj_using_entropy(img_resized, pool, target_area_ratio=info.settings.target_area_ratio)
    elif selection_strategy == "edge_gradient":
        selected_entry, pool_stats, _ = pick_obj_using_edge_gradient(img_resized, pool, target_area_ratio=info.settings.target_area_ratio)
    else:
        selected_entry, pool_stats, _ = pick_obj_using_heuristic(img_resized, pool, target_area_ratio=info.settings.target_area_ratio)

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

    return mask, history, img_resized, segment, info


# --- TTA runner -------------------------------------------------------------------
def load_tta_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    constants = _load_constants()
    pipeline_cfg = load_pipeline_config(MAIN_DIR / constants["pipeline_cfg"])
    tta_cfg = load_tta_config(MAIN_DIR / "configs" / "tta_config.json")

    output_dir = _ensure_output_dir()
    _save_config_snapshot(output_dir, constants, pipeline_cfg, tta_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(constants["model_cfg"], constants["checkpoint"], device=device)
    lora_cfg = tta_cfg.get("lora", {})
    injected_modules: List[str] = []
    if lora_cfg.get("target") == "mask_decoder":
        injected_modules = apply_lora_to_mask_decoder(
            model,
            r=int(lora_cfg.get("rank", 4)),
            lora_alpha=int(lora_cfg.get("alpha", 8)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=lora_cfg.get("target_modules"),
        )
    predictor = SAM2ImagePredictor(model)
    predictor.model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable_params:
        opt_cfg = tta_cfg.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(opt_cfg.get("lr", 1e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
        max_grad_norm = float(opt_cfg.get("max_grad_norm", 1.0))
    else:
        max_grad_norm = 0.0

    datasets = [pipeline_cfg.dataset.name]
    images, gt_masks, image_names = load_dataset(
        pipeline_cfg.dataset.name,
        data_root=None,
        target_long_edge=pipeline_cfg.dataset.target_long_edge,
        return_paths=True,
    )

    predictions: List[np.ndarray] = []
    adapted: List[np.ndarray] = []
    metrics_before: List[Dict[str, float]] = []
    metrics_after: List[Dict[str, float]] = []

    log_lines: List[str] = []
    per_image_metrics: List[Dict[str, float]] = []
    per_image_losses: List[Dict[str, float]] = []

    log_lines.append("=== LoRA Injection ===")
    log_lines.append(f"Injected modules: {injected_modules}")
    log_lines.append(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

    for idx in tqdm(range(len(images)), desc="TTA", unit="img"):
        image = images[idx]
        gt_mask = gt_masks[idx]
        name = image_names[idx]

        base_mask, history, vis_image, segments, info = run_segmentation_with_info(image, pipeline_cfg, predictor)
        predictions.append(base_mask)

        # Metrics before TTA
        gt_aligned = gt_mask
        if gt_mask.shape != base_mask.shape:
            gt_aligned = cv2.resize(gt_mask.astype(np.uint8), (base_mask.shape[1], base_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        inter = np.logical_and(base_mask, gt_aligned).sum()
        union = np.logical_or(base_mask, gt_aligned).sum()
        iou_before = float(inter / union) if union else 1.0
        dice_before = float(2 * inter / (base_mask.sum() + gt_aligned.sum())) if (base_mask.sum() + gt_aligned.sum()) else 1.0
        metrics_before.append({"file": name, "iou": iou_before, "dice": dice_before})

        # Prepare prompts for TTA (use final prompts from last history step)
        final_prompts = history[-1].prompts
        tta_loss_weights = TTALossWeights(
            anchor=float(tta_cfg["loss_weights"]["anchor"]),
            entropy=float(tta_cfg["loss_weights"]["entropy"]),
            consistency=float(tta_cfg["loss_weights"]["consistency"]),
            regularization=float(tta_cfg["loss_weights"].get("regularization", 0.0)),
        )
        augment_fn = default_multi_view_augment(
            scales=tta_cfg["augment"]["scales"],
            do_flip=tta_cfg["augment"]["use_flip"],
        )

        # Run TTA steps (no-op optimizer placeholder)
        def _optimizer_step(total_loss, _losses):
            if optimizer is None:
                return
            if not isinstance(total_loss, torch.Tensor) or not total_loss.requires_grad:
                return
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()

        tta_out = None
        last_losses: Optional[Dict[str, float]] = None
        for _ in range(int(tta_cfg.get("tta_steps", 1))):
            tta_out = run_tta_from_pool(
                predictor,
                vis_image,
                info.get_mask_pool(),
                {
                    "point_coords": final_prompts.points,
                    "point_labels": final_prompts.labels,
                    "box": None,
                    "mask_input": None,
                    "multimask_output": pipeline_cfg.sam.multimask_output,
                },
                loss_weights=tta_loss_weights,
                selection_strategy=tta_cfg.get("pseudo_label", {}).get("strategy", "score_top_k"),
                top_k=int(tta_cfg.get("pseudo_label", {}).get("top_k_masks", 3)),
                augment_fn=augment_fn,
                optimizer_step_fn=_optimizer_step,
            )
            if tta_out is not None:
                last_losses = dict(tta_out["tta_outputs"].losses)

        if tta_out is not None:
            prob_map = tta_out["tta_outputs"].student_probs
            if isinstance(prob_map, torch.Tensor):
                prob_map = prob_map.detach().cpu().numpy()
            adapted_mask = prob_map > 0.5
        else:
            adapted_mask = base_mask
        adapted.append(adapted_mask)

        inter_a = np.logical_and(adapted_mask, gt_aligned).sum()
        union_a = np.logical_or(adapted_mask, gt_aligned).sum()
        iou_after = float(inter_a / union_a) if union_a else 1.0
        dice_after = float(2 * inter_a / (adapted_mask.sum() + gt_aligned.sum())) if (adapted_mask.sum() + gt_aligned.sum()) else 1.0
        metrics_after.append({"file": name, "iou": iou_after, "dice": dice_after})
        per_image_metrics.append(
            {
                "file": name,
                "iou_before": iou_before,
                "dice_before": dice_before,
                "iou_after": iou_after,
                "dice_after": dice_after,
            }
        )
        if last_losses is not None:
            per_image_losses.append({"file": name, **last_losses})

        log_lines.append(
            f"{name} | before IoU={iou_before:.4f} Dice={dice_before:.4f} | "
            f"after IoU={iou_after:.4f} Dice={dice_after:.4f}"
        )

    miou_before, _ = calculate_miou(predictions, gt_masks)
    miou_after, _ = calculate_miou(adapted, gt_masks)
    dice_before_mean, _ = calculate_dice(predictions, gt_masks)
    dice_after_mean, _ = calculate_dice(adapted, gt_masks)
    map_before, _ = calculate_map(predictions, gt_masks)
    map_after, _ = calculate_map(adapted, gt_masks)

    print("\n=== Summary ===")
    print(f"Before TTA: mIoU={miou_before:.4f}, Dice={dice_before_mean:.4f}, mAP={map_before:.4f}")
    print(f"After  TTA: mIoU={miou_after:.4f}, Dice={dice_after_mean:.4f}, mAP={map_after:.4f}")

    summary = {
        "num_samples": len(images),
        "miou_before": float(miou_before),
        "miou_after": float(miou_after),
        "dice_before": float(dice_before_mean),
        "dice_after": float(dice_after_mean),
        "map_before": float(map_before),
        "map_after": float(map_after),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "per_image_metrics.json").write_text(json.dumps(per_image_metrics, indent=2), encoding="utf-8")
    (output_dir / "per_image_losses.json").write_text(json.dumps(per_image_losses, indent=2), encoding="utf-8")
    (output_dir / "train.log").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
