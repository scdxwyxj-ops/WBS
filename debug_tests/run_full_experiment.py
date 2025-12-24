"""Run the unsupervised SAM2 pipeline across all datasets and persist diagnostics."""

from __future__ import annotations

import json
import os
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import numpy as np
import sys
import cv2
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SAM2_DIR = ROOT_DIR.parent / "sam2"
if SAM2_DIR.exists() and str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

from configs.pipeline_config import PipelineConfig, load_pipeline_config
from datasets.dataset import load_dataset
from debug_tests.debug_test import (
    MAIN_DIR,
    _load_constants,
    run_unsupervised_segmentation,
    StepRecord,
)
from metrics.metric import calculate_dice, calculate_map, calculate_miou
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


DATASETS = ["cropped", "dataset_v0"]


def _ensure_output_dir() -> Path:
    override = os.environ.get("PIPELINE_OUTPUT_DIR")
    if override:
        out_dir = Path(override)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = MAIN_DIR / "assets" / f"experiment_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _save_config_snapshot(output_dir: Path, config: PipelineConfig, constants: Dict) -> None:
    snapshot_path = output_dir / "config_snapshot.json"
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "constants": constants,
        "pipeline_config": asdict(config),
        "datasets": DATASETS,
    }
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # Preserve exact config files for reproducibility.
    pipeline_path = MAIN_DIR / constants["pipeline_cfg"]
    pipeline_target = output_dir / Path(constants["pipeline_cfg"]).name
    pipeline_target.parent.mkdir(parents=True, exist_ok=True)
    pipeline_target.write_text(pipeline_path.read_text(encoding="utf-8"), encoding="utf-8")

    constant_src = MAIN_DIR / "CONSTANT.json"
    constant_dst = output_dir / "CONSTANT.json"
    constant_dst.write_text(constant_src.read_text(encoding="utf-8"), encoding="utf-8")


def _prepare_predictor(constants: Dict) -> SAM2ImagePredictor:
    model = build_sam2(
        constants["model_cfg"],
        constants["checkpoint"],
        device="cpu",
    )
    return SAM2ImagePredictor(model)


def _save_history(
    history: Sequence[StepRecord],
    base_dir: Path,
    *,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    image_name: str,
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict] = []

    np.savez_compressed(base_dir / "prediction.npz", mask=prediction.astype(np.uint8))
    np.savez_compressed(base_dir / "ground_truth.npz", mask=ground_truth.astype(np.uint8))

    for step_idx, step in enumerate(history):
        step_dir = base_dir / f"step_{step_idx:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(step_dir / "prompts.npz", points=step.prompts.points, labels=step.prompts.labels)

        if step.prompts.mask_prompt is not None:
            np.savez_compressed(step_dir / "mask_prompt.npz", mask=step.prompts.mask_prompt)
        if step.prompts.low_res_mask is not None:
            np.savez_compressed(step_dir / "low_res_mask.npz", mask=step.prompts.low_res_mask)

        np.savez_compressed(step_dir / "logits.npz", logits=step.logits)
        np.savez_compressed(step_dir / "mask.npz", mask=step.mask.astype(np.uint8))
        np.savez_compressed(step_dir / "positive_mask.npz", mask=step.positive_mask.astype(np.uint8))

        candidate_id = step.candidate_id
        prompt_candidate_id = step.prompts.candidate_id
        pos_points = step.prompts.all_positive_points or []

        summary.append(
            {
                "step": f"step_{step_idx:02d}",
                "iteration": int(step.iteration),
                "stage": str(step.stage),
                "candidate_id": int(candidate_id) if candidate_id is not None else None,
                "prompt_candidate_id": int(prompt_candidate_id) if prompt_candidate_id is not None else None,
                "all_positive_points": [
                    [float(p[0]), float(p[1])] for p in pos_points
                ],
                "num_positive_points": int(np.sum(step.prompts.labels == 1)),
                "num_negative_points": int(np.sum(step.prompts.labels == 0)),
            }
        )

    meta = {
        "image_name": image_name,
        "num_steps": len(history),
        "steps": summary,
    }
    (base_dir / "history.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _run_dataset(
    dataset_name: str,
    config: PipelineConfig,
    predictor: SAM2ImagePredictor,
    output_dir: Path,
) -> Dict:
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    images, gt_masks, image_names = load_dataset(
        dataset_name,
        data_root=None,
        target_long_edge=config.dataset.target_long_edge,
        return_paths=True,
    )

    predictions: List[np.ndarray] = []
    aligned_gt_masks: List[np.ndarray] = []
    histories: List[Sequence[StepRecord]] = []

    per_image_metrics: List[Dict] = []

    num_samples = len(images)
    progress = tqdm(range(num_samples), desc=f"{dataset_name}", unit="img", leave=False)

    for idx in progress:
        image = images[idx]
        gt_mask = gt_masks[idx]
        name = image_names[idx]
        pred_mask, history, _, _ = run_unsupervised_segmentation(image, config, predictor)

        if gt_mask.shape != pred_mask.shape:
            gt_aligned = cv2.resize(
                gt_mask.astype(np.uint8),
                (pred_mask.shape[1], pred_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            gt_aligned = gt_mask

        predictions.append(pred_mask)
        histories.append(history)
        aligned_gt_masks.append(gt_aligned)

        intersection = np.logical_and(pred_mask, gt_aligned).sum()
        union = np.logical_or(pred_mask, gt_aligned).sum()
        iou = float(intersection / union) if union else (1.0 if intersection == 0 else 0.0)

        pred_sum = pred_mask.sum()
        gt_sum = gt_aligned.sum()
        dice = float(2 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) else 1.0

        per_image_metrics.append(
            {
                "index": idx,
                "file": name,
                "iou": iou,
                "dice": dice,
                "num_history_steps": len(history),
            }
        )
    progress.close()

    miou, iou_list = calculate_miou(predictions, aligned_gt_masks)
    dice_mean, dice_list = calculate_dice(predictions, aligned_gt_masks)
    map_mean, ap_list = calculate_map(predictions, aligned_gt_masks)

    # Persist per-image metrics.
    metrics_path = dataset_dir / "per_image_metrics.json"
    metrics_payload = [
        {
            **entry,
            "iou": float(entry["iou"]),
            "dice": float(entry["dice"]),
        }
        for entry in per_image_metrics
    ]
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Save histories for worst 10%.
    if iou_list:
        worst_count = max(1, math.ceil(len(iou_list) * 0.1))
        worst_indices = np.argsort(iou_list)[:worst_count]
        history_dir = dataset_dir / "worst_histories"
        for idx in worst_indices:
            entry = per_image_metrics[idx]
            name = entry["file"]
            stem = Path(name).stem
            folder_name = f"{idx:03d}_{stem}"
            _save_history(
                histories[idx],
                history_dir / folder_name,
                prediction=predictions[idx],
                ground_truth=aligned_gt_masks[idx],
                image_name=name,
            )

    summary = {
        "dataset": dataset_name,
        "num_samples": len(predictions),
        "miou": float(miou),
        "dice": float(dice_mean),
        "map": float(map_mean),
        "ap_thresholds": list(np.arange(0.5, 1.0, 0.05)),
        "ap_scores": [float(v) for v in ap_list],
        "worst_percentage": 0.1,
    }

    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    constants = _load_constants()
    override_cfg = os.environ.get("PIPELINE_CFG")
    if override_cfg:
        constants = dict(constants)
        constants["pipeline_cfg"] = override_cfg
    pipeline_cfg = load_pipeline_config(MAIN_DIR / constants["pipeline_cfg"])
    _set_seed(pipeline_cfg.algorithm.seed)

    output_dir = _ensure_output_dir()
    _save_config_snapshot(output_dir, pipeline_cfg, constants)

    predictor = _prepare_predictor(constants)

    overall_summary: List[Dict] = []

    for dataset_name in DATASETS:
        print(f"\n=== Running dataset: {dataset_name} ===")
        summary = _run_dataset(dataset_name, pipeline_cfg, predictor, output_dir)
        overall_summary.append(summary)
        print(
            f"{dataset_name}: mIoU={summary['miou']:.4f}, "
            f"Dice={summary['dice']:.4f}, mAP={summary['map']:.4f}"
        )

    combined_path = output_dir / "overall_summary.json"
    combined_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
