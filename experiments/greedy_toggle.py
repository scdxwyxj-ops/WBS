"""Greedy toggle search for boolean switches in the pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from configs.pipeline_config import apply_pipeline_overrides, load_pipeline_config
from datasets.dataset import load_dataset
from experiments.runner import prepare_output_dir, save_metadata, set_seed
from metrics.metric import calculate_miou
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from debug_tests.run_tta import run_segmentation_with_info


DEFAULT_TOGGLES = [
    "algorithm.use_subset_points",
    "algorithm.use_convex_hull",
    "algorithm.deduplicate_mask_pool",
]


def _load_constants() -> Dict[str, Any]:
    constants_path = Path("CONSTANT.json")
    return json.loads(constants_path.read_text(encoding="utf-8"))


def _prepare_predictor(constants: Dict[str, Any]) -> SAM2ImagePredictor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(constants["model_cfg"], constants["checkpoint"], device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.to(device)
    predictor.model.eval()
    return predictor


def _evaluate_dataset(
    pipeline_cfg,
    predictor: SAM2ImagePredictor,
    dataset_name: str,
    *,
    max_samples: int | None = None,
) -> float:
    images, gt_masks, _ = load_dataset(
        dataset_name,
        target_long_edge=pipeline_cfg.dataset.target_long_edge,
        return_paths=True,
    )
    if max_samples:
        images = images[:max_samples]
        gt_masks = gt_masks[:max_samples]

    preds = []
    gts = []
    with torch.no_grad():
        for image, gt in tqdm(
            list(zip(images, gt_masks)),
            desc=f"{dataset_name}",
            unit="img",
            leave=False,
        ):
            pred_mask, _, _, _, _ = run_segmentation_with_info(image, pipeline_cfg, predictor)
            if pred_mask.shape != gt.shape:
                import cv2

                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            preds.append(pred_mask)
            gts.append(gt)

    miou, _ = calculate_miou(preds, gts)
    return float(miou)


def _score_config(
    pipeline_cfg,
    predictor: SAM2ImagePredictor,
    datasets: List[str],
    *,
    max_samples: int | None = None,
) -> Dict[str, float]:
    per_dataset = {}
    for name in datasets:
        per_dataset[name] = _evaluate_dataset(
            pipeline_cfg, predictor, name, max_samples=max_samples
        )
    mean_iou = float(np.mean(list(per_dataset.values()))) if per_dataset else 0.0
    return {"mean_iou": mean_iou, "per_dataset": per_dataset}


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy toggle search for pipeline switches.")
    parser.add_argument("--pipeline-cfg", default="configs/pipeline.json")
    parser.add_argument("--datasets", default="dataset_v0,cropped")
    parser.add_argument("--toggles", default=",".join(DEFAULT_TOGGLES))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = prepare_output_dir("greedy_toggles", args.output_dir)
    set_seed(args.seed)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    toggles = [t.strip() for t in args.toggles.split(",") if t.strip()]

    base_cfg = load_pipeline_config(Path(args.pipeline_cfg))
    constants = _load_constants()
    predictor = _prepare_predictor(constants)

    save_metadata(
        output_dir,
        {
            "mode": "greedy_toggle",
            "pipeline_cfg": args.pipeline_cfg,
            "datasets": datasets,
            "toggles": toggles,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "output_dir": str(output_dir),
        },
    )

    history: List[Dict[str, Any]] = []
    current_cfg = base_cfg
    current_score = _score_config(current_cfg, predictor, datasets, max_samples=args.max_samples)

    history.append(
        {
            "step": "baseline",
            "overrides": {},
            "score": current_score,
        }
    )

    for toggle in toggles:
        results = []
        for value in (False, True):
            overrides = {toggle: value}
            cfg = apply_pipeline_overrides(current_cfg, overrides)
            score = _score_config(cfg, predictor, datasets, max_samples=args.max_samples)
            results.append({"override": overrides, "score": score})

        # choose best by mean IoU
        best = max(results, key=lambda r: r["score"]["mean_iou"])
        current_cfg = apply_pipeline_overrides(current_cfg, best["override"])
        current_score = best["score"]

        history.append(
            {
                "step": toggle,
                "choices": results,
                "selected": best,
            }
        )

        (output_dir / "greedy_progress.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    (output_dir / "greedy_final.json").write_text(
        json.dumps(
            {
                "final_config": asdict(current_cfg),
                "final_score": current_score,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
