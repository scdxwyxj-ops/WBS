"""5-fold CV hyperparameter tuning for the pipeline on multiple datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.pipeline_config import (
    PipelineConfig,
    load_pipeline_config,
)
from datasets.dataset import load_dataset
from experiments.runner import prepare_output_dir, save_metadata, set_seed
from metrics.metric import calculate_dice, calculate_miou
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from debug_tests.run_tta import run_segmentation_with_info


DEFAULT_SEARCH_SPACE: Dict[str, List[float]] = {
    "algorithm.threshold.value": [0.45, 0.5, 0.55],
    "algorithm.negative_pct": [0.05, 0.1],
    "preprocessing.slic.compactness": [8.0, 12.0],
}


def _load_constants() -> Dict[str, Any]:
    constants_path = ROOT / "CONSTANT.json"
    return json.loads(constants_path.read_text(encoding="utf-8"))


def _make_folds(num_samples: int, folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_samples)
    return [np.array(split) for split in np.array_split(indices, folds) if len(split) > 0]


def _apply_overrides(cfg: PipelineConfig, overrides: Dict[str, Any]) -> PipelineConfig:
    dataset = cfg.dataset
    preprocessing = cfg.preprocessing
    slic = preprocessing.slic
    algorithm = cfg.algorithm
    threshold = algorithm.threshold

    for key, value in overrides.items():
        if key == "algorithm.threshold.value":
            threshold = replace(threshold, value=float(value))
        elif key == "algorithm.negative_pct":
            algorithm = replace(algorithm, negative_pct=float(value))
        elif key == "preprocessing.slic.compactness":
            slic = replace(slic, compactness=float(value))
        else:
            raise ValueError(f"Unsupported override key: {key}")

    preprocessing = replace(preprocessing, slic=slic)
    algorithm = replace(algorithm, threshold=threshold)
    return replace(cfg, dataset=dataset, preprocessing=preprocessing, algorithm=algorithm)


def _grid_from_space(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not space:
        return [{}]
    keys = list(space.keys())
    values = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _evaluate_images(
    pipeline_cfg: PipelineConfig,
    predictor: SAM2ImagePredictor,
    images: List[np.ndarray],
    masks: List[np.ndarray],
    indices: Iterable[int],
) -> Tuple[float, float]:
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    eval_indices = list(indices)
    with torch.no_grad():
        for idx in tqdm(eval_indices, desc="eval", unit="img", leave=False):
            image = images[idx]
            gt = masks[idx]
            base_mask, _, _, _, _ = run_segmentation_with_info(image, pipeline_cfg, predictor)
            if base_mask.shape != gt.shape:
                base_mask = cv2.resize(
                    base_mask.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            preds.append(base_mask)
            gts.append(gt)
    miou, _ = calculate_miou(preds, gts)
    dice, _ = calculate_dice(preds, gts)
    return float(miou), float(dice)


def _load_search_space(path: str | None) -> Dict[str, List[Any]]:
    if not path:
        return DEFAULT_SEARCH_SPACE
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "search_space" in payload and isinstance(payload["search_space"], dict):
            return payload["search_space"]
        if "search_space_path" in payload:
            nested_path = Path(payload["search_space_path"])
            if not nested_path.is_absolute():
                nested_path = (ROOT / nested_path).resolve()
            nested_payload = json.loads(nested_path.read_text(encoding="utf-8"))
            if "search_space" in nested_payload and isinstance(nested_payload["search_space"], dict):
                return nested_payload["search_space"]
            if isinstance(nested_payload, dict):
                return nested_payload
        if all(isinstance(v, (list, tuple)) for v in payload.values()):
            return payload
    raise ValueError("Invalid search space file: expected a dict of list values or a config with search_space.")


def _make_logger(output_dir: Path):
    log_path = output_dir / "train.log"
    handle = log_path.open("a", encoding="utf-8")

    def _log(message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()

    return _log


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV tuning for pipeline hyperparameters.")
    parser.add_argument("--config", default=None, help="Path to cv_run.json config (preferred).")
    parser.add_argument("--pipeline-cfg", default=str(ROOT / "configs" / "pipeline.json"))
    parser.add_argument("--datasets", default="dataset_v0,cropped")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search", default=None, help="Optional JSON file with search_space dict.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.config:
        cfg_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        args.pipeline_cfg = cfg_payload.get("pipeline_cfg", args.pipeline_cfg)
        datasets = cfg_payload.get("datasets")
        if datasets:
            args.datasets = ",".join(datasets)
        args.folds = int(cfg_payload.get("folds", args.folds))
        args.seed = int(cfg_payload.get("seed", args.seed))
        args.search = cfg_payload.get("search_space_path", args.search)
        if cfg_payload.get("max_samples") is not None:
            args.max_samples = int(cfg_payload.get("max_samples"))

    output_dir = prepare_output_dir("cv_tune_pipeline", args.output_dir)
    log = _make_logger(output_dir)
    set_seed(args.seed)

    base_cfg = load_pipeline_config(Path(args.pipeline_cfg))
    search_space = _load_search_space(args.search)
    grid = _grid_from_space(search_space)

    constants = _load_constants()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(constants["model_cfg"], constants["checkpoint"], device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.to(device)
    predictor.model.eval()

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    all_results: List[Dict[str, Any]] = []

    save_metadata(
        output_dir,
        {
            "mode": "cv_tune_pipeline",
            "seed": args.seed,
            "pipeline_cfg": args.pipeline_cfg,
            "datasets": dataset_names,
            "folds": args.folds,
            "search_space": search_space,
            "output_dir": str(output_dir),
        },
    )
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "pipeline_cfg": str(args.pipeline_cfg),
                "datasets": dataset_names,
                "folds": args.folds,
                "seed": args.seed,
                "search_space": search_space,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Loaded pipeline config: {args.pipeline_cfg}")
    log(f"Datasets: {dataset_names} | folds={args.folds} | seed={args.seed}")
    log(f"Grid size: {len(grid)} runs")

    for run_idx, overrides in enumerate(grid):
        run_name = f"run_{run_idx:03d}"
        cfg = _apply_overrides(base_cfg, overrides)
        log(f"[{run_name}] overrides={overrides}")
        run_summary: Dict[str, Any] = {
            "name": run_name,
            "overrides": overrides,
            "folds": [],
        }

        fold_scores: List[Tuple[float, float]] = []
        for dataset_name in dataset_names:
            images, masks, _ = load_dataset(
                dataset_name,
                target_long_edge=cfg.dataset.target_long_edge,
                return_paths=True,
            )
            if args.max_samples:
                images = images[: args.max_samples]
                masks = masks[: args.max_samples]

            folds = _make_folds(len(images), args.folds, args.seed)
            for fold_idx, val_indices in enumerate(folds):
                fold_cfg = replace(cfg, dataset=replace(cfg.dataset, name=dataset_name))
                miou, dice = _evaluate_images(fold_cfg, predictor, images, masks, val_indices)
                fold_scores.append((miou, dice))
                log(
                    f"[{run_name}] {dataset_name} fold={fold_idx} "
                    f"count={len(val_indices)} mIoU={miou:.4f} Dice={dice:.4f}"
                )
                run_summary["folds"].append(
                    {
                        "dataset": dataset_name,
                        "fold": fold_idx,
                        "count": int(len(val_indices)),
                        "miou": miou,
                        "dice": dice,
                    }
                )

        if fold_scores:
            mean_miou = float(np.mean([s[0] for s in fold_scores]))
            mean_dice = float(np.mean([s[1] for s in fold_scores]))
        else:
            mean_miou = 0.0
            mean_dice = 0.0

        run_summary["mean_miou"] = mean_miou
        run_summary["mean_dice"] = mean_dice
        log(f"[{run_name}] mean mIoU={mean_miou:.4f} Dice={mean_dice:.4f}")
        all_results.append(run_summary)

    all_results.sort(key=lambda r: r["mean_miou"], reverse=True)
    (output_dir / "cv_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    top = all_results[0] if all_results else None
    (output_dir / "cv_best.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    if top:
        log(f"Best run: {top['name']} mIoU={top['mean_miou']:.4f} Dice={top['mean_dice']:.4f}")

    csv_path = output_dir / "cv_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "mean_miou", "mean_dice", "overrides"])
        for row in all_results:
            writer.writerow([row["name"], row["mean_miou"], row["mean_dice"], json.dumps(row["overrides"])])


if __name__ == "__main__":
    main()
