"""Run baseline segmentation comparisons on multiple datasets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
from skimage import filters, segmentation
from skimage.feature import peak_local_max
from skimage.graph import rag_mean_color
from skimage.measure import label
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset import load_dataset
from experiments.runner import prepare_output_dir, save_metadata, set_seed
from metrics.metric import calculate_dice, calculate_hd95, calculate_miou
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    datasets: List[str]
    target_long_edge: Optional[int]
    max_samples: Optional[int]
    method: Dict[str, Any]


def _load_constants() -> Dict[str, Any]:
    constants_path = ROOT / "CONSTANT.json"
    return json.loads(constants_path.read_text(encoding="utf-8"))


def _read_config(path: Path) -> BaselineConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BaselineConfig(
        name=str(payload.get("name", path.stem)),
        datasets=list(payload.get("datasets", [])),
        target_long_edge=payload.get("target_long_edge"),
        max_samples=payload.get("max_samples"),
        method=dict(payload.get("method", {})),
    )


def _make_logger(output_dir: Path):
    log_path = output_dir / "train.log"
    handle = log_path.open("a", encoding="utf-8")

    def _log(message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()

    return _log


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def _otsu_mask(image: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(_ensure_uint8(image), cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    thresh = filters.threshold_otsu(gray)
    return gray >= thresh


def _kmeans_mask(image: np.ndarray, k: int = 2, color_space: str = "lab") -> np.ndarray:
    img = _ensure_uint8(image)
    if color_space.lower() == "lab":
        data = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else:
        data = img
    pixels = data.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    means = []
    for idx in range(k):
        means.append(gray.reshape(-1)[labels == idx].mean() if np.any(labels == idx) else 0.0)
    fg_idx = int(np.argmax(means))
    return labels.reshape(img.shape[:2]) == fg_idx


def _watershed_mask(image: np.ndarray, min_distance: int = 5, compactness: float = 0.0) -> np.ndarray:
    img = _ensure_uint8(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = filters.threshold_otsu(gray)
    binary = gray >= thresh
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=binary)
    markers = np.zeros_like(gray, dtype=np.int32)
    for idx, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = idx
    if markers.max() == 0:
        return binary
    labels_ws = segmentation.watershed(-distance, markers, mask=binary, compactness=compactness)
    center = (gray.shape[0] // 2, gray.shape[1] // 2)
    label_id = labels_ws[center]
    if label_id == 0:
        return binary
    return labels_ws == label_id


def _grabcut_mask(image: np.ndarray, rect_margin: float = 0.1, iterations: int = 5) -> np.ndarray:
    img = _ensure_uint8(image)
    h, w = img.shape[:2]
    margin_h = int(h * rect_margin)
    margin_w = int(w * rect_margin)
    rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)
    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
    return np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)


def _mst_mask(
    image: np.ndarray,
    n_segments: int = 200,
    compactness: float = 10.0,
    sigma: float = 1.0,
    edge_threshold: float = 10.0,
) -> np.ndarray:
    img = _ensure_uint8(image)
    segments = segmentation.slic(
        img,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
    )
    rag = rag_mean_color(img, segments, mode="distance")
    mst = nx.minimum_spanning_tree(rag, weight="weight")
    pruned = mst.copy()
    for u, v, data in list(pruned.edges(data=True)):
        if data.get("weight", 0.0) > edge_threshold:
            pruned.remove_edge(u, v)
    center = (segments.shape[0] // 2, segments.shape[1] // 2)
    center_label = int(segments[center])
    for comp in nx.connected_components(pruned):
        if center_label in comp:
            mask = np.isin(segments, list(comp))
            return mask
    return segments == center_label


def _build_predictor(constants: Dict[str, Any]) -> SAM2ImagePredictor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(constants["model_cfg"], constants["checkpoint"], device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.to(device)
    predictor.model.eval()
    return predictor


def _grid_points(shape: Tuple[int, int], grid_size: int) -> np.ndarray:
    h, w = shape
    ys = np.linspace(0, h - 1, grid_size).astype(int)
    xs = np.linspace(0, w - 1, grid_size).astype(int)
    points = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return points


def _random_points(shape: Tuple[int, int], num_points: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    h, w = shape
    xs = rng.randint(0, w, size=num_points)
    ys = rng.randint(0, h, size=num_points)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _sam_prompt_mask(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    points: np.ndarray,
    multimask_output: bool = False,
) -> np.ndarray:
    predictor.set_image(image)
    labels = np.ones(len(points), dtype=np.int32)
    logits, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=None,
        mask_input=None,
        multimask_output=multimask_output,
        return_logits=True,
    )
    best = 0
    if isinstance(scores, np.ndarray):
        best = int(np.argmax(scores))
    logits = logits[best]
    return logits > 0


def _run_method(
    method_cfg: Dict[str, Any],
    image: np.ndarray,
    predictor: Optional[SAM2ImagePredictor],
) -> np.ndarray:
    method_type = method_cfg.get("type", "").lower()
    if method_type == "otsu":
        return _otsu_mask(image, sigma=float(method_cfg.get("gaussian_sigma", 0.0)))
    if method_type == "kmeans":
        return _kmeans_mask(
            image,
            k=int(method_cfg.get("k", 2)),
            color_space=str(method_cfg.get("color_space", "lab")),
        )
    if method_type == "watershed":
        return _watershed_mask(
            image,
            min_distance=int(method_cfg.get("min_distance", 5)),
            compactness=float(method_cfg.get("compactness", 0.0)),
        )
    if method_type == "grabcut":
        return _grabcut_mask(
            image,
            rect_margin=float(method_cfg.get("rect_margin", 0.1)),
            iterations=int(method_cfg.get("iterations", 5)),
        )
    if method_type == "auto_prompt":
        if predictor is None:
            raise ValueError("SAM2 predictor required for auto_prompt.")
        grid_size = int(method_cfg.get("grid_size", 6))
        points = _grid_points(image.shape[:2], grid_size)
        return _sam_prompt_mask(predictor, image, points, bool(method_cfg.get("multimask_output", False)))
    if method_type == "random_prompt":
        if predictor is None:
            raise ValueError("SAM2 predictor required for random_prompt.")
        num_points = int(method_cfg.get("num_points", 10))
        seed = int(method_cfg.get("seed", 42))
        points = _random_points(image.shape[:2], num_points, seed)
        return _sam_prompt_mask(predictor, image, points, bool(method_cfg.get("multimask_output", False)))
    if method_type == "mst":
        return _mst_mask(
            image,
            n_segments=int(method_cfg.get("n_segments", 200)),
            compactness=float(method_cfg.get("compactness", 10.0)),
            sigma=float(method_cfg.get("sigma", 1.0)),
            edge_threshold=float(method_cfg.get("edge_threshold", 10.0)),
        )
    raise ValueError(f"Unknown baseline method: {method_type}")


def run_from_config(
    config_path: Path,
    *,
    output_root: Optional[str] = None,
    max_samples_override: Optional[int] = None,
) -> Path:
    config = _read_config(config_path)
    if max_samples_override is not None:
        config = BaselineConfig(
            name=config.name,
            datasets=config.datasets,
            target_long_edge=config.target_long_edge,
            max_samples=int(max_samples_override),
            method=config.method,
        )
    output_dir = prepare_output_dir(f"baseline_{config.name}", output_root)
    log = _make_logger(output_dir)
    save_metadata(
        output_dir,
        {
            "mode": "baseline_compare",
            "name": config.name,
            "config_path": str(config_path),
            "datasets": config.datasets,
            "method": config.method,
            "output_dir": str(output_dir),
        },
    )
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "datasets": config.datasets,
                "target_long_edge": config.target_long_edge,
                "max_samples": config.max_samples,
                "method": config.method,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Config: {config_path}")
    log(f"Datasets: {config.datasets}")
    log(f"Method: {config.method}")

    constants = _load_constants()
    method_type = config.method.get("type", "").lower()
    predictor = None
    if method_type in {"auto_prompt", "random_prompt"}:
        predictor = _build_predictor(constants)

    overall_summary: List[Dict[str, Any]] = []

    for dataset_name in config.datasets:
        images, masks, image_names = load_dataset(
            dataset_name,
            target_long_edge=config.target_long_edge,
            return_paths=True,
        )
        if config.max_samples:
            images = images[: config.max_samples]
            masks = masks[: config.max_samples]
            image_names = image_names[: config.max_samples]

        preds: List[np.ndarray] = []
        gts: List[np.ndarray] = []
        per_image: List[Dict[str, Any]] = []

        for image, gt, name in tqdm(list(zip(images, masks, image_names)), desc=dataset_name, unit="img"):
            pred = _run_method(config.method, image, predictor)
            if pred.shape != gt.shape:
                pred = cv2.resize(
                    pred.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            preds.append(pred)
            gts.append(gt)
            per_image.append({"file": name})

        miou, iou_list = calculate_miou(preds, gts)
        dice, dice_list = calculate_dice(preds, gts)
        hd95, hd95_list = calculate_hd95(preds, gts)

        for idx, entry in enumerate(per_image):
            entry.update(
                {
                    "iou": float(iou_list[idx]),
                    "dice": float(dice_list[idx]),
                    "hd95": float(hd95_list[idx]),
                }
            )

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "per_image_metrics.json").write_text(json.dumps(per_image, indent=2), encoding="utf-8")
        summary = {
            "dataset": dataset_name,
            "num_samples": len(preds),
            "miou": float(miou),
            "dice": float(dice),
            "hd95": float(hd95),
        }
        (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        overall_summary.append(summary)
        log(
            f"{dataset_name}: mIoU={summary['miou']:.4f} "
            f"Dice={summary['dice']:.4f} HD95={summary['hd95']:.4f}"
        )

    (output_dir / "overall_summary.json").write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline comparisons.")
    parser.add_argument("--config", required=True, help="Path to baseline config JSON.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_from_config(
        Path(args.config),
        output_root=args.output_dir,
        max_samples_override=args.max_samples,
    )


if __name__ == "__main__":
    main()
