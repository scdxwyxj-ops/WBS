#!/usr/bin/env python3
"""Generate process-illustration figures for SGIP candidate evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

# Make `python tools/...` work from repo root on server.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.pipeline_config import load_pipeline_config
from datasets.dataset import load_dataset
from debug_tests.run_tta import run_segmentation_with_info
from image_processings.image_pre_seg import change_image_type, image_i_segment
from image_processings.info import Candidate, Info, PromptBundle
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class CandidateEval:
    candidate: Candidate
    bundle: PromptBundle
    logits: np.ndarray
    mask: np.ndarray
    score: float


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _prepare_segment_data(pre_segment: image_i_segment) -> Tuple[np.ndarray, np.ndarray]:
    img_resized = change_image_type(pre_segment.image_resized, "np.array")
    segment_tensor = pre_segment.segment_without_padding
    if hasattr(segment_tensor, "cpu"):
        segment = segment_tensor.cpu().numpy()
    else:
        segment = np.array(segment_tensor)
    return _ensure_uint8_image(img_resized), segment


def _select_mask_input(bundle: PromptBundle, info: Info, mask_prompt_source: str) -> Optional[np.ndarray]:
    strategy = mask_prompt_source.lower()
    if strategy in {"none", "off"}:
        return None
    if strategy in {"previous_low_res", "previous"}:
        return info.last_low_res_mask
    if strategy in {"algorithm", "foreground", "slic", "slic_foreground"}:
        return bundle.low_res_mask
    raise ValueError(f"Unknown mask_prompt_source: {mask_prompt_source}")


def _predict_once(
    predictor: SAM2ImagePredictor,
    info: Info,
    bundle: PromptBundle,
    multimask_output: bool,
    mask_prompt_source: str,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    mask_input = _select_mask_input(bundle, info, mask_prompt_source)
    logits, scores, low_res_mask = predictor.predict(
        point_coords=bundle.points,
        point_labels=bundle.labels,
        box=None,
        mask_input=mask_input,
        multimask_output=multimask_output,
        return_logits=True,
    )
    logits = logits[0]
    score_val = scores[0] if isinstance(scores, Sequence) else scores
    score_arr = np.asarray(score_val).reshape(-1)
    score = float(score_arr[0]) if score_arr.size else 0.0
    mask = logits > 0
    return logits, mask, score, low_res_mask


def _collect_candidates(info: Info, top_n: int) -> List[Candidate]:
    """Collect candidates using the exact pipeline API (Info.get_candidates)."""
    return info.get_candidates(top_k=top_n)


def _draw_points(
    ax: plt.Axes,
    points: Sequence[Tuple[float, float]],
    color: str,
    label: str,
    marker: str = "o",
    size: float = 40,
) -> None:
    if not points:
        return
    arr = np.asarray(points, dtype=float)
    ax.scatter(arr[:, 0], arr[:, 1], c=color, s=size, marker=marker, edgecolors="white", linewidths=0.8, label=label)


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    base = _ensure_uint8_image(image).copy()
    overlay = base.copy()
    m = np.asarray(mask, dtype=bool)
    overlay[m] = np.array(color, dtype=np.uint8)
    out = (base.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
    return out


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _logits_to_rgb(logits: np.ndarray, cmap: str = "magma") -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - lo) / (hi - lo)
    cm = plt.get_cmap(cmap)
    rgb = cm(norm)[..., :3]
    return (rgb * 255).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray, image_shape: Tuple[int, int], pad_ratio: float = 0.15, min_pad: int = 12) -> Tuple[int, int, int, int]:
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    h, w = image_shape
    if ys.size == 0:
        return 0, h - 1, 0, w - 1
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    box_h = max(1, y1 - y0 + 1)
    box_w = max(1, x1 - x0 + 1)
    py = max(min_pad, int(round(box_h * pad_ratio)))
    px = max(min_pad, int(round(box_w * pad_ratio)))
    y0 = max(0, y0 - py)
    y1 = min(h - 1, y1 + py)
    x0 = max(0, x0 - px)
    x1 = min(w - 1, x1 + px)
    return y0, y1, x0, x1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate-process figures for one image.")
    parser.add_argument("--pipeline-cfg", type=Path, default=Path("configs/pipeline.json"))
    parser.add_argument("--constants", type=Path, default=Path("CONSTANT.json"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--top3", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("assets/figs"))
    args = parser.parse_args()

    root = ROOT
    pipeline_cfg = load_pipeline_config(root / args.pipeline_cfg)
    constants = json.loads((root / args.constants).read_text(encoding="utf-8"))

    images, _masks = load_dataset(
        pipeline_cfg.dataset.name,
        data_root=None,
        target_long_edge=pipeline_cfg.dataset.target_long_edge,
    )
    if not images:
        raise RuntimeError("Dataset is empty.")

    sample_index = int(np.clip(args.sample_index, 0, len(images) - 1))
    image = images[sample_index]

    predictor = SAM2ImagePredictor(build_sam2(constants["model_cfg"], constants["checkpoint"]))

    pre_segment = image_i_segment(
        image=image,
        new_size_of_image=pipeline_cfg.preprocessing.image_size,
        num_node_for_graph=pipeline_cfg.preprocessing.num_graph_nodes,
        compactness_in_SLIC=pipeline_cfg.preprocessing.slic.compactness,
        sigma_in_SLIC=pipeline_cfg.preprocessing.slic.sigma,
        min_size_factor_in_SLIC=pipeline_cfg.preprocessing.slic.min_size_factor,
        max_size_factor_in_SLIC=pipeline_cfg.preprocessing.slic.max_size_factor,
    )
    img_resized, segment = _prepare_segment_data(pre_segment)

    predictor.set_image(img_resized)

    info = Info(
        segment=segment,
        logits=None,
        image=img_resized,
        graph=pre_segment.graph,
        settings=pipeline_cfg.algorithm,
        debug_mode=False,
        mask_prompt_source=pipeline_cfg.sam.mask_prompt_source,
    )

    # Initial prediction to move to "current timestep" state.
    initial_bundle = info.build_initial_prompts()
    logits, mask, score, low_res_mask = _predict_once(
        predictor,
        info,
        initial_bundle,
        multimask_output=pipeline_cfg.sam.multimask_output,
        mask_prompt_source=pipeline_cfg.sam.mask_prompt_source,
    )
    _ = mask, score
    info.record_low_res_mask(low_res_mask)
    # Snapshot strict previous-step positives before current-step foreground update.
    prev_pos = list(info.positive_point_coords)
    info.update_from_logits(logits)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Pure original image for presentation (no overlays/annotations).
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_resized)
    ax.axis("off")
    _save(fig, out_dir / "00_original_image.png")

    # 1) SLIC segmentation visualization.
    seg_vis = mark_boundaries(img_resized, segment, color=(1, 1, 0), mode="thick")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(seg_vis)
    ax.set_title(f"SLIC Segmentation (sample={sample_index})")
    ax.axis("off")
    _save(fig, out_dir / "01_slic_segmentation.png")

    # 1b) Keep segment boundaries, but use current-step logits heatmap as background.
    logits_bg = _logits_to_rgb(logits, cmap="magma")
    seg_logits_vis = mark_boundaries(logits_bg, segment, color=(1, 1, 0), mode="thick")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(seg_logits_vis)
    ax.set_title("SLIC Boundaries on Current-Step Logits")
    ax.axis("off")
    _save(fig, out_dir / "01b_slic_boundaries_on_logits.png")

    # Candidates at current timestep.
    top_n = max(args.top_candidates, args.top3)
    candidates = _collect_candidates(info, top_n=top_n)
    evals: List[CandidateEval] = []
    top3: List[CandidateEval] = []
    prev_neg = list(info.negative_point_coords)
    candidate_error: Optional[str] = None

    if candidates:
        # 2) Top-k candidates in ONE figure + one zoomed-in candidate.
        top_candidates = candidates[: args.top_candidates]
        cand_colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 180, 255],
                [255, 180, 0],
                [255, 0, 255],
            ],
            dtype=np.float32,
        )
        combined = img_resized.astype(np.float32).copy()
        alpha = 0.40
        for rank, cand in enumerate(top_candidates, start=1):
            mask_i = np.asarray(info.node_list[cand.node_id].mask, dtype=bool)
            color = cand_colors[(rank - 1) % len(cand_colors)]
            combined[mask_i] = combined[mask_i] * (1.0 - alpha) + color * alpha

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.clip(combined, 0, 255).astype(np.uint8))
        ax.axis("off")
        _save(fig, out_dir / "02_candidates_overlay.png")

        # Use the top-1 candidate for zoom-only view.
        zoom_rank = 1
        zoom_candidate = top_candidates[zoom_rank - 1]
        zoom_mask = np.asarray(info.node_list[zoom_candidate.node_id].mask, dtype=bool)
        y0, y1, x0, x1 = _bbox_from_mask(zoom_mask, img_resized.shape[:2], pad_ratio=0.22, min_pad=14)
        zoom_img = img_resized[y0 : y1 + 1, x0 : x1 + 1]
        zoom_mask_crop = zoom_mask[y0 : y1 + 1, x0 : x1 + 1]
        zoom_vis = _overlay_mask(zoom_img, zoom_mask_crop, color=(255, 0, 0), alpha=0.45)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(zoom_vis)
        ax.set_title(
            f"Candidate #{zoom_rank} Zoom-in (node={zoom_candidate.node_id}, logit={zoom_candidate.score:.4f})"
        )
        ax.axis("off")
        _save(fig, out_dir / "02b_candidate_zoom.png")

        # Evaluate candidates with SAM2 and rank by SAM2 score.
        for cand in candidates:
            bundle = info.build_prompts(candidate_id=cand.node_id)
            c_logits, c_mask, c_score, _low_res = _predict_once(
                predictor,
                info,
                bundle,
                multimask_output=pipeline_cfg.sam.multimask_output,
                mask_prompt_source=pipeline_cfg.sam.mask_prompt_source,
            )
            evals.append(CandidateEval(candidate=cand, bundle=bundle, logits=c_logits, mask=c_mask, score=c_score))

        evals.sort(key=lambda x: x.score, reverse=True)
        top3 = evals[: min(args.top3, len(evals))]

        # 3) Previous-step positive prompts over cell image.
        for rank, ev in enumerate(top3, start=1):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_resized)
            _draw_points(ax, prev_pos, color="blue", label="previous positive", marker="o", size=44)
            _draw_points(ax, prev_neg, color="red", label="negative", marker="^", size=42)
            ax.set_title(f"Top-{rank} Candidate Context (SAM2 score={ev.score:.4f})")
            ax.axis("off")
            _save(fig, out_dir / f"03_top{rank}_previous_positive_prompts.png")

        # 4) Candidate superpixel + centroid.
        for rank, ev in enumerate(top3, start=1):
            cand_mask = info.node_list[ev.candidate.node_id].mask
            cx, cy = info.node_list[ev.candidate.node_id].center
            vis = _overlay_mask(img_resized, cand_mask, color=(255, 255, 0), alpha=0.45)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(vis)
            ax.scatter([cx], [cy], c="green", s=70, marker="x", linewidths=2.0)
            ax.set_title(f"Top-{rank} Candidate Superpixel + Centroid")
            ax.axis("off")
            _save(fig, out_dir / f"04_top{rank}_candidate_centroid.png")

        # 5) Candidate prompt set (previous positives + new centroid).
        for rank, ev in enumerate(top3, start=1):
            cx, cy = info.node_list[ev.candidate.node_id].center
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_resized)
            _draw_points(ax, prev_pos, color="blue", label="previous positive", marker="o", size=40)
            _draw_points(ax, prev_neg, color="red", label="negative", marker="^", size=38)
            ax.scatter([cx], [cy], c="green", s=70, marker="x", linewidths=2.0, label="new centroid")
            ax.set_title(f"Top-{rank} Candidate Prompt (add centroid)")
            ax.axis("off")
            _save(fig, out_dir / f"05_top{rank}_candidate_prompt.png")

        # 6) SAM2 masks for selected candidate prompts.
        for rank, ev in enumerate(top3, start=1):
            over = _overlay_mask(img_resized, ev.mask, color=(255, 0, 0), alpha=0.4)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(over)
            ax.set_title(f"Top-{rank} SAM2 Mask (score={ev.score:.4f})")
            ax.axis("off")
            _save(fig, out_dir / f"06_top{rank}_sam2_mask.png")
    else:
        candidate_error = (
            "No candidates at current timestep after initial update. "
            "Saved non-candidate figures and full-pipeline outputs only."
        )
        print(f"[WARN] {candidate_error}")

    # 7) True mask-pool from full iterative pipeline; one image per pool entry.
    _final_mask, _history, vis_full, _seg_full, info_full = run_segmentation_with_info(
        image=image,
        config=pipeline_cfg,
        predictor=predictor,
    )
    # Raw mask pool collected across iterations (before dedup/cluster filtering).
    pool_entries_full = info_full.get_mask_pool(full=True)
    for idx, entry in enumerate(pool_entries_full, start=1):
        pool_mask = np.asarray(entry.get("mask"), dtype=bool)
        pool_score = float(entry.get("score", 0.0))
        pool_iter = int(entry.get("iteration", -1))
        overlay = _overlay_mask(vis_full, pool_mask, color=(0, 255, 255), alpha=0.42)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title(f"Mask Pool (full) #{idx} | iter={pool_iter} | score={pool_score:.4f}")
        ax.axis("off")
        _save(fig, out_dir / f"07_mask_pool_{idx:02d}.png")

    # Filtered pool after dedup/selection-stage preprocessing (for comparison).
    pool_entries_filtered = info_full.get_mask_pool(full=False)
    for idx, entry in enumerate(pool_entries_filtered, start=1):
        pool_mask = np.asarray(entry.get("mask"), dtype=bool)
        pool_score = float(entry.get("score", 0.0))
        pool_iter = int(entry.get("iteration", -1))
        overlay = _overlay_mask(vis_full, pool_mask, color=(255, 165, 0), alpha=0.38)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title(f"Mask Pool (filtered) #{idx} | iter={pool_iter} | score={pool_score:.4f}")
        ax.axis("off")
        _save(fig, out_dir / f"07b_mask_pool_filtered_{idx:02d}.png")

    # 8) Final selected mask result.
    final_overlay = _overlay_mask(vis_full, np.asarray(_final_mask, dtype=bool), color=(255, 0, 0), alpha=0.42)
    final_score = None
    if info_full.selected_entry is not None:
        final_score = float(info_full.selected_entry.get("score", 0.0))
    final_method = str(info_full.selection_metadata.get("method", "unknown"))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(final_overlay)
    if final_score is None:
        ax.set_title(f"Final Mask Selection | method={final_method}")
    else:
        ax.set_title(f"Final Mask Selection | method={final_method} | score={final_score:.4f}")
    ax.axis("off")
    _save(fig, out_dir / "08_final_selected_mask.png")

    # Optional compact summary text.
    summary = {
        "sample_index": sample_index,
        "dataset": pipeline_cfg.dataset.name,
        "candidate_top_n": top_n,
        "num_candidates_found": len(candidates),
        "requested_top3": int(args.top3),
        "candidate_warning": candidate_error,
        "mask_pool_size_full": len(pool_entries_full),
        "mask_pool_size_filtered": len(pool_entries_filtered),
        "final_selection_method": final_method,
        "final_selection_score": final_score,
        "selected_top3_by_sam2_score": [
            {
                "rank": i + 1,
                "node_id": int(e.candidate.node_id),
                "superpixel_logit_score": float(e.candidate.score),
                "sam2_score": float(e.score),
            }
            for i, e in enumerate(top3)
        ],
    }
    (out_dir / "fig_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
