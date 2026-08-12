"""Lightweight single-image bbox-guided SAM2 region-growing inference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes

from . import __version__
from .config import InferenceConfig, load_config, resolved_config_dict
from .geometry import BBox, crop_image, paste_crop_mask, validate_bbox
from .preprocessing import PreprocessedImage, preprocess
from .selection import Proposal, select_proposal


@dataclass(frozen=True)
class PromptBundle:
    points: np.ndarray
    labels: np.ndarray
    low_res_mask: np.ndarray | None
    candidate_id: int | None


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: int | None
    logits: np.ndarray
    mask: np.ndarray
    sam_score: float
    low_res_mask: np.ndarray | None
    prompts: PromptBundle


@dataclass(frozen=True)
class InferenceResult:
    mask: np.ndarray
    bbox_xyxy: BBox
    metadata: dict[str, Any]

    def save(self, mask_path: str | Path, metadata_path: str | Path | None = None, mask_value: int = 255) -> None:
        target = Path(mask_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), np.asarray(self.mask, dtype=np.uint8) * int(mask_value)):
            raise OSError(f"Failed to write mask: {target}")
        if metadata_path is not None:
            meta_target = Path(metadata_path)
            meta_target.parent.mkdir(parents=True, exist_ok=True)
            meta_target.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")


class _State:
    def __init__(self, prepared: PreprocessedImage, config: InferenceConfig) -> None:
        self.prepared = prepared
        self.config = config
        count = len(prepared.regions)
        self.labels = np.full(count, -1, dtype=np.int8)
        self.positive_points: list[tuple[float, float]] = []
        self.negative_points: list[tuple[float, float]] = []
        self.prompt_mask = np.zeros_like(prepared.segments, dtype=bool)
        self.logits: np.ndarray | None = None
        self.sam_mask: np.ndarray | None = None
        self.last_low_res: np.ndarray | None = None
        self.proposals: list[Proposal] = []
        self._initialise()

    def _initialise(self) -> None:
        prompt_cfg = self.config.prompts
        descending = sorted(self.prepared.regions, key=lambda region: (-region.color_score, region.region_id))
        negative_regions = list(reversed(descending))
        negative_quota = max(
            prompt_cfg.initial_negative_min_count,
            int(round(prompt_cfg.initial_negative_fraction * len(self.prepared.regions))),
        )
        for region in negative_regions:
            if len(self.negative_points) >= negative_quota:
                break
            if region.touches_edge and self.labels[region.region_id] != 0:
                self.labels[region.region_id] = 0
                self.negative_points.append(region.point_xy)
        promoted = 0
        for region in descending:
            if promoted >= prompt_cfg.initial_positive_count:
                break
            if self.labels[region.region_id] == 0:
                continue
            if region.touches_edge:
                continue
            if not self._inside_range(region.point_xy, prompt_cfg.initial_positive_center_range):
                continue
            self._promote(region.region_id)
            promoted += 1
        if promoted == 0:
            for region in descending:
                if self.labels[region.region_id] != 0:
                    self._promote(region.region_id)
                    break
        self._refresh_prompt_mask()

    def _inside_range(self, point: tuple[float, float], value_range: tuple[float, float]) -> bool:
        height, width = self.prepared.image.shape[:2]
        low, high = value_range
        return width * low <= point[0] <= width * high and height * low <= point[1] <= height * high

    def _promote(self, region_id: int) -> None:
        self.labels[region_id] = 1
        point = self.prepared.regions[region_id].point_xy
        if point not in self.positive_points:
            self.positive_points.append(point)

    def _refresh_prompt_mask(self, additional_id: int | None = None) -> np.ndarray:
        selected = set(np.flatnonzero(self.labels == 1).tolist())
        if additional_id is not None:
            selected.add(additional_id)
        mask = np.zeros_like(self.prepared.segments, dtype=bool)
        for region_id in selected:
            mask |= self.prepared.regions[int(region_id)].mask
        if additional_id is None:
            self.prompt_mask = mask
        return mask

    def initial_prompts(self) -> PromptBundle:
        points = np.asarray(self.positive_points + self.negative_points, dtype=np.float32)
        labels = np.asarray([1] * len(self.positive_points) + [0] * len(self.negative_points), dtype=np.int64)
        return PromptBundle(points=points, labels=labels, low_res_mask=None, candidate_id=None)

    def _convex_hull(self, mask: np.ndarray) -> np.ndarray:
        prompt_cfg = self.config.prompts
        mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(mask_u8)
        for contour in contours:
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            selected = hull if area / hull_area < prompt_cfg.convex_hull_area_ratio_threshold else contour
            cv2.drawContours(result, [selected], -1, 255, thickness=-1)
        return binary_fill_holes(result > 0)

    def _augmentation_point(self, mask_prompt: np.ndarray) -> tuple[float, float] | None:
        if self.sam_mask is None:
            return None
        existing = np.logical_or(self.sam_mask, self.prompt_mask)
        new_mask = np.logical_and(mask_prompt, np.logical_not(existing)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(new_mask, connectivity=8)
        if count <= 1:
            return None
        region_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        coords = np.argwhere(labels == region_label)
        if not len(coords):
            return None
        center_yx = np.rint(coords.mean(axis=0)).astype(float)
        return float(center_yx[1]), float(center_yx[0])

    def _filter_positive_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        prompt_cfg = self.config.prompts
        if len(points) < 3:
            return points
        centered = [point for point in points if self._inside_range(point, prompt_cfg.subset_center_range)]
        if not centered:
            return points
        filtered: list[tuple[float, float]] = []
        for point in reversed(centered):
            if all(
                np.hypot(point[0] - kept[0], point[1] - kept[1]) >= prompt_cfg.min_point_distance for kept in filtered
            ):
                filtered.append(point)
        filtered.reverse()
        return filtered or centered

    def _bundle(self, candidate_id: int | None, include_augmentation: bool = True) -> PromptBundle:
        prompt_cfg = self.config.prompts
        positive = list(self.positive_points)
        if candidate_id is not None:
            positive.append(self.prepared.regions[candidate_id].point_xy)
        mask_prompt = self._refresh_prompt_mask(candidate_id)
        mask_prompt = self._convex_hull(mask_prompt)
        if include_augmentation:
            point = self._augmentation_point(mask_prompt)
            if point is not None and point not in positive:
                positive.append(point)
        positive = self._filter_positive_points(positive)
        points = np.asarray(positive + self.negative_points, dtype=np.float32)
        labels = np.asarray([1] * len(positive) + [0] * len(self.negative_points), dtype=np.int64)
        size = prompt_cfg.mask_prompt_low_res_size
        low_res = cv2.resize(mask_prompt.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)[None]
        return PromptBundle(points=points, labels=labels, low_res_mask=low_res, candidate_id=candidate_id)

    def update_from_logits(self, logits: np.ndarray) -> None:
        growing = self.config.growing
        self.logits = np.asarray(logits)
        self.sam_mask = self.logits > growing.sam_logit_threshold
        flat_regions = self.prepared.segments.ravel()
        region_area = np.bincount(flat_regions, minlength=len(self.prepared.regions))
        mask_area = np.bincount(flat_regions[self.sam_mask.ravel()], minlength=len(self.prepared.regions))
        occupancy = np.divide(
            mask_area, region_area, out=np.zeros_like(region_area, dtype=float), where=region_area > 0
        )
        threshold = growing.occupancy_threshold
        eligible = [int(index) for index in np.flatnonzero(occupancy >= threshold) if self.labels[int(index)] == -1]
        if eligible:

            def key(region_id: int) -> tuple[float, float, int]:
                region_mask = self.prepared.regions[region_id].mask
                overlap = np.logical_and(region_mask, self.sam_mask)
                values = self.logits[overlap]
                if not values.size:
                    values = self.logits[region_mask]
                score = float(values.sum())
                return score, float(occupancy[region_id]), -region_id

            self._promote(max(eligible, key=key))
            self._refresh_prompt_mask()

    def candidates(self) -> list[int]:
        growing = self.config.growing
        candidate_ids: set[int] = set()
        for region_id in np.flatnonzero(self.labels == 1):
            for neighbor in self.prepared.adjacency[int(region_id)]:
                if self.labels[neighbor] != -1:
                    continue
                if self.prepared.regions[neighbor].touches_edge:
                    continue
                candidate_ids.add(neighbor)
        scored = []
        for region_id in sorted(candidate_ids):
            if self.logits is None:
                continue
            values = self.logits[self.prepared.regions[region_id].mask]
            score = float(values.sum())
            if score >= growing.region_score_lower_bound:
                scored.append((region_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [region_id for region_id, _ in scored[: growing.candidate_top_n]]

    def candidate_bundle(self, region_id: int) -> PromptBundle:
        return self._bundle(region_id)

    def commit(self, region_id: int) -> None:
        if self.labels[region_id] == 0:
            raise ValueError("Cannot promote a negative region")
        self._promote(region_id)
        self._refresh_prompt_mask()


class WBSSegmenter:
    """Reusable model wrapper with a stable ``predict(image, bbox_xyxy)`` API."""

    def __init__(self, predictor: Any, config: InferenceConfig, *, device: str | None = None) -> None:
        self.predictor = predictor
        self.config = config
        self.device = device or self._resolve_device(config.runtime.device)
        self._set_reproducibility()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        config: str | Path | InferenceConfig,
        *,
        device: str | None = None,
    ) -> WBSSegmenter:
        resolved = load_config(config) if not isinstance(config, InferenceConfig) else config
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        actual = cls._sha256(checkpoint_path)
        if actual != resolved.model.checkpoint_sha256:
            raise ValueError(f"Checkpoint SHA256 mismatch: expected {resolved.model.checkpoint_sha256}, got {actual}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError('SAM2 is not installed. Install with: pip install -e ".[sam2]"') from exc
        selected_device = device or cls._resolve_device(resolved.runtime.device)
        model = build_sam2(resolved.model.sam2_model_cfg, str(checkpoint_path), device=selected_device)
        model.to(selected_device)
        model.eval()
        return cls(SAM2ImagePredictor(model), resolved, device=selected_device)

    def _set_reproducibility(self) -> None:
        runtime = self.config.runtime
        seed = runtime.seed
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    def _prediction_context(self) -> ExitStack:
        stack = ExitStack()
        runtime = self.config.runtime
        stack.enter_context(torch.inference_mode())
        if self.device == "cuda" and runtime.amp_dtype != "none":
            dtype = torch.float16 if runtime.amp_dtype == "float16" else torch.bfloat16
            stack.enter_context(torch.autocast("cuda", dtype=dtype))
        return stack

    def _predict_bundle(
        self, state: _State, bundle: PromptBundle, *, force_previous_low_res: bool = False
    ) -> CandidateResult:
        if force_previous_low_res:
            mask_input = state.last_low_res
        else:
            mask_input = bundle.low_res_mask
        with self._prediction_context():
            logits, scores, low_res = self.predictor.predict(
                point_coords=bundle.points,
                point_labels=bundle.labels,
                box=None,
                mask_input=mask_input,
                multimask_output=False,
                return_logits=True,
            )
        logits_arr = np.asarray(logits)[0]
        score_values = np.asarray(scores).reshape(-1)
        score = float(score_values[0]) if score_values.size else 0.0
        return CandidateResult(
            candidate_id=bundle.candidate_id,
            logits=logits_arr,
            mask=logits_arr > self.config.growing.sam_logit_threshold,
            sam_score=score,
            low_res_mask=None if low_res is None else np.asarray(low_res),
            prompts=bundle,
        )

    @staticmethod
    def _proposal(result: CandidateResult, iteration: int) -> Proposal:
        return Proposal(
            mask=np.asarray(result.mask, dtype=bool).copy(),
            sam_score=result.sam_score,
            logits=np.asarray(result.logits).copy(),
            low_res_mask=None if result.low_res_mask is None else np.asarray(result.low_res_mask).copy(),
            iteration=iteration,
            prompts=result.prompts,
            candidate_id=result.candidate_id,
        )

    def _run_crop(self, crop: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        prepared = preprocess(crop, self.config.preprocessing)
        self.predictor.set_image(prepared.image)
        state = _State(prepared, self.config)
        initial = self._predict_bundle(state, state.initial_prompts())
        state.last_low_res = initial.low_res_mask
        state.proposals.append(self._proposal(initial, 0))
        logits = initial.logits
        iterations = 0
        candidate_calls = 0
        for iteration in range(1, self.config.growing.max_iterations + 1):
            state.update_from_logits(logits)
            candidate_ids = state.candidates()
            if not candidate_ids:
                break
            results = []
            for region_id in candidate_ids:
                results.append(self._predict_bundle(state, state.candidate_bundle(region_id)))
                candidate_calls += 1
            best = max(enumerate(results), key=lambda item: (item[1].sam_score, -item[0]))[1]
            if best.candidate_id is None:
                break
            state.commit(best.candidate_id)
            state.last_low_res = best.low_res_mask
            logits = best.logits
            state.proposals.append(self._proposal(best, iteration))
            iterations = iteration
        for refine_index in range(self.config.growing.refine_rounds):
            if state.last_low_res is None:
                break
            refined = self._predict_bundle(
                state,
                state._bundle(None, include_augmentation=False),
                force_previous_low_res=True,
            )
            state.last_low_res = refined.low_res_mask
            logits = refined.logits
            state.proposals.append(self._proposal(refined, iterations + refine_index + 1))
        selected, selection_metadata = select_proposal(prepared.image, state.proposals, self.config.selection)
        metadata = {
            "resized_shape": list(prepared.image.shape[:2]),
            "num_superpixels_effective": len(prepared.regions),
            "growing_iterations": iterations,
            "candidate_decoder_calls": candidate_calls,
            "proposal_count": len(state.proposals),
            "selection": selection_metadata,
        }
        if self.config.output.include_proposal_summary:
            metadata["proposals"] = [
                {
                    "iteration": proposal.iteration,
                    "candidate_id": proposal.candidate_id,
                    "area_ratio": proposal.area_ratio,
                    "sam_score": proposal.sam_score,
                    "heuristic_score": proposal.heuristic_score,
                }
                for proposal in state.proposals
            ]
        return np.asarray(selected.mask, dtype=bool), metadata

    def predict(self, image: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> InferenceResult:
        started = time.perf_counter()
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB HxWx3 image, got {image.shape}")
        bbox = validate_bbox(bbox_xyxy, image.shape)
        crop = crop_image(image, bbox)
        crop_mask, details = self._run_crop(crop)
        full_mask = paste_crop_mask(crop_mask, image.shape, bbox)
        metadata = {
            "config_sha256": self.config.sha256,
            "checkpoint_sha256": self.config.model.checkpoint_sha256,
            "bbox_xyxy": list(bbox),
            "original_shape": list(image.shape[:2]),
            "crop_shape": list(crop.shape[:2]),
            "device": self.device,
            "runtime_seconds": time.perf_counter() - started,
            "software": {
                "wbs_inference": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "torch": torch.__version__,
            },
            "resolved_config": resolved_config_dict(self.config),
            **details,
        }
        return InferenceResult(mask=full_mask, bbox_xyxy=bbox, metadata=metadata)
