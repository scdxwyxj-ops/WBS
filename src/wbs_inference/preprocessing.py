"""Deterministic image resizing, SLIC partitioning, and region graph construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skimage.segmentation import slic
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from .config import PreprocessingConfig


@dataclass(frozen=True)
class Region:
    region_id: int
    mask: np.ndarray
    point_xy: tuple[float, float]
    touches_edge: bool
    color_score: float


@dataclass(frozen=True)
class PreprocessedImage:
    image: np.ndarray
    segments: np.ndarray
    regions: tuple[Region, ...]
    adjacency: tuple[frozenset[int], ...]


def _resize_shape(height: int, width: int, config: PreprocessingConfig) -> tuple[int, int]:
    if width >= height:
        new_w = config.long_edge
        new_h = int(height * config.long_edge / width)
        if new_h % 2:
            new_h -= 1
    else:
        new_h = config.long_edge
        new_w = int(width * config.long_edge / height)
        if new_w % 2:
            new_w -= 1
    return max(1, new_h), max(1, new_w)


def resize_image(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 image, got {image.shape}")
    target = _resize_shape(image.shape[0], image.shape[1], config)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    resized = resize(tensor, list(target), interpolation=InterpolationMode.NEAREST, antialias=True)
    return resized.permute(1, 2, 0).cpu().numpy()


def _normalise_labels(segments: np.ndarray) -> np.ndarray:
    labels = np.unique(segments)
    return np.searchsorted(labels, segments).astype(np.int32)


def _touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _representative_point(mask: np.ndarray) -> tuple[float, float]:
    coords_yx = np.argwhere(mask)
    if coords_yx.size == 0:
        raise ValueError("Cannot choose a representative point for an empty region")
    centroid_yx = coords_yx.mean(axis=0)
    rounded = np.rint(centroid_yx).astype(int)
    if 0 <= rounded[0] < mask.shape[0] and 0 <= rounded[1] < mask.shape[1] and mask[tuple(rounded)]:
        point_yx = rounded
    else:
        distances = np.sum((coords_yx - centroid_yx[None, :]) ** 2, axis=1)
        point_yx = coords_yx[int(np.argmin(distances))]
    return float(point_yx[1]), float(point_yx[0])


def _color_score(image: np.ndarray, mask: np.ndarray) -> float:
    pixels = image[mask].astype(np.float32)
    return 255.0 - float(pixels.mean())


def _build_adjacency(segments: np.ndarray) -> tuple[frozenset[int], ...]:
    count = int(segments.max()) + 1
    neighbors: list[set[int]] = [set() for _ in range(count)]
    pairs = []
    horizontal = segments[:, :-1] != segments[:, 1:]
    if horizontal.any():
        pairs.append(np.stack((segments[:, :-1][horizontal], segments[:, 1:][horizontal]), axis=1))
    vertical = segments[:-1, :] != segments[1:, :]
    if vertical.any():
        pairs.append(np.stack((segments[:-1, :][vertical], segments[1:, :][vertical]), axis=1))
    if pairs:
        for left, right in np.unique(np.concatenate(pairs, axis=0), axis=0):
            a, b = int(left), int(right)
            neighbors[a].add(b)
            neighbors[b].add(a)
    return tuple(frozenset(items) for items in neighbors)


def preprocess(image: np.ndarray, preprocessing: PreprocessingConfig) -> PreprocessedImage:
    resized = resize_image(image, preprocessing)
    segments = slic(
        resized,
        n_segments=preprocessing.num_superpixels,
        compactness=preprocessing.compactness,
        sigma=preprocessing.sigma,
        min_size_factor=preprocessing.min_size_factor,
        max_size_factor=preprocessing.max_size_factor,
        start_label=1,
        slic_zero=False,
        convert2lab=True,
        enforce_connectivity=True,
        channel_axis=-1,
    )
    segments = _normalise_labels(segments)
    regions = tuple(
        Region(
            region_id=region_id,
            mask=(mask := segments == region_id),
            point_xy=_representative_point(mask),
            touches_edge=_touches_edge(mask),
            color_score=_color_score(resized, mask),
        )
        for region_id in range(int(segments.max()) + 1)
    )
    return PreprocessedImage(
        image=resized,
        segments=segments,
        regions=regions,
        adjacency=_build_adjacency(segments),
    )
