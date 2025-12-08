import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:  # Pillow>=9.1
    _BILINEAR = Image.Resampling.BILINEAR
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Fallback for older Pillow versions
    _BILINEAR = Image.BILINEAR
    _NEAREST = Image.NEAREST

_SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "CONSTANT.json"


def load_dataset(
    dataset_name: str,
    data_root: Optional[str] = None,
    target_long_edge: Optional[int] = None,
    *,
    return_paths: bool = False,
) -> Union[
    Tuple[List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray], List[str]],
]:
    """Load image and mask pairs for a dataset.

    Args:
        dataset_name: Name of the dataset folder (e.g. "original", "cropped", "dataset_v0").
        data_root: Optional override for the datasets root directory.
        target_long_edge: Optional size to scale the longest image edge to. Aspect ratio is
            preserved and masks use nearest-neighbour sampling.

    Returns:
        Tuple of images (RGB uint8 arrays) and masks (bool arrays). If ``return_paths`` is
        True, also returns a list of image filenames (strings) corresponding to each pair.
    """
    root = _resolve_data_root(data_root)
    dataset_dir = root / dataset_name
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    image_names: List[str] = []

    for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and _is_supported(p)):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            print(f"[Warn] mask missing: {mask_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if target_long_edge is not None:
            image, mask = _resize_pair(image, mask, target_long_edge)

        images.append(np.array(image))
        masks.append(np.array(mask) > 0)
        image_names.append(image_path.name)

    if return_paths:
        return images, masks, image_names
    return images, masks


def _resize_pair(image: Image.Image, mask: Image.Image, target_long_edge: int) -> Tuple[Image.Image, Image.Image]:
    if target_long_edge <= 0:
        raise ValueError("target_long_edge must be positive")

    width, height = image.size
    if width == 0 or height == 0:
        return image, mask

    current_long_edge = max(width, height)
    if current_long_edge == 0 or current_long_edge == target_long_edge:
        return image, mask

    scale = target_long_edge / current_long_edge
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    image = image.resize((new_width, new_height), _BILINEAR)
    mask = mask.resize((new_width, new_height), _NEAREST)
    return image, mask


def _resolve_data_root(data_root: Optional[str]) -> Path:
    if data_root:
        candidate = _normalise_path(data_root)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Provided data_root does not exist: {candidate}")

    default_root = _load_default_data_root()
    if not default_root.exists():
        raise FileNotFoundError(f"Default data path does not exist: {default_root}")
    return default_root


@lru_cache(maxsize=1)
def _load_default_data_root() -> Path:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing CONSTANT.json at {_CONFIG_PATH}")

    config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_path = config.get("data_path")
    if not raw_path:
        raise KeyError("`data_path` not found in CONSTANT.json")

    return _normalise_path(raw_path)


def _normalise_path(raw_path: str) -> Path:
    normalised = raw_path.replace("\\", "/")
    candidate = Path(normalised)
    if candidate.exists():
        return candidate

    if len(normalised) > 2 and normalised[1:3] == ":/":
        drive_letter = normalised[0].lower()
        remainder = normalised[3:]
        wsl_candidate = Path("/mnt") / drive_letter / remainder
        if wsl_candidate.exists():
            return wsl_candidate
        return wsl_candidate

    return candidate


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in _SUPPORTED_SUFFIXES
