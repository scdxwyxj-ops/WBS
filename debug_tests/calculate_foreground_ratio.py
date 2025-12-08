#!/usr/bin/env python3
"""Compute foreground-area ratios for each dataset split.

For every dataset listed in ``DATASETS`` we load the masks using
``datasets.dataset.load_dataset`` and report summary statistics of the
foreground ratio (mask pixels / total pixels). The loader already honours
the default data root configured in ``CONSTANT.json``.

Usage:
    python debug_tests/calculate_foreground_ratio.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.pipeline_config import load_pipeline_config
from datasets.dataset import load_dataset
from debug_tests.debug_test import MAIN_DIR, _load_constants


DEFAULT_DATASETS = ["cropped", "dataset_v0", "original"]


def _foreground_ratios(masks: Sequence[np.ndarray]) -> np.ndarray:
    ratios: List[float] = []
    for mask in masks:
        mask_bool = np.asarray(mask, dtype=bool)
        total_pixels = mask_bool.size
        if total_pixels == 0:
            ratios.append(0.0)
            continue
        ratios.append(float(mask_bool.sum() / total_pixels))
    return np.asarray(ratios, dtype=float)


def describe_ratio(ratios: np.ndarray) -> dict:
    if ratios.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(ratios.size),
        "mean": float(np.mean(ratios)),
        "std": float(np.std(ratios)),
        "min": float(np.min(ratios)),
        "p25": float(np.percentile(ratios, 25)),
        "p50": float(np.percentile(ratios, 50)),
        "p75": float(np.percentile(ratios, 75)),
        "p90": float(np.percentile(ratios, 90)),
        "max": float(np.max(ratios)),
    }


def run(dataset_names: Iterable[str], *, target_long_edge: int | None) -> dict:
    results = {}
    for name in dataset_names:
        print(f"Processing dataset: {name}")
        _, masks = load_dataset(
            name,
            data_root=None,
            target_long_edge=target_long_edge,
        )
        ratios = _foreground_ratios(masks)
        stats = describe_ratio(ratios)
        results[name] = {
            "stats": stats,
            "ratios": ratios.tolist(),
        }
        print(
            f"  count={stats['count']}, mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, min={stats['min']:.4f}, "
            f"p50={stats['p50'] if 'p50' in stats else stats['mean']:.4f}, "
            f"p90={stats.get('p90', 0.0):.4f}, max={stats['max']:.4f}"
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute foreground ratios for datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to include (default: %(default)s)",
    )
    parser.add_argument(
        "--target-long-edge",
        type=int,
        default=None,
        help="Optional long-edge resize applied when loading masks (matches pipeline config if omitted).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional JSON file path to dump the raw ratios and summary stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    constants = _load_constants()
    pipeline_cfg = load_pipeline_config(MAIN_DIR / constants["pipeline_cfg"])

    target_edge = args.target_long_edge
    if target_edge is None:
        target_edge = pipeline_cfg.dataset.target_long_edge

    results = run(args.datasets, target_long_edge=target_edge)

    if args.save:
        output_path = Path(args.save)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
