"""Shared experiment runner utilities (seed, logdir, config overrides)."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional in lightweight environments
    torch = None

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def prepare_output_dir(prefix: str, output_dir: Optional[str] = None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ASSETS_DIR / f"{prefix}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metadata(output_dir: Path, payload: Dict[str, Any]) -> None:
    (output_dir / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory")
    parser.add_argument("--pipeline-cfg", type=str, default=None, help="Override pipeline config path")
    parser.add_argument("--tta-cfg", type=str, default=None, help="Override TTA config path")


def apply_env_overrides(
    output_dir: Path,
    *,
    pipeline_cfg: Optional[str] = None,
    tta_cfg: Optional[str] = None,
    pipeline_output_env: Optional[str] = None,
    tta_output_env: Optional[str] = None,
) -> None:
    if pipeline_cfg:
        os.environ["PIPELINE_CFG"] = pipeline_cfg
    if tta_cfg:
        os.environ["TTA_CFG"] = tta_cfg
    if pipeline_output_env:
        os.environ["PIPELINE_OUTPUT_DIR"] = str(output_dir)
    if tta_output_env:
        os.environ["TTA_OUTPUT_DIR"] = str(output_dir)
