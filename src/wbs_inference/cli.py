"""Command-line interface for single-image inference and config inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_config, resolved_config_dict
from .pipeline import WBSSegmenter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wbs-infer", description="BBox-guided WBS segmentation with SAM2")
    parser.add_argument("--image", type=Path, help="Input RGB image")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--checkpoint", type=Path, help="SAM2.1 checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Complete JSON inference config")
    parser.add_argument("--output", type=Path, help="Output mask PNG")
    parser.add_argument("--metadata", type=Path, help="Output run metadata JSON")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Runtime device override")
    parser.add_argument("--print-config", action="store_true", help="Validate and print resolved config, then exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.print_config:
        print(json.dumps(resolved_config_dict(config), indent=2, ensure_ascii=False))
        return
    missing = [name for name in ("image", "bbox", "checkpoint", "output") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required inference arguments: {', '.join('--' + name for name in missing)}")
    image = np.asarray(Image.open(args.image).convert("RGB"))
    segmenter = WBSSegmenter.from_checkpoint(
        args.checkpoint,
        config,
        device=args.device,
    )
    result = segmenter.predict(image, tuple(args.bbox))
    metadata_path = args.metadata
    if metadata_path is None and config.output.save_metadata:
        metadata_path = args.output.with_suffix(".json")
    result.save(args.output, metadata_path=metadata_path, mask_value=config.output.mask_value)
    print(
        json.dumps(
            {"mask": str(args.output), "metadata": str(metadata_path) if metadata_path else None}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
