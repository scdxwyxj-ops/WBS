"""Quick debug run: minimal pipeline execution for sanity checks."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import add_common_args, apply_env_overrides, prepare_output_dir, save_metadata, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug run (small-scale)")
    add_common_args(parser)
    args = parser.parse_args()

    output_dir = prepare_output_dir("debug", args.output_dir)
    set_seed(args.seed)
    apply_env_overrides(
        output_dir,
        pipeline_cfg=args.pipeline_cfg,
        pipeline_output_env="PIPELINE_OUTPUT_DIR",
    )
    # Limit to a small batch for end-to-end TTA sanity checks.
    if "DEBUG_MAX_SAMPLES" not in os.environ:
        os.environ["DEBUG_MAX_SAMPLES"] = "10"
    save_metadata(
        output_dir,
        {
            "mode": "debug",
            "seed": args.seed,
            "pipeline_cfg": args.pipeline_cfg,
            "output_dir": str(output_dir),
        },
    )

    from debug_tests.run_tta import main as run_main

    run_main()


if __name__ == "__main__":
    main()
