"""Run the pipeline (train/eval/infer) with unified runner settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import add_common_args, apply_env_overrides, prepare_output_dir, save_metadata, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 pipeline experiment")
    add_common_args(parser)
    args = parser.parse_args()

    output_dir = prepare_output_dir("pipeline", args.output_dir)
    set_seed(args.seed)
    apply_env_overrides(
        output_dir,
        pipeline_cfg=args.pipeline_cfg,
        pipeline_output_env="PIPELINE_OUTPUT_DIR",
    )
    save_metadata(
        output_dir,
        {
            "mode": "pipeline",
            "seed": args.seed,
            "pipeline_cfg": args.pipeline_cfg,
            "output_dir": str(output_dir),
        },
    )

    from debug_tests.run_full_experiment import main as run_main

    run_main()


if __name__ == "__main__":
    main()
