"""Run pipeline + TTA using unified runner settings."""

from __future__ import annotations

import argparse

from experiments.runner import add_common_args, apply_env_overrides, prepare_output_dir, save_metadata, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 pipeline + TTA experiment")
    add_common_args(parser)
    args = parser.parse_args()

    output_dir = prepare_output_dir("tta", args.output_dir)
    set_seed(args.seed)
    apply_env_overrides(
        output_dir,
        pipeline_cfg=args.pipeline_cfg,
        tta_cfg=args.tta_cfg,
        tta_output_env="TTA_OUTPUT_DIR",
    )
    save_metadata(
        output_dir,
        {
            "mode": "pipeline_tta",
            "seed": args.seed,
            "pipeline_cfg": args.pipeline_cfg,
            "tta_cfg": args.tta_cfg,
            "output_dir": str(output_dir),
        },
    )

    from debug_tests.run_tta import main as run_main

    run_main()


if __name__ == "__main__":
    main()
