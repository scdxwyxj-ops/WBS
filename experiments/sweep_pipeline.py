"""Sweep pipeline hyperparameters based on a JSON run list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runner import apply_env_overrides, prepare_output_dir, save_metadata, set_seed


def _load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep pipeline runs")
    parser.add_argument("--sweep", required=True, help="Path to JSON list of run specs")
    args = parser.parse_args()

    runs = _load_runs(Path(args.sweep))
    for idx, run in enumerate(runs):
        name = run.get("name", f"run_{idx:03d}")
        output_dir = prepare_output_dir(f"pipeline_{name}", run.get("output_dir"))
        set_seed(run.get("seed"))
        apply_env_overrides(
            output_dir,
            pipeline_cfg=run.get("pipeline_cfg"),
            pipeline_output_env="PIPELINE_OUTPUT_DIR",
        )
        save_metadata(
            output_dir,
            {
                "mode": "sweep_pipeline",
                "name": name,
                "seed": run.get("seed"),
                "pipeline_cfg": run.get("pipeline_cfg"),
                "output_dir": str(output_dir),
            },
        )

        from debug_tests.run_full_experiment import main as run_main

        run_main()


if __name__ == "__main__":
    main()
