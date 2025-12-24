"""Run ablation sets defined in a JSON list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import apply_env_overrides, prepare_output_dir, save_metadata, set_seed


def _load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation table")
    parser.add_argument("--table", required=True, help="Path to JSON list of ablation specs")
    args = parser.parse_args()

    runs = _load_runs(Path(args.table))
    for idx, run in enumerate(runs):
        name = run.get("name", f"ablation_{idx:03d}")
        output_dir = prepare_output_dir(name, run.get("output_dir"))
        set_seed(run.get("seed"))
        apply_env_overrides(
            output_dir,
            pipeline_cfg=run.get("pipeline_cfg"),
            tta_cfg=run.get("tta_cfg"),
            tta_output_env="TTA_OUTPUT_DIR" if run.get("mode") == "tta" else None,
            pipeline_output_env="PIPELINE_OUTPUT_DIR" if run.get("mode") == "pipeline" else None,
        )
        save_metadata(
            output_dir,
            {
                "mode": run.get("mode", "pipeline"),
                "name": name,
                "seed": run.get("seed"),
                "pipeline_cfg": run.get("pipeline_cfg"),
                "tta_cfg": run.get("tta_cfg"),
                "output_dir": str(output_dir),
            },
        )

        if run.get("mode") == "tta":
            from debug_tests.run_tta import main as run_main
        else:
            from debug_tests.run_full_experiment import main as run_main

        run_main()


if __name__ == "__main__":
    main()
