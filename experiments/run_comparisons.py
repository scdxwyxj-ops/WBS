"""Run all baseline comparison configs from compartments/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_baselines import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline comparisons for all configs.")
    parser.add_argument("--comparisons-dir", default=str(ROOT / "configs" / "comparisons"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg_dir = Path(args.comparisons_dir)
    if not cfg_dir.exists():
        raise FileNotFoundError(
            f"Missing comparisons directory: {cfg_dir}"
        )

    configs = sorted(cfg_dir.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No config JSON files found in {cfg_dir}")

    for cfg in configs:
        run_from_config(cfg, output_root=args.output_dir, max_samples_override=args.max_samples)


if __name__ == "__main__":
    main()
