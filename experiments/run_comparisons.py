"""Run all baseline comparison configs from compartments/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from experiments.compare_baselines import run_from_config
from experiments.runner import prepare_output_dir


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

    output_root = prepare_output_dir("comparisons", args.output_dir)
    summaries = []
    for cfg in configs:
        run_dir = run_from_config(
            cfg,
            output_root=str(output_root),
            max_samples_override=args.max_samples,
        )
        summary_path = run_dir / "overall_summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            for entry in payload:
                summaries.append(
                    {
                        "method": run_dir.name,
                        "dataset": entry.get("dataset"),
                        "miou": entry.get("miou"),
                        "dice": entry.get("dice"),
                        "hd95": entry.get("hd95"),
                    }
                )

    (output_root / "comparisons_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    csv_path = output_root / "comparisons_summary.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("method,dataset,miou,dice,hd95\n")
        for row in summaries:
            handle.write(
                f"{row['method']},{row['dataset']},{row['miou']},{row['dice']},{row['hd95']}\n"
            )


if __name__ == "__main__":
    main()
