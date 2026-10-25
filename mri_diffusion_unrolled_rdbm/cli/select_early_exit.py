"""Select one fixed early-exit stage using validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.validation_summary.read_text())
    candidates = [
        {"stage": int(stage), **metrics}
        for stage, metrics in summary["stages"].items()
    ]
    best = max(candidates, key=lambda row: (float(row["psnr"]), float(row["ssim"])))
    result = {
        "selection_metric": "volume-averaged validation PSNR (SSIM tie-break)",
        "checkpoint": summary["checkpoint"],
        "checkpoint_step": summary["checkpoint_step"],
        "completed_epochs": summary["completed_epochs"],
        "acceleration": summary["acceleration"],
        "selected_stage": best["stage"],
        "validation_metrics": {
            key: value for key, value in best.items() if key != "stage"
        },
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
