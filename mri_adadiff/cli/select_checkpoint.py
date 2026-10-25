"""Select AdaDiff prior checkpoint by volume-averaged 4x validation PSNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for summary_path in sorted(args.validation_dir.glob("epoch-*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if int(summary["acceleration"]) != 4 or summary["split"] != "val":
            raise ValueError(f"selection input is not 4x validation: {summary_path}")
        candidates.append(
            {
                "completed_epochs": int(summary["completed_epochs"]),
                "checkpoint_step": int(summary["checkpoint_step"]),
                "checkpoint": str(Path(summary["checkpoint"]).resolve()),
                "adadiff": summary["adadiff"],
            }
        )
    if not candidates:
        raise ValueError(f"no AdaDiff validation summaries under {args.validation_dir}")
    best = max(candidates, key=lambda item: float(item["adadiff"]["psnr"]))
    selection = {
        "selection_metric": "volume-averaged 4x validation PSNR",
        "shared_prior_across_accelerations": True,
        **best,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
