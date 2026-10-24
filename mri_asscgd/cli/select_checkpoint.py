"""Select the ASSCGD checkpoint with the best volume-averaged validation PSNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        help="Restrict selection to summaries from these completed epochs.",
    )
    args = parser.parse_args()
    selected_epochs = set(args.epochs or [])
    candidates = []
    for summary_path in sorted(args.validation_dir.glob("step-*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("split") != "val":
            raise ValueError(f"not a validation summary: {summary_path}")
        completed_epochs = int(summary["completed_epochs"])
        if selected_epochs and completed_epochs not in selected_epochs:
            continue
        candidates.append(
            {
                "checkpoint_step": int(summary["checkpoint_step"]),
                "completed_epochs": completed_epochs,
                "checkpoint": summary["checkpoint"],
                "asscgd": summary["asscgd"],
            }
        )
    if not candidates:
        raise ValueError(f"no validation summaries found under {args.validation_dir}")
    if selected_epochs:
        found_epochs = {item["completed_epochs"] for item in candidates}
        missing_epochs = selected_epochs - found_epochs
        if missing_epochs:
            raise ValueError(
                "missing validation summaries for epochs: "
                + ", ".join(str(epoch) for epoch in sorted(missing_epochs))
            )
    best = max(candidates, key=lambda item: (item["asscgd"]["psnr"], item["asscgd"]["ssim"]))
    selection = {
        "selection_metric": "volume-averaged validation PSNR (SSIM tie-break)",
        "checkpoint": best["checkpoint"],
        "checkpoint_step": best["checkpoint_step"],
        "completed_epochs": best["completed_epochs"],
        "validation_metrics": best["asscgd"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
