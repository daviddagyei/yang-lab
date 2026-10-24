"""Select the full diffusion-unrolled checkpoint by validation PSNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHOD = "diffusion_unrolled_rdbm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    args = parser.parse_args()
    requested = set(args.epochs)
    candidates = []
    for path in sorted(args.validation_dir.glob("step-*/summary.json")):
        summary = json.loads(path.read_text())
        epoch = int(summary["completed_epochs"])
        if summary.get("split") != "val" or epoch not in requested:
            continue
        candidates.append(
            {
                "checkpoint_step": int(summary["checkpoint_step"]),
                "completed_epochs": epoch,
                "checkpoint": summary["checkpoint"],
                METHOD: summary[METHOD],
            }
        )
    found = {row["completed_epochs"] for row in candidates}
    if found != requested:
        raise ValueError(f"missing validation epochs: {sorted(requested - found)}")
    best = max(candidates, key=lambda row: (row[METHOD]["psnr"], row[METHOD]["ssim"]))
    selection = {
        "selection_metric": "volume-averaged validation PSNR (SSIM tie-break)",
        "checkpoint": best["checkpoint"],
        "checkpoint_step": best["checkpoint_step"],
        "completed_epochs": best["completed_epochs"],
        "validation_metrics": best[METHOD],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
