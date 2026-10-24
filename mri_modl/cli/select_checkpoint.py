"""Select the MoDL checkpoint with the best volume-averaged validation PSNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for summary_path in sorted(args.validation_dir.glob("step-*/summary.json")):
        summary = json.loads(summary_path.read_text())
        step = int(summary["checkpoint_step"])
        candidates.append(
            {
                "checkpoint_step": step,
                "checkpoint": str((args.checkpoint_dir / f"checkpoint-{step:07d}.pt").resolve()),
                "modl": summary["modl"],
            }
        )
    if not candidates:
        raise ValueError(f"no validation summaries found under {args.validation_dir}")
    best = max(candidates, key=lambda item: float(item["modl"]["psnr"]))
    selection = {
        "selection_metric": "volume-averaged validation PSNR",
        "checkpoint": best["checkpoint"],
        "checkpoint_step": best["checkpoint_step"],
        "validation_metrics": best["modl"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
