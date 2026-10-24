"""Select the checkpoint with the best volume-averaged validation PSNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = [json.loads(path.read_text()) for path in args.summaries]
    if any(candidate.get("split") != "val" for candidate in candidates):
        raise ValueError("checkpoint selection may only use validation summaries")
    best = max(candidates, key=lambda item: (item["rdbm"]["psnr"], item["rdbm"]["ssim"]))
    result = {
        "selection_metric": "volume-averaged validation PSNR",
        "checkpoint": best["checkpoint"],
        "checkpoint_step": best["checkpoint_step"],
        "validation_metrics": best["rdbm"],
        "candidates": [
            {
                "checkpoint_step": item["checkpoint_step"],
                "checkpoint": item["checkpoint"],
                "rdbm": item["rdbm"],
            }
            for item in sorted(candidates, key=lambda item: item["checkpoint_step"])
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
