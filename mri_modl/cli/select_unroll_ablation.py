"""Summarize validation-only MoDL checkpoint and early-exit ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm, rm = left["metrics"], right["metrics"]
    no_worse = lm["psnr"] >= rm["psnr"] and lm["ssim"] >= rm["ssim"] and lm["nmse"] <= rm["nmse"]
    strictly_better = lm["psnr"] > rm["psnr"] or lm["ssim"] > rm["ssim"] or lm["nmse"] < rm["nmse"]
    return no_worse and strictly_better


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--psnr-tolerance", type=float, default=0.25)
    args = parser.parse_args()

    candidates = []
    for path in sorted(args.ablation_dir.glob("step-*/summary.json")):
        summary = json.loads(path.read_text())
        for stage, values in summary["stages"].items():
            candidates.append(
                {
                    "checkpoint_step": int(summary["checkpoint_step"]),
                    "checkpoint": str(
                        (args.checkpoint_dir / f"checkpoint-{int(summary['checkpoint_step']):07d}.pt").resolve()
                    ),
                    "unroll": int(stage),
                    "metrics": values["modl"],
                }
            )
    if not candidates:
        raise ValueError(f"no ablation summaries found under {args.ablation_dir}")
    best_psnr = max(candidates, key=lambda item: item["metrics"]["psnr"])
    best_ssim = max(candidates, key=lambda item: item["metrics"]["ssim"])
    best_nmse = min(candidates, key=lambda item: item["metrics"]["nmse"])
    near_best = [
        item for item in candidates
        if item["metrics"]["psnr"] >= best_psnr["metrics"]["psnr"] - args.psnr_tolerance
    ]
    balanced = max(
        near_best,
        key=lambda item: (item["metrics"]["ssim"], -item["metrics"]["nmse"]),
    )
    pareto = [
        candidate for candidate in candidates
        if not any(dominates(other, candidate) for other in candidates if other is not candidate)
    ]
    pareto.sort(key=lambda item: (-item["metrics"]["psnr"], -item["metrics"]["ssim"]))
    output = {
        "status": "validation_only_post_hoc_exploration",
        "selection_rule": (
            f"among configurations within {args.psnr_tolerance:.2f} dB of best validation PSNR, "
            "maximize validation SSIM and use NMSE as tie-breaker"
        ),
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_nmse": best_nmse,
        "balanced_candidate": balanced,
        "pareto_front": pareto,
        "candidate_count": len(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
