"""Quick evaluation script to test multiple threshold values efficiently."""

import subprocess
import sys
from pathlib import Path


def run_evaluation(threshold: float, limit: int = None) -> None:
    """Run evaluation with specific threshold and optionally limit samples."""
    cmd = [
        "poetry",
        "run",
        "python",
        "-m",
        "technical.pipeline.evaluate_dataset",
        "--dataset-root",
        "technical/dataset/test_processed",
        "--threshold",
        str(threshold),
        "--device",
        "cuda",
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"\n{'='*60}")
    print(f"Testing threshold: {threshold}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, check=False)


def main():
    """Run threshold sweep to find optimal value."""
    # Quick test on subset first
    print("PHASE 1: Quick threshold sweep on 2000 samples")
    print("-" * 60)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]

    for threshold in thresholds:
        run_evaluation(threshold, limit=2000)

    print("\n" + "=" * 60)
    print("Phase 1 complete!")
    print("=" * 60)
    print("\nReview results above and pick the best threshold.")
    print("Then run full evaluation:")
    print("  poetry run python -m technical.pipeline.evaluate_dataset \\")
    print("      --dataset-root technical/dataset/test_processed \\")
    print("      --threshold <BEST_VALUE> \\")
    print("      --device cuda")


if __name__ == "__main__":
    main()
