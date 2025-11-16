"""
Convenience script for multi-resolution cyclic training.

Automates the full workflow:
1. Initial DSR training (100 epochs)
2. EdgeFace fine-tuning (35 epochs)  
3. DSR cyclic fine-tuning (20 additional epochs from checkpoint with early stopping patience of 8)

Usage:
    # Train all resolutions with full cycle
    python -m technical.tools.cyclic_train

    # Train only specific resolutions
    python -m technical.tools.cyclic_train --vlr-sizes 32

    # Skip initial training (if models already exist)
    python -m technical.tools.cyclic_train --skip-initial --skip-edgeface

    # Dry run (print commands without executing)
    python -m technical.tools.cyclic_train --dry-run
    
Note:
    The cyclic fine-tuning step (Step 3) now uses --additional-epochs to ensure all models
    are fine-tuned for exactly 20 epochs regardless of their checkpoint's epoch count.
    This prevents inconsistent fine-tuning durations (e.g., one model getting 15 epochs
    while another gets 5 epochs due to different starting points).
"""

import subprocess
import argparse
from pathlib import Path


def run_cmd(cmd: str, dry_run: bool = False):
    """Run command and stream output."""
    print(f"\n{'='*70}\n{cmd}\n{'='*70}\n")
    
    if dry_run:
        print("(DRY RUN - command not executed)")
        return
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Automated cyclic training pipeline for DSR + EdgeFace"
    )
    parser.add_argument(
        "--vlr-sizes", 
        nargs="+", 
        type=int, 
        default=[16, 24, 32],
        help="VLR resolutions to train (default: 16 24 32)"
    )
    parser.add_argument(
        "--device", 
        default="cuda",
        help="Device for training (default: cuda)"
    )
    parser.add_argument(
        "--skip-initial", 
        action="store_true", 
        help="Skip initial DSR training (use existing checkpoints)"
    )
    parser.add_argument(
        "--skip-edgeface", 
        action="store_true", 
        help="Skip EdgeFace fine-tuning (use existing checkpoints)"
    )
    parser.add_argument(
        "--skip-cyclic", 
        action="store_true",
        help="Skip DSR cyclic fine-tuning (only do initial training + EdgeFace FT)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--frontal-only",
        action="store_true",
        help="Use frontal-only filtered dataset"
    )
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    frontal_flag = "--frontal-only" if args.frontal_only else ""
    
    print(f"\n{'#'*70}")
    print(f"# CYCLIC TRAINING PIPELINE")
    print(f"# Resolutions: {args.vlr_sizes}")
    print(f"# Device: {args.device}")
    print(f"# Skip initial: {args.skip_initial}")
    print(f"# Skip EdgeFace: {args.skip_edgeface}")
    print(f"# Skip cyclic: {args.skip_cyclic}")
    print(f"# Dataset: {'frontal-only' if args.frontal_only else 'full'}")
    print(f"{'#'*70}\n")
    
    for vlr_size in args.vlr_sizes:
        print(f"\n\n{'#'*70}")
        print(f"# TRAINING {vlr_size}×{vlr_size} RESOLUTION")
        print(f"{'#'*70}\n")
        
        dsr_path = base_dir / "dsr" / f"dsr{vlr_size}.pth"
        edgeface_path = base_dir / "facial_rec" / "edgeface_weights" / f"edgeface_finetuned_{vlr_size}.pth"
        
        # Step 1: Initial DSR training
        if not args.skip_initial:
            print(f"\n📊 STEP 1: Initial DSR Training ({vlr_size}×{vlr_size})")
            run_cmd(
                f"python -m technical.dsr.train_dsr "
                f"--vlr-size {vlr_size} "
                f"--device {args.device} "
                f"{frontal_flag}",
                dry_run=args.dry_run
            )
        else:
            print(f"\n⏭️  STEP 1: Skipped (using existing {dsr_path})")
            if not args.dry_run and not dsr_path.exists():
                print(f"⚠️  WARNING: {dsr_path} does not exist!")
        
        # Step 2: EdgeFace fine-tuning
        if not args.skip_edgeface:
            print(f"\n🎯 STEP 2: EdgeFace Fine-Tuning ({vlr_size}×{vlr_size})")
            run_cmd(
                f"python -m technical.facial_rec.finetune_edgeface "
                f"--vlr-size {vlr_size} "
                f"--device {args.device}",
                dry_run=args.dry_run
            )
        else:
            print(f"\n⏭️  STEP 2: Skipped (using existing {edgeface_path})")
            if not args.dry_run and not edgeface_path.exists():
                print(f"⚠️  WARNING: {edgeface_path} does not exist!")
        
        # Step 3: DSR cyclic fine-tuning
        if not args.skip_cyclic:
            print(f"\n🔄 STEP 3: DSR Cyclic Fine-Tuning ({vlr_size}×{vlr_size})")
            
            # Resolution-specific hyperparameters for cyclic training
            # All models train for 20 additional epochs from their current checkpoint
            # with early stopping patience of 8 epochs
            if vlr_size == 16:
                additional_epochs = 20
                lr = 9e-5
                lambda_id = 0.68
                lambda_fm = 0.22
            elif vlr_size == 24:
                additional_epochs = 20
                lr = 8.5e-5
                lambda_id = 0.65
                lambda_fm = 0.20
            else:  # 32
                additional_epochs = 20
                lr = 8e-5
                lambda_id = 0.65
                lambda_fm = 0.20
            
            run_cmd(
                f"python -m technical.dsr.train_dsr "
                f"--vlr-size {vlr_size} "
                f"--device {args.device} "
                f"--resume {dsr_path} "
                f"--edgeface edgeface_finetuned_{vlr_size}.pth "
                f"--additional-epochs {additional_epochs} "
                f"--patience 8 "
                f"--learning-rate {lr} "
                f"--lambda-identity {lambda_id} "
                f"--lambda-feature-match {lambda_fm} "
                f"{frontal_flag}",
                dry_run=args.dry_run
            )
        else:
            print(f"\n⏭️  STEP 3: Skipped (no cyclic fine-tuning)")
        
        print(f"\n✅ Completed pipeline for {vlr_size}×{vlr_size}\n")
    
    print(f"\n{'#'*70}")
    print(f"# 🎉 ALL RESOLUTIONS COMPLETE!")
    print(f"{'#'*70}\n")
    
    if not args.dry_run:
        print("Next steps:")
        print("1. Evaluate models with: python -m technical.pipeline.evaluate_gui")
        print("2. Compare to baseline metrics")
        print("3. Generate visualizations for paper")
        print("\nSee CYCLIC_VS_RETRAINING_ANALYSIS.md for expected improvements.\n")


if __name__ == "__main__":
    main()
