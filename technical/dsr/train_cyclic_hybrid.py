"""Cyclic training script for Hybrid DSR and EdgeFace models.

This script performs full cyclic training:
1. Train hybrid DSR models (16, 24, 32) with base EdgeFace XXS
2. Finetune EdgeFace models (16, 24, 32) using hybrid DSR outputs
3. Retrain hybrid DSR models with finetuned EdgeFace weights

Each cycle improves both the super-resolution and face recognition components.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Import needed for unpickling finetuned EdgeFace checkpoints
try:
    from ..facial_rec.finetune_edgeface import FinetuneConfig
except ImportError:
    from technical.facial_rec.finetune_edgeface import FinetuneConfig


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        cmd: Command as list of strings
        description: Description to print
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user\n")
        return False


def train_hybrid_dsr_phase(vlr_sizes: list[int], edgeface_base: str, resume: bool = False) -> bool:
    """Train hybrid DSR models for all resolutions.
    
    Args:
        vlr_sizes: List of VLR sizes to train (e.g., [16, 24, 32])
        edgeface_base: Base EdgeFace model name (e.g., "edgeface_xxs.pt")
        resume: Resume from checkpoints if they exist
        
    Returns:
        True if all models trained successfully
    """
    print(f"\n{'#'*80}")
    print(f"# PHASE 1: TRAIN HYBRID DSR MODELS")
    print(f"# EdgeFace: {edgeface_base}")
    print(f"# Resolutions: {vlr_sizes}")
    print(f"{'#'*80}\n")
    
    for vlr_size in vlr_sizes:
        cmd = [
            "poetry", "run", "python", "-m", "technical.dsr.train_hybrid_dsr",
            "--vlr-size", str(vlr_size),
            "--edgeface", edgeface_base,
            "--num-workers", "8"
        ]
        
        if resume:
            cmd.append("--resume")
        
        success = run_command(
            cmd,
            f"Train Hybrid DSR {vlr_size}×{vlr_size} → 112×112"
        )
        
        if not success:
            response = input(f"Continue to next model? (y/n): ")
            if response.lower() != 'y':
                return False
    
    return True


def finetune_edgeface_phase(vlr_sizes: list[int]) -> bool:
    """Finetune EdgeFace models using hybrid DSR outputs.
    
    Args:
        vlr_sizes: List of VLR sizes to finetune
        
    Returns:
        True if all models finetuned successfully
    """
    print(f"\n{'#'*80}")
    print(f"# PHASE 2: FINETUNE EDGEFACE MODELS")
    print(f"# Using Hybrid DSR outputs for each resolution")
    print(f"# Resolutions: {vlr_sizes}")
    print(f"{'#'*80}\n")
    
    for vlr_size in vlr_sizes:
        # Use edgeface_finetune dataset (same subjects in train/val for proper fine-tuning)
        cmd = [
            "poetry", "run", "python", "-m", "technical.facial_rec.finetune_edgeface",
            "--vlr-size", str(vlr_size),
            "--dsr-weights", f"technical/dsr/hybrid_dsr{vlr_size}.pth",
            "--train-dir", "technical/dataset/edgeface_finetune/train",
            "--val-dir", "technical/dataset/edgeface_finetune/val"
        ]
        
        success = run_command(
            cmd,
            f"Finetune EdgeFace for {vlr_size}×{vlr_size}"
        )
        
        if not success:
            response = input(f"Continue to next model? (y/n): ")
            if response.lower() != 'y':
                return False
    
    return True


def retrain_hybrid_dsr_phase(vlr_sizes: list[int]) -> bool:
    """Retrain hybrid DSR models with finetuned EdgeFace weights.
    
    Args:
        vlr_sizes: List of VLR sizes to retrain
        
    Returns:
        True if all models retrained successfully
    """
    print(f"\n{'#'*80}")
    print(f"# PHASE 3: RETRAIN HYBRID DSR WITH FINETUNED EDGEFACE")
    print(f"# Using finetuned EdgeFace models for identity losses")
    print(f"# Resolutions: {vlr_sizes}")
    print(f"{'#'*80}\n")
    
    base_dir = Path(__file__).resolve().parent
    
    for vlr_size in vlr_sizes:
        # Move old checkpoint to backup
        old_checkpoint = base_dir / f"hybrid_dsr{vlr_size}.pth"
        backup_checkpoint = base_dir / f"hybrid_dsr{vlr_size}_base_edgeface.pth"
        
        if old_checkpoint.exists():
            print(f"Backing up {old_checkpoint.name} → {backup_checkpoint.name}")
            old_checkpoint.rename(backup_checkpoint)
        
        # Train with finetuned EdgeFace
        cmd = [
            "poetry", "run", "python", "-m", "technical.dsr.train_hybrid_dsr",
            "--vlr-size", str(vlr_size),
            "--edgeface", f"edgeface_finetuned_{vlr_size}.pth",
            "--num-workers", "8"
        ]
        
        success = run_command(
            cmd,
            f"Retrain Hybrid DSR {vlr_size}×{vlr_size} with finetuned EdgeFace"
        )
        
        if not success:
            response = input(f"Continue to next model? (y/n): ")
            if response.lower() != 'y':
                return False
    
    return True


def verify_prerequisites() -> bool:
    """Verify that required files and directories exist.
    
    Returns:
        True if all prerequisites are met
    """
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent
    
    print("Verifying prerequisites...")
    
    # Check datasets
    train_dir = base_dir.parent / "dataset" / "train_processed"
    val_dir = base_dir.parent / "dataset" / "val_processed"
    
    if not train_dir.exists():
        print(f"❌ Training dataset not found: {train_dir}")
        return False
    if not val_dir.exists():
        print(f"❌ Validation dataset not found: {val_dir}")
        return False
    
    # Check base EdgeFace model
    edgeface_base = base_dir.parent / "facial_rec" / "edgeface_weights" / "edgeface_xxs.pt"
    if not edgeface_base.exists():
        print(f"❌ Base EdgeFace model not found: {edgeface_base}")
        return False
    
    print("✓ All prerequisites met\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Cyclic training for Hybrid DSR and EdgeFace models"
    )
    
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[16, 24, 32],
        choices=[16, 24, 32],
        help="VLR resolutions to train (default: 16 24 32)"
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip phase 1 (hybrid DSR with base EdgeFace)"
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip phase 2 (EdgeFace finetuning)"
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip phase 3 (hybrid DSR with finetuned EdgeFace)"
    )
    parser.add_argument(
        "--phase1-resume",
        action="store_true",
        help="Resume phase 1 training from checkpoints"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of complete cycles to run (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Verify prerequisites
    if not verify_prerequisites():
        print("\n❌ Prerequisites not met. Please ensure datasets and base EdgeFace model exist.")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"CYCLIC HYBRID DSR TRAINING")
    print(f"{'='*80}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Cycles: {args.cycles}")
    print(f"Skip phase 1: {args.skip_phase1}")
    print(f"Skip phase 2: {args.skip_phase2}")
    print(f"Skip phase 3: {args.skip_phase3}")
    print(f"{'='*80}\n")
    
    for cycle in range(args.cycles):
        print(f"\n{'*'*80}")
        print(f"* CYCLE {cycle + 1}/{args.cycles}")
        print(f"{'*'*80}\n")
        
        # Determine EdgeFace model to use
        if cycle == 0:
            edgeface_model = "edgeface_xxs.pt"
        else:
            # Use finetuned models from previous cycle
            edgeface_model = None  # Will use resolution-specific finetuned models
        
        # Phase 1: Train hybrid DSR with base or finetuned EdgeFace
        if not args.skip_phase1:
            if cycle == 0:
                success = train_hybrid_dsr_phase(
                    args.resolutions,
                    edgeface_model,
                    resume=args.phase1_resume
                )
            else:
                # Use finetuned EdgeFace from previous cycle
                for vlr_size in args.resolutions:
                    finetuned_edgeface = f"edgeface_finetuned_{vlr_size}.pth"
                    success = train_hybrid_dsr_phase(
                        [vlr_size],
                        finetuned_edgeface,
                        resume=False
                    )
                    if not success:
                        break
            
            if not success:
                print(f"\n❌ Cycle {cycle + 1} failed during Phase 1")
                sys.exit(1)
        
        # Phase 2: Finetune EdgeFace using hybrid DSR outputs
        if not args.skip_phase2:
            success = finetune_edgeface_phase(args.resolutions)
            if not success:
                print(f"\n❌ Cycle {cycle + 1} failed during Phase 2")
                sys.exit(1)
        
        # Phase 3: Retrain hybrid DSR with finetuned EdgeFace
        if not args.skip_phase3:
            success = retrain_hybrid_dsr_phase(args.resolutions)
            if not success:
                print(f"\n❌ Cycle {cycle + 1} failed during Phase 3")
                sys.exit(1)
        
        print(f"\n{'*'*80}")
        print(f"* CYCLE {cycle + 1}/{args.cycles} COMPLETED")
        print(f"{'*'*80}\n")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"ALL CYCLES COMPLETED SUCCESSFULLY")
    print(f"{'='*80}\n")
    
    base_dir = Path(__file__).resolve().parent
    print("Final models:")
    for vlr_size in args.resolutions:
        dsr_path = base_dir / f"hybrid_dsr{vlr_size}.pth"
        edgeface_path = base_dir.parent / "facial_rec" / "edgeface_weights" / f"edgeface_finetuned_{vlr_size}.pth"
        
        if dsr_path.exists():
            size_mb = dsr_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Hybrid DSR {vlr_size}×{vlr_size}: {dsr_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ Hybrid DSR {vlr_size}×{vlr_size}: NOT FOUND")
        
        if edgeface_path.exists():
            size_mb = edgeface_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ EdgeFace {vlr_size}×{vlr_size}: {edgeface_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ EdgeFace {vlr_size}×{vlr_size}: NOT FOUND")
    
    print()


if __name__ == "__main__":
    main()
