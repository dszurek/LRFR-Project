"""Sequential training script for all three Hybrid DSR models.

Trains models for 16x16, 24x24, and 32x32 VLR inputs sequentially
using the full train_processed dataset (not frontal_only).
"""

import argparse
import sys
from pathlib import Path

# Import needed for unpickling finetuned EdgeFace checkpoints
try:
    from ..facial_rec.finetune_edgeface import FinetuneConfig
except ImportError:
    from technical.facial_rec.finetune_edgeface import FinetuneConfig

from .train_hybrid_dsr import HybridTrainConfig, train


class Args:
    """Mock args namespace for sequential training."""
    
    def __init__(
        self,
        vlr_size: int,
        edgeface: str,
        frontal_only: bool,
        resume: bool,
        seed: int,
        num_workers: int
    ):
        self.vlr_size = vlr_size
        self.edgeface = edgeface
        self.frontal_only = frontal_only
        self.resume = resume
        self.seed = seed
        self.num_workers = num_workers


def train_all_models(
    resume: bool = False,
    seed: int = 42,
    num_workers: int = 4,
    skip_completed: bool = True
):
    """Train all three DSR models sequentially.
    
    Args:
        resume: Resume training from existing checkpoints
        seed: Random seed for reproducibility
        num_workers: Number of data loader workers
        skip_completed: Skip models that already have checkpoints (unless resume=True)
    """
    
    base_dir = Path(__file__).resolve().parent
    
    # Model configurations: (vlr_size, edgeface_model)
    models = [
        (16, "edgeface_finetuned_16.pth"),
        (24, "edgeface_finetuned_24.pth"),
        (32, "edgeface_finetuned_32.pth"),
    ]
    
    print("=" * 80)
    print("SEQUENTIAL HYBRID DSR TRAINING")
    print("=" * 80)
    print(f"Training all models on FULL train_processed dataset")
    print(f"Models to train: {len(models)}")
    print(f"  - 16x16 -> 112x112")
    print(f"  - 24x24 -> 112x112")
    print(f"  - 32x32 -> 112x112")
    print(f"Resume mode: {resume}")
    print(f"Random seed: {seed}")
    print("=" * 80)
    print()
    
    # Train each model
    for i, (vlr_size, edgeface_model) in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(models)}: {vlr_size}x{vlr_size} -> 112x112")
        print(f"{'='*80}\n")
        
        # Check if model already exists
        save_path = base_dir / f"hybrid_dsr{vlr_size}.pth"
        if save_path.exists() and skip_completed and not resume:
            print(f"⏭️  Checkpoint already exists at {save_path}")
            print(f"   Skipping (use --resume to continue training or --no-skip to retrain)")
            continue
        
        # Create configuration
        config = HybridTrainConfig.for_resolution(vlr_size)
        
        # Create args for this model (FULL dataset, not frontal_only)
        args = Args(
            vlr_size=vlr_size,
            edgeface=edgeface_model,
            frontal_only=False,  # Use full dataset
            resume=resume,
            seed=seed,
            num_workers=num_workers
        )
        
        try:
            # Train this model
            train(config, args)
            print(f"\n✓ Completed training for {vlr_size}x{vlr_size} model")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user for {vlr_size}x{vlr_size} model")
            response = input("Continue to next model? (y/n): ")
            if response.lower() != 'y':
                print("Stopping all training.")
                sys.exit(0)
            print("Continuing to next model...")
            
        except Exception as e:
            print(f"\n❌ Error training {vlr_size}x{vlr_size} model: {e}")
            response = input("Continue to next model? (y/n): ")
            if response.lower() != 'y':
                print("Stopping all training.")
                sys.exit(1)
            print("Continuing to next model...")
    
    print("\n" + "=" * 80)
    print("ALL MODELS TRAINING COMPLETE")
    print("=" * 80)
    print("\nTrained models:")
    for vlr_size, _ in models:
        save_path = base_dir / f"hybrid_dsr{vlr_size}.pth"
        if save_path.exists():
            size_mb = save_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ hybrid_dsr{vlr_size}.pth ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ hybrid_dsr{vlr_size}.pth (not found)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train all three Hybrid DSR models sequentially on full dataset"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoints"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Retrain models even if checkpoints exist (unless --resume is used)"
    )
    
    args = parser.parse_args()
    
    # Train all models
    train_all_models(
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
        skip_completed=not args.no_skip
    )


if __name__ == "__main__":
    main()
