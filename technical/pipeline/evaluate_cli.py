"""Command-line interface for multi-resolution evaluation.

This bypasses the GUI and allows direct command-line evaluation.

Usage:
    python -m technical.pipeline.evaluate_cli
    python -m technical.pipeline.evaluate_cli --test-root technical/dataset/test_processed
    python -m technical.pipeline.evaluate_cli --test-root technical/dataset/frontal_only/test --resolutions 16 24 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the evaluator class
from .evaluate_gui import MultiResolutionEvaluator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive multi-resolution evaluation (CLI mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all resolutions on test_processed dataset
  python -m technical.pipeline.evaluate_cli
  
  # Evaluate specific resolutions on frontal_only dataset
  python -m technical.pipeline.evaluate_cli --test-root technical/dataset/frontal_only/test --resolutions 16 32
  
  # Include gallery for identification metrics
  python -m technical.pipeline.evaluate_cli --gallery-root technical/dataset/train_processed
  
  # Save results to custom directory
  python -m technical.pipeline.evaluate_cli --output-dir my_results
"""
    )
    
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("technical/dataset/test_processed"),
        help="Test dataset root directory (default: technical/dataset/test_processed)"
    )
    
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=None,
        help="Gallery dataset root for identification metrics (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--resolutions",
        nargs='+',
        type=int,
        default=[16, 24, 32],
        choices=[16, 24, 32],
        help="VLR resolutions to evaluate (default: 16 24 32)"
    )
    
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of test samples (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Validate test root exists
    if not args.test_root.exists():
        print(f"❌ Error: Test root not found: {args.test_root}")
        print(f"\nAvailable dataset directories:")
        dataset_dir = Path("technical/dataset")
        if dataset_dir.exists():
            for subdir in dataset_dir.iterdir():
                if subdir.is_dir():
                    print(f"  - {subdir}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"Low-Resolution Face Recognition - Comprehensive Evaluation")
    print(f"{'='*70}")
    print(f"Test Root: {args.test_root}")
    print(f"Gallery Root: {args.gallery_root or 'None (verification only)'}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Device: {args.device}")
    if args.num_samples:
        print(f"Sample Limit: {args.num_samples}")
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create evaluator
    evaluator = MultiResolutionEvaluator(device=args.device)
    
    # Evaluate each resolution
    for vlr_size in args.resolutions:
        try:
            metrics = evaluator.evaluate_resolution(
                vlr_size=vlr_size,
                test_root=args.test_root,
                gallery_root=args.gallery_root,
                num_samples=args.num_samples,
            )
            
            print(f"\n✅ {vlr_size}×{vlr_size} Evaluation Results:")
            print(f"\n   Image Quality:")
            print(f"   - PSNR: {metrics.psnr_mean:.2f} ± {metrics.psnr_std:.2f} dB")
            print(f"   - SSIM: {metrics.ssim_mean:.4f} ± {metrics.ssim_std:.4f}")
            print(f"   - Feature Preservation: {metrics.feature_preservation_mean:.4f} ± {metrics.feature_preservation_std:.4f}")
            print(f"\n   Verification Metrics (1:1 Matching):")
            print(f"   - EER: {metrics.eer:.4f} @ threshold {metrics.eer_threshold:.4f}")
            print(f"   - d-prime: {metrics.d_prime:.4f} (separability)")
            print(f"   - ROC AUC: {metrics.roc_auc:.4f}")
            print(f"   - TAR @ FAR=0.01%: {metrics.tar_at_far_0001:.4f}")
            print(f"   - TAR @ FAR=0.1%: {metrics.tar_at_far_001:.4f}")
            print(f"   - TAR @ FAR=1.0%: {metrics.tar_at_far_01:.4f}")
            print(f"   - TAR @ FAR=10%: {metrics.tar_at_far_10:.4f}")
            if args.gallery_root:
                print(f"\n   Identification Metrics:")
                print(f"   - 1:1   Rank-1: {metrics.rank1_accuracy_1v1:.2%}")
                print(f"   - 1:10  Rank-1: {metrics.rank1_accuracy_1v10:.2%}")
                print(f"   - 1:100 Rank-1: {metrics.rank1_accuracy_1v100:.2%}")
                print(f"   - 1:N   Rank-1: {metrics.rank1_accuracy_1vN:.2%}")
                print(f"   - 1:N   Rank-5: {metrics.rank5_accuracy_1vN:.2%}")
                print(f"   - 1:N   Rank-10: {metrics.rank10_accuracy_1vN:.2%}")
                print(f"   - 1:N   Rank-20: {metrics.rank20_accuracy_1vN:.2%}")
            
        except Exception as e:
            print(f"\n❌ Error evaluating {vlr_size}×{vlr_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparative plots
    if evaluator.results:
        print(f"\n{'='*70}")
        print("📊 Generating comparative plots and reports...")
        print(f"{'='*70}")
        
        try:
            evaluator.generate_comparative_plots(args.output_dir)
            evaluator.export_results(args.output_dir / "results.json")
            
            print(f"\n✅ Evaluation complete!")
            print(f"📁 Results saved to: {args.output_dir.absolute()}")
            print(f"\nGenerated files:")
            print(f"  - results.json (raw metrics)")
            print(f"  - quality_metrics.png (PSNR/SSIM/Identity comparison)")
            print(f"  - verification_metrics.png (ROC curves)")
            print(f"  - identification_metrics.png (Rank-N accuracy)")
            print(f"  - score_distributions.png (genuine/impostor distributions)")
            print(f"  - summary_table.png (comprehensive metrics table)")
            print(f"  - comprehensive_report.pdf (all plots in one PDF)")
            
        except Exception as e:
            print(f"\n❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ No results to plot! Check that:")
        print("  1. Test dataset exists and has the correct structure")
        print("  2. VLR images exist in vlr_images_16x16/, vlr_images_24x24/, vlr_images_32x32/")
        print("  3. HR images exist in hr_images/")
        print("  4. DSR and EdgeFace checkpoints are available")


if __name__ == "__main__":
    main()
