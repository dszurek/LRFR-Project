"""Regenerate VLR dataset with 32×32 resolution.

This script re-downsamples the HR images to 32×32 VLR resolution,
preserving the dataset structure and filenames. It validates that
HR images exist before processing and reports statistics.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple

from PIL import Image
from tqdm import tqdm


def downsample_image(hr_path: Path, vlr_size: Tuple[int, int]) -> Image.Image:
    """Downsample HR image to VLR resolution using high-quality Lanczos.

    Args:
        hr_path: Path to high-resolution image
        vlr_size: Target (width, height) for VLR image

    Returns:
        Downsampled PIL Image
    """
    img = Image.open(hr_path).convert("RGB")
    # Use LANCZOS for downsampling (best quality antialiasing)
    vlr_img = img.resize(vlr_size, Image.Resampling.LANCZOS)
    return vlr_img


def regenerate_vlr_split(
    dataset_root: Path,
    split: str,
    vlr_size: Tuple[int, int],
    backup: bool = True,
) -> dict:
    """Regenerate VLR images for one dataset split.

    Args:
        dataset_root: Root directory containing dataset splits
        split: Split name (e.g., 'train_processed', 'val_processed')
        vlr_size: Target (width, height) for VLR images
        backup: Whether to backup old VLR images

    Returns:
        Dictionary with processing statistics
    """

    hr_dir = dataset_root / split / "hr_images"
    vlr_dir = dataset_root / split / "vlr_images"

    if not hr_dir.exists():
        print(f"❌ HR directory not found: {hr_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    if not vlr_dir.exists():
        print(f"❌ VLR directory not found: {vlr_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    # Backup old VLR images
    if backup and vlr_dir.exists():
        backup_dir = vlr_dir.parent / f"vlr_images_old_14x16"
        if not backup_dir.exists():
            print(f"📦 Backing up old VLR images to: {backup_dir.name}")
            shutil.copytree(vlr_dir, backup_dir, dirs_exist_ok=False)
        else:
            print(f"⏭️  Backup already exists: {backup_dir.name}")

    # Get all HR images
    hr_images = sorted(hr_dir.glob("*.png"))

    if not hr_images:
        print(f"⚠️  No HR images found in {hr_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    print(f"\n🔄 Processing {split} split ({len(hr_images)} images)")
    print(f"   Target VLR size: {vlr_size[0]}×{vlr_size[1]}")

    for hr_path in tqdm(hr_images, desc=f"  {split}", unit="img"):
        vlr_path = vlr_dir / hr_path.name

        try:
            # Generate new VLR image
            vlr_img = downsample_image(hr_path, vlr_size)

            # Save with same filename
            vlr_img.save(vlr_path, "PNG")
            stats["processed"] += 1

        except Exception as e:
            print(f"\n❌ Error processing {hr_path.name}: {e}")
            stats["errors"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate VLR dataset with 32×32 resolution"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("technical/dataset"),
        help="Root directory containing train_processed, val_processed, test_processed",
    )
    parser.add_argument(
        "--vlr-width",
        type=int,
        default=32,
        help="Target VLR width (default: 32)",
    )
    parser.add_argument(
        "--vlr-height",
        type=int,
        default=32,
        help="Target VLR height (default: 32)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_processed", "val_processed", "test_processed"],
        help="Dataset splits to regenerate (default: all)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up old VLR images",
    )

    args = parser.parse_args()

    vlr_size = (args.vlr_width, args.vlr_height)

    print("=" * 60)
    print("VLR Dataset Regeneration")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Target VLR resolution: {vlr_size[0]}×{vlr_size[1]}")
    print(f"Upscaling factor to 128×128: {128 / vlr_size[0]:.2f}×")
    print(f"Backup old VLR: {not args.no_backup}")
    print(f"Splits: {', '.join(args.splits)}")
    print("=" * 60)

    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    for split in args.splits:
        split_path = args.dataset_root / split
        if not split_path.exists():
            print(f"\n⚠️  Split directory not found: {split_path}")
            continue

        stats = regenerate_vlr_split(
            args.dataset_root,
            split,
            vlr_size,
            backup=not args.no_backup,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Processed: {total_stats['processed']} images")
    print(f"⏭️  Skipped:   {total_stats['skipped']} images")
    print(f"❌ Errors:    {total_stats['errors']} images")
    print("=" * 60)

    if total_stats["processed"] > 0:
        print("\n✨ Dataset regeneration complete!")
        print(f"   New VLR resolution: {vlr_size[0]}×{vlr_size[1]}")
        print(f"\n📋 Next steps:")
        print(f"   1. Retrain DSR model:")
        print(f"      poetry run python -m technical.dsr.train_dsr --device cuda")
        print(f"   2. Re-evaluate pipeline:")
        print(f"      poetry run python -m technical.pipeline.evaluate_dataset \\")
        print(f"        --dataset-root technical/dataset/test_processed \\")
        print(f"        --threshold 0.35 --device cuda")
        print(f"\n💡 Expected accuracy improvement: +15-25% (from ~35% to 50-60%)")


if __name__ == "__main__":
    main()
