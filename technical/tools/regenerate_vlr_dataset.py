"""Regenerate VLR dataset for one or more target resolutions.

This script manages VLR directories with a consistent naming scheme:
- ALL resolutions use format: vlr_images_{W}x{H} (e.g., vlr_images_32x32, vlr_images_24x24)
- Automatically detects and renames legacy "vlr_images" folder to vlr_images_32x32
- Creates new resolution folders without touching existing ones
- Prompts for confirmation before overwriting existing resolution folders

Features:
- Auto-detection of image resolution for misnamed folders
- Clean y/n prompts for overwriting existing data
- Non-destructive to other resolutions
- Preserves dataset structure and filenames
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


def parse_size_token(token: str) -> Tuple[int, int]:
    """Parse a CLI size token into a (width, height) pair."""

    cleaned = token.lower().replace("×", "x")
    if "x" in cleaned:
        width_str, height_str = cleaned.split("x", maxsplit=1)
    else:
        width_str = height_str = cleaned

    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise ValueError(f"Invalid VLR size token '{token}'") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"VLR dimensions must be positive: {token}")

    return (width, height)


def format_size(vlr_size: Tuple[int, int]) -> str:
    """Return a WxH representation for logs and folder names."""

    return f"{vlr_size[0]}x{vlr_size[1]}"


def resolve_vlr_dir(split_root: Path, vlr_size: Tuple[int, int]) -> Path:
    """Determine the target VLR directory within a split.
    
    Always uses format: vlr_images_{W}x{H}
    No special case for 32x32 anymore.
    """
    return split_root / f"vlr_images_{format_size(vlr_size)}"


def detect_image_size(image_dir: Path) -> Optional[Tuple[int, int]]:
    """Detect resolution of images in a directory by reading the first image."""
    try:
        first_image = next(image_dir.glob("*.png"), None)
        if not first_image:
            return None
        with Image.open(first_image) as img:
            return img.size
    except Exception:
        return None


def check_and_rename_legacy_folders(split_root: Path) -> None:
    """Check for legacy 'vlr_images' folder and rename to vlr_images_32x32.
    
    Also checks any vlr_images_* folders and renames them to match actual resolution.
    """
    # Check for legacy "vlr_images" (assumed to be 32x32)
    legacy_dir = split_root / "vlr_images"
    if legacy_dir.exists() and legacy_dir.is_dir():
        detected_size = detect_image_size(legacy_dir)
        if detected_size:
            target_name = f"vlr_images_{detected_size[0]}x{detected_size[1]}"
            target_dir = split_root / target_name
            
            if not target_dir.exists():
                print(f"🔄 Renaming legacy 'vlr_images' → '{target_name}' (detected: {detected_size[0]}×{detected_size[1]})")
                legacy_dir.rename(target_dir)
            else:
                print(f"⚠️  Legacy 'vlr_images' exists but '{target_name}' already present. Skipping rename.")
    
    # Check all vlr_images_* folders and ensure names match content
    for vlr_dir in split_root.glob("vlr_images_*"):
        if not vlr_dir.is_dir():
            continue
            
        # Extract expected size from folder name
        try:
            name_parts = vlr_dir.name.replace("vlr_images_", "")
            if "x" in name_parts and not name_parts.startswith("old"):
                expected_w, expected_h = map(int, name_parts.split("x"))
                expected_size = (expected_w, expected_h)
                
                # Detect actual size
                detected_size = detect_image_size(vlr_dir)
                if detected_size and detected_size != expected_size:
                    correct_name = f"vlr_images_{detected_size[0]}x{detected_size[1]}"
                    correct_dir = split_root / correct_name
                    
                    if not correct_dir.exists():
                        print(f"🔄 Renaming '{vlr_dir.name}' → '{correct_name}' (detected: {detected_size[0]}×{detected_size[1]})")
                        vlr_dir.rename(correct_dir)
                    else:
                        print(f"⚠️  '{vlr_dir.name}' has wrong size but '{correct_name}' already exists. Manual intervention needed.")
        except (ValueError, IndexError):
            # Skip folders with unexpected naming patterns
            continue


def prompt_overwrite(vlr_dir: Path, vlr_size: Tuple[int, int]) -> bool:
    """Prompt user for confirmation to overwrite existing VLR directory."""
    print(f"\n⚠️  Directory already exists: {vlr_dir}")
    
    # Count images
    image_count = len(list(vlr_dir.glob("*.png")))
    print(f"   Contains {image_count} images at {format_size(vlr_size)} resolution")
    
    response = input(f"   Overwrite and regenerate? (y/n): ").strip().lower()
    return response in ("y", "yes")


def downsample_image(hr_path: Path, vlr_size: Tuple[int, int]) -> Image.Image:
    """Downsample an HR image to the requested VLR resolution."""

    with Image.open(hr_path) as img:
        vlr_img = img.convert("RGB").resize(vlr_size, Image.Resampling.LANCZOS)
    return vlr_img


def regenerate_vlr_split(
    dataset_root: Path,
    split: str,
    vlr_size: Tuple[int, int],
    force: bool,
) -> Optional[Dict[str, int]]:
    """Regenerate VLR images for a specific dataset split.
    
    Returns None if user declined overwrite, otherwise returns stats dict.
    """

    split_root = dataset_root / split
    hr_dir = split_root / "hr_images"
    vlr_dir = resolve_vlr_dir(split_root, vlr_size)

    if not hr_dir.exists():
        print(f"❌ HR directory not found: {hr_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    # Check for legacy folders and rename them
    check_and_rename_legacy_folders(split_root)

    # Check if VLR directory exists
    if vlr_dir.exists() and any(vlr_dir.glob("*.png")):
        if not force:
            if not prompt_overwrite(vlr_dir, vlr_size):
                print(f"   ⏭️  Skipping {split} split")
                return None
        
        # Remove existing directory
        print(f"   🗑️  Removing existing directory...")
        shutil.rmtree(vlr_dir)
    
    vlr_dir.mkdir(parents=True, exist_ok=True)

    hr_images = sorted(hr_dir.glob("*.png"))
    if not hr_images:
        print(f"⚠️  No HR images found in {hr_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    stats: Dict[str, int] = {"processed": 0, "skipped": 0, "errors": 0}

    print(f"\n🔄 Processing {split} split ({len(hr_images)} images)")
    print(f"   Target VLR size: {format_size(vlr_size)}")
    print(f"   Output directory: {vlr_dir.relative_to(dataset_root)}")

    for hr_path in tqdm(hr_images, desc=f"  {split}", unit="img"):
        vlr_path = vlr_dir / hr_path.name

        try:
            vlr_img = downsample_image(hr_path, vlr_size)
            vlr_img.save(vlr_path, "PNG")
            stats["processed"] += 1
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"\n❌ Error processing {hr_path.name}: {exc}")
            stats["errors"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate VLR dataset for one or more resolutions"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("technical/dataset"),
        help="Root directory containing train_processed, val_processed, test_processed",
    )
    parser.add_argument(
        "--vlr-sizes",
        nargs="+",
        default=["32"],
        help=(
            "One or more target VLR sizes. Provide integers for square sizes "
            "(e.g., 32 24 16) or explicit WxH tokens (e.g., 32x32 24x24)."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Dataset splits to regenerate (default: auto-detect from dataset-root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts and overwrite existing directories",
    )

    args = parser.parse_args()

    try:
        vlr_sizes: List[Tuple[int, int]] = [
            parse_size_token(token) for token in args.vlr_sizes
        ]
    except ValueError as exc:  # pragma: no cover - argparse guards
        parser.error(str(exc))

    # Auto-detect splits if not provided
    if args.splits is None:
        dataset_root = Path(args.dataset_root)
        potential_splits = [
            "train_processed", "val_processed", "test_processed",  # Standard format
            "train", "val", "test"  # Frontal-only format
        ]
        args.splits = [
            split for split in potential_splits
            if (dataset_root / split).exists()
        ]
        
        if not args.splits:
            print(f"❌ Error: No valid splits found in {dataset_root}")
            print(f"   Checked: {', '.join(potential_splits)}")
            return

    assumed_hr_size = 112  # Current super-resolution target

    print("=" * 60)
    print("VLR Dataset Regeneration")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Splits: {', '.join(args.splits)}")
    print(
        "Target VLR resolutions: " + ", ".join(format_size(size) for size in vlr_sizes)
    )
    print(f"Force overwrite: {args.force}")
    print(f"Naming format: vlr_images_{{W}}x{{H}} (consistent for all resolutions)")
    print("=" * 60)

    for vlr_size in vlr_sizes:
        total_stats: Dict[str, int] = {"processed": 0, "skipped": 0, "errors": 0}

        print("\n" + "-" * 60)
        print(f"Processing VLR size {format_size(vlr_size)}")
        print("-" * 60)
        print(
            f"Estimated scale factor to {assumed_hr_size}x{assumed_hr_size}: "
            f"{assumed_hr_size / vlr_size[0]:.2f}x"
        )

        for split in args.splits:
            split_path = args.dataset_root / split
            if not split_path.exists():
                print(f"\n⚠️  Split directory not found: {split_path}")
                continue

            stats = regenerate_vlr_split(
                args.dataset_root,
                split,
                vlr_size,
                force=args.force,
            )

            # None means user declined overwrite
            if stats is None:
                continue

            for key in total_stats:
                total_stats[key] += stats[key]

        print("\nSummary")
        print(f"✅ Processed: {total_stats['processed']} images")
        print(f"⏭️  Skipped:   {total_stats['skipped']} images")
        print(f"❌ Errors:    {total_stats['errors']} images")

    print("\n" + "=" * 60)
    print("✅ All requested VLR regenerations complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train DSR: python -m technical.dsr.train_dsr --vlr-size {N} --device cuda")
    print("  2. Fine-tune EdgeFace: python -m technical.facial_rec.finetune_edgeface --vlr-size {N}")
    print("  3. Evaluate: python -m technical.pipeline.evaluate_gui --resolutions {N}")
    print("\n💡 Tip: All VLR folders now use format vlr_images_{W}x{H} (including 32x32)")


if __name__ == "__main__":
    main()
