"""Create a dedicated dataset for EdgeFace fine-tuning.

This script reorganizes the frontal_only dataset to have the same subjects
in both train and val splits, with different images per subject.

WHY CREATE A SEPARATE DATASET?
- Current frontal_only dataset: Different people in train/val/test (good for DSR)
- EdgeFace fine-tuning needs: Same people in train/val (good for recognition)
- Solution: Create edgeface_finetune/ with proper train/val split

STRATEGY:
- For each subject, split their images 80/20 into train/val
- Ensures classification accuracy is meaningful during validation
- Maintains embedding similarity as a metric for generalization
- Filters subjects with < 5 images (not enough for proper splitting)

USAGE:
    poetry run python -m technical.facial_rec.create_finetuning_dataset

OUTPUT:
    technical/dataset/edgeface_finetune/
    ├── train/
    │   ├── vlr_images/   # 32x32 low-res
    │   └── hr_images/    # 112x112 high-res
    └── val/
        ├── vlr_images/
        └── hr_images/
"""

from pathlib import Path
import shutil
from collections import defaultdict
import random
from tqdm import tqdm


def extract_subject_id(filename: str) -> str:
    """Extract subject ID from filename.

    Formats:
    - CMU: 001_01_01_041_00_crop_128.png -> subject '001'
    - LFW: 1234_5678_lfw.png -> subject '1234'
    """
    stem = filename.replace(".png", "")
    if "_lfw" in stem:
        return stem.split("_")[0]
    else:
        return stem.split("_")[0]


def main():
    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    source_train = base_dir / "technical" / "dataset" / "frontal_only" / "train"
    source_val = base_dir / "technical" / "dataset" / "frontal_only" / "val"

    output_dir = base_dir / "technical" / "dataset" / "edgeface_finetune"
    output_train = output_dir / "train"
    output_val = output_dir / "val"

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / split / "vlr_images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "hr_images").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Creating EdgeFace Fine-tuning Dataset")
    print("=" * 70)
    print(f"Source train: {source_train}")
    print(f"Source val:   {source_val}")
    print(f"Output:       {output_dir}")
    print()

    # Collect all images by subject
    print("Step 1: Collecting images by subject...")
    subject_to_files = defaultdict(list)

    for source_dir in [source_train, source_val]:
        vlr_dir = source_dir / "vlr_images"
        hr_dir = source_dir / "hr_images"

        if not vlr_dir.exists():
            print(f"Warning: {vlr_dir} does not exist, skipping...")
            continue

        for vlr_path in tqdm(
            list(vlr_dir.glob("*.png")), desc=f"Scanning {source_dir.name}"
        ):
            hr_path = hr_dir / vlr_path.name
            if not hr_path.exists():
                continue

            subject_id = extract_subject_id(vlr_path.name)
            subject_to_files[subject_id].append((vlr_path, hr_path))

    print(f"Found {len(subject_to_files)} unique subjects")
    print(f"Total images: {sum(len(files) for files in subject_to_files.values())}")
    print()

    # Filter subjects with enough images
    min_images_per_subject = 5
    filtered_subjects = {
        subject: files
        for subject, files in subject_to_files.items()
        if len(files) >= min_images_per_subject
    }

    print(
        f"Step 2: Filtering subjects with at least {min_images_per_subject} images..."
    )
    print(f"Kept {len(filtered_subjects)} subjects")
    print(f"Removed {len(subject_to_files) - len(filtered_subjects)} subjects")
    print()

    # Split each subject's images 80/20
    print("Step 3: Splitting images 80/20 train/val...")
    random.seed(42)

    train_count = 0
    val_count = 0

    for subject, files in tqdm(filtered_subjects.items(), desc="Splitting subjects"):
        # Shuffle files for this subject
        files = list(files)
        random.shuffle(files)

        # 80/20 split
        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # Copy train files
        for vlr_src, hr_src in train_files:
            vlr_dst = output_train / "vlr_images" / vlr_src.name
            hr_dst = output_train / "hr_images" / hr_src.name
            shutil.copy2(vlr_src, vlr_dst)
            shutil.copy2(hr_src, hr_dst)
            train_count += 1

        # Copy val files
        for vlr_src, hr_src in val_files:
            vlr_dst = output_val / "vlr_images" / vlr_src.name
            hr_dst = output_val / "hr_images" / hr_src.name
            shutil.copy2(vlr_src, vlr_dst)
            shutil.copy2(hr_src, hr_dst)
            val_count += 1

    print()
    print("=" * 70)
    print("✅ Dataset Creation Complete!")
    print("=" * 70)
    print(f"Subjects:        {len(filtered_subjects)}")
    print(f"Train images:    {train_count}")
    print(f"Val images:      {val_count}")
    print(
        f"Split ratio:     {train_count/(train_count+val_count):.1%} train / {val_count/(train_count+val_count):.1%} val"
    )
    print()
    print(f"📁 Output directory: {output_dir}")
    print()
    print("🚀 To use this dataset for fine-tuning:")
    print()
    print("  poetry run python -m technical.facial_rec.finetune_edgeface \\")
    print(f'    --train-dir "{output_train}" \\')
    print(f'    --val-dir "{output_val}" \\')
    print("    --device cuda \\")
    print("    --edgeface edgeface_xxs.pt")
    print()
    print("📊 Expected improvements with this dataset:")
    print("  ✅ Classification accuracy will be meaningful (not 0%)")
    print("  ✅ Embedding similarity will track DSR quality")
    print("  ✅ Can monitor per-subject performance")
    print("  ✅ Better overfitting detection")
    print()


if __name__ == "__main__":
    main()
