"""Quick check of subject ID overlap between train and val."""

from pathlib import Path
from collections import Counter


def main():
    train_hr = Path("technical/dataset/edgeface_finetune/train/hr_images")
    val_vlr = Path("technical/dataset/edgeface_finetune/val/vlr_images")

    print("=" * 70)
    print("SUBJECT ID OVERLAP CHECK")
    print("=" * 70)

    # Extract subjects
    train_subjects = set()
    for p in train_hr.glob("*.png"):
        subject = p.stem.split("_")[0]
        train_subjects.add(subject)

    val_subjects = set()
    val_subject_counts = Counter()
    for p in val_vlr.glob("*.png"):
        subject = p.stem.split("_")[0]
        val_subjects.add(subject)
        val_subject_counts[subject] += 1

    overlap = train_subjects & val_subjects
    missing = val_subjects - train_subjects

    print(f"\nTrain subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Overlap: {len(overlap)} ({len(overlap)/len(val_subjects)*100:.1f}%)")

    if missing:
        print(f"\n⚠️  WARNING: {len(missing)} val subjects NOT in training:")
        print(f"Missing: {sorted(list(missing))[:20]}")
    else:
        print(f"\n✓ All validation subjects are in training set!")

    # Show some examples
    print("\n" + "=" * 70)
    print("EXAMPLE FILENAMES")
    print("=" * 70)

    print("\nTraining HR (first 5):")
    for p in sorted(train_hr.glob("*.png"))[:5]:
        subject = p.stem.split("_")[0]
        print(f"  {p.name} → Subject: '{subject}'")

    print("\nValidation VLR (first 5):")
    for p in sorted(val_vlr.glob("*.png"))[:5]:
        subject = p.stem.split("_")[0]
        print(f"  {p.name} → Subject: '{subject}'")

    # Show subject with most validation images
    print("\n" + "=" * 70)
    print("VALIDATION SUBJECT DISTRIBUTION")
    print("=" * 70)

    top_5 = val_subject_counts.most_common(5)
    print("\nTop 5 subjects by validation image count:")
    for subject, count in top_5:
        in_train = "✓" if subject in train_subjects else "✗"
        print(f"  {in_train} Subject {subject}: {count} images")

    bottom_5 = val_subject_counts.most_common()[-5:]
    print("\nBottom 5 subjects by validation image count:")
    for subject, count in bottom_5:
        in_train = "✓" if subject in train_subjects else "✗"
        print(f"  {in_train} Subject {subject}: {count} images")


if __name__ == "__main__":
    main()
