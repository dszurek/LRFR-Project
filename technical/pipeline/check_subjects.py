"""Check subject ordering in frontal_only/test."""

from pathlib import Path
from collections import defaultdict


def check_subjects():
    hr_dir = Path("technical/dataset/frontal_only/test/hr_images")

    # Count images per subject
    subjects = defaultdict(int)
    for p in hr_dir.glob("*.png"):
        subject = p.stem.split("_")[0]
        subjects[subject] += 1

    # Sort subjects
    sorted_subjects = sorted(subjects.keys())

    print(f"Total subjects: {len(sorted_subjects)}")
    print(f"First 20 subjects: {sorted_subjects[:20]}")
    print(f"\nLooking for subject 007...")

    if "007" in sorted_subjects:
        position = sorted_subjects.index("007")
        print(f"  Subject 007 is at position: {position}")
    else:
        print(f"  Subject 007 NOT FOUND")

    print(f"\nLooking for subject 2142...")
    if len(sorted_subjects) > 2142:
        print(f"  Subject at position 2142: {sorted_subjects[2142]}")
    else:
        print(f"  Position 2142 is out of range (only {len(sorted_subjects)} subjects)")

    print(f"\nLooking for subject '2142' directly...")
    if "2142" in sorted_subjects:
        position = sorted_subjects.index("2142")
        print(f"  Subject 2142 is at position: {position}")
        print(f"  Number of images: {subjects['2142']}")
    else:
        print(f"  Subject 2142 NOT FOUND in this dataset")


if __name__ == "__main__":
    check_subjects()
