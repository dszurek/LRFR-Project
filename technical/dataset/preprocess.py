# Save this file as technical/dataset/preprocess.py

import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

# --- Configuration ---
SOURCE_HR_DIR = "train/HR"
SOURCE_LR_DIR = "train/LR"
TRAIN_DIR = "train_processed"
VAL_DIR = "val_processed"
TEST_DIR = "test_processed"
HR_IMG_SIZE = (160, 160)
VLR_IMG_SIZE = (14, 16)
TEST_SPLIT_RATIO = 0.10
VAL_SPLIT_RATIO = 0.10
# --- End of Configuration ---


def process_and_copy(file_list, source_dir, dest_dir, target_size, desc="Processing"):
    """Processes a list of image files and copies them to the destination with a progress bar."""
    os.makedirs(dest_dir, exist_ok=True)
    # *** MODIFICATION HERE: tqdm is now wrapped around the file list iterator ***
    for filename in tqdm(file_list, desc=desc):
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        img = cv2.imread(src_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(dest_path, img_resized)


def main():
    """Main function to orchestrate the dataset splitting and processing."""
    print("Starting dataset preprocessing for COLOR images...")

    all_hr_files = os.listdir(SOURCE_HR_DIR)
    subjects = sorted(list(set([f.split("_")[0] for f in all_hr_files])))
    print(f"Found {len(subjects)} unique subjects.")

    train_val_subjects, test_subjects = train_test_split(
        subjects, test_size=TEST_SPLIT_RATIO, random_state=42
    )
    val_split_adjusted = VAL_SPLIT_RATIO / (1 - TEST_SPLIT_RATIO)
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=val_split_adjusted, random_state=42
    )

    print(
        f"Splitting subjects: {len(train_subjects)} train, {len(val_subjects)} validation, {len(test_subjects)} test."
    )

    subject_map = {s: "train" for s in train_subjects}
    subject_map.update({s: "val" for s in val_subjects})
    subject_map.update({s: "test" for s in test_subjects})

    file_sets = {"train": [], "val": [], "test": []}
    for filename in all_hr_files:
        subject_id = filename.split("_")[0]
        if subject_id in subject_map:
            set_name = subject_map[subject_id]
            file_sets[set_name].append(filename)

    print("Processing and copying files...")
    for set_name in ["train", "val", "test"]:
        file_list = file_sets[set_name]
        output_base = globals()[f"{set_name.upper()}_DIR"]

        # Call the function which now contains its own progress bar
        process_and_copy(
            file_list,
            SOURCE_HR_DIR,
            os.path.join(output_base, "hr_images"),
            HR_IMG_SIZE,
            desc=f"Processing {set_name} HR",
        )
        process_and_copy(
            file_list,
            SOURCE_LR_DIR,
            os.path.join(output_base, "vlr_images"),
            VLR_IMG_SIZE,
            desc=f"Processing {set_name} LR",
        )

    print("\n--- Preprocessing Complete! ---")
    print(
        f"Color data has been split and processed into '{TRAIN_DIR}', '{VAL_DIR}', and '{TEST_DIR}' folders."
    )


if __name__ == "__main__":
    main()
