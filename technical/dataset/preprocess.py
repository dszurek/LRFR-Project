# Save this code as preprocess_data.py

import cv2
import os
import glob
from mtcnn import MTCNN
from tqdm import tqdm

# --- Configuration ---
RAW_IMAGE_DIR = "raw_dataset"
OUTPUT_DIR = "processed_dataset"
HR_IMG_SIZE = (160, 160)
VLR_IMG_SIZE = (14, 16)
DOWNSAMPLE_INTERPOLATION = cv2.INTER_CUBIC
# --- End of Configuration ---


def create_directories():
    """Creates the necessary output directories if they don't exist."""
    global hr_dir, vlr_dir
    hr_dir = os.path.join(OUTPUT_DIR, "hr_images")
    vlr_dir = os.path.join(OUTPUT_DIR, "vlr_images")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(vlr_dir, exist_ok=True)
    print(f"✅ Output directories created at: {os.path.abspath(OUTPUT_DIR)}")


def process_images(detector):
    """Finds all images, detects faces, and creates HR/VLR pairs."""
    image_paths = glob.glob(os.path.join(RAW_IMAGE_DIR, "**", "*.*"), recursive=True)
    image_paths = [
        path
        for path in image_paths
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".ppm"))
    ]

    if not image_paths:
        print(f"❌ Error: No images found in '{RAW_IMAGE_DIR}'. Please check the path.")
        return

    print(f"Found {len(image_paths)} images to process. Starting...")

    skipped_count = 0
    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                skipped_count += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)

            if not results:
                skipped_count += 1
                continue

            x1, y1, width, height = results[0]["box"]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = img[y1:y2, x1:x2]

            hr_face = cv2.resize(face, HR_IMG_SIZE, interpolation=cv2.INTER_AREA)
            hr_face_gray = cv2.cvtColor(hr_face, cv2.COLOR_BGR2GRAY)
            vlr_face_gray = cv2.resize(
                hr_face_gray, VLR_IMG_SIZE, interpolation=DOWNSAMPLE_INTERPOLATION
            )

            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            parent_dir_name = os.path.basename(os.path.dirname(img_path))
            save_filename = f"{parent_dir_name}_{base_filename}.png"

            cv2.imwrite(os.path.join(hr_dir, save_filename), hr_face_gray)
            cv2.imwrite(os.path.join(vlr_dir, save_filename), vlr_face_gray)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped_count += 1

    print("\n--- Processing Complete! ---")
    print(f"✅ Successfully processed {len(image_paths) - skipped_count} images.")
    print(f"⚠️ Skipped {skipped_count} images (no face detected or error).")


if __name__ == "__main__":
    print("Initializing MTCNN face detector...")
    detector = MTCNN(min_face_size=20)
    create_directories()
    process_images(detector)
