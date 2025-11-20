"""Register a new user in the face recognition system.

This script demonstrates how to add a new user to the gallery WITHOUT
retraining the EdgeFace model. Users provide multiple photos, and the
system extracts and averages their embeddings for robust recognition.

Example usage:
    # Register user with multiple photos
    poetry run python -m technical.pipeline.register_user \
        --user-id john_doe \
        --photos photo1.jpg photo2.jpg photo3.jpg \
        --device cuda

    # Register user from webcam (capture 5 photos)
    poetry run python -m technical.pipeline.register_user \
        --user-id jane_smith \
        --webcam \
        --num-captures 5 \
        --device cuda

The registered embeddings are saved to a gallery file that can be loaded
during inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

try:
    from .pipeline import FaceRecognitionPipeline, PipelineConfig
except ImportError:
    from pipeline import FaceRecognitionPipeline, PipelineConfig


def register_user_from_photos(
    pipeline: FaceRecognitionPipeline,
    user_id: str,
    photo_paths: List[Path],
    min_photos: int = 3,
) -> torch.Tensor:
    """Register a user by providing multiple photos.

    Args:
        pipeline: Face recognition pipeline
        user_id: Unique identifier for the user
        photo_paths: List of paths to user photos
        min_photos: Minimum number of photos required

    Returns:
        Mean embedding vector for the user

    Raises:
        ValueError: If too few photos provided
    """
    if len(photo_paths) < min_photos:
        raise ValueError(f"Need at least {min_photos} photos, got {len(photo_paths)}")

    print(f"\n{'='*60}")
    print(f"Registering user: {user_id}")
    print(f"{'='*60}")

    embeddings: List[torch.Tensor] = []

    for i, photo_path in enumerate(photo_paths, 1):
        print(f"[{i}/{len(photo_paths)}] Processing {photo_path.name}...", end=" ")

        try:
            # Load image
            image = Image.open(photo_path).convert("RGB")

            # Run through pipeline: VLR -> DSR -> EdgeFace
            sr_tensor = pipeline.upscale(image)
            embedding = pipeline.infer_embedding(sr_tensor)

            embeddings.append(embedding.cpu())
            print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    if not embeddings:
        raise RuntimeError("Failed to process any photos")

    # Average embeddings for robustness
    stacked = torch.stack(embeddings)
    mean_embedding = torch.mean(stacked, dim=0)

    # Register in gallery
    pipeline.gallery.add(user_id, mean_embedding)

    print(f"\n✓ Successfully registered {user_id} with {len(embeddings)} embeddings")
    print(f"  Mean embedding shape: {mean_embedding.shape}")
    print(f"  Gallery now contains {pipeline.gallery.size} users")

    return mean_embedding


def register_user_from_webcam(
    pipeline: FaceRecognitionPipeline,
    user_id: str,
    num_captures: int = 5,
) -> torch.Tensor:
    """Register a user by capturing photos from webcam.

    Args:
        pipeline: Face recognition pipeline
        user_id: Unique identifier for the user
        num_captures: Number of photos to capture

    Returns:
        Mean embedding vector for the user
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "OpenCV (cv2) is required for webcam capture. "
            "Install with: pip install opencv-python"
        )

    print(f"\n{'='*60}")
    print(f"Registering user: {user_id} (Webcam Mode)")
    print(f"{'='*60}")
    print(f"\nInstructions:")
    print(f"  1. Look at the camera")
    print(f"  2. Press SPACE to capture a photo")
    print(f"  3. Capture {num_captures} photos from different angles")
    print(f"  4. Press ESC to cancel")
    print(f"\nCaptures needed: {num_captures}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    embeddings: List[torch.Tensor] = []
    captures = 0

    try:
        while captures < num_captures:
            ret, frame = cap.read()
            if not ret:
                continue

            # Display preview
            display = frame.copy()
            cv2.putText(
                display,
                f"Captures: {captures}/{num_captures}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display,
                "Press SPACE to capture, ESC to cancel",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Register User", display)

            key = cv2.waitKey(1) & 0xFF

            # SPACE - capture photo
            if key == ord(" "):
                print(f"\n[{captures + 1}/{num_captures}] Capturing...", end=" ")

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Run through pipeline
                try:
                    sr_tensor = pipeline.upscale(pil_image)
                    embedding = pipeline.infer_embedding(sr_tensor)
                    embeddings.append(embedding.cpu())
                    captures += 1
                    print("✓")
                except Exception as e:
                    print(f"✗ Error: {e}")

            # ESC - cancel
            elif key == 27:
                print("\n\n✗ Registration cancelled")
                return None

        # Average embeddings
        stacked = torch.stack(embeddings)
        mean_embedding = torch.mean(stacked, dim=0)

        # Register in gallery
        pipeline.gallery.add(user_id, mean_embedding)

        print(
            f"\n✓ Successfully registered {user_id} with {len(embeddings)} embeddings"
        )
        print(f"  Gallery now contains {pipeline.gallery.size} users")

        return mean_embedding

    finally:
        cap.release()
        cv2.destroyAllWindows()


def save_gallery(
    pipeline: FaceRecognitionPipeline,
    output_path: Path,
) -> None:
    """Save gallery embeddings to a file.

    Args:
        pipeline: Face recognition pipeline
        output_path: Path to save gallery (e.g., 'gallery.pt')
    """
    gallery_data = {
        "labels": pipeline.gallery._labels,
        "embeddings": [emb for emb in pipeline.gallery._embeddings],
        "threshold": pipeline.gallery.threshold,
        "normalize_embeddings": pipeline.gallery.normalize_embeddings,
    }

    torch.save(gallery_data, output_path)
    print(f"\n✓ Gallery saved to {output_path}")
    print(f"  Total users: {len(gallery_data['labels'])}")


def load_gallery(
    pipeline: FaceRecognitionPipeline,
    gallery_path: Path,
) -> None:
    """Load gallery embeddings from a file.

    Args:
        pipeline: Face recognition pipeline
        gallery_path: Path to saved gallery file
    """
    gallery_data = torch.load(gallery_path)

    pipeline.gallery._labels = gallery_data["labels"]
    pipeline.gallery._embeddings = gallery_data["embeddings"]
    pipeline.gallery.threshold = gallery_data.get(
        "threshold", pipeline.gallery.threshold
    )
    pipeline.gallery.normalize_embeddings = gallery_data.get(
        "normalize_embeddings", pipeline.gallery.normalize_embeddings
    )

    print(f"✓ Loaded gallery from {gallery_path}")
    print(f"  Total users: {len(pipeline.gallery._labels)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a new user in the face recognition system"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Unique identifier for the user (e.g., 'john_doe')",
    )
    parser.add_argument(
        "--photos",
        nargs="+",
        type=Path,
        help="Paths to user photos (minimum 3 recommended)",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Capture photos from webcam instead of file paths",
    )
    parser.add_argument(
        "--num-captures",
        type=int,
        default=5,
        help="Number of photos to capture from webcam (default: 5)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=Path("gallery.pt"),
        help="Path to save/load gallery embeddings (default: gallery.pt)",
    )
    parser.add_argument(
        "--edgeface-weights",
        type=Path,
        default=None,
        help="Path to EdgeFace model weights (default: uses config default)",
    )
    parser.add_argument(
        "--dsr-weights",
        type=Path,
        default=None,
        help="Path to DSR model weights (default: uses config default)",
    )

    args = parser.parse_args()

    # Validate input mode
    if not args.webcam and not args.photos:
        parser.error("Must specify either --photos or --webcam")

    if args.webcam and args.photos:
        parser.error("Cannot use both --photos and --webcam")

    # Build pipeline
    print("Loading models...")
    config = PipelineConfig(device=args.device)

    if args.edgeface_weights:
        config.edgeface_weights_path = args.edgeface_weights
    if args.dsr_weights:
        config.dsr_weights_path = args.dsr_weights

    from pipeline import build_pipeline

    pipeline = build_pipeline(config)

    # Load existing gallery if available
    if args.gallery_path.exists():
        load_gallery(pipeline, args.gallery_path)

    # Register user
    if args.webcam:
        embedding = register_user_from_webcam(pipeline, args.user_id, args.num_captures)
    else:
        embedding = register_user_from_photos(pipeline, args.user_id, args.photos)

    if embedding is not None:
        # Save updated gallery
        save_gallery(pipeline, args.gallery_path)

        print(f"\n{'='*60}")
        print("Registration Complete!")
        print(f"{'='*60}")
        print(f"\nUser '{args.user_id}' can now be recognized by the system.")
        print(f"Gallery saved to: {args.gallery_path}")
        print(f"\nTo use this gallery in your pipeline:")
        print(f"  1. Load gallery: load_gallery(pipeline, '{args.gallery_path}')")
        print(f"  2. Run inference: pipeline.run(image)")


if __name__ == "__main__":
    main()
