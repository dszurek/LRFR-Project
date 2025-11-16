"""Test if fine-tuned model is being loaded correctly."""

from pathlib import Path


def check_model_file():
    """Check if the fine-tuned model file exists and is recent."""
    print("=" * 70)
    print("MODEL FILE CHECK")
    print("=" * 70)

    # Check paths
    finetuned_path = Path(
        "technical/facial_rec/edgeface_weights/edgeface_finetuned.pth"
    )
    pretrained_path = Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt")

    for name, path in [("Fine-tuned", finetuned_path), ("Pretrained", pretrained_path)]:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"\n{name}: {path}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Modified: {path.stat().st_mtime}")
        else:
            print(f"\n{name}: {path}")
            print(f"  ⚠️  NOT FOUND")

    # Check which is newer
    if finetuned_path.exists() and pretrained_path.exists():
        if finetuned_path.stat().st_mtime > pretrained_path.stat().st_mtime:
            print(f"\n✓ Fine-tuned model is NEWER than pretrained")
        else:
            print(f"\n⚠️  Fine-tuned model is OLDER than pretrained")


if __name__ == "__main__":
    check_model_file()
