"""Verify dataset and model setup before fine-tuning and evaluation.

This script checks:
1. Dataset structure is correct
2. Filenames follow expected format
3. Models are accessible
4. Architecture can be detected
"""

from pathlib import Path
import sys


def check_dataset(name: str, root: Path) -> bool:
    """Check if dataset structure is valid."""
    print(f"\n{'='*60}")
    print(f"Checking {name} dataset: {root}")
    print("=" * 60)

    if not root.exists():
        print(f"❌ Dataset root not found: {root}")
        return False

    hr_dir = root / "hr_images"
    vlr_dir = root / "vlr_images"

    # Check directories
    if not hr_dir.exists():
        print(f"❌ HR directory not found: {hr_dir}")
        return False
    if not vlr_dir.exists():
        print(f"❌ VLR directory not found: {vlr_dir}")
        return False

    # Count images
    hr_images = list(hr_dir.glob("*.png"))
    vlr_images = list(vlr_dir.glob("*.png"))

    print(f"✓ HR images: {len(hr_images)}")
    print(f"✓ VLR images: {len(vlr_images)}")

    if len(hr_images) == 0:
        print(f"❌ No HR images found in {hr_dir}")
        return False
    if len(vlr_images) == 0:
        print(f"❌ No VLR images found in {vlr_dir}")
        return False

    # Check filename format
    print("\nChecking filename format...")
    sample_files = hr_images[:5]
    subjects = set()

    for img_path in sample_files:
        filename = img_path.name
        if "_" not in filename:
            print(f"⚠️  Warning: No underscore in filename: {filename}")
            print(f"   Expected format: {{subject}}_{{id}}.png")
        else:
            subject = filename.split("_")[0]
            subjects.add(subject)
            print(f"✓ {filename} → subject: {subject}")

    print(f"\n✓ Found {len(subjects)} subjects in sample")

    # Extract all subjects
    all_subjects = set()
    for img_path in hr_images:
        if "_" in img_path.name:
            subject = img_path.name.split("_")[0]
            all_subjects.add(subject)

    print(f"✓ Total subjects: {len(all_subjects)}")

    return True


def check_model(path: Path) -> bool:
    """Check if model file exists and is valid."""
    print(f"\n{'='*60}")
    print(f"Checking model: {path}")
    print("=" * 60)

    if not path.exists():
        print(f"❌ Model not found: {path}")
        return False

    print(f"✓ Model exists: {path}")
    print(f"  Size: {path.stat().st_size / 1024 / 1024:.1f} MB")

    # Try to detect type
    try:
        import torch

        # Try TorchScript
        try:
            model = torch.jit.load(str(path), map_location="cpu")
            print(f"✓ Type: TorchScript (.pt)")
            return True
        except:
            pass

        # Try state dict
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            print(f"✓ Type: State dict checkpoint (.pth)")

            # Check contents
            if isinstance(checkpoint, dict):
                print(f"  Keys: {list(checkpoint.keys())[:5]}")

                if "backbone_state_dict" in checkpoint:
                    print(f"  ✓ Contains backbone_state_dict (fine-tuned)")
                    state_dict = checkpoint["backbone_state_dict"]
                elif "state_dict" in checkpoint:
                    print(f"  ✓ Contains state_dict")
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # Detect architecture
                sample_keys = list(state_dict.keys())[:10]
                if any("stem" in k or "stages" in k for k in sample_keys):
                    print(f"  ✓ Architecture: ConvNeXt (edgeface_xxs)")
                elif any("features" in k for k in sample_keys):
                    print(f"  ✓ Architecture: LDC (edgeface_s)")
                else:
                    print(f"  ⚠️  Unknown architecture")

                print(f"  Sample keys: {sample_keys[:3]}")

            return True
        except Exception as e:
            # Try without weights_only
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                print(f"✓ Type: State dict checkpoint (.pth, requires pickle)")
                return True
            except:
                pass

        print(f"❌ Could not load model: {e}")
        return False

    except ImportError:
        print(f"⚠️  PyTorch not available, skipping model check")
        return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("DATASET AND MODEL VERIFICATION")
    print("=" * 60)

    all_ok = True

    # Check fine-tuning datasets
    finetune_train = Path("technical/dataset/edgeface_finetune/train")
    finetune_val = Path("technical/dataset/edgeface_finetune/val")

    if finetune_train.exists():
        all_ok &= check_dataset("Fine-tuning (train)", finetune_train)
    else:
        print(f"\n⚠️  Fine-tuning train dataset not found: {finetune_train}")
        print(
            "   Run: poetry run python technical/dataset/create_finetuning_dataset.py"
        )

    if finetune_val.exists():
        all_ok &= check_dataset("Fine-tuning (val)", finetune_val)
    else:
        print(f"\n⚠️  Fine-tuning val dataset not found: {finetune_val}")

    # Check evaluation dataset
    eval_test = Path("technical/dataset/frontal_only/test")
    if eval_test.exists():
        all_ok &= check_dataset("Evaluation (test)", eval_test)
    else:
        print(f"\n❌ Evaluation test dataset not found: {eval_test}")
        all_ok = False

    # Check models
    pretrained_model = Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt")
    if pretrained_model.exists():
        all_ok &= check_model(pretrained_model)
    else:
        print(f"\n❌ Pretrained model not found: {pretrained_model}")
        all_ok = False

    finetuned_model = Path("edgeface_finetuned.pth")
    if finetuned_model.exists():
        all_ok &= check_model(finetuned_model)
    else:
        print(f"\n⚠️  Fine-tuned model not found: {finetuned_model}")
        print("   This is expected if you haven't run fine-tuning yet")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    if all_ok:
        print("✅ All checks passed! You're ready to run:")
        print("\n1. Fine-tuning:")
        print("   poetry run python -m technical.facial_rec.finetune_edgeface \\")
        print('       --train-dir "technical/dataset/edgeface_finetune/train" \\')
        print('       --val-dir "technical/dataset/edgeface_finetune/val" \\')
        print("       --device cuda \\")
        print("       --edgeface edgeface_xxs.pt")
        print("\n2. Evaluation (after fine-tuning):")
        print("   poetry run python -m technical.pipeline.evaluate_dataset \\")
        print("       --dataset-root technical/dataset/frontal_only/test \\")
        print("       --device cuda \\")
        print("       --edgeface-weights edgeface_finetuned.pth \\")
        print("       --threshold 0.35")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
