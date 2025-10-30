"""Quick test to verify train/val datasets use same subject_to_id mapping."""

from pathlib import Path
import torch
from technical.dsr import load_dsr_model
from technical.facial_rec.finetune_edgeface import DSROutputDataset

device = torch.device("cpu")
base_dir = Path(__file__).resolve().parents[2]

# Load DSR
dsr_weights = base_dir / "technical" / "dsr" / "dsr.pth"
dsr_model = load_dsr_model(dsr_weights, device=device)
dsr_model.eval()

# Build datasets
train_dir = base_dir / "technical" / "dataset" / "frontal_only" / "train"
val_dir = base_dir / "technical" / "dataset" / "frontal_only" / "val"

print("Building training dataset...")
train_dataset = DSROutputDataset(train_dir, dsr_model, device, augment=False)

print("\nBuilding validation dataset (with train mapping)...")
val_dataset = DSROutputDataset(
    val_dir, dsr_model, device, augment=False, subject_to_id=train_dataset.subject_to_id
)

print(f"\n✓ Train subjects: {len(train_dataset.subject_to_id)}")
print(f"✓ Val subjects (using train mapping): {len(val_dataset.subject_to_id)}")
print(f"✓ Val samples: {len(val_dataset)} (may be less if some subjects not in train)")

# Show sample mapping
print("\nSample train subject_to_id:")
for i, (subj, id_) in enumerate(list(train_dataset.subject_to_id.items())[:5]):
    print(f"  '{subj}' → ID {id_}")

print("\nChecking if val uses same mapping...")
sample_subjects = list(train_dataset.subject_to_id.keys())[:5]
for subj in sample_subjects:
    train_id = train_dataset.subject_to_id.get(subj)
    val_id = val_dataset.subject_to_id.get(subj)
    match = "✓" if train_id == val_id else "✗"
    print(f"  {match} Subject '{subj}': Train ID {train_id} | Val ID {val_id}")

print("\n✓ Label mapping test complete!")
