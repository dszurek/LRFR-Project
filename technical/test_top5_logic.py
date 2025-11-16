"""Test Top-5 accuracy calculation logic."""

import torch

# Simulate a batch
batch_size = 8
num_classes = 518

# Create fake logits and labels
logits = torch.randn(batch_size, num_classes)
labels = torch.randint(0, num_classes, (batch_size,))

print("Testing Top-5 accuracy calculation")
print("=" * 60)
print(f"Batch size: {batch_size}")
print(f"Num classes: {num_classes}")
print(f"\nLabels: {labels}")

# Get top-5 predictions
_, top5_pred = logits.topk(5, dim=1)
print(f"\nTop-5 predictions shape: {top5_pred.shape}")
print(f"Top-5 predictions:\n{top5_pred}")

# Method 1: Using explicit loop (current implementation)
correct_top5_v1 = 0
for i in range(labels.size(0)):
    if labels[i] in top5_pred[i]:
        correct_top5_v1 += 1
        print(
            f"  Sample {i}: label={labels[i].item()}, top5={top5_pred[i].tolist()} -> HIT"
        )
    else:
        print(
            f"  Sample {i}: label={labels[i].item()}, top5={top5_pred[i].tolist()} -> MISS"
        )

print(f"\nMethod 1 (loop): {correct_top5_v1}/{batch_size} correct")

# Method 2: Using vectorized approach
labels_expanded = labels.unsqueeze(1).expand_as(top5_pred)
matches = (top5_pred == labels_expanded).any(dim=1)
correct_top5_v2 = matches.sum().item()

print(f"Method 2 (vectorized): {correct_top5_v2}/{batch_size} correct")

# Verify they match
assert correct_top5_v1 == correct_top5_v2, "Methods don't match!"
print("\n[SUCCESS] Both methods give the same result!")
