import torch
import sys
from pathlib import Path

def inspect_checkpoint():
    path = Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt")
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    try:
        # Load with weights_only=False to handle potential full model objects or legacy formats
        state_dict = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    print(f"Total keys: {len(state_dict)}")
    
    # Print first 50 keys to see structure
    print("First 50 keys:")
    for i, key in enumerate(sorted(state_dict.keys())):
        if i >= 50: break
        shape = state_dict[key].shape
        print(f"{key}: {shape}")
        
    # Check values of temperature and gamma
    print("\nChecking parameter values:")
    for key in sorted(state_dict.keys()):
        if "temperature" in key:
            val = state_dict[key]
            print(f"{key}: shape={val.shape}, mean={val.mean().item():.4f}, min={val.min().item():.4f}, max={val.max().item():.4f}")
            print(f"Values: {val.flatten().tolist()}")
            break # Just one example
            
    for key in sorted(state_dict.keys()):
        if "gamma" in key and "xca" not in key: # Block gamma
            val = state_dict[key]
            print(f"{key}: shape={val.shape}, mean={val.mean().item():.4f}")
            # Don't print all values, too many
            break

if __name__ == "__main__":
    inspect_checkpoint()
