import torch

state = torch.load("facial_rec/edgeface_weights/edgeface_xxs.pt", map_location="cpu")

print("Stage 1 XCA block (stages.1.blocks.1) all keys:")
xca_keys = sorted([k for k in state.keys() if "stages.1.blocks.1" in k])
for k in xca_keys:
    print(f"  {k}: {state[k].shape}")
