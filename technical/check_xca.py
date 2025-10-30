import torch

state = torch.load("facial_rec/edgeface_weights/edgeface_xxs.pt", map_location="cpu")

print("XCA positional encoding:")
for k in state.keys():
    if "pos_embd.token_projection.weight" in k:
        print(f"  {k}: {state[k].shape}")

print("\nXCA num_heads (check temperature shape):")
for k in state.keys():
    if "xca.temperature" in k:
        print(f"  {k}: {state[k].shape} -> {state[k].shape[0]} heads")

print("\nXCA conv channels (in XCABlock):")
for k in state.keys():
    if "stages.1.blocks.1.convs.0.weight" in k:
        print(f"  Stage 1 XCA convs[0]: {state[k].shape}")
for k in state.keys():
    if "stages.2.blocks.5.convs.0.weight" in k:
        print(f"  Stage 2 XCA convs[0]: {state[k].shape}")
    if "stages.2.blocks.5.convs.1.weight" in k:
        print(f"  Stage 2 XCA convs[1]: {state[k].shape}")
for k in state.keys():
    if "stages.3.blocks.1.convs.0.weight" in k:
        print(f"  Stage 3 XCA convs[0]: {state[k].shape}")
    if "stages.3.blocks.1.convs.2.weight" in k:
        print(f"  Stage 3 XCA convs[2]: {state[k].shape}")
