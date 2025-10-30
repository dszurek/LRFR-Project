import torch

state = torch.load("facial_rec/edgeface_weights/edgeface_xxs.pt", map_location="cpu")

print("Actual dimensions in edgeface_xxs.pt:")
for i in range(4):
    gamma_keys = [
        k
        for k in state.keys()
        if f"stages.{i}.blocks.0.gamma" in k and "gamma_xca" not in k
    ]
    if gamma_keys:
        dim = state[gamma_keys[0]].shape[0]
        print(f"  Stage {i}: {dim} channels")

print("\nDW conv sizes:")
for k in state.keys():
    if "conv_dw.weight" in k and "stages.1.blocks.0" in k:
        print(f"  Stage 1 block 0: {state[k].shape}")
        break

for k in state.keys():
    if "conv_dw.weight" in k and "stages.2.blocks.0" in k:
        print(f"  Stage 2 block 0: {state[k].shape}")
        break

for k in state.keys():
    if "conv_dw.weight" in k and "stages.3.blocks.0" in k:
        print(f"  Stage 3 block 0: {state[k].shape}")
        break

print("\nHead structure:")
head_keys = [k for k in state.keys() if k.startswith("model.head.")]
for k in sorted(head_keys)[:6]:
    print(f"  {k}: {state[k].shape}")
