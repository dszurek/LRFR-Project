import torch, sys, pathlib

p = pathlib.Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt")
obj = torch.load(p, map_location="cpu")
print(type(obj))
if isinstance(obj, dict):
    print("keys:", list(obj.keys())[:20])
else:
    print(
        "object has methods:",
        [m for m in dir(obj) if m.endswith("state_dict") or m == "state_dict"],
    )
