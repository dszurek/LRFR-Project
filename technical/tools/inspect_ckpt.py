import sys
from pathlib import Path
import torch

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dsr/dsr.pth")
print("Inspecting", path)
ckpt = torch.load(path, map_location="cpu")
print("top type:", type(ckpt))
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print("top keys count", len(keys))
    print("top keys (first 40):", keys[:40])
    # prefer explicit names
    for candidate in ("state_dict", "model_state_dict", "state_dict_ema", "model"):
        if candidate in ckpt:
            sd = ckpt[candidate]
            print(
                f"found candidate '{candidate}' of type {type(sd)} len {len(sd) if hasattr(sd,'__len__') else 'N/A'}"
            )
            if isinstance(sd, dict):
                for i, (k, v) in enumerate(sd.items()):
                    if i < 80:
                        print(k, getattr(v, "shape", None))
                    else:
                        break
                sys.exit(0)
    # fallback: list tensor items at top-level
    tensor_items = [
        (k, getattr(v, "shape", None)) for k, v in ckpt.items() if hasattr(v, "shape")
    ]
    print("tensor-like top-level items count", len(tensor_items))
    for i, (k, shape) in enumerate(tensor_items[:120]):
        print(k, shape)
else:
    # object - try state_dict
    try:
        sd = ckpt.state_dict()
        print("object.state_dict len", len(sd))
        for i, (k, v) in enumerate(sd.items()):
            if i < 80:
                print(k, getattr(v, "shape", None))
            else:
                break
    except Exception as e:
        print("Could not extract state_dict:", e)
