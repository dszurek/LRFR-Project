"""Check if EdgeFace weights are actually loading."""

from pathlib import Path
from technical.pipeline.pipeline import build_pipeline, PipelineConfig
import torch

config = PipelineConfig(
    device="cuda",
    edgeface_weights_path=Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt"),
)
pipeline = build_pipeline(config)

print(f"Model type: {type(pipeline.recognition_model)}")

if hasattr(pipeline.recognition_model, "model"):
    stem_weight = pipeline.recognition_model.model.stem[0].weight.data
elif hasattr(pipeline.recognition_model, "stem"):
    stem_weight = pipeline.recognition_model.stem[0].weight.data
else:
    print("ERROR: Cannot find stem!")
    exit(1)

print(f"Stem weight mean: {stem_weight.mean():.20f}")
print(f"Stem weight shape: {stem_weight.shape}")

# Load the pretrained and check its mean
state = torch.load(
    "technical/facial_rec/edgeface_weights/edgeface_xxs.pt",
    map_location="cpu",
    weights_only=False,
)
if "model.stem.0.weight" in state:
    pretrained_mean = state["model.stem.0.weight"].mean()
    print(f"Pretrained stem weight mean: {pretrained_mean:.20f}")
    print(
        f'Weights match: {torch.allclose(stem_weight.cpu(), state["model.stem.0.weight"])}'
    )
