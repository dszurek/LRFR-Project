# LRFR Project

End-to-end research prototype for low-resolution facial recognition. The
repository now contains a reusable pipeline that stitches together the custom
Deep Super Resolution (DSR) model and the ultra-light EdgeFace `xxs_q` weights
for identity inference on resource-constrained hardware, such as a Raspberry Pi
5 (4 GB RAM) running Ubuntu.

## 📦 Project layout

- `technical/dsr/` – DSR training scripts and reusable model definitions.
- `technical/facial_rec/` – EdgeFace backbone utilities and weights.
- `technical/pipeline/` – Production-focused inference pipeline.
- `tests/` – Lightweight unit checks for core utilities.

## 🧠 How the DSR model works

At a glance, the Deep Super Resolution (DSR) network turns a blurry,
very-low-resolution face crop into a sharper 128×128 RGB image. It is a compact
convolutional model (`DSRColor`) built from residual blocks, so each layer learns
to refine the previous approximation instead of starting from scratch. During
training we feed the model paired examples of matching low-resolution (VLR) and
high-resolution (HR) faces. The script at `technical/dsr/train_dsr.py` augments
each pair (random flips, slight rotations, mild colour tweaks) to make the model
robust to real-world capture noise.

The optimisation objective mixes four ideas:

1. **Pixel fidelity (L1 loss):** keeps the generated image close to the HR target.
2. **Perceptual similarity (VGG19 features):** compares intermediate features of
   a pretrained classification network so textures look natural, not just accurate
   per pixel.
3. **Identity consistency (EdgeFace embeddings):** runs both the SR output and
   the ground-truth HR image through the lightweight EdgeFace recogniser and
   maximises the cosine similarity of their embeddings. This keeps the person’s
   identity intact.
4. **Total variation regularisation:** discourages noisy checkerboard artefacts.

Training uses AdamW with cosine learning-rate decay, mixed precision (AMP), and
an Exponential Moving Average (EMA) copy of the weights for stable validation.
Each epoch ends with PSNR and loss reporting, and the EMA weights are saved to
`technical/dsr/dsr.pth` whenever we beat the previous best validation score.

## 🔄 How the full pipeline is stitched together

The pipeline in `technical/pipeline/pipeline.py` orchestrates the end-to-end
recognition flow:

1. **Upscale:** load the trained DSR weights and run them on the incoming VLR
   probe image to create a sharper version.
2. **Embed:** feed the super-resolved image through EdgeFace to obtain a compact
   512-dimensional embedding vector.
3. **Compare:** compute cosine similarity between the probe embedding and every
   registered gallery embedding. The gallery entries can be the raw HR images or
   the same DSR output, depending on configuration.
4. **Decide:** pick the gallery identity with the highest similarity and emit it
   when the score crosses the configured threshold; otherwise report "unknown".

`PipelineConfig` lets you pick the device (CPU/GPU), thread count, thresholds,
and the file paths to the stored DSR/EdgeFace weights. The helper methods
`register_identity()` and `run()` wrap the common tasks so a CLI or service can
add new identities and query the recogniser with just a few lines of code.

## 🚀 Pipeline quick-start

The pipeline is designed to run fully on CPU. Install dependencies with Poetry
(recommended) or pip.

```powershell
poetry install --with dev
```

On Raspberry Pi, replace the local Windows-only Torch wheels in
`pyproject.toml` with the appropriate ARM builds or install torch separately
before running `poetry install`.

### Minimal example

```python
from technical.pipeline import PipelineConfig, build_pipeline

config = PipelineConfig(
	dsr_weights_path="technical/dsr/dsr.pth",
	edgeface_weights_path="technical/facial_rec/edgeface_weights/edgeface_xxs_q.pt",
	device="cpu",
	num_threads=2,  # tune for Raspberry Pi
)

pipeline = build_pipeline(config)
pipeline.register_identity("Alice", "path/to/alice_low_res.png")

result = pipeline.run("path/to/low_res_input.png")
print(result["identity"], result["score"])
result["sr_image"].save("upscaled.png")
```

### Integrating with a CLI capture loop

The `FaceRecognitionPipeline` exposes `run()` and `register_identity()` helper
methods so that a future CLI can:

1. Capture a frame from a USB camera.
2. Pass the frame directly (NumPy array or PIL image) to `run()`.
3. Display the returned identity score and store the upscaled image if desired.

See `technical/pipeline/pipeline.py` for more hooks that can be imported into a
CLI entry point.

## 📊 Dataset evaluation

Once the dataset has been preprocessed into `*_processed/hr_images` and
`*_processed/vlr_images` folders, you can benchmark the full pipeline end to end
with the bundled CLI:

```powershell
poetry run python -m technical.pipeline.evaluate_dataset `
	--dataset-root technical/dataset/test_processed
```

Key options:

- `--device`: select the inference device (`cpu`, `cuda`, etc.).
- `--threshold`: override the recognition threshold defined in
  `PipelineConfig`.
- `--gallery-via-dsr`: run the DSR model over the HR gallery when registering
  identities (default is to embed HR images directly).
- `--dump-results`: save a CSV with per-image predictions for further analysis.

See `technical/EVALUATION.md` for a more detailed walkthrough.

## 🧪 Tests

Execute unit tests locally (requires PyTorch):

```powershell
poetry run pytest
```

`tests/test_identity_database.py` is skipped automatically when torch is not
installed, keeping CI lean.

## 🛠 Raspberry Pi deployment notes

- Install official CPU wheels for PyTorch/torchvision (`pip install torch torchvision`)
  before running `poetry install` so Poetry can reuse the existing packages.
- Start the pipeline with `PipelineConfig(num_threads=2, force_full_precision=True)`
  to keep memory pressure predictable on the Pi 5.
- For USB camera capture in a future CLI, prefer `opencv-python` (`cv2.VideoCapture`)
  or `libcamera` bindings and pass the captured frame (as NumPy) directly to
  `pipeline.run(frame)`.
