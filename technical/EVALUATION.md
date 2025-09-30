# Dataset evaluation workflow

This guide explains how to benchmark the super-resolution (DSR) + EdgeFace
recognition pipeline on the preprocessed HR/VLR evaluation splits.

## Prerequisites

- Install project dependencies with Poetry. On Windows, the provided local
  Torch wheels are used automatically; on Linux/ARM, replace them with
  platform-appropriate builds before running `poetry install`.

  ```powershell
  poetry install --with dev
  ```

- Ensure the dataset has been processed with `technical/dataset/preprocess.py`.
  The evaluation script expects the following structure:

  ```text
  <dataset_root>/
    hr_images/
      123_...
    vlr_images/
      123_...
  ```

## Running the evaluation

Invoke the CLI from the project root. By default it targets the
`technical/dataset/test_processed` split and runs fully on CPU.

```powershell
poetry run python -m technical.pipeline.evaluate_dataset `
  --dataset-root technical/dataset/test_processed `
  --device cpu
```

The script performs the following steps:

1. Builds an identity gallery by averaging embeddings from all HR images that
   share the same subject prefix.
2. Runs every VLR probe through the DSR super-resolution model followed by the
   EdgeFace recognizer.
3. Reports aggregate accuracy, per-subject statistics (top 10 by probe count),
   and a short list of misidentifications.

### Useful flags

- `--threshold <float>` – override the recognition threshold used by the
  pipeline when accepting matches.
- `--gallery-via-dsr` – run the DSR model when registering the HR gallery
  images (disabled by default because HR images already match the recognizer
  resolution).
- `--limit <N>` – evaluate only the first _N_ VLR probes (handy for quick smoke
  tests).
- `--dump-results results.csv` – save a CSV with per-image predictions for
  offline analysis or visualisation.

## Output interpretation

- **Aggregate metrics** – overall accuracy and the proportion of probes marked
  as "unknown" because they fell below the recognition threshold.
- **Per-subject accuracy** – top ten identities by probe count, showing how
  consistent the recognizer is for more frequent subjects.
- **Sample misidentifications** – a quick sanity check to inspect the most
  confident mistakes.

When the `--dump-results` option is supplied, the resulting CSV contains four
columns: `filename`, `truth`, `prediction`, and `score` (the cosine similarity
reported by the identity database). This file can be ingested by pandas, Excel,
or any downstream visualisation tool for deeper analysis.
