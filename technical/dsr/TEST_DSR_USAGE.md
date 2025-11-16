# DSR Test Script - Multi-Resolution Support

## Overview

The `test_dsr.py` script now supports testing all three DSR model types:

- **16×16** DSR model
- **24×24** DSR model
- **32×32** DSR model (default)

## Features

### 1. Resolution Selection

- **GUI Dropdown**: Select resolution dynamically from the GUI
- **Command-Line Argument**: Specify initial resolution with `--vlr-size`
- **Auto-Loading**: Model automatically loads when resolution changes

### 2. Model Checkpoint Paths

The script looks for models at:

```
technical/dsr/dsr_16x16.pth  # 16×16 model
technical/dsr/dsr_24x24.pth  # 24×24 model
technical/dsr/dsr.pth        # 32×32 model (default)
```

### 3. Dataset Directory Detection

For each resolution, the script searches for VLR datasets in:

```
technical/dataset/frontal_only/test/vlr_images_{size}x{size}/
technical/dataset/test_processed/vlr_images_{size}x{size}/
technical/dataset/frontal_only/test/vlr_images/  # Legacy 32×32 fallback
technical/dataset/test_processed/vlr_images/     # Legacy 32×32 fallback
```

## Usage

### Launch with Default Resolution (32×32)

```bash
poetry run python -m technical.dsr.test_dsr
```

### Launch with 16×16 Model

```bash
poetry run python -m technical.dsr.test_dsr --vlr-size 16
```

### Launch with 24×24 Model

```bash
poetry run python -m technical.dsr.test_dsr --vlr-size 24
```

### Help

```bash
poetry run python -m technical.dsr.test_dsr --help
```

## GUI Usage

1. **Select Resolution**: Use the dropdown to choose 16×16, 24×24, or 32×32
2. **Test Random Database Image**: Process a random image from the test dataset
3. **Upload Your Image**: Process your own image with face detection

## Testing Workflow

### Test All Three Models

```bash
# Test 16×16 model
poetry run python -m technical.dsr.test_dsr --vlr-size 16

# Test 24×24 model
poetry run python -m technical.dsr.test_dsr --vlr-size 24

# Test 32×32 model
poetry run python -m technical.dsr.test_dsr --vlr-size 32
```

### Switch Models in GUI

1. Launch the application
2. Use the resolution dropdown to switch between 16, 24, and 32
3. Model will automatically reload
4. Test images with the new resolution

## Model Output

The script displays:

- **VLR Input**: Original low-resolution input ({size}×{size})
- **DSR Output**: Super-resolved output (112×112)
- **Original HR**: Ground truth high-resolution image (112×112)

All images are displayed side-by-side for comparison.

## Troubleshooting

### "Model not found for {size}×{size}"

**Solution**: Train the DSR model for that resolution:

```bash
poetry run python -m technical.dsr.train_dsr --vlr-size {size} --device cuda --frontal-only
```

### "No VLR directory found for {size}×{size}"

**Solution**: Generate the VLR dataset:

```bash
poetry run python -m technical.tools.regenerate_vlr_dataset --vlr-sizes {size} --dataset-root technical/dataset/frontal_only
```

### Model loads but shows wrong resolution

**Check**: Verify the checkpoint was trained with the correct resolution and saved at the expected path.

## Implementation Details

### Key Changes

1. **Dynamic Model Paths**: `MODEL_PATHS` dictionary maps sizes to checkpoint paths
2. **Resolution Parameter**: All functions now accept `vlr_size` parameter
3. **Auto-Detection**: `get_candidate_lr_dirs()` generates resolution-specific paths
4. **GUI Dropdown**: `ttk.Combobox` for resolution selection
5. **Command-Line Args**: `argparse` for initial resolution selection

### Function Signatures

```python
def robust_load_checkpoint(path: Path, device: torch.device, vlr_size: int)
def process_user_image(model, img_path: str, vlr_size: int)
def process_random_database(model, vlr_size: int)
def find_lr_dir(vlr_size: int) -> Optional[Path]
def get_candidate_lr_dirs(vlr_size: int) -> list[Path]
```

## Example Output

```
Loading 24×24 model from dsr_24x24.pth...
[load] matched keys: 142, mismatched/ignored: 0
[load] Using saved target HR size: 112×112
[load] Model configured for VLR input size: 24×24
Model loaded successfully.
DSR output size: 112×112
Displaying result. Press any key to close.
```
