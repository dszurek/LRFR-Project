"""
Quantize DSR and EdgeFace models for deployment on edge devices (Raspberry Pi, etc.)

Supports:
- PyTorch dynamic quantization (INT8)
- Static quantization (INT8, requires calibration data)
- ONNX export with quantization
- Model size comparison and inference speed benchmarking

Usage:
    # Quantize all DSR models (16, 24, 32) with dynamic quantization
    python -m technical.tools.quantize_models --models dsr --method dynamic
    
    # Quantize EdgeFace models with static quantization
    python -m technical.tools.quantize_models --models edgeface --method static
    
    # Quantize specific model and export to ONNX
    python -m technical.tools.quantize_models --models dsr --vlr-sizes 32 --export-onnx
    
    # Benchmark quantized vs original
    python -m technical.tools.quantize_models --models dsr --benchmark
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technical.dsr.models import DSRColor, DSRConfig
from technical.facial_rec.edgeface_weights.edgeface import EdgeFace
# Import FinetuneConfig to unpickle checkpoints
try:
    from technical.facial_rec.finetune_edgeface import FinetuneConfig
except ImportError:
    FinetuneConfig = None  # For older checkpoints without config


class CalibrationDataset(Dataset):
    """Simple dataset for quantization calibration."""
    
    def __init__(self, num_samples: int = 100, image_size: tuple[int, int] = (32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size
        # Generate random samples for calibration
        self.samples = [
            torch.randn(3, *image_size) for _ in range(num_samples)
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_dsr_model(vlr_size: int, device: torch.device) -> DSRColor:
    """Load DSR model from checkpoint."""
    model_path = ROOT / "technical" / "dsr" / f"dsr{vlr_size}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"DSR model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Infer config from checkpoint
    if isinstance(checkpoint, dict):
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            config = DSRConfig(
                base_channels=config_dict.get("base_channels", 120),
                residual_blocks=config_dict.get("residual_blocks", 16),
            )
        else:
            # Default config for given VLR size
            if vlr_size == 16:
                config = DSRConfig(base_channels=132, residual_blocks=20)
            elif vlr_size == 24:
                config = DSRConfig(base_channels=126, residual_blocks=18)
            else:
                config = DSRConfig(base_channels=120, residual_blocks=16)
        
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        config = DSRConfig()
        state_dict = checkpoint
    
    # Set output size to 112x112
    config.output_size = (112, 112)
    
    model = DSRColor(config=config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def load_edgeface_model(vlr_size: int, device: torch.device) -> nn.Module:
    """Load fine-tuned EdgeFace model."""
    model_path = ROOT / "technical" / "facial_rec" / "edgeface_weights" / f"edgeface_finetuned_{vlr_size}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"EdgeFace model not found: {model_path}")
    
    # Create EdgeFace model with correct backbone
    model = EdgeFace(embedding_size=512, back="edgeface_xxs")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle nested state dict formats
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("backbone_state_dict", 
                                   checkpoint.get("model_state_dict", 
                                                 checkpoint.get("state_dict", checkpoint)))
    else:
        state_dict = checkpoint
    
    # Strip common prefixes (like "model." or "backbone.")
    cleaned_state = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
        cleaned_state[new_key] = value
    
    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()
    
    return model


def dynamic_quantize_model(model: nn.Module, dtype=torch.qint8) -> nn.Module:
    """Apply dynamic quantization to model (weights quantized, activations quantized at runtime)."""
    print(f"  Applying dynamic quantization (dtype={dtype})...")
    quantized_model = quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=dtype
    )
    return quantized_model


def prepare_static_quantization(
    model: nn.Module, 
    calibration_loader: DataLoader,
    device: torch.device
) -> nn.Module:
    """Apply static quantization (requires calibration data)."""
    print("  Preparing static quantization...")
    
    # Set quantization config
    model.qconfig = quantization.get_default_qconfig('x86')  # or 'fbgemm' for x86, 'qnnpack' for ARM
    
    # Prepare model
    quantization.prepare(model, inplace=True)
    
    # Calibrate with sample data
    print("  Calibrating with sample data...")
    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            batch = batch.to(device)
            model(batch)
    
    # Convert to quantized model
    print("  Converting to quantized model...")
    quantization.convert(model, inplace=True)
    
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int],
    opset_version: int = 13
):
    """Export model to ONNX format for cross-platform deployment."""
    print(f"  Exporting to ONNX: {output_path}")
    
    dummy_input = torch.randn(1, 3, *input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"  ✓ ONNX model saved: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def benchmark_model(
    model: nn.Module,
    input_size: tuple[int, int],
    device: torch.device,
    num_runs: int = 100
) -> dict:
    """Benchmark model inference speed."""
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def quantize_dsr_models(
    vlr_sizes: list[int],
    method: Literal['dynamic', 'static'],
    device: torch.device,
    export_onnx: bool = False,
    benchmark: bool = False
):
    """Quantize DSR models for specified VLR sizes."""
    output_dir = ROOT / "technical" / "dsr" / "quantized"
    output_dir.mkdir(exist_ok=True)
    
    for vlr_size in vlr_sizes:
        print(f"\n{'='*70}")
        print(f"Quantizing DSR model: {vlr_size}×{vlr_size}")
        print(f"{'='*70}")
        
        # Load original model
        model = load_dsr_model(vlr_size, device)
        original_size = get_model_size(model)
        print(f"Original model size: {original_size:.2f} MB")
        
        if benchmark:
            print("Benchmarking original model...")
            original_perf = benchmark_model(model, (vlr_size, vlr_size), device)
            print(f"  Mean inference: {original_perf['mean_ms']:.2f} ms (±{original_perf['std_ms']:.2f})")
        
        # Quantize
        if method == 'dynamic':
            quantized_model = dynamic_quantize_model(model)
        else:
            # Static quantization requires calibration
            calib_dataset = CalibrationDataset(num_samples=100, image_size=(vlr_size, vlr_size))
            calib_loader = DataLoader(calib_dataset, batch_size=10, shuffle=False)
            quantized_model = prepare_static_quantization(model, calib_loader, device)
        
        quantized_size = get_model_size(quantized_model)
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        if benchmark:
            print("Benchmarking quantized model...")
            quantized_perf = benchmark_model(quantized_model, (vlr_size, vlr_size), device)
            print(f"  Mean inference: {quantized_perf['mean_ms']:.2f} ms (±{quantized_perf['std_ms']:.2f})")
            speedup = original_perf['mean_ms'] / quantized_perf['mean_ms']
            print(f"  Speedup: {speedup:.2f}x")
        
        # Save quantized model
        output_path = output_dir / f"dsr{vlr_size}_quantized_{method}.pth"
        torch.save(quantized_model.state_dict(), output_path)
        print(f"✓ Saved: {output_path}")
        
        # Export to ONNX if requested
        if export_onnx:
            onnx_path = output_dir / f"dsr{vlr_size}_quantized_{method}.onnx"
            export_to_onnx(quantized_model, onnx_path, (vlr_size, vlr_size))


def quantize_edgeface_models(
    vlr_sizes: list[int],
    method: Literal['dynamic', 'static'],
    device: torch.device,
    export_onnx: bool = False,
    benchmark: bool = False
):
    """Quantize EdgeFace models for specified VLR sizes."""
    output_dir = ROOT / "technical" / "facial_rec" / "edgeface_weights" / "quantized"
    output_dir.mkdir(exist_ok=True)
    
    for vlr_size in vlr_sizes:
        print(f"\n{'='*70}")
        print(f"Quantizing EdgeFace model: {vlr_size}×{vlr_size}")
        print(f"{'='*70}")
        
        # Load original model
        model = load_edgeface_model(vlr_size, device)
        original_size = get_model_size(model)
        print(f"Original model size: {original_size:.2f} MB")
        
        if benchmark:
            print("Benchmarking original model...")
            original_perf = benchmark_model(model, (112, 112), device)
            print(f"  Mean inference: {original_perf['mean_ms']:.2f} ms (±{original_perf['std_ms']:.2f})")
        
        # Quantize
        if method == 'dynamic':
            quantized_model = dynamic_quantize_model(model)
        else:
            # Static quantization requires calibration
            calib_dataset = CalibrationDataset(num_samples=100, image_size=(112, 112))
            calib_loader = DataLoader(calib_dataset, batch_size=10, shuffle=False)
            quantized_model = prepare_static_quantization(model, calib_loader, device)
        
        quantized_size = get_model_size(quantized_model)
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        if benchmark:
            print("Benchmarking quantized model...")
            quantized_perf = benchmark_model(quantized_model, (112, 112), device)
            print(f"  Mean inference: {quantized_perf['mean_ms']:.2f} ms (±{quantized_perf['std_ms']:.2f})")
            speedup = original_perf['mean_ms'] / quantized_perf['mean_ms']
            print(f"  Speedup: {speedup:.2f}x")
        
        # Save quantized model
        output_path = output_dir / f"edgeface_finetuned_{vlr_size}_quantized_{method}.pth"
        torch.save(quantized_model.state_dict(), output_path)
        print(f"✓ Saved: {output_path}")
        
        # Export to ONNX if requested
        if export_onnx:
            onnx_path = output_dir / f"edgeface_finetuned_{vlr_size}_quantized_{method}.onnx"
            export_to_onnx(quantized_model, onnx_path, (112, 112))


def main():
    parser = argparse.ArgumentParser(
        description="Quantize DSR and EdgeFace models for edge deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['dsr', 'edgeface', 'all'],
        default=['all'],
        help="Which models to quantize (default: all)"
    )
    parser.add_argument(
        '--vlr-sizes',
        nargs='+',
        type=int,
        choices=[16, 24, 32],
        default=[16, 24, 32],
        help="VLR sizes to process (default: all)"
    )
    parser.add_argument(
        '--method',
        choices=['dynamic', 'static'],
        default='dynamic',
        help="Quantization method (default: dynamic)"
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help="Device for quantization (default: cpu, recommended for deployment target)"
    )
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        help="Also export quantized models to ONNX format"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help="Benchmark original vs quantized model performance"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    models_to_process = args.models if 'all' not in args.models else ['dsr', 'edgeface']
    
    print(f"\n{'='*70}")
    print(f"Model Quantization Tool")
    print(f"{'='*70}")
    print(f"Models: {models_to_process}")
    print(f"VLR sizes: {args.vlr_sizes}")
    print(f"Method: {args.method}")
    print(f"Device: {device}")
    print(f"Export ONNX: {args.export_onnx}")
    print(f"Benchmark: {args.benchmark}")
    
    if 'dsr' in models_to_process:
        quantize_dsr_models(
            args.vlr_sizes,
            args.method,
            device,
            args.export_onnx,
            args.benchmark
        )
    
    if 'edgeface' in models_to_process:
        quantize_edgeface_models(
            args.vlr_sizes,
            args.method,
            device,
            args.export_onnx,
            args.benchmark
        )
    
    print(f"\n{'='*70}")
    print("✓ Quantization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
