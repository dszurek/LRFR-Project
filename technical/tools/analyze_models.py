"""Model Analysis Tool - Display size and parameter counts for all models.

This script scans the project for model files and provides detailed information about:
- File size on disk
- Total parameters
- Trainable parameters
- Model architecture configuration
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from technical.dsr.models import DSRColor, DSRConfig, load_dsr_model
from technical.facial_rec.edgeface_weights.edgeface import EdgeFace


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def analyze_dsr_model(model_path: Path, vlr_size: int) -> Dict:
    """Analyze a DSR model checkpoint."""
    if not model_path.exists():
        return {
            "exists": False,
            "path": str(model_path),
            "vlr_size": vlr_size,
            "file_size_mb": 0.0,
        }
    
    try:
        # Load checkpoint to inspect
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Determine config
        config = None
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config_dict = checkpoint["config"]
            config = DSRConfig(
                base_channels=config_dict.get("base_channels", 120),
                residual_blocks=config_dict.get("residual_blocks", 16),
            )
        else:
            # Use default configs based on VLR size
            if vlr_size == 16:
                config = DSRConfig(base_channels=132, residual_blocks=20)
            elif vlr_size == 24:
                config = DSRConfig(base_channels=126, residual_blocks=18)
            else:
                config = DSRConfig(base_channels=120, residual_blocks=16)
        
        # Load model to count parameters
        model = load_dsr_model(model_path, device="cpu", config=config, strict=False)
        total_params, trainable_params = count_parameters(model)
        
        return {
            "exists": True,
            "path": str(model_path),
            "vlr_size": vlr_size,
            "file_size_mb": get_file_size_mb(model_path),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "base_channels": config.base_channels,
            "residual_blocks": config.residual_blocks,
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(model_path),
            "vlr_size": vlr_size,
            "file_size_mb": get_file_size_mb(model_path),
            "error": str(e),
        }


def analyze_edgeface_model(model_path: Path, vlr_size: int) -> Dict:
    """Analyze an EdgeFace model checkpoint."""
    if not model_path.exists():
        return {
            "exists": False,
            "path": str(model_path),
            "vlr_size": vlr_size,
            "file_size_mb": 0.0,
        }
    
    try:
        # Create model
        model = EdgeFace(embedding_size=512, back="edgeface_xxs")
        
        # Load checkpoint
        from dataclasses import dataclass
        import types
        
        # Handle FinetuneConfig for unpickling
        try:
            from technical.facial_rec.finetune_edgeface import FinetuneConfig
        except ImportError:
            @dataclass
            class FinetuneConfig:
                pass
            
            module_name = "technical.facial_rec.finetune_edgeface"
            fake_mod = types.ModuleType(module_name)
            fake_mod.FinetuneConfig = FinetuneConfig
            sys.modules[module_name] = fake_mod
        
        import __main__
        if not hasattr(__main__, 'FinetuneConfig'):
            __main__.FinetuneConfig = FinetuneConfig
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("backbone_state_dict",
                                       checkpoint.get("model_state_dict",
                                                     checkpoint.get("state_dict", checkpoint)))
        else:
            state_dict = checkpoint
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        
        return {
            "exists": True,
            "path": str(model_path),
            "vlr_size": vlr_size,
            "file_size_mb": get_file_size_mb(model_path),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "backbone": "edgeface_xxs",
            "embedding_size": 512,
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(model_path),
            "vlr_size": vlr_size,
            "file_size_mb": get_file_size_mb(model_path),
            "error": str(e),
        }


def format_params(params: int) -> str:
    """Format parameter count in human-readable form."""
    if params >= 1_000_000:
        return f"{params/1_000_000:.2f}M"
    elif params >= 1_000:
        return f"{params/1_000:.1f}K"
    else:
        return str(params)


def print_model_info(info: Dict, model_type: str):
    """Print formatted model information."""
    if not info["exists"]:
        print(f"  ❌ NOT FOUND: {info['path']}")
        return
    
    if "error" in info:
        print(f"  ⚠️  ERROR loading model:")
        print(f"     Path: {info['path']}")
        print(f"     Size: {info['file_size_mb']:.2f} MB")
        print(f"     Error: {info['error']}")
        return
    
    print(f"  ✅ {model_type} {info['vlr_size']}×{info['vlr_size']}")
    print(f"     Path: {Path(info['path']).name}")
    print(f"     Size: {info['file_size_mb']:.2f} MB")
    print(f"     Total params: {format_params(info['total_params'])} ({info['total_params']:,})")
    print(f"     Trainable: {format_params(info['trainable_params'])} ({info['trainable_params']:,})")
    
    if model_type == "DSR":
        print(f"     Config: {info['base_channels']} channels, {info['residual_blocks']} blocks")
    elif model_type == "EdgeFace":
        print(f"     Backbone: {info['backbone']}, Embedding: {info['embedding_size']}-d")
    
    print()


def main():
    """Analyze all models in the project."""
    print("=" * 70)
    print("MODEL ANALYSIS REPORT")
    print("=" * 70)
    print()
    
    # DSR Models
    print("─" * 70)
    print("DSR MODELS (Super-Resolution)")
    print("─" * 70)
    
    dsr_base = PROJECT_ROOT / "technical" / "dsr"
    dsr_regular = {
        16: dsr_base / "dsr16.pth",
        24: dsr_base / "dsr24.pth",
        32: dsr_base / "dsr32.pth",
    }
    
    dsr_quantized = {
        16: dsr_base / "quantized" / "dsr16_quantized_dynamic.pth",
        24: dsr_base / "quantized" / "dsr24_quantized_dynamic.pth",
        32: dsr_base / "quantized" / "dsr32_quantized_dynamic.pth",
    }
    
    print("\nRegular DSR Models:")
    for vlr_size, path in dsr_regular.items():
        info = analyze_dsr_model(path, vlr_size)
        print_model_info(info, "DSR")
    
    print("\nQuantized DSR Models:")
    for vlr_size, path in dsr_quantized.items():
        info = analyze_dsr_model(path, vlr_size)
        print_model_info(info, "DSR")
    
    # EdgeFace Models
    print("─" * 70)
    print("EDGEFACE MODELS (Face Recognition)")
    print("─" * 70)
    
    edgeface_base = PROJECT_ROOT / "technical" / "facial_rec" / "edgeface_weights"
    edgeface_regular = {
        16: edgeface_base / "edgeface_finetuned_16.pth",
        24: edgeface_base / "edgeface_finetuned_24.pth",
        32: edgeface_base / "edgeface_finetuned_32.pth",
    }
    
    edgeface_quantized = {
        16: edgeface_base / "quantized" / "edgeface_finetuned_16_quantized_dynamic.pth",
        24: edgeface_base / "quantized" / "edgeface_finetuned_24_quantized_dynamic.pth",
        32: edgeface_base / "quantized" / "edgeface_finetuned_32_quantized_dynamic.pth",
    }
    
    print("\nRegular EdgeFace Models:")
    for vlr_size, path in edgeface_regular.items():
        info = analyze_edgeface_model(path, vlr_size)
        print_model_info(info, "EdgeFace")
    
    print("\nQuantized EdgeFace Models:")
    for vlr_size, path in edgeface_quantized.items():
        info = analyze_edgeface_model(path, vlr_size)
        print_model_info(info, "EdgeFace")
    
    # Pretrained EdgeFace Models
    print("─" * 70)
    print("PRETRAINED EDGEFACE MODELS (Not fine-tuned)")
    print("─" * 70)
    print()
    
    pretrained_models = {
        "EdgeFace XXS": edgeface_base / "edgeface_xxs.pt",
        "EdgeFace XXS Quantized": edgeface_base / "edgeface_xxs_q.pt",
        "EdgeFace S (gamma=0.5)": edgeface_base / "edgeface_s_gamma_05.pt",
    }
    
    for name, path in pretrained_models.items():
        if path.exists():
            print(f"  ✅ {name}")
            print(f"     Path: {path.name}")
            print(f"     Size: {get_file_size_mb(path):.2f} MB")
            print()
        else:
            print(f"  ❌ NOT FOUND: {name}")
            print(f"     Path: {path}")
            print()
    
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()
