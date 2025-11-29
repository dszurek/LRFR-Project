
import torch
import torch.nn as nn
from technical.dsr.hybrid_model import create_hybrid_dsr

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vlr_size = 16
    model = create_hybrid_dsr(vlr_size).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, vlr_size, vlr_size).to(device)
    
    print("Running forward pass...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
