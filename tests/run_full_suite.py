import argparse
import torch
from pathlib import Path
from tabulate import tabulate
import pandas as pd

from tests.benchmark_dsr import benchmark
from tests.evaluate_dataset import run_evaluation, parse_args as parse_eval_args

def run_suite(device="cuda"):
    print("=" * 80)
    print("RUNNING FULL DSR + EDGEFACE TEST SUITE")
    print("=" * 80)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    base_dir = Path(__file__).resolve().parents[1]
    dataset_root = base_dir / "technical" / "dataset" / "test_processed"
    dsr_dir = base_dir / "technical" / "dsr"
    edgeface_dir = base_dir / "technical" / "facial_rec" / "edgeface_weights"
    
    # Define configurations
    resolutions = [16, 24, 32]
    
    results = []

    # 1. Evaluate DSR Models (PSNR, SSIM, Entropy)
    print("\n" + "-" * 40)
    print("PHASE 1: DSR Model Benchmarking")
    print("-" * 40)
    
    dsr_metrics = {}
    
    for res in resolutions:
        print(f"\nBenchmarking DSR {res}x{res}...")
        model_path = dsr_dir / f"hybrid_dsr{res}.pth"
        
        # Mock args for benchmark
        args = argparse.Namespace(
            vlr_size=res,
            model_path=str(model_path),
            dataset_path=str(dataset_root),
            batch_size=16,
            device=str(device)
        )
            
        try:
            psnr, ssim, entropy = benchmark(args)
            dsr_metrics[res] = {"PSNR": psnr, "SSIM": ssim, "Entropy": entropy}
        except Exception as e:
            print(f"Error benchmarking DSR {res}: {e}")
            dsr_metrics[res] = {"PSNR": 0.0, "SSIM": 0.0, "Entropy": 0.0}

    # 2. Evaluate Recognition Accuracy
    print("\n" + "-" * 40)
    print("PHASE 2: Recognition Accuracy Evaluation")
    print("-" * 40)

    # Helper to run evaluation
    def evaluate(res, dsr_model, edgeface_model, description):
        print(f"\nEvaluating: {description}")
        
        # Construct args for evaluate_dataset
        # Note: evaluate_dataset expects a list of strings for parse_args, 
        # but we can construct the namespace directly or pass a list to parse_args
        
        cmd_args = [
            "--dataset-root", str(dataset_root),
            "--device", str(device),
            "--vlr-size", str(res),
        ]
        
        if dsr_model:
            cmd_args.extend(["--dsr-weights", str(dsr_model)])
        else:
            cmd_args.append("--skip-dsr")
            
        if edgeface_model:
            cmd_args.extend(["--edgeface-weights", str(edgeface_model)])
            
        args = parse_eval_args(cmd_args)
        
        try:
            acc = run_evaluation(args)
            return acc
        except Exception as e:
            print(f"Error evaluating {description}: {e}")
            return 0.0

    # Default EdgeFace Weights
    default_edgeface = edgeface_dir / "edgeface_xxs.pt"
    
    # A. Ground Truth (HR)
    print("\n--- Ground Truth Evaluation ---")
    # For GT, we skip DSR. We can use any VLR size arg as it just sets up paths, 
    # but we need to ensure we are testing HR images against HR gallery.
    # evaluate_dataset builds gallery from HR. 
    # If we skip DSR, it feeds VLR images directly to EdgeFace.
    # To test GT, we ideally want HR probes. 
    # BUT evaluate_dataset is designed for VLR probes.
    # However, if we use --skip-dsr, it uses the VLR images.
    # To test "Ground Truth", we should theoretically use HR images as probes.
    # The current script doesn't explicitly support HR-as-probe easily without modification,
    # OR we can point --vlr-dir to the HR directory!
    
    print("Evaluating Ground Truth (HR Probes) + Default EdgeFace...")
    # Hack: Point vlr-dir to hr_images to simulate GT testing
    gt_args = [
        "--dataset-root", str(dataset_root),
        "--device", str(device),
        "--vlr-size", "32", # Dummy
        "--vlr-dir", str(dataset_root / "hr_images"),
        "--edgeface-weights", str(default_edgeface),
        "--skip-dsr"
    ]
    gt_acc = run_evaluation(parse_eval_args(gt_args))
    
    results.append({
        "Input": "Ground Truth (HR)",
        "DSR Model": "N/A",
        "EdgeFace Model": "Default (XXS)",
        "PSNR": "Inf",
        "SSIM": "1.0000",
        "Entropy": "N/A",
        "Accuracy": f"{gt_acc:.2%}"
    })

    # B. DSR + Default EdgeFace
    print("\n--- DSR + Default EdgeFace ---")
    for res in resolutions:
        dsr_path = dsr_dir / f"hybrid_dsr{res}.pth"
        acc = evaluate(res, dsr_path, default_edgeface, f"DSR {res}x{res} + Default EdgeFace")
        
        metrics = dsr_metrics.get(res, {"PSNR": 0, "SSIM": 0, "Entropy": 0})
        results.append({
            "Input": f"{res}x{res}",
            "DSR Model": "HybridDSR",
            "EdgeFace Model": "Default (XXS)",
            "PSNR": f"{metrics['PSNR']:.2f}",
            "SSIM": f"{metrics['SSIM']:.4f}",
            "Entropy": f"{metrics['Entropy']:.4f}",
            "Accuracy": f"{acc:.2%}"
        })

    # C. DSR + Finetuned EdgeFace
    print("\n--- DSR + Finetuned EdgeFace ---")
    for res in resolutions:
        dsr_path = dsr_dir / f"hybrid_dsr{res}.pth"
        finetuned_path = edgeface_dir / f"edgeface_finetuned_{res}.pth"
        
        if not finetuned_path.exists():
            print(f"Warning: Finetuned model for {res}x{res} not found at {finetuned_path}. Skipping.")
            continue
            
        acc = evaluate(res, dsr_path, finetuned_path, f"DSR {res}x{res} + Finetuned EdgeFace")
        
        metrics = dsr_metrics.get(res, {"PSNR": 0, "SSIM": 0, "Entropy": 0})
        results.append({
            "Input": f"{res}x{res}",
            "DSR Model": "HybridDSR",
            "EdgeFace Model": f"Finetuned ({res}x{res})",
            "PSNR": f"{metrics['PSNR']:.2f}",
            "SSIM": f"{metrics['SSIM']:.4f}",
            "Entropy": f"{metrics['Entropy']:.4f}",
            "Accuracy": f"{acc:.2%}"
        })

    # 3. Report Generation
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK REPORT")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print(tabulate(df, headers="keys", tablefmt="grid"))
    
    # Save to CSV
    report_path = base_dir / "benchmark_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full DSR + EdgeFace test suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    run_suite(args.device)
