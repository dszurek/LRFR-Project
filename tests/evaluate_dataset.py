"""Command-line evaluation for the DSR + EdgeFace pipeline on prepared datasets.

This utility builds a facial identity gallery from the provided high-resolution
(HR) images and evaluates very-low-resolution (VLR) probes by running them
through the full super-resolution and recognition pipeline.

Example usage (from the project root):

    poetry run python -m technical.pipeline.evaluate_dataset \
        --dataset-root technical/dataset/test_processed \
        --device cpu

The script prints aggregate accuracy statistics and can optionally dump the raw
per-image results to CSV for deeper analysis.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

try:  # torch and torchvision are optional at import time for static analysis
    import torch
    from torchvision import transforms
except ModuleNotFoundError as exc:  # pragma: no cover - executed only when deps missing
    raise RuntimeError(
        "technical.pipeline.evaluate_dataset requires torch and torchvision.\n"
        "Install the wheels listed in pyproject.toml (or platform-specific builds)"
    ) from exc

from technical.pipeline.pipeline import PipelineConfig, build_pipeline


@dataclass
class EvaluationResult:
    """Stores the outcome for a single probe image."""

    filename: str
    truth: str
    prediction: Optional[str]
    score: Optional[float]


def _iter_image_files(directory: Path) -> Iterator[Path]:
    return (path for path in sorted(directory.glob("*.png")) if path.is_file())


def _subject_from_filename(path: Path) -> str:
    return path.stem.split("_")[0]


def _load_hr_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    pil = Image.open(image_path).convert("RGB")
    tensor = transforms.functional.to_tensor(pil)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _group_by_subject(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        grouped[_subject_from_filename(path)].append(path)
    return grouped


def build_gallery(
    pipeline,
    hr_directory: Path,
    use_dsr: bool,
) -> Dict[str, List[Path]]:
    grouped = _group_by_subject(_iter_image_files(hr_directory))
    if not grouped:
        raise RuntimeError(
            f"No HR .png files found under {hr_directory}. Check the dataset path."
        )

    for subject, images in tqdm(
        grouped.items(),
        desc="Registering gallery identities",
        unit="subject",
    ):
        embeddings: List[torch.Tensor] = []
        for image_path in images:
            if use_dsr:
                sr = pipeline.upscale(image_path)
                embedding = pipeline.infer_embedding(sr)
            else:
                tensor = _load_hr_tensor(image_path, pipeline.device)
                embedding = pipeline.infer_embedding(tensor)
            embeddings.append(embedding.cpu())

        stacked = torch.stack(embeddings)
        mean_embedding = torch.mean(stacked, dim=0)
        pipeline.gallery.add(subject, mean_embedding)

    return grouped


def evaluate_vlr(
    pipeline,
    vlr_directory: Path,
    limit: Optional[int] = None,
) -> Tuple[List[EvaluationResult], Counter, Counter, int]:
    paths = list(_iter_image_files(vlr_directory))
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        raise RuntimeError(
            f"No VLR .png files found under {vlr_directory}. Check the dataset path."
        )

    subject_totals: Counter = Counter()
    subject_correct: Counter = Counter()
    unknown_predictions = 0
    results: List[EvaluationResult] = []

    for path in tqdm(paths, desc="Evaluating probes", unit="image"):
        truth = _subject_from_filename(path)
        subject_totals[truth] += 1
        inference = pipeline.run(path)
        prediction = inference["identity"]
        score = inference["score"]
        results.append(
            EvaluationResult(
                filename=path.name, truth=truth, prediction=prediction, score=score
            )
        )

        if prediction is None:
            unknown_predictions += 1
        if prediction == truth:
            subject_correct[truth] += 1

    return results, subject_totals, subject_correct, unknown_predictions


def summarise_results(
    results: List[EvaluationResult],
    subject_totals: Counter,
    subject_correct: Counter,
    unknown_predictions: int,
    gallery_subjects: Optional[set] = None,
) -> None:
    total_probes = len(results)
    correct = sum(1 for r in results if r.prediction == r.truth)
    accuracy = correct / total_probes if total_probes else 0.0

    print("\n=== Aggregate metrics ===")
    print(f"Total probes evaluated : {total_probes}")
    print(f"Correct predictions    : {correct} ({accuracy:.2%})")
    if total_probes:
        print(
            f"Predicted as unknown : {unknown_predictions} "
            f"({unknown_predictions / total_probes:.2%})"
        )

    # Separate known vs unknown subject performance
    if gallery_subjects is not None:
        known_results = [r for r in results if r.truth in gallery_subjects]
        unknown_results = [r for r in results if r.truth not in gallery_subjects]

        if known_results:
            known_correct = sum(1 for r in known_results if r.prediction == r.truth)
            known_accuracy = known_correct / len(known_results)
            print(f"\nKnown subjects (in gallery):")
            print(
                f"  Probes: {len(known_results)}, Correct: {known_correct} ({known_accuracy:.2%})"
            )

        if unknown_results:
            unknown_correct = sum(1 for r in unknown_results if r.prediction == r.truth)
            unknown_accuracy = (
                unknown_correct / len(unknown_results) if unknown_results else 0.0
            )
            unknown_rejected = sum(1 for r in unknown_results if r.prediction is None)
            print(f"\nUnknown subjects (NOT in gallery):")
            print(
                f"  Probes: {len(unknown_results)}, Correct: {unknown_correct} ({unknown_accuracy:.2%})"
            )
            print(
                f"  Correctly rejected as unknown: {unknown_rejected} ({unknown_rejected/len(unknown_results):.2%})"
            )

    print("\n=== Per-subject accuracy (top 10 by probe count) ===")
    ranked = subject_totals.most_common()
    for subject, count in ranked[:10]:
        correct_count = subject_correct.get(subject, 0)
        subject_accuracy = correct_count / count if count else 0.0
        in_gallery = (
            "(in gallery)"
            if gallery_subjects and subject in gallery_subjects
            else "(NOT in gallery)"
        )
        print(
            f"{subject:>6} | {correct_count:>3}/{count:<3} | {subject_accuracy:.2%} {in_gallery}"
        )

    misses = [r for r in results if r.prediction not in (None, r.truth)]
    if misses:
        print("\n=== Sample misidentifications ===")
        for sample in misses[:10]:
            print(
                f"{sample.filename}: truth={sample.truth}, "
                f"pred={sample.prediction}, score={sample.score:.3f}"
                if sample.score
                else ""
            )


def dump_results_csv(path: Path, results: Iterable[EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "truth", "prediction", "score"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "filename": row.filename,
                    "truth": row.truth,
                    "prediction": row.prediction,
                    "score": "" if row.score is None else f"{row.score:.6f}",
                }
            )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the DSR + EdgeFace pipeline on a prepared dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("technical/dataset/test_processed"),
        help="Directory containing hr_images/ and vlr_images/ subfolders.",
    )
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=None,
        help="Root directory for gallery enrollment (e.g., edgeface_finetune/train). Uses HR images from here.",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=None,
        help="Root directory for test probes (e.g., frontal_only/test). Uses VLR images from here.",
    )
    parser.add_argument(
        "--hr-dir",
        type=Path,
        default=None,
        help="Override the HR gallery directory (defaults to dataset-root/hr_images or gallery-root/hr_images).",
    )
    parser.add_argument(
        "--vlr-dir",
        type=Path,
        default=None,
        help="Override the VLR probe directory (defaults to dataset-root/vlr_images or probe-root/vlr_images).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--edgeface-weights",
        type=Path,
        default=None,
        help="Path to EdgeFace model weights (e.g., edgeface_xxs.pt, edgeface_finetuned.pth).",
    )
    parser.add_argument(
        "--dsr-weights",
        type=Path,
        default=None,
        help="Path to DSR model weights (e.g., technical/dsr/hybrid_dsr16.pth).",
    )
    parser.add_argument(
        "--vlr-size",
        type=int,
        default=32,
        choices=[16, 24, 32],
        help="VLR input size (used to select default DSR model if not specified).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the recognition threshold defined in PipelineConfig.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N probe images (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--gallery-via-dsr",
        action="store_true",
        help="Run the DSR model when registering gallery images (default: HR direct).",
    )
    parser.add_argument(
        "--dump-results",
        type=Path,
        default=None,
        help="Optional CSV output path for per-image predictions.",
    )
    parser.add_argument(
        "--skip-dsr",
        action="store_true",
        help="Bypass the DSR model entirely (feed images directly to EdgeFace).",
    )
    return parser.parse_args(argv)


def run_evaluation(args) -> float:
    # Determine gallery and probe directories
    if args.gallery_root and args.probe_root:
        # Split-dataset mode: enroll from one dataset, test on another
        gallery_root = args.gallery_root
        probe_root = args.probe_root
        hr_dir = args.hr_dir or gallery_root / "hr_images"
        vlr_dir = args.vlr_dir or probe_root / "vlr_images"
        print("=" * 70)
        print("SPLIT-DATASET EVALUATION MODE")
        print("=" * 70)
        print(f"Gallery (known identities): {gallery_root}")
        print(f"Probes (unseen images):     {probe_root}")
        print()
    else:
        # Original mode: same dataset for gallery and probes
        dataset_root: Path = args.dataset_root
        hr_dir: Path = args.hr_dir or dataset_root / "hr_images"
        
        # Support resolution-specific VLR folders
        if args.vlr_dir:
            vlr_dir = args.vlr_dir
        else:
            # Try specific resolution folder first
            vlr_dir_candidate = dataset_root / f"vlr_images_{args.vlr_size}x{args.vlr_size}"
            if vlr_dir_candidate.exists():
                vlr_dir = vlr_dir_candidate
            else:
                vlr_dir = dataset_root / "vlr_images"

    config = PipelineConfig(device=args.device)
    if args.threshold is not None:
        config.recognition_threshold = args.threshold
    if args.edgeface_weights is not None:
        config.edgeface_weights_path = args.edgeface_weights
    
    # Set DSR weights
    if args.dsr_weights is not None:
        config.dsr_weights_path = args.dsr_weights
    else:
        # Default to hybrid model for the specified size
        base_dir = Path(__file__).resolve().parents[1]
        default_dsr = base_dir / "technical" / "dsr" / f"hybrid_dsr{args.vlr_size}.pth"
        if default_dsr.exists():
            config.dsr_weights_path = default_dsr
            
    if args.skip_dsr:
        config.skip_dsr = True

    pipeline = build_pipeline(config)

    print("Building gallery from HR images ...")
    gallery_subjects = build_gallery(pipeline, hr_dir, use_dsr=args.gallery_via_dsr)
    print(f"Enrolled {len(gallery_subjects)} identities in gallery")

    print("\nEvaluating VLR probes ...")
    results, subject_totals, subject_correct, unknown_predictions = evaluate_vlr(
        pipeline, vlr_dir, limit=args.limit
    )

    # Calculate how many test subjects are in the gallery (known vs unknown)
    test_subjects = set(subject_totals.keys())
    known_subjects = test_subjects & set(gallery_subjects.keys())
    unknown_subjects = test_subjects - set(gallery_subjects.keys())

    print(f"\n=== Test Set Composition ===")
    print(f"Total test subjects:    {len(test_subjects)}")
    print(
        f"Known (in gallery):     {len(known_subjects)} ({len(known_subjects)/len(test_subjects):.1%})"
    )
    print(
        f"Unknown (not in gallery): {len(unknown_subjects)} ({len(unknown_subjects)/len(test_subjects):.1%})"
    )

    summarise_results(
        results,
        subject_totals,
        subject_correct,
        unknown_predictions,
        gallery_subjects=set(gallery_subjects.keys()),
    )

    if args.dump_results:
        dump_results_csv(args.dump_results, results)
        print(f"\nDetailed results written to {args.dump_results}")

    total_probes = len(results)
    correct = sum(1 for r in results if r.prediction == r.truth)
    accuracy = correct / total_probes if total_probes else 0.0
    return accuracy


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
