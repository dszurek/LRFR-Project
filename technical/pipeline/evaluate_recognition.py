"""Evaluate EdgeFace recognition in verification (1:1) or identification (1:N) modes."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from technical.pipeline.pipeline import PipelineConfig, build_pipeline

KNOWN_EDGEFACE_WEIGHTS = {
    "xxs": Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt"),
    "finetuned": Path("technical/facial_rec/edgeface_weights/edgeface_finetuned.pth"),
    "s-gamma": Path("technical/facial_rec/edgeface_weights/edgeface_s_gamma_05.pt"),
}


def _subject_from_filename(path: Path) -> str:
    return path.stem.split("_")[0]


def _load_hr_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = transforms.functional.to_tensor(image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _collect_images(directory: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(directory.glob("*.png")):
        grouped[_subject_from_filename(path)].append(path)
    return grouped


def _compute_hr_embedding(
    pipeline, path: Path, device: torch.device, cache: Dict[Path, torch.Tensor]
) -> torch.Tensor:
    if path not in cache:
        tensor = _load_hr_tensor(path, device)
        embedding = pipeline.infer_embedding(tensor).detach().cpu()
        cache[path] = embedding
    return cache[path]


def _compute_dsr_embedding(
    pipeline, path: Path, cache: Dict[Path, torch.Tensor]
) -> torch.Tensor:
    if path not in cache:
        sr = pipeline.upscale(path)
        embedding = pipeline.infer_embedding(sr).detach().cpu()
        cache[path] = embedding
    return cache[path]


def _build_verification_pairs(
    positives: List[Tuple[Path, Path]],
    hr_by_subject: Dict[str, List[Path]],
    rng: random.Random,
) -> List[Tuple[Path, Path, int]]:
    subjects = list(hr_by_subject.keys())
    pairs: List[Tuple[Path, Path, int]] = []
    for vlr_path, hr_path in positives:
        pairs.append((vlr_path, hr_path, 1))
        if len(subjects) > 1:
            anchor_subject = _subject_from_filename(vlr_path)
            distractor_subjects = [s for s in subjects if s != anchor_subject]
            other_subject = rng.choice(distractor_subjects)
            other_hr = rng.choice(hr_by_subject[other_subject])
            pairs.append((vlr_path, other_hr, 0))
    return pairs


def _compute_verification_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    thresholds = np.linspace(-1.0, 1.0, num=2001)
    best_accuracy = -1.0
    best_index = 0
    tprs: List[float] = []
    fprs: List[float] = []
    accuracies: List[float] = []

    positives = labels == 1
    negatives = labels == 0

    for idx, threshold in enumerate(thresholds):
        predictions = scores >= threshold
        tp = np.logical_and(predictions, positives).sum()
        tn = np.logical_and(~predictions, negatives).sum()
        fp = np.logical_and(predictions, negatives).sum()
        fn = np.logical_and(~predictions, positives).sum()

        tpr = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
        accuracy = (tp + tn) / labels.size

        tprs.append(tpr)
        fprs.append(fpr)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_index = idx

    auc = float(np.trapz(tprs, fprs))
    return {
        "best_threshold": float(thresholds[best_index]),
        "best_accuracy": best_accuracy,
        "tpr": tprs[best_index],
        "fpr": fprs[best_index],
        "roc_auc": auc,
    }


def run_verification(
    pipeline,
    hr_by_subject: Dict[str, List[Path]],
    vlr_by_subject: Dict[str, List[Path]],
    limit: int | None,
    seed: int,
    threshold_override: float | None,
) -> None:
    rng = random.Random(seed)

    positives: List[Tuple[Path, Path]] = []
    for subject, vlr_paths in vlr_by_subject.items():
        for vlr_path in vlr_paths:
            hr_candidates = hr_by_subject.get(subject)
            if not hr_candidates:
                continue
            hr_path = None
            for candidate in hr_candidates:
                if candidate.name == vlr_path.name:
                    hr_path = candidate
                    break
            hr_path = hr_path or rng.choice(hr_candidates)
            positives.append((vlr_path, hr_path))
    if limit is not None:
        positives = positives[:limit]

    pairs = _build_verification_pairs(positives, hr_by_subject, rng)
    if not pairs:
        print("No verification pairs available.")
        return

    hr_cache: Dict[Path, torch.Tensor] = {}
    dsr_cache: Dict[Path, torch.Tensor] = {}
    scores: List[float] = []
    labels: List[int] = []

    for vlr_path, hr_path, label in tqdm(pairs, desc="Pairs", unit="pair"):
        dsr_embedding = _compute_dsr_embedding(pipeline, vlr_path, dsr_cache)
        hr_embedding = _compute_hr_embedding(
            pipeline, hr_path, pipeline.device, hr_cache
        )
        similarity = float(torch.dot(dsr_embedding, hr_embedding))
        scores.append(similarity)
        labels.append(label)

    scores_np = np.array(scores)
    labels_np = np.array(labels)
    metrics = _compute_verification_metrics(scores_np, labels_np)

    print("\nVERIFICATION METRICS")
    print(f"Pairs evaluated      : {len(pairs)}")
    print(f"Positives / negatives: {labels_np.sum()} / {(labels_np == 0).sum()}")
    print(f"ROC-AUC              : {metrics['roc_auc']:.4f}")
    print(f"Best threshold       : {metrics['best_threshold']:.3f}")
    print(
        f"Accuracy @ best      : {metrics['best_accuracy']:.4f}"
        f" (TPR={metrics['tpr']:.4f}, FPR={metrics['fpr']:.4f})"
    )
    if threshold_override is not None:
        thr = threshold_override
        predictions = scores_np >= thr
        positives = labels_np == 1
        negatives = labels_np == 0
        tp = np.logical_and(predictions, positives).sum()
        tn = np.logical_and(~predictions, negatives).sum()
        fp = np.logical_and(predictions, negatives).sum()
        fn = np.logical_and(~predictions, positives).sum()
        tpr = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
        accuracy = (tp + tn) / labels_np.size
        print(
            f"Accuracy @ {thr:.3f}    : {accuracy:.4f}"
            f" (TPR={tpr:.4f}, FPR={fpr:.4f})"
        )


def run_identification(
    pipeline,
    hr_by_subject: Dict[str, List[Path]],
    vlr_by_subject: Dict[str, List[Path]],
    limit: int | None,
    n_candidates: int,
    seed: int,
) -> None:
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    subjects = list(hr_by_subject.keys())
    if len(subjects) < n_candidates:
        raise ValueError("Dataset does not have enough subjects for requested N")

    rng = random.Random(seed)

    hr_cache: Dict[Path, torch.Tensor] = {}
    dsr_cache: Dict[Path, torch.Tensor] = {}

    subject_prototypes: Dict[str, torch.Tensor] = {}
    for subject, paths in hr_by_subject.items():
        embeddings = [
            _compute_hr_embedding(pipeline, path, pipeline.device, hr_cache)
            for path in paths
        ]
        stacked = torch.stack(embeddings)
        mean_embedding = torch.mean(stacked, dim=0)
        subject_prototypes[subject] = F.normalize(mean_embedding, dim=0).cpu()

    evaluations: List[Tuple[str, List[str], str, float]] = []

    all_samples: List[Tuple[str, Path]] = []
    for subject, vlr_paths in vlr_by_subject.items():
        for path in vlr_paths:
            all_samples.append((subject, path))
    if limit is not None:
        all_samples = all_samples[:limit]

    for subject, vlr_path in tqdm(all_samples, desc="Probes", unit="image"):
        candidates = {subject}
        distractors = [s for s in subjects if s != subject]
        if len(distractors) < n_candidates - 1:
            continue
        candidates.update(rng.sample(distractors, n_candidates - 1))
        candidate_list = sorted(candidates)

        dsr_embedding = _compute_dsr_embedding(pipeline, vlr_path, dsr_cache)

        best_subject = None
        best_score = -1.0
        for candidate in candidate_list:
            score = float(torch.dot(dsr_embedding, subject_prototypes[candidate]))
            if score > best_score:
                best_score = score
                best_subject = candidate

        evaluations.append((subject, candidate_list, best_subject or "", best_score))

    if not evaluations:
        print("No identification probes evaluated.")
        return

    correct = sum(1 for truth, _, pred, _ in evaluations if pred == truth)
    print("\nIDENTIFICATION METRICS")
    print(f"Probes evaluated : {len(evaluations)}")
    print(f"Gallery size (N) : {n_candidates}")
    print(f"Top-1 accuracy   : {correct / len(evaluations):.4f}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EdgeFace recognition metrics."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing hr_images/ and vlr_images/.",
    )
    parser.add_argument(
        "--metric",
        choices=["verification", "identification"],
        default="verification",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=5,
        help="Number of gallery candidates for identification (includes true subject).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples/pairs for faster testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling negatives and distractors.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--edgeface",
        choices=sorted(KNOWN_EDGEFACE_WEIGHTS.keys()),
        default="xxs",
        help="Choose a bundled EdgeFace checkpoint to evaluate.",
    )
    parser.add_argument(
        "--edgeface-weights",
        type=Path,
        default=None,
        help="Override and load EdgeFace weights from a custom path.",
    )
    parser.add_argument(
        "--dsr-weights",
        type=Path,
        default=Path("technical/dsr/dsr.pth"),
        help="Path to DSR weights.",
    )
    parser.add_argument(
        "--skip-dsr",
        action="store_true",
        help="Bypass DSR (feed HR tensors directly).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed threshold for verification (overrides sweep).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    hr_dir = args.dataset_root / "hr_images"
    vlr_dir = args.dataset_root / "vlr_images"
    if not hr_dir.exists() or not vlr_dir.exists():
        raise RuntimeError(
            "Dataset root must contain hr_images/ and vlr_images/ subfolders."
        )

    if args.edgeface_weights is not None:
        edgeface_path = args.edgeface_weights
    else:
        edgeface_path = KNOWN_EDGEFACE_WEIGHTS[args.edgeface]

    config = PipelineConfig(
        device=args.device,
        dsr_weights_path=args.dsr_weights,
        edgeface_weights_path=edgeface_path,
    )
    config.skip_dsr = args.skip_dsr

    pipeline = build_pipeline(config)

    hr_by_subject = _collect_images(hr_dir)
    vlr_by_subject = _collect_images(vlr_dir)

    if args.metric == "verification":
        run_verification(
            pipeline,
            hr_by_subject,
            vlr_by_subject,
            args.limit,
            args.seed,
            args.threshold,
        )
    else:
        run_identification(
            pipeline,
            hr_by_subject,
            vlr_by_subject,
            args.limit,
            args.n_candidates,
            args.seed,
        )


if __name__ == "__main__":
    main()
