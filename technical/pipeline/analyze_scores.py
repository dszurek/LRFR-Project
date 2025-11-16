"""Analyze the distribution of similarity scores to understand threshold impact."""

from pathlib import Path
import argparse


def parse_dump_csv(csv_path: Path):
    """Parse the dumped results CSV to analyze score distribution."""
    import csv

    print("=" * 70)
    print("SIMILARITY SCORE DISTRIBUTION")
    print("=" * 70)

    correct_scores = []
    incorrect_scores = []
    rejected_scores = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            truth = row["truth"]
            prediction = row["prediction"]
            score = float(row["score"]) if row["score"] else None

            if prediction == "UNKNOWN":
                if score is not None:
                    rejected_scores.append(score)
            elif prediction == truth:
                if score is not None:
                    correct_scores.append(score)
            else:
                if score is not None:
                    incorrect_scores.append(score)

    print(
        f"\nTotal samples: {len(correct_scores) + len(incorrect_scores) + len(rejected_scores)}"
    )
    print(f"  Correct: {len(correct_scores)}")
    print(f"  Incorrect: {len(incorrect_scores)}")
    print(f"  Rejected (UNKNOWN): {len(rejected_scores)}")

    if correct_scores:
        print(f"\nCorrect match scores:")
        print(f"  Min:  {min(correct_scores):.4f}")
        print(f"  Max:  {max(correct_scores):.4f}")
        print(f"  Mean: {sum(correct_scores)/len(correct_scores):.4f}")

        # Show distribution
        bins = [0.0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"\n  Distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in correct_scores if bins[i] <= s < bins[i + 1])
            pct = count / len(correct_scores) * 100
            print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:4d} ({pct:5.1f}%)")

    if incorrect_scores:
        print(f"\nIncorrect match scores:")
        print(f"  Min:  {min(incorrect_scores):.4f}")
        print(f"  Max:  {max(incorrect_scores):.4f}")
        print(f"  Mean: {sum(incorrect_scores)/len(incorrect_scores):.4f}")

        # Show distribution
        bins = [0.0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print(f"\n  Distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in incorrect_scores if bins[i] <= s < bins[i + 1])
            pct = count / len(incorrect_scores) * 100
            print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:4d} ({pct:5.1f}%)")

    # Threshold analysis
    print(f"\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY")
    print("=" * 70)

    all_matches = [(s, True) for s in correct_scores] + [
        (s, False) for s in incorrect_scores
    ]

    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        accepted = [
            is_correct for score, is_correct in all_matches if score >= threshold
        ]
        if accepted:
            accuracy = sum(accepted) / len(accepted) * 100
            total = len(accepted)
            correct = sum(accepted)
        else:
            accuracy = 0.0
            total = 0
            correct = 0

        print(
            f"  Threshold {threshold:.2f}: {correct:4d}/{total:4d} = {accuracy:5.1f}% accuracy"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="Path to dumped results CSV")
    args = parser.parse_args()

    parse_dump_csv(args.csv_path)
