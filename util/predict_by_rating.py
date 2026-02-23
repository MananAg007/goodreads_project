"""
Predict user preferences based on average book ratings.

Reads dataset.jsonl and predicts that users prefer the book with higher average rating.
Calculates the accuracy of this simple baseline method.
"""

import argparse
import json
from pathlib import Path


def load_dataset(dataset_path):
    """Load dataset from jsonl file.

    Args:
        dataset_path: Path to dataset.jsonl file

    Returns:
        list: List of data entries
    """
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def predict_by_rating(entry):
    """Predict preference based on average book ratings.

    Args:
        entry: Dataset entry with book_a and book_b information

    Returns:
        str: Predicted preference ('A' or 'B')
    """
    rating_a = entry['book_a']['average_rating']
    rating_b = entry['book_b']['average_rating']

    # Predict the book with higher average rating
    if rating_a > rating_b:
        return 'A'
    elif rating_b > rating_a:
        return 'B'
    else:
        # If ratings are equal, default to 'A'
        return 'A'


def calculate_accuracy(dataset):
    """Calculate accuracy of rating-based prediction.

    Args:
        dataset: List of dataset entries

    Returns:
        dict: Metrics including accuracy, correct count, and total count
    """
    correct = 0
    total = len(dataset)

    for entry in dataset:
        prediction = predict_by_rating(entry)
        actual = entry['preferred']

        if prediction == actual:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict preferences based on average book ratings"
    )
    parser.add_argument(
        "--dataset",
        default="/home/mananaga/goodreads/data/book_preference_dataset.jsonl",
        help="Path to dataset.jsonl file (default: /home/mananaga/goodreads/data/book_preference_dataset.jsonl)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} entries")

    # Calculate accuracy
    print("\nCalculating accuracy of rating-based prediction...")
    metrics = calculate_accuracy(dataset)

    # Print results
    print("\n" + "="*50)
    print("RATING-BASED PREDICTION RESULTS")
    print("="*50)
    print(f"Total entries:    {metrics['total']}")
    print(f"Correct:          {metrics['correct']}")
    print(f"Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print("="*50)

    return metrics


if __name__ == "__main__":
    main()
