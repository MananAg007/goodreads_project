#!/usr/bin/env python3
"""
Script to evaluate additional baseline models on Goodreads dataset.

Evaluates:
1. Random baseline - on entire dataset (train + test concatenated)
2. Book average baseline - on train and test separately (70/30 split)
3. User average baseline - on train and test separately (70/30 split)
4. Corpus mean baseline - on train and test separately (70/30 split)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.additional_baselines import (
    RandomBaseline,
    UserBookAverageBaseline,
    CorpusMeanBaseline
)
from data.dataloader import GoodreadsDataLoader


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary with metric names and values
        prefix: Optional prefix string
    """
    if prefix:
        print(f"  {prefix}")
    print(f"    MAE:      {metrics['mae']:.4f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")


def evaluate_random_baseline(
    train_ratings: np.ndarray,
    test_ratings: np.ndarray,
    random_seed: int = 86
) -> None:
    """
    Evaluate random baseline on entire dataset.
    
    Args:
        train_ratings: Training ratings
        test_ratings: Test ratings
        random_seed: Random seed for reproducibility
    """
    print("\n" + "=" * 60)
    print("BASELINE 1: Random Prediction (1-5)")
    print("=" * 60)
    
    # Concatenate train and test
    all_ratings = np.concatenate([train_ratings, test_ratings])
    
    print(f"\nTotal samples: {len(all_ratings):,}")
    
    # Initialize and evaluate random baseline
    print("\nEvaluating random baseline...")
    model = RandomBaseline(random_seed=random_seed)
    metrics = model.evaluate(all_ratings)
    
    print("\nResults on entire dataset:")
    print_metrics(metrics)


def evaluate_user_book_average_baseline(
    train_entity_ids: np.ndarray,
    train_ratings: np.ndarray,
    test_entity_ids: np.ndarray,
    test_ratings: np.ndarray,
    is_book: bool = True
) -> None:
    """
    Evaluate user/book average baseline on train and test sets.
    
    Args:
        train_entity_ids: Training entity IDs (user IDs or book IDs)
        train_ratings: Training ratings
        test_entity_ids: Test entity IDs (user IDs or book IDs)
        test_ratings: Test ratings
        is_book: If True, use book averages. If False, use user averages.
    """
    entity_type = "Book" if is_book else "User"
    print("\n" + "=" * 60)
    print(f"BASELINE 2: {entity_type} Average Ratings")
    print("=" * 60)
    
    # Initialize and train user/book average baseline
    print(f"\nTraining {entity_type.lower()} average baseline...")
    model = UserBookAverageBaseline(is_book=is_book)
    model.fit(train_entity_ids, train_ratings)
    
    # Evaluate on train set
    print("\nEvaluating on train set...")
    train_metrics = model.evaluate(train_entity_ids, train_ratings)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = model.evaluate(test_entity_ids, test_ratings)
    
    # Report results
    print("\nResults:")
    print_metrics(train_metrics, "Train:")
    print_metrics(test_metrics, "Test:")


def evaluate_corpus_mean_baseline(
    train_ratings: np.ndarray,
    test_ratings: np.ndarray
) -> None:
    """
    Evaluate corpus mean baseline on train and test sets.
    
    Args:
        train_ratings: Training ratings
        test_ratings: Test ratings
    """
    print("\n" + "=" * 60)
    print("BASELINE 3: Corpus Mean Prediction")
    print("=" * 60)
    
    # Initialize and train corpus mean baseline
    print("\nTraining corpus mean baseline...")
    model = CorpusMeanBaseline()
    model.fit(train_ratings)
    
    # Evaluate on train set
    print("\nEvaluating on train set...")
    train_metrics = model.evaluate(train_ratings)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = model.evaluate(test_ratings)
    
    # Report results
    print("\nResults:")
    print_metrics(train_metrics, "Train:")
    print_metrics(test_metrics, "Test:")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate additional baseline models on Goodreads dataset"
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/data/user_data/sheels/Spring2026/10718_mlip/data/goodreads_interactions.csv",
        help="Path to the interactions CSV file"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=86,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["random", "book_avg", "user_avg", "corpus_mean", "all"],
        default="all",
        help="Which baseline(s) to evaluate"
    )
    parser.add_argument(
        "--use_map",
        action="store_true",
        help="Load user_id_map.csv and book_id_map.csv from CSV directory"
    )
    
    args = parser.parse_args()
    
    # Create single dataloader with 70/0/30 split
    print("=" * 60)
    print("Loading data with 70/0/30 split...")
    print("=" * 60)
    
    dataloader = GoodreadsDataLoader(
        csv_path=args.csv_path,
        batch_size=512,
        train_ratio=0.7,
        val_ratio=0.0,
        test_ratio=0.3,
        random_seed=args.random_seed,
        use_map=args.use_map
    )
    
    # Get train and test data (val is empty)
    train_users, train_books, train_ratings = dataloader.get_split_data('train')
    test_users, test_books, test_ratings = dataloader.get_split_data('test')
    
    print(f"\nDataset loaded successfully!")
    print(f"  Train: {len(train_ratings):,} samples")
    print(f"  Test:  {len(test_ratings):,} samples")
    
    # Run selected baselines
    if args.baseline in ["random", "all"]:
        evaluate_random_baseline(
            train_ratings,
            test_ratings,
            args.random_seed
        )
    
    if args.baseline in ["book_avg", "all"]:
        evaluate_user_book_average_baseline(
            train_books, train_ratings,
            test_books, test_ratings,
            is_book=True
        )
    
    if args.baseline in ["user_avg", "all"]:
        evaluate_user_book_average_baseline(
            train_users, train_ratings,
            test_users, test_ratings,
            is_book=False
        )
    
    if args.baseline in ["corpus_mean", "all"]:
        evaluate_corpus_mean_baseline(
            train_ratings,
            test_ratings
        )
    
    print("\n" + "=" * 60)
    print("All baselines evaluated successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
