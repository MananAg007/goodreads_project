#!/usr/bin/env python3
"""
Frequency analysis script for Goodreads dataset.

Plots histograms of:
- User frequency (number of books read per user)
- Book frequency (number of times each book was read)

Only considers records with is_read = 1 (True).
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_frequencies(csv_path: str) -> tuple:
    """
    Load user and book frequencies from the interactions CSV.
    
    Args:
        csv_path: Path to goodreads_interactions.csv
        
    Returns:
        Tuple of (user_counts, book_counts) as Counter objects
    """
    user_counts = Counter()
    book_counts = Counter()
    
    total_rows = 0
    read_rows = 0
    
    print(f"Loading data from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_rows += 1
            
            # Only count records where is_read = 1
            is_read = int(row['is_read'])
            if is_read != 1:
                continue
            
            read_rows += 1
            user_id = int(row['user_id'])
            book_id = int(row['book_id'])
            
            user_counts[user_id] += 1
            book_counts[book_id] += 1
    
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with is_read=1: {read_rows:,}")
    print(f"Unique users: {len(user_counts):,}")
    print(f"Unique books: {len(book_counts):,}")
    
    return user_counts, book_counts


def plot_histogram(
    counts: Counter,
    title: str,
    xlabel: str,
    output_path: str,
    total_count: int,
    log_scale: bool = True,
    n_bins: int = 50
) -> None:
    """
    Plot a histogram of frequency counts.
    
    Args:
        counts: Counter object with frequency counts
        title: Title for the plot
        xlabel: Label for x-axis
        output_path: Path to save the plot
        total_count: Total number of unique items (users or books)
        log_scale: Whether to use log scale for y-axis
        n_bins: Number of bins for histogram
    """
    values = list(counts.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Linear scale histogram
    ax1 = axes[0]
    ax1.hist(values, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'{title} (Linear Scale)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics annotation
    mean_val = np.mean(values)
    median_val = np.median(values)
    max_val = max(values)
    min_val = min(values)
    
    stats_text = f'Total: {total_count:,}\nMean: {mean_val:.1f}\nMedian: {median_val:.1f}\nMax: {max_val}\nMin: {min_val}'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right plot: Log-log scale histogram
    ax2 = axes[1]
    
    # Use log-spaced bins for better visualization
    log_bins = np.logspace(np.log10(max(1, min_val)), np.log10(max_val), n_bins)
    ax2.hist(values, bins=log_bins, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'{title} (Log-Log Scale)', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze user and book frequencies in Goodreads dataset"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/user_data/sheels/Spring2026/10718_mlip/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save plots (default: plots/survey/ relative to project root)"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="Number of bins for histograms"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    csv_path = Path(args.data_dir) / "goodreads_interactions.csv"
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: plots/survey/ relative to project root
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "plots" / "survey"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Goodreads Frequency Analysis")
    print("=" * 60)
    
    # Load frequencies
    user_counts, book_counts = load_frequencies(str(csv_path))
    
    # Plot user frequency
    print("\n" + "-" * 40)
    print("Plotting user frequency distribution...")
    plot_histogram(
        user_counts,
        title="User Activity Distribution",
        xlabel="Number of Books Read per User",
        output_path=str(output_dir / "user_frequency.png"),
        total_count=len(user_counts),
        n_bins=args.n_bins
    )
    
    # Plot book frequency
    print("\n" + "-" * 40)
    print("Plotting book frequency distribution...")
    plot_histogram(
        book_counts,
        title="Book Popularity Distribution",
        xlabel="Number of Times Book Was Read",
        output_path=str(output_dir / "book_frequency.png"),
        total_count=len(book_counts),
        n_bins=args.n_bins
    )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    user_vals = list(user_counts.values())
    book_vals = list(book_counts.values())
    
    print("\nUser Activity (books read per user):")
    print(f"  Mean:   {np.mean(user_vals):.2f}")
    print(f"  Median: {np.median(user_vals):.2f}")
    print(f"  Std:    {np.std(user_vals):.2f}")
    print(f"  Min:    {min(user_vals)}")
    print(f"  Max:    {max(user_vals)}")
    print(f"  25th percentile: {np.percentile(user_vals, 25):.0f}")
    print(f"  75th percentile: {np.percentile(user_vals, 75):.0f}")
    print(f"  90th percentile: {np.percentile(user_vals, 90):.0f}")
    print(f"  99th percentile: {np.percentile(user_vals, 99):.0f}")
    
    print("\nBook Popularity (times read per book):")
    print(f"  Mean:   {np.mean(book_vals):.2f}")
    print(f"  Median: {np.median(book_vals):.2f}")
    print(f"  Std:    {np.std(book_vals):.2f}")
    print(f"  Min:    {min(book_vals)}")
    print(f"  Max:    {max(book_vals)}")
    print(f"  25th percentile: {np.percentile(book_vals, 25):.0f}")
    print(f"  75th percentile: {np.percentile(book_vals, 75):.0f}")
    print(f"  90th percentile: {np.percentile(book_vals, 90):.0f}")
    print(f"  99th percentile: {np.percentile(book_vals, 99):.0f}")
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
