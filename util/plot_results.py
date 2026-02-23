"""
Plot evaluation results from different num_book_reviews and num_user_reviews configurations.

Reads metrics.json files from subdirectories in the results directory and creates plots
showing how accuracy varies with different parameter settings.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_directory_name(dir_name):
    """
    Parse directory name to extract num_book_reviews and num_user_reviews.

    Expected format: num_book_reviews_{X}_num_user_reviews_{Y}

    Returns:
        tuple: (num_book_reviews, num_user_reviews) or None if parsing fails
    """
    pattern = r"num_book_reviews_(\d+)_num_user_reviews_(\d+)"
    match = re.match(pattern, dir_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_results(results_dir):
    """
    Load all metrics.json files from the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        list: List of dicts with keys: num_book_reviews, num_user_reviews, metrics
    """
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return results

    # Iterate through subdirectories
    for subdir in results_path.iterdir():
        if not subdir.is_dir():
            continue

        # Parse directory name
        parsed = parse_directory_name(subdir.name)
        if parsed is None:
            print(f"Warning: Skipping directory with unexpected name: {subdir.name}")
            continue

        num_book_reviews, num_user_reviews = parsed

        # Load metrics.json
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: No metrics.json found in {subdir.name}")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        results.append({
            "num_book_reviews": num_book_reviews,
            "num_user_reviews": num_user_reviews,
            "metrics": metrics
        })

    print(f"Loaded {len(results)} result files")
    return results


def plot_accuracy_vs_book_reviews(results, output_dir):
    """Plot accuracy vs num_book_reviews for each num_user_reviews value."""
    # Group by num_user_reviews
    grouped = defaultdict(list)
    for r in results:
        grouped[r["num_user_reviews"]].append(r)

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    # Get all unique book review values
    all_book_reviews = sorted(set(r["num_book_reviews"] for r in results))
    num_groups = len(all_book_reviews)
    num_series = len(grouped)
    bar_width = 0.7 / num_series

    # Generate x positions for bars
    x_base = np.arange(num_groups)

    for idx, num_user_reviews in enumerate(sorted(grouped.keys())):
        data = grouped[num_user_reviews]
        # Sort by num_book_reviews
        data.sort(key=lambda x: x["num_book_reviews"])

        y = [r["metrics"]["accuracy"] for r in data]
        x_positions = x_base + (idx - num_series/2 + 0.5) * bar_width

        bars = ax.bar(x_positions, y, bar_width,
                     label=f"num_user_reviews={num_user_reviews}",
                     color=colors[idx % len(colors)],
                     alpha=0.85,
                     edgecolor='white',
                     linewidth=1.5)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Number of Book Reviews (Community Reviews)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')
    ax.set_title("Accuracy vs Number of Community Reviews per Book",
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_base)
    ax.set_xticklabels(all_book_reviews, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Aesthetic improvements
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_vs_book_reviews.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_accuracy_vs_user_reviews(results, output_dir):
    """Plot accuracy vs num_user_reviews for each num_book_reviews value."""
    # Group by num_book_reviews
    grouped = defaultdict(list)
    for r in results:
        grouped[r["num_book_reviews"]].append(r)

    # Only plot if we have variation in num_user_reviews
    if len(set(r["num_user_reviews"] for r in results)) < 2:
        print("Skipping accuracy vs user reviews plot (no variation in num_user_reviews)")
        return

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    # Get all unique user review values
    all_user_reviews = sorted(set(r["num_user_reviews"] for r in results))
    num_groups = len(all_user_reviews)
    num_series = len(grouped)
    bar_width = 0.7 / num_series

    # Generate x positions for bars
    x_base = np.arange(num_groups)

    for idx, num_book_reviews in enumerate(sorted(grouped.keys())):
        data = grouped[num_book_reviews]
        # Sort by num_user_reviews
        data.sort(key=lambda x: x["num_user_reviews"])

        y = [r["metrics"]["accuracy"] for r in data]
        x_positions = x_base + (idx - num_series/2 + 0.5) * bar_width

        bars = ax.bar(x_positions, y, bar_width,
                     label=f"num_book_reviews={num_book_reviews}",
                     color=colors[idx % len(colors)],
                     alpha=0.85,
                     edgecolor='white',
                     linewidth=1.5)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Number of User Reviews (Reference Reviews)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')
    ax.set_title("Accuracy vs Number of User Reference Reviews",
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_base)
    ax.set_xticklabels(all_user_reviews, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Aesthetic improvements
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_vs_user_reviews.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Book Reviews':<15} {'User Reviews':<15} {'Accuracy':<12} {'Parsed':<12} {'Total':<10}")
    print("-"*70)

    # Sort by num_book_reviews, then num_user_reviews
    sorted_results = sorted(results, key=lambda x: (x["num_book_reviews"], x["num_user_reviews"]))

    for r in sorted_results:
        m = r["metrics"]
        print(f"{r['num_book_reviews']:<15} {r['num_user_reviews']:<15} "
              f"{m['accuracy']:.4f}      {m['n_parsed']:<12} {m['n_total']:<10}")

    print("="*70)

    # Find best configuration
    best = max(results, key=lambda x: x["metrics"]["accuracy"])
    print(f"\nBest configuration:")
    print(f"  num_book_reviews={best['num_book_reviews']}, "
          f"num_user_reviews={best['num_user_reviews']}")
    print(f"  Accuracy: {best['metrics']['accuracy']:.4f}")
    print(f"  Parsed: {best['metrics']['n_parsed']}/{best['metrics']['n_total']}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation results from different parameter configurations"
    )
    parser.add_argument(
        "--results_dir",
        default="/home/mananaga/goodreads/results",
        help="Directory containing result subdirectories (default: /home/mananaga/goodreads/results)"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mananaga/goodreads/plots",
        help="Directory to save plots (default: /home/mananaga/goodreads/plots)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    results = load_results(args.results_dir)

    if len(results) == 0:
        print("No results found. Exiting.")
        return

    # Print summary
    print_summary(results)

    # Create plots
    print(f"\nGenerating plots in {args.output_dir}...")
    plot_accuracy_vs_book_reviews(results, args.output_dir)
    #plot_accuracy_vs_user_reviews(results, args.output_dir)
    #plot_parse_success_rate(results, args.output_dir)
    #plot_heatmap(results, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
