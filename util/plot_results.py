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
    Parse directory name to extract model_size, num_book_reviews and num_user_reviews.

    Expected formats:
        - model_{SIZE}_num_book_reviews_{X}_num_user_reviews_{Y}
        - num_book_reviews_{X}_num_user_reviews_{Y} (legacy format)

    Returns:
        tuple: (model_size, num_book_reviews, num_user_reviews) or None if parsing fails
        model_size will be None for legacy format
    """
    # Try new format with model size
    pattern_with_model = r"model_(\w+)_num_book_reviews_(\d+)_num_user_reviews_(\d+)"
    match = re.match(pattern_with_model, dir_name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    # Try legacy format without model size
    pattern = r"num_book_reviews_(\d+)_num_user_reviews_(\d+)"
    match = re.match(pattern, dir_name)
    if match:
        return None, int(match.group(1)), int(match.group(2))

    return None


def load_results(results_dir):
    """
    Load all metrics.json files from the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        list: List of dicts with keys: model_size, num_book_reviews, num_user_reviews, metrics
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

        model_size, num_book_reviews, num_user_reviews = parsed

        # Load metrics.json
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: No metrics.json found in {subdir.name}")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        results.append({
            "model_size": model_size,
            "num_book_reviews": num_book_reviews,
            "num_user_reviews": num_user_reviews,
            "metrics": metrics
        })

    print(f"Loaded {len(results)} result files")
    return results


def plot_accuracy_vs_book_reviews(results, output_dir, num_book_reviews_list=None, num_user_reviews_list=None):
    """Plot accuracy vs num_book_reviews for each num_user_reviews value.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save the plot
        num_book_reviews_list: List of num_book_reviews values to include (None = all)
        num_user_reviews_list: List of num_user_reviews values to include (None = all)
    """
    # Filter results based on input lists
    filtered_results = results
    if num_book_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_book_reviews"] in num_book_reviews_list]
    if num_user_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_user_reviews"] in num_user_reviews_list]

    if not filtered_results:
        print("No results to plot after filtering")
        return

    # Group by num_user_reviews
    grouped = defaultdict(list)
    for r in filtered_results:
        grouped[r["num_user_reviews"]].append(r)

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    # Get all unique book review values
    all_book_reviews = sorted(set(r["num_book_reviews"] for r in filtered_results))
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


def plot_accuracy_vs_user_reviews(results, output_dir, num_book_reviews_list=None, num_user_reviews_list=None):
    """Plot accuracy vs num_user_reviews for each num_book_reviews value.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save the plot
        num_book_reviews_list: List of num_book_reviews values to include (None = all)
        num_user_reviews_list: List of num_user_reviews values to include (None = all)
    """
    # Filter results based on input lists
    filtered_results = results
    if num_book_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_book_reviews"] in num_book_reviews_list]
    if num_user_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_user_reviews"] in num_user_reviews_list]

    if not filtered_results:
        print("No results to plot after filtering")
        return

    # Only plot if we have variation in num_user_reviews
    if len(set(r["num_user_reviews"] for r in filtered_results)) < 2:
        print("Skipping accuracy vs user reviews plot (no variation in num_user_reviews)")
        return

    # Group by num_book_reviews
    grouped = defaultdict(list)
    for r in filtered_results:
        grouped[r["num_book_reviews"]].append(r)

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    # Get all unique user review values
    all_user_reviews = sorted(set(r["num_user_reviews"] for r in filtered_results))
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


def plot_accuracy_vs_llm_model(results, output_dir, num_book_reviews=None, num_user_reviews=None):
    """Plot accuracy vs LLM model for a specific configuration.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save the plot
        num_book_reviews: Specific num_book_reviews value to compare (required)
        num_user_reviews: Specific num_user_reviews value to compare (required)
    """
    # Filter results that have model_size information
    filtered_results = [r for r in results if r["model_size"] is not None]

    if not filtered_results:
        print("No results with model information found. Skipping accuracy vs LLM model plot.")
        return

    # Filter by specific num_book_reviews and num_user_reviews
    if num_book_reviews is not None:
        filtered_results = [r for r in filtered_results if r["num_book_reviews"] == num_book_reviews]
    if num_user_reviews is not None:
        filtered_results = [r for r in filtered_results if r["num_user_reviews"] == num_user_reviews]

    if not filtered_results:
        print(f"No results found for num_book_reviews={num_book_reviews}, num_user_reviews={num_user_reviews}")
        return

    # Only plot if we have variation in model_size
    if len(set(r["model_size"] for r in filtered_results)) < 2:
        print("Skipping accuracy vs LLM model plot (no variation in model sizes)")
        return

    # Sort by model size for consistent ordering
    filtered_results.sort(key=lambda x: x["model_size"])

    # Extract data
    model_sizes = [r["model_size"] for r in filtered_results]
    accuracies = [r["metrics"]["accuracy"] for r in filtered_results]

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    # Create bars
    x_positions = np.arange(len(model_sizes))
    bars = ax.bar(x_positions, accuracies,
                 color=[colors[i % len(colors)] for i in range(len(model_sizes))],
                 alpha=0.85,
                 edgecolor='white',
                 linewidth=1.5)

    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{acc:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("LLM Model Size", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')

    title = f"Accuracy vs LLM Model Size"
    if num_book_reviews is not None and num_user_reviews is not None:
        title += f"\n(Book Reviews={num_book_reviews}, User Reviews={num_user_reviews})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_sizes, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Aesthetic improvements
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_vs_llm_model.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "="*85)
    print("RESULTS SUMMARY")
    print("="*85)
    print(f"{'Model':<12} {'Book Reviews':<15} {'User Reviews':<15} {'Accuracy':<12} {'Parsed':<12} {'Total':<10}")
    print("-"*85)

    # Sort by model_size, num_book_reviews, then num_user_reviews
    sorted_results = sorted(results, key=lambda x: (x["model_size"] or "", x["num_book_reviews"], x["num_user_reviews"]))

    for r in sorted_results:
        m = r["metrics"]
        model_str = r["model_size"] if r["model_size"] else "N/A"
        print(f"{model_str:<12} {r['num_book_reviews']:<15} {r['num_user_reviews']:<15} "
              f"{m['accuracy']:.4f}      {m['n_parsed']:<12} {m['n_total']:<10}")

    print("="*85)

    # Find best configuration
    best = max(results, key=lambda x: x["metrics"]["accuracy"])
    print(f"\nBest configuration:")
    model_info = f"model={best['model_size']}, " if best['model_size'] else ""
    print(f"  {model_info}num_book_reviews={best['num_book_reviews']}, "
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

    # Hardcoded filters for each plot
    plot_accuracy_vs_book_reviews(results, args.output_dir,
                                  num_book_reviews_list=[1, 2, 4, 8],
                                  num_user_reviews_list=[1])
    plot_accuracy_vs_user_reviews(results, args.output_dir,
                                  num_book_reviews_list=[4],
                                  num_user_reviews_list=[1, 2, 4])
    plot_accuracy_vs_llm_model(results, args.output_dir,
                              num_book_reviews=8,
                              num_user_reviews=1)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
