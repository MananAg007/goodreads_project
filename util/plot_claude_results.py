"""
Plot evaluation results from Claude API runs with different num_book_reviews and num_user_reviews configurations.

Reads metrics.json files from subdirectories in the Claude results directory and creates plots
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
    Parse directory name to extract model_name, num_book_reviews and num_user_reviews.

    Expected formats:
        - model_{MODEL_NAME}_num_book_reviews_{X}_num_user_reviews_{Y}
        where MODEL_NAME can be: haiku, sonnet, opus, etc.

    Returns:
        tuple: (model_name, num_book_reviews, num_user_reviews) or None if parsing fails
    """
    # Try format with model name (e.g., haiku, sonnet, opus)
    pattern = r"model_([a-zA-Z0-9_-]+)_num_book_reviews_(\d+)_num_user_reviews_(\d+)"
    match = re.match(pattern, dir_name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    return None


def parse_qwen_directory_name(dir_name):
    """
    Parse Qwen directory name to extract model_size, num_book_reviews and num_user_reviews.

    Expected formats:
        - model_{SIZE}_num_book_reviews_{X}_num_user_reviews_{Y}
        where SIZE can be: 0.5B, 1.5B, 3B, 7B, etc.

    Returns:
        tuple: (model_size, num_book_reviews, num_user_reviews) or None if parsing fails
    """
    # Try format with model size (handles decimal sizes like "0.5B", "1.5B", "3B", etc.)
    pattern = r"model_([\d.]+B)_num_book_reviews_(\d+)_num_user_reviews_(\d+)"
    match = re.match(pattern, dir_name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    return None


def extract_numeric_model_size(model_size):
    """
    Extract numeric value from model size string for sorting.

    Args:
        model_size: String like "0.5B", "1.5B", "3B", "7B", etc.

    Returns:
        float: Numeric value (e.g., 0.5, 1.5, 3.0, 7.0) or 0 if invalid
    """
    match = re.match(r'([\d.]+)B', model_size)
    return float(match.group(1)) if match else 0


def load_results(results_dir):
    """
    Load all metrics.json files from the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        list: List of dicts with keys: model_name, num_book_reviews, num_user_reviews, metrics
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

        model_name, num_book_reviews, num_user_reviews = parsed

        # Load metrics.json
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: No metrics.json found in {subdir.name}")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        results.append({
            "model_name": model_name,
            "num_book_reviews": num_book_reviews,
            "num_user_reviews": num_user_reviews,
            "metrics": metrics
        })

    print(f"Loaded {len(results)} result files")
    return results


def load_qwen_results(qwen_results_dir, num_book_reviews=8, num_user_reviews=1):
    """
    Load Qwen model results from the main results directory.

    Args:
        qwen_results_dir: Path to Qwen results directory (e.g., /home/mananaga/goodreads/results)
        num_book_reviews: Filter for specific num_book_reviews
        num_user_reviews: Filter for specific num_user_reviews

    Returns:
        list: List of dicts with keys: model_name, accuracy
    """
    results = []
    results_path = Path(qwen_results_dir)

    if not results_path.exists():
        print(f"Warning: Qwen results directory not found: {qwen_results_dir}")
        return results

    # Iterate through subdirectories
    for subdir in results_path.iterdir():
        if not subdir.is_dir():
            continue

        # Parse directory name
        parsed = parse_qwen_directory_name(subdir.name)
        if parsed is None:
            continue

        model_size, n_book, n_user = parsed

        # Filter by num_book_reviews and num_user_reviews
        if n_book != num_book_reviews or n_user != num_user_reviews:
            continue

        # Skip 0.5B model
        if model_size == "0.5B":
            continue

        # Load metrics.json
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        results.append({
            "model_name": f"Qwen-{model_size}",
            "model_size": model_size,
            "accuracy": metrics["accuracy"]
        })

    return results


def plot_accuracy_vs_model(claude_results_dir, qwen_results_dir, output_dir, num_book_reviews=8, num_user_reviews=1):
    """
    Plot accuracy comparison across different models (both Qwen and Claude).

    Args:
        claude_results_dir: Path to Claude results directory
        qwen_results_dir: Path to Qwen results directory
        output_dir: Directory to save the plot
        num_book_reviews: Specific num_book_reviews value to compare
        num_user_reviews: Specific num_user_reviews value to compare
    """
    all_models = []

    # Load Qwen results
    print(f"\n  Loading Qwen models from {qwen_results_dir}...")
    qwen_results = load_qwen_results(qwen_results_dir, num_book_reviews, num_user_reviews)
    print(f"  Found {len(qwen_results)} Qwen models")
    all_models.extend(qwen_results)

    # Load Claude results
    print(f"  Loading Claude models from {claude_results_dir}...")
    claude_count = 0
    claude_results_path = Path(claude_results_dir)
    if claude_results_path.exists():
        for subdir in claude_results_path.iterdir():
            if not subdir.is_dir():
                continue

            parsed = parse_directory_name(subdir.name)
            if parsed is None:
                continue

            model_name, n_book, n_user = parsed

            # Filter by num_book_reviews and num_user_reviews
            if n_book != num_book_reviews or n_user != num_user_reviews:
                continue

            # Load metrics.json
            metrics_file = subdir / "metrics.json"
            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                metrics = json.load(f)

            all_models.append({
                "model_name": f"Claude-{model_name}",
                "accuracy": metrics["accuracy"]
            })
            claude_count += 1

    print(f"  Found {claude_count} Claude models")

    if not all_models:
        print(f"No models found for num_book_reviews={num_book_reviews}, num_user_reviews={num_user_reviews}")
        return

    # Sort models: Qwen models by size, then Claude models
    def model_sort_key(model):
        name = model["model_name"]
        if name.startswith("Qwen"):
            # Extract size for sorting
            size = model.get("model_size", "0B")
            return (0, extract_numeric_model_size(size))
        elif name.startswith("Claude"):
            # Claude models after Qwen
            return (1, name)
        return (2, name)

    all_models.sort(key=model_sort_key)

    # Extract data for plotting
    model_names = [m["model_name"] for m in all_models]
    accuracies = [m["accuracy"] for m in all_models]

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8E7DBE']

    # Create bars
    x_positions = np.arange(len(model_names))
    bars = ax.bar(x_positions, accuracies,
                 color=[colors[i % len(colors)] for i in range(len(model_names))],
                 alpha=0.85,
                 edgecolor='white',
                 linewidth=1.5)

    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{acc:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Model", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')

    title = f"Accuracy vs Model"
    title += f"\n(Book Reviews={num_book_reviews}, User Reviews={num_user_reviews})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, fontsize=11, rotation=0)
    ax.tick_params(axis='y', labelsize=11)

    # Add random baseline
    ax.axhline(0.5, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=10)

    # Aesthetic improvements
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0.45, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_vs_model.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved plot with {len(all_models)} models: {output_path}")
    plt.close()


def plot_accuracy_vs_book_reviews(results, output_dir, num_book_reviews_list=None, num_user_reviews_list=None, model_name=None):
    """Plot accuracy vs num_book_reviews for each num_user_reviews value.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save the plot
        num_book_reviews_list: List of num_book_reviews values to include (None = all)
        num_user_reviews_list: List of num_user_reviews values to include (None = all)
        model_name: Specific model name to filter by (e.g., "haiku", "sonnet") (None = all)
    """
    # Filter results based on input lists
    filtered_results = results
    if num_book_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_book_reviews"] in num_book_reviews_list]
    if num_user_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_user_reviews"] in num_user_reviews_list]
    if model_name is not None:
        filtered_results = [r for r in filtered_results if r["model_name"] == model_name]

    if not filtered_results:
        print("No results to plot after filtering")
        return

    # Group by (num_user_reviews, model_name) to avoid mixing models
    grouped = defaultdict(list)
    for r in filtered_results:
        key = (r["num_user_reviews"], r["model_name"])
        grouped[key].append(r)

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

    for idx, (num_user_reviews, model) in enumerate(sorted(grouped.keys())):
        data = grouped[(num_user_reviews, model)]
        # Sort by num_book_reviews
        data.sort(key=lambda x: x["num_book_reviews"])

        y = [r["metrics"]["accuracy"] for r in data]
        x_positions = x_base + (idx - num_series/2 + 0.5) * bar_width

        label = f"num_user_reviews={num_user_reviews}"
        if len(set(r["model_name"] for r in filtered_results)) > 1:
            label += f" ({model})"

        bars = ax.bar(x_positions, y, bar_width,
                     label=label,
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

    title = "Accuracy vs Number of Community Reviews per Book"
    if model_name is not None:
        title += f"\n(Model: Claude-{model_name})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_base)
    ax.set_xticklabels(all_book_reviews, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Add random baseline on top of bars
    ax.axhline(0.5, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=10)

    # Aesthetic improvements
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0.45, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f"claude_accuracy_vs_book_reviews_{model_name}.png" if model_name else "claude_accuracy_vs_book_reviews.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_accuracy_vs_user_reviews(results, output_dir, num_book_reviews_list=None, num_user_reviews_list=None, model_name=None):
    """Plot accuracy vs num_user_reviews for each num_book_reviews value.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save the plot
        num_book_reviews_list: List of num_book_reviews values to include (None = all)
        num_user_reviews_list: List of num_user_reviews values to include (None = all)
        model_name: Specific model name to filter by (e.g., "haiku", "sonnet") (None = all)
    """
    # Filter results based on input lists
    filtered_results = results
    if num_book_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_book_reviews"] in num_book_reviews_list]
    if num_user_reviews_list is not None:
        filtered_results = [r for r in filtered_results if r["num_user_reviews"] in num_user_reviews_list]
    if model_name is not None:
        filtered_results = [r for r in filtered_results if r["model_name"] == model_name]

    if not filtered_results:
        print("No results to plot after filtering")
        return

    # Only plot if we have variation in num_user_reviews
    if len(set(r["num_user_reviews"] for r in filtered_results)) < 2:
        print("Skipping accuracy vs user reviews plot (no variation in num_user_reviews)")
        return

    # Group by (num_book_reviews, model_name) to avoid mixing models
    grouped = defaultdict(list)
    for r in filtered_results:
        key = (r["num_book_reviews"], r["model_name"])
        grouped[key].append(r)

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

    for idx, (num_book_reviews, model) in enumerate(sorted(grouped.keys())):
        data = grouped[(num_book_reviews, model)]
        # Sort by num_user_reviews
        data.sort(key=lambda x: x["num_user_reviews"])

        y = [r["metrics"]["accuracy"] for r in data]
        x_positions = x_base + (idx - num_series/2 + 0.5) * bar_width

        label = f"num_book_reviews={num_book_reviews}"
        if len(set(r["model_name"] for r in filtered_results)) > 1:
            label += f" ({model})"

        bars = ax.bar(x_positions, y, bar_width,
                     label=label,
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

    title = "Accuracy vs Number of User Reference Reviews"
    if model_name is not None:
        title += f"\n(Model: Claude-{model_name})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_base)
    ax.set_xticklabels(all_user_reviews, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Add random baseline on top of bars
    ax.axhline(0.5, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=10)

    # Aesthetic improvements
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0.45, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f"claude_accuracy_vs_user_reviews_{model_name}.png" if model_name else "claude_accuracy_vs_user_reviews.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_accuracy_vs_algorithm(output_dir):
    """Plot accuracy comparison across different algorithms with hardcoded values.

    Args:
        output_dir: Directory to save the plot
    """
    # Hardcoded algorithm names and their accuracies
    algorithms = ['random', 'collaborative-filtering', 'book-rating', 'llm']
    accuracies = [0.50, 0.61, 0.64, 0.71]  # TODO: Replace with actual values

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Create bars
    x_positions = np.arange(len(algorithms))
    bars = ax.bar(x_positions, accuracies,
                 color=colors,
                 alpha=0.85,
                 edgecolor='white',
                 linewidth=1.5)

    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{acc:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Algorithm", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')
    ax.set_title("Accuracy vs Algorithm", fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Add random baseline on top of bars
    ax.axhline(0.5, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=10)

    # Aesthetic improvements
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0.45, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "claude_accuracy_vs_algorithm.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "="*85)
    print("CLAUDE RESULTS SUMMARY")
    print("="*85)
    print(f"{'Model':<15} {'Book Reviews':<15} {'User Reviews':<15} {'Accuracy':<12} {'Parsed':<12} {'Total':<10}")
    print("-"*85)

    # Sort by model_name, num_book_reviews, then num_user_reviews
    sorted_results = sorted(results, key=lambda x: (x["model_name"], x["num_book_reviews"], x["num_user_reviews"]))

    for r in sorted_results:
        m = r["metrics"]
        print(f"{r['model_name']:<15} {r['num_book_reviews']:<15} {r['num_user_reviews']:<15} "
              f"{m['accuracy']:.4f}      {m['n_parsed']:<12} {m['n_total']:<10}")

    print("="*85)

    # Find best configuration
    best = max(results, key=lambda x: x["metrics"]["accuracy"])
    print(f"\nBest configuration:")
    print(f"  model={best['model_name']}, num_book_reviews={best['num_book_reviews']}, "
          f"num_user_reviews={best['num_user_reviews']}")
    print(f"  Accuracy: {best['metrics']['accuracy']:.4f}")
    print(f"  Parsed: {best['metrics']['n_parsed']}/{best['metrics']['n_total']}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Claude API evaluation results from different parameter configurations"
    )
    parser.add_argument(
        "--results_dir",
        default="/home/mananaga/goodreads/results/claude",
        help="Directory containing Claude result subdirectories (default: /home/mananaga/goodreads/results/claude)"
    )
    parser.add_argument(
        "--qwen_results_dir",
        default="/home/mananaga/goodreads/results",
        help="Directory containing Qwen result subdirectories (default: /home/mananaga/goodreads/results)"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mananaga/goodreads/plots/claude",
        help="Directory to save plots (default: /home/mananaga/goodreads/plots/claude)"
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

    # Plot based on what's actually being run: book_reviews=[1,2,4], user_reviews=[1]
    plot_accuracy_vs_book_reviews(results, args.output_dir,
                                  num_book_reviews_list=[1, 2, 4, 8],
                                  num_user_reviews_list=[1])

    # If we get user_reviews variation later, this will work
    plot_accuracy_vs_user_reviews(results, args.output_dir,
                                  num_book_reviews_list=[8],
                                  num_user_reviews_list=[1, 2, 4])

    # Plot algorithm comparison
    plot_accuracy_vs_algorithm(args.output_dir)

    # Plot model comparison (Qwen + Claude)
    print("\nGenerating model comparison plot (Qwen + Claude)...")
    plot_accuracy_vs_model(args.results_dir, args.qwen_results_dir, args.output_dir,
                          num_book_reviews=8, num_user_reviews=1)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
