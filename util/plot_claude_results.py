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

    Does NOT match directories with _avg_ratings_ suffix (those are handled by parse_directory_name_with_avg_ratings).

    Returns:
        tuple: (model_name, num_book_reviews, num_user_reviews) or None if parsing fails
    """
    # Skip directories with avg_ratings in the name
    if "_avg_ratings_" in dir_name:
        return None

    # Try format with model name (e.g., haiku, sonnet, opus)
    # Match the pattern and ensure it ends after num_user_reviews (no additional suffix)
    pattern = r"^model_([a-zA-Z0-9_-]+)_num_book_reviews_(\d+)_num_user_reviews_(\d+)$"
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


def parse_directory_name_with_avg_ratings(dir_name):
    """
    Parse directory name to extract model_name, num_book_reviews, num_user_reviews, and avg_ratings_mode.
    Extends parse_directory_name to handle average ratings experiments.

    Expected formats:
        - model_{MODEL_NAME}_num_book_reviews_{X}_num_user_reviews_{Y}_avg_ratings_{MODE}
        where MODEL_NAME can be: haiku, sonnet, opus, etc.
        MODE can be: true, random, flipped, unavailable

    Returns:
        tuple: (model_name, num_book_reviews, num_user_reviews, avg_ratings_mode)
               or None if parsing fails
    """
    pattern = r"model_([a-zA-Z0-9_-]+)_num_book_reviews_(\d+)_num_user_reviews_(\d+)_avg_ratings_([a-z]+)"
    match = re.match(pattern, dir_name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
    return None


def parse_directory_name_with_reviews_filter(dir_name):
    """
    Parse directory name to extract model_name, num_book_reviews, num_user_reviews, and reviews_filter_mode.
    Extends parse_directory_name to handle review filtering experiments.

    Expected formats:
        - model_{MODEL_NAME}_num_book_reviews_{X}_num_user_reviews_{Y}_avg_ratings_{MODE}_filter_{FILTER}
        where MODEL_NAME can be: haiku, sonnet, opus, etc.
        MODE can be: true, random, flipped, unavailable
        FILTER can be: none, most_popular, least_popular

    Returns:
        tuple: (model_name, num_book_reviews, num_user_reviews, avg_ratings_mode, reviews_filter_mode)
               or None if parsing fails
    """
    pattern = r"model_([a-zA-Z0-9_-]+)_num_book_reviews_(\d+)_num_user_reviews_(\d+)_avg_ratings_([a-z]+)_filter_([a-z_]+)"
    match = re.match(pattern, dir_name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3)), match.group(4), match.group(5)
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


def load_results_with_avg_ratings(results_dir):
    """
    Load all metrics.json files from results directory, including average_ratings_mode variants.

    Args:
        results_dir: Path to results directory

    Returns:
        list: List of dicts with keys: model_name, num_book_reviews, num_user_reviews, avg_ratings_mode, metrics
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

        # Try parsing with average_ratings_mode
        parsed = parse_directory_name_with_avg_ratings(subdir.name)
        if parsed is None:
            continue  # Skip directories that don't have avg_ratings in name

        model_name, num_book_reviews, num_user_reviews, avg_ratings_mode = parsed

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
            "avg_ratings_mode": avg_ratings_mode,
            "metrics": metrics
        })

    print(f"Loaded {len(results)} result files with average_ratings modes")
    return results


def load_results_with_reviews_filter(results_dir):
    """
    Load all metrics.json files from results directory, including reviews_filter_mode variants.

    Args:
        results_dir: Path to results directory

    Returns:
        list: List of dicts with keys: model_name, num_book_reviews, num_user_reviews, avg_ratings_mode, reviews_filter_mode, metrics
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

        # Try parsing with reviews_filter_mode
        parsed = parse_directory_name_with_reviews_filter(subdir.name)
        if parsed is None:
            continue  # Skip directories that don't have filter in name

        model_name, num_book_reviews, num_user_reviews, avg_ratings_mode, reviews_filter_mode = parsed

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
            "avg_ratings_mode": avg_ratings_mode,
            "reviews_filter_mode": reviews_filter_mode,
            "metrics": metrics
        })

    print(f"Loaded {len(results)} result files with reviews_filter modes")
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
    fig, ax = plt.subplots(figsize=(10, 8))

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
    fig, ax = plt.subplots(figsize=(10, 8))

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
    fig, ax = plt.subplots(figsize=(10, 8))

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


def plot_reasoning_quality(output_dir):
    """Plot reasoning quality comparison for correct vs incorrect predictions with hardcoded values.

    Args:
        output_dir: Directory to save the plot
    """
    # Hardcoded values from reasoning evaluation
    # Correct predictions: Sound: 8, Vague: 2
    # Incorrect predictions: Sound: 7, Vague: 3
    # Categories renamed: Sound -> Strong, Vague -> Weak
    categories = ['Strong', 'Weak']
    correct_counts = [8, 2]
    incorrect_counts = [7, 3]

    # Convert to percentages
    total_correct = sum(correct_counts)
    total_incorrect = sum(incorrect_counts)
    correct_percentages = [c / total_correct * 100 for c in correct_counts]
    incorrect_percentages = [c / total_incorrect * 100 for c in incorrect_counts]

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72']  # Blue for correct, Purple for incorrect

    # Set up bar positions
    x_positions = np.arange(len(categories))
    bar_width = 0.35

    # Create grouped bars
    bars1 = ax.bar(x_positions - bar_width/2, correct_percentages, bar_width,
                   label='Correct Predictions',
                   color=colors[0],
                   alpha=0.85,
                   edgecolor='white',
                   linewidth=1.5)

    bars2 = ax.bar(x_positions + bar_width/2, incorrect_percentages, bar_width,
                   label='Incorrect Predictions',
                   color=colors[1],
                   alpha=0.85,
                   edgecolor='white',
                   linewidth=1.5)

    # Add value labels on top of bars
    for bar, count, pct in zip(bars1, correct_counts, correct_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{count}\n({pct:.0f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, count, pct in zip(bars2, incorrect_counts, incorrect_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{count}\n({pct:.0f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Reasoning Quality Category", fontsize=13, fontweight='bold')
    ax.set_ylabel("Percentage (%)", fontsize=13, fontweight='bold')
    ax.set_title("Reasoning Quality: Correct vs Incorrect Predictions", fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Aesthetic improvements
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "reasoning_quality_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_accuracy_vs_avg_ratings_mode(results, output_dir, num_book_reviews=8, num_user_reviews=1, model_name=None):
    """
    Plot accuracy vs average_ratings_mode to show robustness to corrupted/missing average ratings.

    Args:
        results: List of result dictionaries (from load_results_with_avg_ratings)
        output_dir: Directory to save the plot
        num_book_reviews: Filter for specific num_book_reviews
        num_user_reviews: Filter for specific num_user_reviews
        model_name: Filter for specific model (e.g., "haiku") (None = all)
    """
    # Filter results
    filtered_results = results
    if num_book_reviews is not None:
        filtered_results = [r for r in filtered_results if r["num_book_reviews"] == num_book_reviews]
    if num_user_reviews is not None:
        filtered_results = [r for r in filtered_results if r["num_user_reviews"] == num_user_reviews]
    if model_name is not None:
        filtered_results = [r for r in filtered_results if r["model_name"] == model_name]

    if not filtered_results:
        print(f"No results found for robustness plot (books={num_book_reviews}, users={num_user_reviews}, model={model_name})")
        return

    # Group by model_name to create separate series if multiple models
    grouped = defaultdict(list)
    for r in filtered_results:
        grouped[r["model_name"]].append(r)

    # Define mode order and display names
    mode_order = ["flipped", "random", "unavailable", "provided"]
    mode_display = {"flipped": "flipped", "random": "random", "unavailable": "unavailable", "provided": "provided"}
    # Map true -> provided for display
    mode_mapping = {"true": "provided"}

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    num_modes = len(mode_order)
    num_models = len(grouped)
    bar_width = 0.7 / num_models if num_models > 1 else 0.6

    # Generate x positions for bars
    x_base = np.arange(num_modes)

    for idx, model in enumerate(sorted(grouped.keys())):
        data = grouped[model]
        # Map true -> provided for display
        data_by_mode = {}
        for r in data:
            mode = r["avg_ratings_mode"]
            # Map "true" to "provided" for consistent naming
            display_mode = mode_mapping.get(mode, mode)
            data_by_mode[display_mode] = r

        y = []
        for mode in mode_order:
            if mode in data_by_mode:
                y.append(data_by_mode[mode]["metrics"]["accuracy"])
            else:
                y.append(0)  # Placeholder if mode missing

        # Calculate x positions
        if num_models > 1:
            x_positions = x_base + (idx - num_models/2 + 0.5) * bar_width
        else:
            x_positions = x_base

        label = f"Claude-{model}"

        bars = ax.bar(x_positions, y, bar_width,
                     label=label,
                     color=colors[idx % len(colors)],
                     alpha=0.85,
                     edgecolor='white',
                     linewidth=1.5)

        # Add value labels on top of bars
        for bar, acc in zip(bars, y):
            if acc > 0:  # Only label non-zero values
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{acc:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Average Ratings Mode", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')

    title = f"Robustness: Accuracy vs Average Ratings Mode"
    title += f"\n(Book Reviews={num_book_reviews}, User Reviews={num_user_reviews})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_base)
    ax.set_xticklabels(mode_order, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Add random baseline (no legend label)
    ax.axhline(0.5, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=10)

    # Aesthetic improvements
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0.45, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f"claude_robustness_avg_ratings_{model_name}.png" if model_name else "claude_robustness_avg_ratings.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved robustness plot: {output_path}")
    plt.close()


def plot_accuracy_vs_reviews_filter_mode(results, output_dir, avg_ratings_mode="true", num_book_reviews=8, num_user_reviews=1, model_name=None):
    """
    Plot accuracy vs reviews_filter_mode to show robustness to different review popularity tiers.

    Args:
        results: List of result dictionaries (from load_results_with_reviews_filter)
        output_dir: Directory to save the plot
        avg_ratings_mode: Filter for specific avg_ratings_mode (default: "true")
        num_book_reviews: Filter for specific num_book_reviews
        num_user_reviews: Filter for specific num_user_reviews
        model_name: Filter for specific model (e.g., "haiku") (None = all)
    """
    # Filter results for specific avg_ratings_mode
    filtered_results = results
    if avg_ratings_mode is not None:
        filtered_results = [r for r in filtered_results if r["avg_ratings_mode"] == avg_ratings_mode]
    if num_book_reviews is not None:
        filtered_results = [r for r in filtered_results if r["num_book_reviews"] == num_book_reviews]
    if num_user_reviews is not None:
        filtered_results = [r for r in filtered_results if r["num_user_reviews"] == num_user_reviews]
    if model_name is not None:
        filtered_results = [r for r in filtered_results if r["model_name"] == model_name]

    if not filtered_results:
        print(f"No results found for reviews_filter plot (avg_ratings={avg_ratings_mode}, books={num_book_reviews}, users={num_user_reviews}, model={model_name})")
        return

    # Group by model_name to create separate series if multiple models
    grouped = defaultdict(list)
    for r in filtered_results:
        grouped[r["model_name"]].append(r)

    # Define filter mode order
    filter_order = ["none", "most_popular", "least_popular"]

    # Square-shaped figure with better aesthetics
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define aesthetic color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    num_filters = len(filter_order)
    num_models = len(grouped)
    bar_width = 0.7 / num_models if num_models > 1 else 0.6

    # Generate x positions for bars
    x_base = np.arange(num_filters)

    for idx, model in enumerate(sorted(grouped.keys())):
        data = grouped[model]
        # Organize by filter mode
        data_by_filter = {}
        for r in data:
            filter_mode = r["reviews_filter_mode"]
            data_by_filter[filter_mode] = r

        y = []
        for filter_mode in filter_order:
            if filter_mode in data_by_filter:
                y.append(data_by_filter[filter_mode]["metrics"]["accuracy"])
            else:
                y.append(0)  # Placeholder if mode missing

        # Calculate x positions
        if num_models > 1:
            x_positions = x_base + (idx - num_models/2 + 0.5) * bar_width
        else:
            x_positions = x_base

        label = f"Claude-{model}"

        bars = ax.bar(x_positions, y, bar_width,
                     label=label,
                     color=colors[idx % len(colors)],
                     alpha=0.85,
                     edgecolor='white',
                     linewidth=1.5)

        # Add value labels on top of bars
        for bar, acc in zip(bars, y):
            if acc > 0:  # Only label non-zero values
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{acc:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Review Filter Mode (by n_votes)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')

    title = f"Robustness: Accuracy vs Review Filter Mode"
    title += f"\n(Book Reviews={num_book_reviews}, User Reviews={num_user_reviews}, Avg Ratings={avg_ratings_mode})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_base)
    ax.set_xticklabels(["prefix (none)", "most popular", "least popular"], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Add random baseline (no legend label)
    ax.axhline(0.5, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=10)

    # Aesthetic improvements
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_ylim(0.45, 0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f"claude_robustness_reviews_filter_{model_name}.png" if model_name else "claude_robustness_reviews_filter.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved reviews filter robustness plot: {output_path}")
    plt.close()


def parse_directory_name_with_adversarial(dir_name):
    """
    Parse directory name to extract all parameters including adversarial_example.

    Expected format:
        model_{MODEL}_num_book_reviews_{X}_num_user_reviews_{Y}_avg_ratings_{MODE}
        _filter_{FILTER}_user_filter_{USER_FILTER}_adversarial_{ADV}

    Returns:
        tuple: (model_name, num_book_reviews, num_user_reviews, avg_ratings_mode,
                reviews_filter_mode, user_reviews_filter_mode, adversarial_example)
               or None if parsing fails
    """
    pattern = (
        r"model_([a-zA-Z0-9_-]+)_num_book_reviews_(\d+)_num_user_reviews_(\d+)"
        r"_avg_ratings_([a-z]+)_filter_(none|most_popular|least_popular)"
        r"_user_filter_([a-z]+)_adversarial_(none|positive|negative)$"
    )
    match = re.match(pattern, dir_name)
    if match:
        return (
            match.group(1), int(match.group(2)), int(match.group(3)),
            match.group(4), match.group(5), match.group(6), match.group(7),
        )
    return None


def load_results_with_adversarial(results_dir):
    """
    Load all metrics.json files from results directory, including adversarial_example variants.

    Returns:
        list: List of dicts with keys: model_name, num_book_reviews, num_user_reviews,
              avg_ratings_mode, reviews_filter_mode, user_reviews_filter_mode,
              adversarial_example, metrics
    """
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return results

    for subdir in results_path.iterdir():
        if not subdir.is_dir():
            continue

        parsed = parse_directory_name_with_adversarial(subdir.name)
        if parsed is None:
            continue

        model_name, num_book_reviews, num_user_reviews, avg_ratings_mode, \
            reviews_filter_mode, user_reviews_filter_mode, adversarial_example = parsed

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
            "avg_ratings_mode": avg_ratings_mode,
            "reviews_filter_mode": reviews_filter_mode,
            "user_reviews_filter_mode": user_reviews_filter_mode,
            "adversarial_example": adversarial_example,
            "metrics": metrics,
        })

    print(f"Loaded {len(results)} result files with adversarial_example variants")
    return results


def plot_accuracy_vs_adversarial_example(results, output_dir, avg_ratings_mode="true",
                                         reviews_filter_mode="none", user_reviews_filter_mode="prefix",
                                         num_book_reviews=8, num_user_reviews=1, model_name=None):
    """
    Plot accuracy as 3 bars: none (baseline), negative, positive adversarial modes.

    Args:
        results: List of result dicts from load_results_with_adversarial
        output_dir: Directory to save the plot
        avg_ratings_mode: Filter for specific avg_ratings_mode (default: "true")
        reviews_filter_mode: Filter for specific reviews_filter_mode (default: "none")
        user_reviews_filter_mode: Filter for specific user_reviews_filter_mode (default: "prefix")
        num_book_reviews: Filter for specific num_book_reviews (default: 8)
        num_user_reviews: Filter for specific num_user_reviews (default: 1)
        model_name: Filter for specific model (e.g., "haiku") (None = all)
    """
    filtered = [r for r in results
                if r["avg_ratings_mode"] == avg_ratings_mode
                and r["reviews_filter_mode"] == reviews_filter_mode
                and r["user_reviews_filter_mode"] == user_reviews_filter_mode
                and r["num_book_reviews"] == num_book_reviews
                and r["num_user_reviews"] == num_user_reviews]
    if model_name is not None:
        filtered = [r for r in filtered if r["model_name"] == model_name]

    if not filtered:
        print(f"No adversarial results found for avg_ratings={avg_ratings_mode}, "
              f"filter={reviews_filter_mode}, user_filter={user_reviews_filter_mode}, "
              f"books={num_book_reviews}, users={num_user_reviews}, model={model_name}")
        return

    adv_order = ["negative", "positive", "none"]
    adv_colors = ["#C73E1D", "#6A994E", "#2E86AB"]

    data_by_adv = {r["adversarial_example"]: r for r in filtered}
    accuracies = [data_by_adv[adv]["metrics"]["accuracy"] if adv in data_by_adv else 0
                  for adv in adv_order]

    x_positions = np.arange(len(adv_order))
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.bar(x_positions, accuracies, bar_width,
                  color=adv_colors,
                  alpha=0.85,
                  edgecolor='white',
                  linewidth=1.5)

    for bar, acc in zip(bars, accuracies):
        if acc > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("Adversarial Example Mode", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')

    title = "Adversarial Attack: Accuracy by Injection Type"
    title += f"\n(Book Reviews={num_book_reviews}, User Reviews={num_user_reviews}, Avg Ratings={avg_ratings_mode})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["negative", "positive", "original"], fontsize=12)
    ax.tick_params(axis='y', labelsize=11)

    ax.axhline(0.5, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=10)

    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)

    all_accs = [r["metrics"]["accuracy"] for r in filtered if r["metrics"]["accuracy"] > 0]
    y_min = max(0.3, min(all_accs) - 0.05) if all_accs else 0.3
    y_max = min(1.0, max(all_accs) + 0.1) if all_accs else 0.8
    ax.set_ylim(y_min, y_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    output_path = os.path.join(output_dir, f"claude_adversarial_example{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved adversarial plot: {output_path}")
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

    # Plot reasoning quality comparison
    print("\nGenerating reasoning quality comparison plot...")
    plot_reasoning_quality(args.output_dir)

    # Plot robustness experiment (if average_ratings results available)
    print("\nChecking for robustness (average_ratings) experiment results...")
    avg_ratings_results = load_results_with_avg_ratings(args.results_dir)
    if len(avg_ratings_results) > 0:
        print(f"Found {len(avg_ratings_results)} robustness experiment runs")
        print("\nGenerating average_ratings robustness plots...")
        plot_accuracy_vs_avg_ratings_mode(avg_ratings_results, args.output_dir,
                                         num_book_reviews=8, num_user_reviews=1)
    else:
        print("No robustness (average_ratings) experiment results found")

    # Plot reviews filter robustness experiment (if results available)
    print("\nChecking for robustness (reviews_filter) experiment results...")
    reviews_filter_results = load_results_with_reviews_filter(args.results_dir)
    if len(reviews_filter_results) > 0:
        print(f"Found {len(reviews_filter_results)} reviews_filter experiment runs")
        print("\nGenerating reviews_filter robustness plots...")
        plot_accuracy_vs_reviews_filter_mode(reviews_filter_results, args.output_dir,
                                            avg_ratings_mode="true", num_book_reviews=8, num_user_reviews=1)
    else:
        print("No robustness (reviews_filter) experiment results found")

    # Plot adversarial example experiment (if results available)
    print("\nChecking for adversarial example experiment results...")
    adversarial_results = load_results_with_adversarial(args.results_dir)
    if len(adversarial_results) > 0:
        print(f"Found {len(adversarial_results)} adversarial experiment runs")
        print("\nGenerating adversarial example plots...")
        plot_accuracy_vs_adversarial_example(adversarial_results, args.output_dir,
                                             avg_ratings_mode="true", reviews_filter_mode="none",
                                             user_reviews_filter_mode="prefix", num_book_reviews=8,
                                             num_user_reviews=1)
    else:
        print("No adversarial example experiment results found")

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
