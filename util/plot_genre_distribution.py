"""
Plot genre distribution from the processed dataset.

Reads the processed parquet (output of process_data.py), deduplicates by book_id,
and plots a histogram of genre frequencies in decreasing order.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_genre_counts(input_file, top_n=25):
    """
    Load genre counts from the processed parquet, counting each unique book once.

    Args:
        input_file: Path to processed parquet file
        top_n: Number of top genres to return (None = all)

    Returns:
        List of (genre, count) tuples sorted by count descending
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file, columns=["book_id", "genres"])
    print(f"  Loaded {len(df):,} rows, {df['book_id'].nunique():,} unique books")

    # One genre per book
    books = df.drop_duplicates(subset=["book_id"])
    total_books = len(books)
    books_with_genre = books["genres"].notna().sum()
    print(f"  Books with genre: {books_with_genre:,} / {total_books:,}")

    counts = books["genres"].dropna().value_counts()

    # Print top-5 fraction before truncating
    top5_count = counts.head(5).sum()
    top5_fraction = top5_count / books_with_genre
    print(f"  Top-5 genres cover {top5_count:,} / {books_with_genre:,} books "
          f"({top5_fraction:.1%} of books with a genre)")
    print(f"  Top-5 genres: {list(counts.head(5).index)}")

    if top_n is not None:
        counts = counts.head(top_n)

    return list(zip(counts.index, counts.values)), total_books


def plot_genre_distribution(input_file, output_dir, top_n=25):
    """
    Plot a bar chart of genre frequencies in decreasing order.

    Args:
        input_file: Path to processed parquet file
        output_dir: Directory to save the plot
        top_n: Number of top genres to show
    """
    genre_counts, total_books = load_genre_counts(input_file, top_n=top_n)

    if not genre_counts:
        print("No genre data found.")
        return

    genres = [g for g, _ in genre_counts]
    counts = [c for _, c in genre_counts]

    # Cycle through aesthetic color palette
    palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8E7DBE']
    colors = [palette[i % len(palette)] for i in range(len(genres))]

    fig, ax = plt.subplots(figsize=(14, 8))

    x_positions = np.arange(len(genres))
    bars = ax.bar(x_positions, counts,
                  color=colors,
                  alpha=0.85,
                  edgecolor='white',
                  linewidth=1.5)

    # Value labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.005,
                f'{count:,}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    ax.set_xlabel("Genre", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Books", fontsize=13, fontweight='bold')
    ax.set_title(f"Genre Distribution (Top {len(genres)} Genres)\n"
                 f"({total_books:,} unique books)",
                 fontsize=15, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(genres, fontsize=10, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=11)

    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "genre_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot genre distribution from the processed dataset"
    )
    parser.add_argument(
        "--input",
        default="/home/mananaga/goodreads_dataset/processed_dataset.parquet",
        help="Processed parquet file (output of process_data.py)"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mananaga/goodreads/plots",
        help="Directory to save the plot"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Number of top genres to show (default: 25)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_genre_distribution(args.input, args.output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
