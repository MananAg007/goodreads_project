"""
Filter Goodreads reviews to only include specified book series.
"""
import pandas as pd
import argparse
import json
from pathlib import Path


def load_book_metadata(metadata_file, series_names=None):
    """
    Load book metadata that maps book_id to title and series information.
    Optimized for large files by reading in chunks and filtering early.

    Args:
        metadata_file: Path to book metadata JSON file
        series_names: Optional list of series to filter for (speeds up loading)

    Returns:
        DataFrame with book_id, title, and series information
    """
    print(f"Loading book metadata from {metadata_file}...")

    if metadata_file.endswith('.csv'):
        books_df = pd.read_csv(metadata_file)
    elif metadata_file.endswith('.parquet'):
        books_df = pd.read_parquet(metadata_file)
    elif metadata_file.endswith('.json'):
        # For JSON, use chunked reading for efficiency
        print("Using chunked reading for large JSON file...")

        chunks = []
        chunk_size = 100000
        total_books = 0

        for i, chunk in enumerate(pd.read_json(metadata_file, lines=True, chunksize=chunk_size)):
            total_books += len(chunk)

            # If we're filtering for specific series, do it early to save memory
            if series_names:
                # Check both series and title columns
                mask = pd.Series([False] * len(chunk), index=chunk.index)
                for col in ['series', 'title']:
                    if col in chunk.columns:
                        for series_name in series_names:
                            mask |= chunk[col].astype(str).str.contains(
                                series_name, case=False, na=False
                            )

                chunk = chunk[mask]

            # Only keep columns we need to save memory (after filtering)
            cols_to_keep = ['book_id']
            if 'title' in chunk.columns:
                cols_to_keep.append('title')
            if 'series' in chunk.columns:
                cols_to_keep.append('series')

            chunk = chunk[cols_to_keep]

            if len(chunk) > 0:
                chunks.append(chunk)

            if (i + 1) % 10 == 0:
                print(f"  Processed {total_books:,} books...")

        if len(chunks) == 0:
            print("Warning: No matching books found!")
            return pd.DataFrame(columns=['book_id', 'title'])

        books_df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(books_df):,} relevant books from {total_books:,} total")
    else:
        raise ValueError(f"Unsupported file format: {metadata_file}")

    return books_df


def find_books_in_series(books_df, series_names, series_column='title'):
    """
    Find all book IDs that belong to the specified series.

    Args:
        books_df: DataFrame with book metadata
        series_names: List of series names to filter for
        series_column: Name of column containing series information (default: title)

    Returns:
        Set of book IDs in the specified series
    """
    if series_column not in books_df.columns:
        # Try to find series info in title
        print(f"Warning: '{series_column}' column not found. Trying 'title'...")
        series_column = 'title'

    book_ids = set()

    for series_name in series_names:
        print(f"\nSearching for series: {series_name}")

        # Case-insensitive search
        mask = books_df[series_column].astype(str).str.contains(
            series_name, case=False, na=False
        )

        series_books = books_df[mask]
        series_book_ids = set(series_books['book_id'].values)

        print(f"  Found {len(series_book_ids):,} books")
        if len(series_book_ids) > 0 and len(series_book_ids) <= 20:
            # Show titles if not too many
            if 'title' in series_books.columns:
                print(f"  Sample titles: {list(series_books['title'].head(5).values)}")

        book_ids.update(series_book_ids)

    return book_ids


def filter_reviews_by_books(reviews_file, book_ids, books_df, output_file):
    """
    Filter reviews to only include specified books and merge with book titles.

    Args:
        reviews_file: Path to reviews file (CSV or parquet)
        book_ids: Set of book IDs to keep
        books_df: DataFrame with book metadata (must have book_id and title columns)
        output_file: Path to save filtered reviews
    """
    print(f"\nLoading reviews from {reviews_file}...")

    # Load reviews
    if reviews_file.endswith('.csv'):
        reviews_df = pd.read_csv(reviews_file)
    elif reviews_file.endswith('.parquet'):
        reviews_df = pd.read_parquet(reviews_file)
    else:
        raise ValueError(f"Unsupported file format: {reviews_file}")

    print(f"Loaded {len(reviews_df):,} reviews")

    # Filter reviews
    print(f"Filtering to {len(book_ids):,} books...")
    filtered_df = reviews_df[reviews_df['book_id'].isin(book_ids)]

    print(f"Kept {len(filtered_df):,} reviews ({len(filtered_df)/len(reviews_df)*100:.1f}%)")
    print(f"Unique users: {filtered_df['user_id'].nunique():,}")
    print(f"Unique books: {filtered_df['book_id'].nunique():,}")

    # Merge with book titles
    if 'title' in books_df.columns:
        print("\nMerging book titles...")
        # Only keep book_id and title from books_df
        books_info = books_df[['book_id', 'title']].drop_duplicates()
        filtered_df = filtered_df.merge(books_info, on='book_id', how='left')
        print(f"Added 'title' column to reviews")

    # Save filtered reviews
    print(f"\nSaving to {output_file}...")
    if output_file.endswith('.parquet'):
        filtered_df.to_parquet(output_file, index=False)
    else:
        filtered_df.to_csv(output_file, index=False)

    print("Done!")
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter Goodreads reviews to specific book series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (Harry Potter and Percy Jackson)
  python filter_series.py

  # Filter using custom series names
  python filter_series.py --series "Twilight" "Hunger Games"

  # Filter using book IDs directly
  python filter_series.py --book-ids 1 2 3 4 5

  # Fully customized
  python filter_series.py --series "Harry Potter" \\
      --metadata /path/to/books.json \\
      --input reviews.parquet \\
      --output filtered_reviews.parquet
        """
    )

    # Series filtering
    parser.add_argument(
        '--series',
        nargs='+',
        default=["Harry Potter", "Percy Jackson"],
        help='Series names to filter for (default: ["Harry Potter", "Percy Jackson"])'
    )
    parser.add_argument(
        '--metadata',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/goodreads_books.json',
        help='Path to book metadata file (JSON, CSV, or parquet)'
    )
    parser.add_argument(
        '--series-column',
        default='title',
        help='Column name containing series information (default: title)'
    )

    # Direct book ID filtering
    parser.add_argument(
        '--book-ids',
        nargs='+',
        type=int,
        help='Book IDs to filter for (alternative to --series)'
    )

    # Input/output
    parser.add_argument(
        '--input',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/subsampled_good_reviews.csv',
        help='Input reviews file (CSV or parquet)'
    )
    parser.add_argument(
        '--output',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/series_filtered_reviews.parquet',
        help='Output file path (CSV or parquet)'
    )

    args = parser.parse_args()

    # Determine book IDs to filter
    if args.book_ids:
        # Direct book IDs provided
        book_ids = set(args.book_ids)
        print(f"Filtering to {len(book_ids)} specified book IDs")
        # Load minimal book metadata for titles
        books_df = load_book_metadata(args.metadata)
    else:
        # Use series (either default or user-specified)
        print(f"Filtering by series: {args.series}")

        # Pass series names to optimize loading for large files
        books_df = load_book_metadata(args.metadata, series_names=args.series)
        book_ids = find_books_in_series(books_df, args.series, args.series_column)

        if len(book_ids) == 0:
            print("Error: No books found for specified series!")
            return

    # Filter reviews and merge with book titles
    filter_reviews_by_books(args.input, book_ids, books_df, args.output)


if __name__ == "__main__":
    main()
