"""
Preprocess raw data files by merging reviews with book metadata.
This creates an efficient intermediate file that prepare_dataset.py can read quickly.

Reads:
- Reviews file (CSV or parquet)
- Book metadata (JSON)

Outputs:
- Merged parquet file with reviews + book metadata (title, average_rating)
"""
import pandas as pd
import argparse


def load_book_metadata(metadata_file, needed_book_ids):
    """
    Load book metadata efficiently, keeping only needed columns and books.
    Only loads metadata for books that appear in needed_book_ids set.

    Args:
        metadata_file: Path to book metadata JSON file
        needed_book_ids: Set of book_ids to load metadata for

    Returns:
        DataFrame with book_id, title, average_rating for requested books only
    """
    print(f"Loading book metadata from {metadata_file}...")
    print(f"  Only loading metadata for {len(needed_book_ids):,} books that appear in reviews")

    chunks = []
    chunk_size = 100000
    total_books_scanned = 0
    total_books_kept = 0

    for i, chunk in enumerate(pd.read_json(metadata_file, lines=True, chunksize=chunk_size)):
        total_books_scanned += len(chunk)

        # Filter to only books we need EARLY (before selecting columns)
        chunk = chunk[chunk['book_id'].isin(needed_book_ids)]
        total_books_kept += len(chunk)

        if len(chunk) == 0:
            # Skip empty chunks after filtering
            if (i + 1) % 10 == 0:
                print(f"  Scanned {total_books_scanned:,} books, kept {total_books_kept:,}...")
            continue

        # Only keep columns we need to save memory
        cols_to_keep = ['book_id']
        if 'title' in chunk.columns:
            cols_to_keep.append('title')
        if 'average_rating' in chunk.columns:
            cols_to_keep.append('average_rating')
        if 'n_votes' in chunk.columns:
            cols_to_keep.append('n_votes')

        chunk = chunk[cols_to_keep]
        chunks.append(chunk)

        if (i + 1) % 10 == 0:
            print(f"  Scanned {total_books_scanned:,} books, kept {total_books_kept:,}...")

    print(f"  Scanned {total_books_scanned:,} total books")
    print(f"  Kept {total_books_kept:,} books that appear in reviews")

    if len(chunks) == 0:
        print("Warning: No matching books found in metadata!")
        return pd.DataFrame(columns=['book_id', 'title', 'average_rating', 'n_votes'])

    metadata_df = pd.concat(chunks, ignore_index=True)

    # Drop duplicates (keep first occurrence)
    metadata_df = metadata_df.drop_duplicates(subset=['book_id'], keep='first')
    print(f"  Unique books after deduplication: {len(metadata_df):,}")

    return metadata_df


def process_data(reviews_file, metadata_file, output_file):
    """
    Load reviews and metadata, merge them, and save to efficient format.
    Optimized to only load metadata for books that appear in reviews.

    Args:
        reviews_file: Path to reviews file (CSV or parquet)
        metadata_file: Path to book metadata JSON file
        output_file: Path to save processed parquet file
    """
    # Load reviews FIRST to get the set of book_ids we need
    print(f"Loading reviews from {reviews_file}...")
    if reviews_file.endswith('.csv'):
        reviews_df = pd.read_csv(reviews_file)
    elif reviews_file.endswith('.parquet'):
        reviews_df = pd.read_parquet(reviews_file)
    else:
        raise ValueError(f"Unsupported file format: {reviews_file}")

    print(f"Loaded {len(reviews_df):,} reviews")
    print(f"  Unique users: {reviews_df['user_id'].nunique():,}")
    print(f"  Unique books: {reviews_df['book_id'].nunique():,}")

    # Get set of book_ids we need metadata for
    needed_book_ids = set(reviews_df['book_id'].unique())
    print(f"  Need metadata for {len(needed_book_ids):,} unique books")

    # Load book metadata ONLY for books in reviews (much faster!)
    print(f"\nLoading book metadata...")
    metadata_df = load_book_metadata(metadata_file, needed_book_ids)

    # Filter to ratings 1-5 only (exclude 0 ratings)
    print(f"\nFiltering to ratings 1-5...")
    original_count = len(reviews_df)
    reviews_df = reviews_df[(reviews_df['rating'] >= 1) & (reviews_df['rating'] <= 5)]
    filtered_count = original_count - len(reviews_df)
    print(f"  Removed {filtered_count:,} reviews with rating 0")
    print(f"  Kept {len(reviews_df):,} reviews")

    # Merge metadata with reviews
    print(f"\nMerging book metadata with reviews...")
    merged_df = reviews_df.merge(
        metadata_df,
        on='book_id',
        how='left'
    )

    print(f"  Merged shape: {merged_df.shape}")
    print(f"  Columns: {list(merged_df.columns)}")

    # Check for missing metadata
    missing_title = merged_df['title'].isna().sum()
    missing_avg_rating = merged_df['average_rating'].isna().sum() if 'average_rating' in merged_df.columns else 0

    print(f"\nMetadata coverage:")
    print(f"  Reviews with title: {len(merged_df) - missing_title:,} / {len(merged_df):,}")
    print(f"  Reviews with avg rating: {len(merged_df) - missing_avg_rating:,} / {len(merged_df):,}")

    # Fill missing titles with placeholder
    if missing_title > 0:
        merged_df['title'] = merged_df['title'].fillna('Unknown Title')

    # Save to parquet
    print(f"\nSaving to {output_file}...")
    merged_df.to_parquet(output_file, index=False, compression='snappy')

    # Verify saved file
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Saved {len(merged_df):,} reviews")

    print("\nDone! You can now use this file with prepare_dataset.py")
    print(f"  python util/prepare_dataset.py --reviews {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess reviews and metadata into efficient format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python process_data.py

  # Custom paths
  python process_data.py \\
      --reviews /path/to/reviews.csv \\
      --metadata /path/to/books.json \\
      --output ./processed_reviews.parquet
        """
    )

    parser.add_argument(
        '--reviews',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/subsampled_good_reviews.csv',
        help='Input reviews file (CSV or parquet)'
    )
    parser.add_argument(
        '--metadata',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/goodreads_books.json',
        help='Book metadata JSON file'
    )
    parser.add_argument(
        '--output',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/processed_reviews.parquet',
        help='Output parquet file'
    )

    args = parser.parse_args()

    process_data(
        reviews_file=args.reviews,
        metadata_file=args.metadata,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
