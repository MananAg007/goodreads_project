"""
Preprocess raw data files by merging reviews with book metadata.
This creates an efficient intermediate file that prepare_dataset.py can read quickly.

Reads:
- Reviews file (CSV or parquet)
- Book metadata (JSON)

Outputs:
- Merged parquet file with reviews + book metadata (title, average_rating, n_votes)
- Filtered metadata cache (parquet) with all original fields including popular_shelves for on-demand genre extraction
"""
import pandas as pd
import argparse
import os


def load_book_metadata(metadata_file, needed_book_ids, cache_file=None):
    """
    Load book metadata efficiently, keeping only needed books.
    Only loads metadata for books that appear in needed_book_ids set.
    Keeps all original fields including popular_shelves for on-demand genre extraction.

    If cache_file is provided and exists, loads from cache instead of JSON.

    Args:
        metadata_file: Path to book metadata JSON file
        needed_book_ids: Set of book_ids to load metadata for
        cache_file: Optional path to save/load filtered metadata parquet

    Returns:
        DataFrame with all original metadata fields for requested books
    """
    # Try loading from cache if provided
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached book metadata from {cache_file}...")
        metadata_df = pd.read_parquet(cache_file)
        print(f"  Loaded {len(metadata_df):,} books from cache")
        return metadata_df

    print(f"Loading book metadata from {metadata_file}...")
    print(f"  Only loading metadata for {len(needed_book_ids):,} books that appear in reviews")

    chunks = []
    chunk_size = 100000
    total_books_scanned = 0
    total_books_kept = 0

    for i, chunk in enumerate(pd.read_json(metadata_file, lines=True, chunksize=chunk_size)):
        total_books_scanned += len(chunk)

        # Filter to only books we need EARLY (before selecting columns)
        chunk = chunk[chunk['book_id'].isin(needed_book_ids)].copy()
        total_books_kept += len(chunk)

        if len(chunk) == 0:
            # Skip empty chunks after filtering
            if (i + 1) % 10 == 0:
                print(f"  Scanned {total_books_scanned:,} books, kept {total_books_kept:,}...")
            continue

        # Keep all original fields including popular_shelves (for genre extraction on-demand)
        chunks.append(chunk)

        if (i + 1) % 10 == 0:
            print(f"  Scanned {total_books_scanned:,} books, kept {total_books_kept:,}...")

    print(f"  Scanned {total_books_scanned:,} total books")
    print(f"  Kept {total_books_kept:,} books that appear in reviews")

    if len(chunks) == 0:
        print("Warning: No matching books found in metadata!")
        return pd.DataFrame()

    metadata_df = pd.concat(chunks, ignore_index=True)

    # Drop duplicates (keep first occurrence)
    metadata_df = metadata_df.drop_duplicates(subset=['book_id'], keep='first')
    print(f"  Unique books after deduplication: {len(metadata_df):,}")

    # Save to cache if provided
    if cache_file:
        print(f"Saving filtered metadata cache to {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        metadata_df.to_parquet(cache_file, index=False, compression='snappy')
        print(f"  Cache saved: {len(metadata_df):,} books with {len(metadata_df.columns)} fields (including popular_shelves for genre extraction)")

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
    # Use cache file for faster iteration on future runs
    cache_file = os.path.join(os.path.dirname(metadata_file), 'book_metadata_filtered.parquet')
    print(f"\nLoading book metadata...")
    metadata_df = load_book_metadata(metadata_file, needed_book_ids, cache_file)

    # Extract genres from popular_shelves
    if 'popular_shelves' in metadata_df.columns:
        print(f"Extracting genres from popular_shelves...")

        EXCLUDE_SHELVES = {
            'to-read', 'currently-reading', 'read', 'did-not-finish',
            'dnf', 'have-it', 'own', 'owned', 'wishlist', 'borrowed',
            're-read', 're-reading', 'rereading', 'favourites', 'favorites'
        }

        def get_top_shelf(shelves):
            """Extract the top genre shelf, excluding reading status shelves."""
            if shelves is None:
                return None
            if isinstance(shelves, list) and len(shelves) == 0:
                return None
            try:
                if isinstance(shelves, list):
                    shelf_list = shelves
                else:
                    shelf_list = list(shelves)

                # Filter out reading status shelves
                genre_shelves = [
                    s for s in shelf_list
                    if isinstance(s, dict) and s.get('name', '').lower() not in EXCLUDE_SHELVES
                ]
                if not genre_shelves:
                    return None
                # Find shelf with highest count
                top_shelf = max(genre_shelves, key=lambda x: int(x.get('count', 0)))
                return top_shelf.get('name')
            except (ValueError, TypeError, AttributeError):
                return None

        metadata_df['genres'] = metadata_df['popular_shelves'].apply(get_top_shelf)
        genres_extracted = metadata_df['genres'].notna().sum()
        print(f"  Extracted genres: {genres_extracted:,} / {len(metadata_df):,}")

    # Select columns for merge (core fields + genres)
    cols_to_keep = ['book_id']
    if 'title' in metadata_df.columns:
        cols_to_keep.append('title')
    if 'average_rating' in metadata_df.columns:
        cols_to_keep.append('average_rating')
    if 'n_votes' in metadata_df.columns:
        cols_to_keep.append('n_votes')
    if 'genres' in metadata_df.columns:
        cols_to_keep.append('genres')

    metadata_df = metadata_df[cols_to_keep]

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
        default='/home/mananaga/goodreads_dataset/subsampled_good_reviews.csv',
        help='Input reviews file (CSV or parquet)'
    )
    parser.add_argument(
        '--metadata',
        default='/home/mananaga/goodreads_dataset/goodreads_books.json',
        help='Book metadata JSON file'
    )
    parser.add_argument(
        '--output',
        default='/home/mananaga/goodreads_dataset/processed_dataset.parquet',
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
