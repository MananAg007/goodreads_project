"""
End-to-end dataset preparation for LLM-based book preference prediction.
Samples random books instead of filtering for specific series.

For each user who has reviewed at least 3 books:
- Select 1 reference book with the user's review
- Select 2 other books (A and B) that the user has rated DIFFERENTLY (no ties)
- Get sample reviews from other users for each of books A and B
- Create JSONL entry with all this information

The LLM task: Given a user's reference review and sample reviews of books A and B,
predict which book the user will prefer (higher rating).

Output includes:
- Reference book: title, user's rating, review text, average_rating
- Book A: title, user's rating, average_rating, sample reviews
- Book B: title, user's rating, average_rating, sample reviews
- Preferred book (A or B)
- Rating difference

Notes:
- Only reviews with ratings 1-5 are included (rating 0 is filtered out)
- Only samples where the user rated book A and B differently are included
- Average rating from Goodreads metadata included if available
- Books are sampled randomly for diversity
"""
import pandas as pd
import argparse
import json
import random
from collections import defaultdict
from typing import List, Dict, Optional


def load_book_metadata(metadata_file):
    """
    Load book metadata that maps book_id to title and average_rating.

    Args:
        metadata_file: Path to book metadata JSON file

    Returns:
        Dictionary mapping book_id to {title, average_rating}
    """
    print(f"Loading book metadata from {metadata_file}...")

    book_metadata = {}
    chunk_size = 100000
    total_books = 0

    for i, chunk in enumerate(pd.read_json(metadata_file, lines=True, chunksize=chunk_size)):
        total_books += len(chunk)

        # Extract needed columns
        for _, row in chunk.iterrows():
            book_id = row.get('book_id')
            if pd.isna(book_id):
                continue

            book_metadata[int(book_id)] = {
                'title': row.get('title', 'Unknown Title'),
                'average_rating': row.get('average_rating')
            }

        if (i + 1) % 10 == 0:
            print(f"  Processed {total_books:,} books...")

    print(f"Loaded metadata for {len(book_metadata):,} books")
    return book_metadata


def get_reviews_for_book(df: pd.DataFrame, book_id: int, exclude_user_id: str,
                         num_reviews: int = 10, prefer_with_text: bool = True) -> List[Dict]:
    """
    Get sample reviews for a book, excluding a specific user.

    Args:
        df: DataFrame with all reviews
        book_id: Book to get reviews for
        exclude_user_id: User to exclude from samples
        num_reviews: Number of reviews to sample
        prefer_with_text: If True, prefer reviews with text

    Returns:
        List of review dictionaries
    """
    # Get all reviews for this book, excluding the target user
    book_reviews = df[(df['book_id'] == book_id) & (df['user_id'] != exclude_user_id)]

    if len(book_reviews) == 0:
        return []

    # Prefer reviews with text if available
    if prefer_with_text:
        reviews_with_text = book_reviews[
            book_reviews['review_text'].notna() &
            (book_reviews['review_text'].str.strip() != '')
        ]

        if len(reviews_with_text) >= num_reviews:
            sample_reviews = reviews_with_text.sample(n=num_reviews, random_state=42)
        elif len(reviews_with_text) > 0:
            # Get all with text, then fill with reviews without text
            sample_reviews = pd.concat([
                reviews_with_text,
                book_reviews[~book_reviews.index.isin(reviews_with_text.index)].sample(
                    n=min(num_reviews - len(reviews_with_text),
                          len(book_reviews) - len(reviews_with_text)),
                    random_state=42
                )
            ])
        else:
            # No reviews with text, just sample
            sample_reviews = book_reviews.sample(n=min(num_reviews, len(book_reviews)), random_state=42)
    else:
        sample_reviews = book_reviews.sample(n=min(num_reviews, len(book_reviews)), random_state=42)

    # Convert to list of dicts
    reviews_list = []
    for _, review in sample_reviews.iterrows():
        review_dict = {
            'rating': int(review['rating']),
            'review_text': review['review_text'] if pd.notna(review['review_text']) else ""
        }
        reviews_list.append(review_dict)

    return reviews_list


def create_dataset(reviews_file: str, metadata_file: str, output_file: str,
                   dataset_size: int = 100, min_books_per_user: int = 3,
                   num_reviews_per_book: int = 10, min_reviews_per_book: int = 10,
                   seed: int = 42):
    """
    Create LLM training dataset from book reviews with random book sampling.
    Creates one dataset point per user.

    Args:
        reviews_file: Path to input reviews file (CSV or parquet)
        metadata_file: Path to book metadata JSON file
        output_file: Path to save JSONL output
        dataset_size: Total number of dataset points to create (default 100)
        min_books_per_user: Minimum books a user must have reviewed (default 3)
        num_reviews_per_book: Number of sample reviews to include per book
        min_reviews_per_book: Minimum reviews a book must have to be eligible
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load book metadata
    book_metadata = load_book_metadata(metadata_file)

    # Load reviews
    print(f"\nLoading reviews from {reviews_file}...")
    if reviews_file.endswith('.csv'):
        df = pd.read_csv(reviews_file)
    elif reviews_file.endswith('.parquet'):
        df = pd.read_parquet(reviews_file)
    else:
        raise ValueError(f"Unsupported file format: {reviews_file}")

    print(f"Loaded {len(df):,} reviews")

    # Filter to only include reviews with ratings 1-5 (skip 0 ratings)
    print(f"\nFiltering to ratings 1-5 only...")
    print(f"  Reviews before filtering: {len(df):,}")
    df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
    print(f"  Reviews after filtering: {len(df):,}")

    # Add book titles and average ratings from metadata
    print(f"\nMerging book metadata...")
    df['title'] = df['book_id'].map(lambda x: book_metadata.get(x, {}).get('title', 'Unknown Title'))
    df['average_rating'] = df['book_id'].map(lambda x: book_metadata.get(x, {}).get('average_rating'))

    print(f"\nDataset statistics:")
    print(f"  Unique users: {df['user_id'].nunique():,}")
    print(f"  Unique books: {df['book_id'].nunique():,}")

    # Filter books that have enough reviews to sample from
    print(f"\nFiltering books with at least {min_reviews_per_book} reviews...")
    book_review_counts = df['book_id'].value_counts()
    eligible_books = set(book_review_counts[book_review_counts >= min_reviews_per_book].index)
    print(f"  Found {len(eligible_books):,} books with sufficient reviews")

    # Filter to only reviews of eligible books
    df = df[df['book_id'].isin(eligible_books)]
    print(f"  Kept {len(df):,} reviews")

    # Group by user to find users with enough books
    print(f"\nFinding users with at least {min_books_per_user} eligible books...")
    user_books = df.groupby('user_id')['book_id'].nunique()
    eligible_users = user_books[user_books >= min_books_per_user].index.tolist()
    print(f"  Found {len(eligible_users):,} eligible users")

    # Shuffle users for random selection
    random.shuffle(eligible_users)

    # Generate training samples (1 per user)
    print(f"\nGenerating dataset...")
    print(f"  Target: {dataset_size} data points (1 per user)")

    dataset = []
    users_processed = 0
    users_skipped = 0

    for user_id in eligible_users:
        # Stop if we've reached the target dataset size
        if len(dataset) >= dataset_size:
            break

        user_reviews = df[df['user_id'] == user_id]
        user_book_ids = user_reviews['book_id'].unique().tolist()

        if len(user_book_ids) < min_books_per_user:
            users_skipped += 1
            continue

        # Get reviews with text for reference
        reviews_with_text = user_reviews[
            user_reviews['review_text'].notna() &
            (user_reviews['review_text'].str.strip() != '')
        ]

        # Try to create one sample for this user
        max_attempts = 10  # Avoid infinite loops
        attempts = 0
        success = False

        while not success and attempts < max_attempts:
            attempts += 1

            # Select reference book (prefer one with review text)
            if len(reviews_with_text) > 0:
                ref_book_candidates = reviews_with_text['book_id'].unique().tolist()
            else:
                ref_book_candidates = user_book_ids

            if len(ref_book_candidates) < 1:
                break

            ref_book_id = random.choice(ref_book_candidates)

            # Get remaining books for A and B
            remaining_books = [b for b in user_book_ids if b != ref_book_id]
            if len(remaining_books) < 2:
                break

            # Randomly select books A and B
            book_a_id, book_b_id = random.sample(remaining_books, 2)

            # Get user's reference review
            ref_review_row = user_reviews[user_reviews['book_id'] == ref_book_id].iloc[0]
            ref_review_text = ref_review_row['review_text'] if pd.notna(ref_review_row['review_text']) else ""
            ref_rating = int(ref_review_row['rating'])
            ref_title = ref_review_row['title']
            ref_avg = ref_review_row['average_rating']

            # Get user's ratings for A and B
            rating_a = int(user_reviews[user_reviews['book_id'] == book_a_id]['rating'].iloc[0])
            rating_b = int(user_reviews[user_reviews['book_id'] == book_b_id]['rating'].iloc[0])

            # Skip if ratings are equal (we only want clear preferences)
            if rating_a == rating_b:
                continue

            # Get book info for A and B
            book_a_row = user_reviews[user_reviews['book_id'] == book_a_id].iloc[0]
            book_b_row = user_reviews[user_reviews['book_id'] == book_b_id].iloc[0]
            title_a = book_a_row['title']
            title_b = book_b_row['title']
            avg_a = book_a_row['average_rating']
            avg_b = book_b_row['average_rating']

            # Get sample reviews for books A and B (from other users)
            reviews_a = get_reviews_for_book(df, book_a_id, user_id, num_reviews_per_book)
            reviews_b = get_reviews_for_book(df, book_b_id, user_id, num_reviews_per_book)

            # Skip if we don't have enough reviews
            if len(reviews_a) < num_reviews_per_book or len(reviews_b) < num_reviews_per_book:
                continue

            # Determine which book the user preferred
            if rating_a > rating_b:
                preferred = "A"
            else:  # rating_b > rating_a
                preferred = "B"

            # Create dataset entry
            # Handle NaN average ratings
            ref_avg_val = float(ref_avg) if pd.notna(ref_avg) else None
            a_avg_val = float(avg_a) if pd.notna(avg_a) else None
            b_avg_val = float(avg_b) if pd.notna(avg_b) else None

            entry = {
                'user_id': user_id,
                'reference_book': {
                    'book_id': int(ref_book_id),
                    'title': ref_title,
                    'rating': ref_rating,
                    'review_text': ref_review_text,
                    'average_rating': ref_avg_val
                },
                'book_a': {
                    'book_id': int(book_a_id),
                    'title': title_a,
                    'user_rating': rating_a,
                    'average_rating': a_avg_val,
                    'sample_reviews': reviews_a
                },
                'book_b': {
                    'book_id': int(book_b_id),
                    'title': title_b,
                    'user_rating': rating_b,
                    'average_rating': b_avg_val,
                    'sample_reviews': reviews_b
                },
                'preferred': preferred,
                'rating_difference': abs(rating_a - rating_b)
            }

            dataset.append(entry)
            success = True

        if success:
            users_processed += 1
        else:
            users_skipped += 1

        if users_processed % 50 == 0 and users_processed > 0:
            print(f"  Generated {len(dataset):,}/{dataset_size} data points")

    print(f"\nDataset generation complete!")
    print(f"  Total users processed: {users_processed:,}")
    print(f"  Total users skipped: {users_skipped:,}")
    print(f"  Total data points generated: {len(dataset):,}")

    # Analyze preference distribution
    preference_counts = defaultdict(int)
    for entry in dataset:
        preference_counts[entry['preferred']] += 1

    print(f"\nPreference distribution (ties excluded):")
    print(f"  Preferred A: {preference_counts['A']:,} ({preference_counts['A']/len(dataset)*100:.1f}%)")
    print(f"  Preferred B: {preference_counts['B']:,} ({preference_counts['B']/len(dataset)*100:.1f}%)")

    # Analyze rating difference distribution
    rating_diffs = [entry['rating_difference'] for entry in dataset]
    print(f"\nRating difference distribution:")
    for diff in sorted(set(rating_diffs)):
        count = rating_diffs.count(diff)
        print(f"  Difference {diff}: {count:,} ({count/len(rating_diffs)*100:.1f}%)")

    # Analyze book diversity
    unique_books = set()
    for entry in dataset:
        unique_books.add(entry['reference_book']['book_id'])
        unique_books.add(entry['book_a']['book_id'])
        unique_books.add(entry['book_b']['book_id'])
    print(f"\nBook diversity:")
    print(f"  Unique books in dataset: {len(unique_books):,}")

    # Save to JSONL
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Done! Saved {len(dataset):,} samples to {output_file}")

    # Save a human-readable sample
    if len(dataset) > 0:
        sample_file = output_file.replace('.jsonl', '_sample.json')
        print(f"\nSaving sample entry to {sample_file}...")
        with open(sample_file, 'w') as f:
            json.dump(dataset[0], f, indent=2)
        print("Sample saved for inspection.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for LLM book preference prediction with random book sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python prepare_dataset.py

  # Custom settings
  python prepare_dataset.py \\
      --reviews /path/to/reviews.csv \\
      --metadata /path/to/books.json \\
      --output ./book_preference_dataset.jsonl \\
      --dataset-size 100 \\
      --min-books 3 \\
      --reviews-per-book 10
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
        default='./book_preference_dataset.jsonl',
        help='Output JSONL file for LLM training'
    )
    parser.add_argument(
        '--dataset-size',
        type=int,
        default=100,
        help='Total number of data points to create (default: 100)'
    )
    parser.add_argument(
        '--min-books',
        type=int,
        default=3,
        help='Minimum books a user must have reviewed (default: 3)'
    )
    parser.add_argument(
        '--reviews-per-book',
        type=int,
        default=10,
        help='Number of sample reviews per book (default: 10)'
    )
    parser.add_argument(
        '--min-reviews-per-book',
        type=int,
        default=10,
        help='Minimum reviews a book must have to be eligible (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    create_dataset(
        reviews_file=args.reviews,
        metadata_file=args.metadata,
        output_file=args.output,
        dataset_size=args.dataset_size,
        min_books_per_user=args.min_books,
        num_reviews_per_book=args.reviews_per_book,
        min_reviews_per_book=args.min_reviews_per_book,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
