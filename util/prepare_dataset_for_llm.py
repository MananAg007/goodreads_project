"""
Prepare dataset for LLM-based book preference prediction.

For each user who has reviewed at least 3 books:
- Select 1 reference book with the user's review
- Select 2 other books (A and B) that the user has rated DIFFERENTLY (no ties)
- Get 10 reviews from other users for each of books A and B
- Create JSONL entry with all this information

The LLM task: Given a user's reference review and 10 reviews each of books A and B,
predict which book the user will prefer (higher rating).

Output includes:
- Reference book: title, user's rating, review text, average_rating
- Book A: title, user's rating, average_rating, 10 sample reviews
- Book B: title, user's rating, average_rating, 10 sample reviews

Notes:
- Only reviews with ratings 1-5 are included (rating 0 is filtered out)
- Only samples where the user rated book A and B differently are included
- Average rating from Goodreads metadata included if available
"""
import pandas as pd
import argparse
import json
import random
from collections import defaultdict
from typing import List, Dict, Optional


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


def create_dataset(input_file: str, output_file: str, min_books_per_user: int = 3,
                   num_samples_per_user: int = 5, num_reviews_per_book: int = 10,
                   seed: int = 42):
    """
    Create LLM training dataset from book reviews.

    Args:
        input_file: Path to input parquet file with core book reviews
        output_file: Path to save JSONL output
        min_books_per_user: Minimum books a user must have reviewed (default 3)
        num_samples_per_user: Number of training samples to create per user
        num_reviews_per_book: Number of sample reviews to include per book
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print(f"Loading reviews from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} reviews")

    # Filter to only include reviews with ratings 1-5 (skip 0 ratings)
    print(f"\nFiltering to ratings 1-5 only...")
    print(f"  Reviews before filtering: {len(df):,}")
    print(f"  Rating distribution before filtering:")
    for rating in sorted(df['rating'].unique()):
        count = len(df[df['rating'] == rating])
        print(f"    Rating {rating}: {count:,}")

    df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
    print(f"  Reviews after filtering: {len(df):,}")
    print(f"  Removed {len(pd.read_parquet(input_file)) - len(df):,} reviews with rating 0")

    print(f"\nDataset statistics:")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique books: {df['book_id'].nunique():,}")

    # Group by user to find users with enough books
    print(f"\nFinding users with at least {min_books_per_user} books...")
    user_books = df.groupby('user_id')['book_id'].nunique()
    eligible_users = user_books[user_books >= min_books_per_user].index.tolist()

    print(f"Found {len(eligible_users):,} eligible users")

    # Create book title and average rating mappings
    book_id_to_title = df[['book_id', 'title']].drop_duplicates().set_index('book_id')['title'].to_dict()

    # Check if average_rating column exists
    if 'average_rating' in df.columns:
        # Convert to numeric in case it's stored as string
        df_ratings = df[['book_id', 'average_rating']].drop_duplicates().copy()
        df_ratings['average_rating'] = pd.to_numeric(df_ratings['average_rating'], errors='coerce')
        book_id_to_avg_rating = df_ratings.set_index('book_id')['average_rating'].to_dict()
        print("  Average rating data available")
    else:
        book_id_to_avg_rating = {}
        print("  Warning: No average_rating column found in data")

    # Generate training samples
    print(f"\nGenerating training samples...")
    print(f"Target: {num_samples_per_user} samples per user")

    dataset = []
    users_processed = 0
    users_skipped = 0

    for user_id in eligible_users:
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

        # Try to create samples for this user
        user_samples = 0
        max_attempts = num_samples_per_user * 3  # Avoid infinite loops
        attempts = 0

        while user_samples < num_samples_per_user and attempts < max_attempts:
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

            # Get user's ratings for A and B
            rating_a = int(user_reviews[user_reviews['book_id'] == book_a_id]['rating'].iloc[0])
            rating_b = int(user_reviews[user_reviews['book_id'] == book_b_id]['rating'].iloc[0])

            # Get sample reviews for books A and B (from other users)
            reviews_a = get_reviews_for_book(df, book_a_id, user_id, num_reviews_per_book)
            reviews_b = get_reviews_for_book(df, book_b_id, user_id, num_reviews_per_book)

            # Skip if we don't have enough reviews
            if len(reviews_a) < num_reviews_per_book or len(reviews_b) < num_reviews_per_book:
                continue

            # Determine which book the user preferred
            # Skip if ratings are equal (we only want clear preferences)
            if rating_a == rating_b:
                continue
            elif rating_a > rating_b:
                preferred = "A"
            else:  # rating_b > rating_a
                preferred = "B"

            # Create dataset entry
            # Get average ratings, handling NaN values
            ref_avg = book_id_to_avg_rating.get(ref_book_id)
            ref_avg = float(ref_avg) if ref_avg is not None and pd.notna(ref_avg) else None

            a_avg = book_id_to_avg_rating.get(book_a_id)
            a_avg = float(a_avg) if a_avg is not None and pd.notna(a_avg) else None

            b_avg = book_id_to_avg_rating.get(book_b_id)
            b_avg = float(b_avg) if b_avg is not None and pd.notna(b_avg) else None

            entry = {
                'user_id': user_id,
                'reference_book': {
                    'book_id': int(ref_book_id),
                    'title': book_id_to_title[ref_book_id],
                    'rating': ref_rating,
                    'review_text': ref_review_text,
                    'average_rating': ref_avg
                },
                'book_a': {
                    'book_id': int(book_a_id),
                    'title': book_id_to_title[book_a_id],
                    'user_rating': rating_a,
                    'average_rating': a_avg,
                    'sample_reviews': reviews_a
                },
                'book_b': {
                    'book_id': int(book_b_id),
                    'title': book_id_to_title[book_b_id],
                    'user_rating': rating_b,
                    'average_rating': b_avg,
                    'sample_reviews': reviews_b
                },
                'preferred': preferred,
                'rating_difference': abs(rating_a - rating_b)
            }

            dataset.append(entry)
            user_samples += 1

        if user_samples > 0:
            users_processed += 1

        if users_processed % 100 == 0:
            print(f"  Processed {users_processed:,} users, generated {len(dataset):,} samples")

    print(f"\nDataset generation complete!")
    print(f"  Total users processed: {users_processed:,}")
    print(f"  Total samples generated: {len(dataset):,}")
    print(f"  Average samples per user: {len(dataset)/users_processed:.2f}")

    # Analyze preference distribution
    preference_counts = defaultdict(int)
    for entry in dataset:
        preference_counts[entry['preferred']] += 1

    print(f"\nPreference distribution (ties excluded):")
    print(f"  Preferred A: {preference_counts['A']:,} ({preference_counts['A']/len(dataset)*100:.1f}%)")
    print(f"  Preferred B: {preference_counts['B']:,} ({preference_counts['B']/len(dataset)*100:.1f}%)")

    # Save to JSONL
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Done! Saved {len(dataset):,} samples to {output_file}")

    # Save a human-readable sample
    sample_file = output_file.replace('.jsonl', '_sample.json')
    print(f"\nSaving sample entry to {sample_file}...")
    with open(sample_file, 'w') as f:
        json.dump(dataset[0], f, indent=2)
    print("Sample saved for inspection.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for LLM book preference prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python prepare_dataset_for_llm.py

  # Custom settings
  python prepare_dataset_for_llm.py \\
      --input /path/to/core_books_reviews.parquet \\
      --output ./book_preference_dataset.jsonl \\
      --min-books 3 \\
      --samples-per-user 5 \\
      --reviews-per-book 10
        """
    )

    parser.add_argument(
        '--input',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/core_books_reviews.parquet',
        help='Input parquet file with core book reviews'
    )
    parser.add_argument(
        '--output',
        default='./book_preference_dataset.jsonl',
        help='Output JSONL file for LLM training'
    )
    parser.add_argument(
        '--min-books',
        type=int,
        default=3,
        help='Minimum books a user must have reviewed (default: 3)'
    )
    parser.add_argument(
        '--samples-per-user',
        type=int,
        default=5,
        help='Number of training samples per user (default: 5)'
    )
    parser.add_argument(
        '--reviews-per-book',
        type=int,
        default=10,
        help='Number of sample reviews per book (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    create_dataset(
        input_file=args.input,
        output_file=args.output,
        min_books_per_user=args.min_books,
        num_samples_per_user=args.samples_per_user,
        num_reviews_per_book=args.reviews_per_book,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
