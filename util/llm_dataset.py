"""
End-to-end dataset preparation for LLM-based book preference prediction.
Samples random books instead of filtering for specific series.

Prerequisites:
- Run process_data.py first to create preprocessed reviews file with metadata

For each user who has reviewed at least 7 books:
- Select 5 reference books with the user's reviews
- Select 2 other books (A and B) that the user has rated DIFFERENTLY (no ties)
- Get sample reviews from other users for each of books A and B
- Create JSONL entry with all this information

The LLM task: Given a user's reference reviews and sample reviews of books A and B,
predict which book the user will prefer (higher rating).

Output includes:
- Reference books: list of 5 books with title, user's rating, review text, average_rating, genres (if available)
- Book A: title, user's rating, user_review, average_rating, genres (if available), sample reviews (with n_votes if available)
- Book B: title, user's rating, user_review, average_rating, genres (if available), sample reviews (with n_votes if available)
- Preferred book (A or B)
- Rating difference

Notes:
- Only reviews with ratings 1-5 are included (already filtered in process_data.py)
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
        # Add n_votes if available
        if 'n_votes' in review.index:
            n_votes = review['n_votes']
            review_dict['n_votes'] = int(n_votes) if pd.notna(n_votes) else 0
        reviews_list.append(review_dict)

    return reviews_list


def create_dataset(reviews_file: str, output_file: str,
                   dataset_size: int = 100, min_books_per_user: int = 7,
                   num_reviews_per_book: int = 10, min_reviews_per_book: int = 10,
                   seed: int = 42):
    """
    Create LLM training dataset from preprocessed book reviews with random book sampling.
    Creates one dataset point per user.

    Args:
        reviews_file: Path to preprocessed reviews file (parquet with metadata merged)
        output_file: Path to save JSONL output
        dataset_size: Total number of dataset points to create (default 100)
        min_books_per_user: Minimum books a user must have reviewed (default 7: 5 reference + 2 comparison)
        num_reviews_per_book: Number of sample reviews to include per book
        min_reviews_per_book: Minimum reviews a book must have to be eligible
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load preprocessed reviews (already has metadata merged and filtered to ratings 1-5)
    print(f"Loading preprocessed reviews from {reviews_file}...")
    df = pd.read_parquet(reviews_file)
    print(f"Loaded {len(df):,} reviews")

    # Verify expected columns exist
    required_cols = ['user_id', 'book_id', 'rating', 'review_text', 'title']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Did you run process_data.py first?")

    # Verify ratings are already filtered
    if df['rating'].min() < 1 or df['rating'].max() > 5:
        print("Warning: Found ratings outside 1-5 range. Filtering...")
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

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

            # Select 5 reference books (prefer ones with review text)
            num_ref_books = 5
            if len(reviews_with_text) >= num_ref_books:
                ref_book_candidates = reviews_with_text['book_id'].unique().tolist()
            else:
                ref_book_candidates = user_book_ids

            if len(ref_book_candidates) < num_ref_books:
                break

            ref_book_ids = random.sample(ref_book_candidates, num_ref_books)

            # Get remaining books for A and B
            remaining_books = [b for b in user_book_ids if b not in ref_book_ids]
            if len(remaining_books) < 2:
                break

            # Randomly select books A and B
            book_a_id, book_b_id = random.sample(remaining_books, 2)

            # Get user's reference reviews (5 books)
            reference_books = []
            for ref_book_id in ref_book_ids:
                ref_review_row = user_reviews[user_reviews['book_id'] == ref_book_id].iloc[0]
                ref_review_text = ref_review_row['review_text'] if pd.notna(ref_review_row['review_text']) else ""
                ref_rating = int(ref_review_row['rating'])
                ref_title = ref_review_row['title']
                ref_avg = ref_review_row['average_rating']
                ref_avg_val = float(ref_avg) if pd.notna(ref_avg) else None
                ref_genres = ref_review_row['genres'] if 'genres' in ref_review_row.index and pd.notna(ref_review_row['genres']) else None

                reference_books.append({
                    'book_id': int(ref_book_id),
                    'title': ref_title,
                    'rating': ref_rating,
                    'review_text': ref_review_text,
                    'average_rating': ref_avg_val,
                    'genres': ref_genres
                })

            # Get user's ratings and reviews for A and B
            book_a_row = user_reviews[user_reviews['book_id'] == book_a_id].iloc[0]
            book_b_row = user_reviews[user_reviews['book_id'] == book_b_id].iloc[0]

            rating_a = int(book_a_row['rating'])
            rating_b = int(book_b_row['rating'])
            review_a = book_a_row['review_text'] if pd.notna(book_a_row['review_text']) else ""
            review_b = book_b_row['review_text'] if pd.notna(book_b_row['review_text']) else ""

            # Skip if ratings are equal (we only want clear preferences)
            if rating_a == rating_b:
                continue

            # Get book info for A and B
            title_a = book_a_row['title']
            title_b = book_b_row['title']
            avg_a = book_a_row['average_rating']
            avg_b = book_b_row['average_rating']
            genres_a = book_a_row['genres'] if 'genres' in book_a_row.index and pd.notna(book_a_row['genres']) else None
            genres_b = book_b_row['genres'] if 'genres' in book_b_row.index and pd.notna(book_b_row['genres']) else None

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
            a_avg_val = float(avg_a) if pd.notna(avg_a) else None
            b_avg_val = float(avg_b) if pd.notna(avg_b) else None

            entry = {
                'user_id': user_id,
                'reference_books': reference_books,
                'book_a': {
                    'book_id': int(book_a_id),
                    'title': title_a,
                    'user_rating': rating_a,
                    'user_review': review_a,
                    'average_rating': a_avg_val,
                    'genres': genres_a,
                    'sample_reviews': reviews_a
                },
                'book_b': {
                    'book_id': int(book_b_id),
                    'title': title_b,
                    'user_rating': rating_b,
                    'user_review': review_b,
                    'average_rating': b_avg_val,
                    'genres': genres_b,
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
        for ref_book in entry['reference_books']:
            unique_books.add(ref_book['book_id'])
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
      --reviews ./processed_reviews.parquet \\
      --output ./book_preference_dataset.jsonl \\
      --dataset-size 100 \\
      --min-books 7 \\
      --reviews-per-book 10

Note: You must run process_data.py first to create the preprocessed reviews file.
        """
    )

    parser.add_argument(
        '--reviews',
        default='/home/mananaga/goodreads_dataset/processed_reviews.parquet',
        help='Preprocessed reviews file (output from process_data.py)'
    )
    parser.add_argument(
        '--output',
        default='../data/book_preference_dataset_new.jsonl',
        help='Output JSONL file for LLM training'
    )
    parser.add_argument(
        '--dataset-size',
        type=int,
        default=10,
        help='Total number of data points to create (default: 100)'
    )
    parser.add_argument(
        '--min-books',
        type=int,
        default=7,
        help='Minimum books a user must have reviewed (default: 7: 5 reference + 2 comparison)'
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
        output_file=args.output,
        dataset_size=args.dataset_size,
        min_books_per_user=args.min_books,
        num_reviews_per_book=args.reviews_per_book,
        min_reviews_per_book=args.min_reviews_per_book,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
