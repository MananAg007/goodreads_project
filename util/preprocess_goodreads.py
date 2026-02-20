"""
Goodreads data preprocessing script.
Filters users and books, then creates a balanced dataset by rating.
"""
import pandas as pd
import argparse

def preprocess_goodreads(input_file, output_file, user_threshold=10, book_threshold=100, sample_size=165555):
    """
    Preprocess Goodreads reviews data.

    Args:
        input_file: Path to input JSON file (lines format)
        output_file: Path to output file (will save as parquet)
        user_threshold: Minimum reviews per user to keep
        book_threshold: Minimum reviews per book to keep
        sample_size: Number of reviews to sample per rating category
    """
    print("Loading raw reviews...")
    df = pd.read_json(input_file, lines=True)
    print(f"Loaded {len(df):,} reviews")

    # Filter users with enough reviews
    print(f"\nFiltering users with >{user_threshold} reviews...")
    user_counts = df.user_id.value_counts()
    good_users = user_counts[user_counts > user_threshold].index
    print(f"Kept {len(good_users):,} users")

    # Filter books with enough reviews
    print(f"Filtering books with >{book_threshold} reviews...")
    book_counts = df.book_id.value_counts()
    good_books = book_counts[book_counts > book_threshold].index
    print(f"Kept {len(good_books):,} books")

    # Keep only good users and good books
    print("\nFiltering reviews...")
    good_reviews = df[(df['user_id'].isin(good_users)) & (df['book_id'].isin(good_books))]
    print(f"Filtered dataset: {len(good_reviews):,} reviews")
    print(f"Rating distribution:\n{good_reviews.rating.value_counts().sort_index()}")

    # Balance dataset by sampling equal numbers from each rating
    print(f"\nBalancing dataset (sampling {sample_size:,} reviews per rating)...")
    sub_dfs = []
    ratings = sorted(good_reviews.rating.unique())

    for rating in ratings:
        rating_df = good_reviews[good_reviews['rating'] == rating]
        n_available = len(rating_df)
        n_sample = min(sample_size, n_available)

        if n_sample < sample_size:
            print(f"  Warning: Rating {rating} has only {n_available:,} reviews (< {sample_size:,})")

        sampled = rating_df.sample(n=n_sample, random_state=42)
        sub_dfs.append(sampled)
        print(f"  Rating {rating}: sampled {n_sample:,} reviews")

    # Concatenate all subsamples
    final_df = pd.concat(sub_dfs, ignore_index=True)
    print(f"\nFinal dataset: {len(final_df):,} reviews")
    print(f"Unique users: {final_df.user_id.nunique():,}")
    print(f"Unique books: {final_df.book_id.nunique():,}")

    # Save to parquet (efficient and easy to load)
    print(f"\nSaving to {output_file}...")
    final_df.to_parquet(output_file, index=False)
    print("Done!")

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Goodreads reviews data")
    parser.add_argument(
        "--input",
        default="/data/user_data/sheels/Spring2026/10718_mlip/data/goodreads_reviews_dedup.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        default="/data/user_data/sheels/Spring2026/10718_mlip/data/subsampled_good_reviews.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--user-threshold",
        type=int,
        default=10,
        help="Minimum reviews per user"
    )
    parser.add_argument(
        "--book-threshold",
        type=int,
        default=100,
        help="Minimum reviews per book"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=165555,
        help="Number of reviews to sample per rating"
    )

    args = parser.parse_args()

    preprocess_goodreads(
        args.input,
        args.output,
        args.user_threshold,
        args.book_threshold,
        args.sample_size
    )
