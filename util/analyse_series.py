"""
Filter reviews to only include core books from each series.
For Harry Potter: Books 1-7 (excludes Cursed Child, prequels, companion books)
For Percy Jackson: Books 1-5 (main Olympians series only)
"""
import pandas as pd
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def identify_core_books(df):
    """
    Identify core books from the dataset based on title patterns.

    Returns:
        dict: Mapping of book_id to title for core books only
    """
    core_books = {}

    for book_id, title in df[['book_id', 'title']].drop_duplicates().values:
        title_lower = title.lower()

        # Harry Potter core series (books 1-7)
        hp_patterns = [
            # English editions - use specific titles to avoid catching translations
            (r"harry potter and the (?:sorcerer's|philosopher's) stone.*#1", 1),
            (r"harry potter and the chamber of secrets.*#2", 2),
            (r"harry potter and the prisoner of azkaban.*#3", 3),
            (r"harry potter and the goblet of fire.*#4", 4),
            (r"harry potter and the order of the phoenix.*#5", 5),
            (r"harry potter and the half-blood prince.*#6", 6),
            (r"harry potter and the deathly hallows.*#7", 7),
        ]

        for pattern, book_num in hp_patterns:
            if re.search(pattern, title_lower):
                # Exclude translations, box sets, graphic novels, fan fiction
                if any(exclude in title_lower for exclude in [
                    'spanish', 'y el', 'y la',  # Spanish
                    'boxset', 'box set', 'collection',
                    'graphic novel', 'illustrated',
                    'methods of rationality',  # Fan fiction
                    'prequel', 'cursed child',
                    'greek', 'companion', 'history'
                ]):
                    continue

                # Prefer main editions (not special anniversary/house editions)
                # But still include them if they're the only version
                core_books[book_id] = title
                break

        # Percy Jackson core Olympians series (books 1-5)
        pj_patterns = [
            (r"the lightning thief.*(?:percy jackson|olympians).*#1", 1),
            (r"the sea of monsters.*(?:percy jackson|olympians).*#2", 2),
            (r"the titan'?s curse.*(?:percy jackson|olympians).*#3", 3),
            (r"the battle of the labyrinth.*(?:percy jackson|olympians).*#4", 4),
            (r"the last olympian.*(?:percy jackson|olympians).*#5", 5),
        ]

        for pattern, book_num in pj_patterns:
            if re.search(pattern, title_lower):
                # Exclude translations, graphic novels, companion books
                if any(exclude in title_lower for exclude in [
                    'spanish', 'el ladrón', 'y los',  # Spanish
                    'graphic novel',
                    'greek gods', 'greek heroes',  # Companion books
                    'crossover', 'kane chronicles',  # Crossover novellas
                    'demigod files', 'ultimate guide'
                ]):
                    continue

                core_books[book_id] = title
                break

    return core_books


def filter_to_core_books(input_file, output_file):
    """
    Filter reviews to only include core series books.
    Deduplicates by keeping only the book_id with most reviews for each title.

    Args:
        input_file: Path to input parquet file with all series books
        output_file: Path to save filtered parquet with only core books
    """
    print(f"Loading reviews from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} reviews")
    print(f"Unique titles: {df['title'].nunique()}")
    print(f"Unique books: {df['book_id'].nunique()}")

    # Identify core books
    print("\nIdentifying core books (HP #1-7, Percy Jackson #1-5)...")
    core_books = identify_core_books(df)

    print(f"\nFound {len(core_books)} core book editions:")

    # Count reviews per book_id
    book_review_counts = {}
    for book_id in core_books.keys():
        count = len(df[df['book_id'] == book_id])
        book_review_counts[book_id] = count

    # Extract book number from title (e.g., "#1", "#2", etc.)
    def extract_series_and_number(title):
        """Extract series name and book number from title."""
        title_lower = title.lower()

        # Extract book number
        match = re.search(r'#(\d+)', title)
        book_num = int(match.group(1)) if match else None

        # Determine series
        if 'harry potter' in title_lower:
            series = 'Harry Potter'
        elif 'percy jackson' in title_lower or 'lightning thief' in title_lower or 'sea of monsters' in title_lower or 'titan' in title_lower or 'labyrinth' in title_lower or 'last olympian' in title_lower:
            series = 'Percy Jackson'
        else:
            series = 'Unknown'

        return (series, book_num)

    # Group by series and book number, find the book_id with most reviews
    series_num_to_best_book = {}
    for book_id, title in core_books.items():
        series_num = extract_series_and_number(title)

        if series_num not in series_num_to_best_book:
            series_num_to_best_book[series_num] = (book_id, title, book_review_counts[book_id])
        else:
            current_best_id, current_title, current_best_count = series_num_to_best_book[series_num]
            if book_review_counts[book_id] > current_best_count:
                series_num_to_best_book[series_num] = (book_id, title, book_review_counts[book_id])

    # Keep only the best book_id for each series/number combo
    deduped_book_ids = {book_id for book_id, title, count in series_num_to_best_book.values()}

    print(f"\nAfter deduplication (keeping edition with most reviews per series/book number):")
    print(f"  {len(deduped_book_ids)} unique books (from {len(core_books)} editions)")

    for (series, num), (book_id, title, count) in sorted(series_num_to_best_book.items()):
        print(f"  {book_id:8d} - {count:4d} reviews - {title}")

    # Filter to deduplicated core books
    print(f"\nFiltering to deduplicated core books...")
    filtered_df = df[df['book_id'].isin(deduped_book_ids)]

    print(f"\nFiltered results:")
    print(f"  Total reviews: {len(filtered_df):,} (was {len(df):,})")
    print(f"  Unique books: {filtered_df['book_id'].nunique()} (was {df['book_id'].nunique()})")
    print(f"  Unique titles: {filtered_df['title'].nunique()}")
    print(f"  Unique users: {filtered_df['user_id'].nunique():,}")
    print(f"  Kept {len(filtered_df)/len(df)*100:.1f}% of reviews")

    # Rating distribution
    print(f"\nRating distribution:")
    for rating in sorted(filtered_df['rating'].unique()):
        count = len(filtered_df[filtered_df['rating'] == rating])
        print(f"  Rating {rating}: {count:,} reviews")

    # Show per-book breakdown
    print(f"\nReviews per book:")
    for title in sorted(filtered_df['title'].unique()):
        count = len(filtered_df[filtered_df['title'] == title])
        print(f"  {count:4d} reviews - {title}")

    # Sample one review per book
    print(f"\n{'='*80}")
    print("SAMPLE REVIEW FOR EACH BOOK")
    print('='*80)
    for book_id in sorted(deduped_book_ids):
        book_reviews = filtered_df[filtered_df['book_id'] == book_id]
        if len(book_reviews) > 0:
            # Get a random sample review (with text if available)
            reviews_with_text = book_reviews[book_reviews['review_text'].notna() & (book_reviews['review_text'].str.strip() != '')]
            if len(reviews_with_text) > 0:
                sample = reviews_with_text.sample(n=1).iloc[0]
            else:
                sample = book_reviews.sample(n=1).iloc[0]

            print(f"\n📚 Book: {sample['title']}")
            print(f"   Rating: {sample['rating']}/5")
            print(f"   User: {sample['user_id']}")
            if pd.notna(sample['review_text']) and sample['review_text'].strip():
                review_text = sample['review_text'].strip()
                # Truncate if too long
                if len(review_text) > 300:
                    review_text = review_text[:300] + "..."
                print(f"   Review: {review_text}")
            else:
                print(f"   Review: (No review text available)")

    # Group reviews by user and analyze reading patterns
    print(f"\n{'='*80}")
    print("USER READING PATTERNS")
    print('='*80)

    user_book_counts = filtered_df.groupby('user_id')['book_id'].nunique()

    print(f"\nTotal unique users: {len(user_book_counts):,}")
    print(f"Average books read per user: {user_book_counts.mean():.2f}")
    print(f"Median books read per user: {user_book_counts.median():.1f}")
    print(f"Max books read by a single user: {user_book_counts.max()}")
    print(f"Min books read by a single user: {user_book_counts.min()}")

    # Distribution summary
    print(f"\nDistribution of books read per user:")
    for num_books in sorted(user_book_counts.unique()):
        num_users = (user_book_counts == num_books).sum()
        percentage = (num_users / len(user_book_counts)) * 100
        print(f"  {num_books:2d} book(s): {num_users:6,} users ({percentage:5.1f}%)")

    # Create histogram
    print(f"\nGenerating histogram...")
    plt.figure(figsize=(12, 6))

    # Create histogram
    bins = np.arange(0.5, user_book_counts.max() + 1.5, 1)
    plt.hist(user_book_counts, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

    plt.xlabel('Number of Unique Books Read', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.title('Distribution of Unique Books Read Per User\n(Harry Potter 1-7 & Percy Jackson 1-5)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis to show integer values
    plt.xticks(range(1, int(user_book_counts.max()) + 1))

    # Add stats text box
    stats_text = f'Total Users: {len(user_book_counts):,}\nMean: {user_book_counts.mean():.2f}\nMedian: {user_book_counts.median():.1f}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    histogram_file = output_file.replace('.parquet', '_user_distribution.png')
    plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {histogram_file}")
    plt.close()

    # Save filtered data
    print(f"\nSaving filtered reviews to {output_file}...")
    filtered_df.to_parquet(output_file, index=False)
    print("Done!")

    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter to core Harry Potter (1-7) and Percy Jackson (1-5) books only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python filter_core_books.py

  # Custom paths
  python filter_core_books.py \\
      --input series_filtered_reviews.parquet \\
      --output core_books_reviews.parquet
        """
    )

    parser.add_argument(
        '--input',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/series_filtered_reviews.parquet',
        help='Input parquet file with all series books'
    )
    parser.add_argument(
        '--output',
        default='/data/user_data/sheels/Spring2026/10718_mlip/data/core_books_reviews.parquet',
        help='Output parquet file with only core books'
    )

    args = parser.parse_args()

    filter_to_core_books(args.input, args.output)


if __name__ == "__main__":
    main()
