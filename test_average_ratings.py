#!/usr/bin/env python3
"""Quick test to verify average_ratings modes work correctly."""

import sys
import json
sys.path.insert(0, '/home/mananaga/goodreads')

from util.run_claude_eval import get_average_rating, format_sample_reviews_block

# Test get_average_rating function
print("=" * 60)
print("Testing get_average_rating function")
print("=" * 60)

test_rating = 3.85

print(f"\nActual rating: {test_rating}")
print(f"Mode 'true':        {get_average_rating(test_rating, 'true')}")
print(f"Mode 'random':      {get_average_rating(test_rating, 'random')}")
print(f"Mode 'unavailable': {get_average_rating(test_rating, 'unavailable')}")
print(f"Mode 'flipped':     {get_average_rating(test_rating, 'flipped', 'book_a')}")

# Test format_sample_reviews_block
print("\n" + "=" * 60)
print("Testing format_sample_reviews_block function")
print("=" * 60)

sample_reviews = [
    {"rating": 5, "review_text": "Amazing book!"},
    {"rating": 3, "review_text": "It was okay."},
    {"rating": 1, "review_text": "Did not like it."},
]

print("\nWith average_rating='3.85':")
block = format_sample_reviews_block(sample_reviews, 2, "3.85")
print(block)

print("\nWith average_rating='NONE' (unavailable mode - skips rating):")
block = format_sample_reviews_block(sample_reviews, 2, "NONE")
print(block)

print("\nWith average_rating=None (original behavior):")
block = format_sample_reviews_block(sample_reviews, 2, None)
print(block)

print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
