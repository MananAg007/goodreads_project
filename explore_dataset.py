#!/usr/bin/env python3
"""
Script to explore Goodreads dataset JSON files.
Loads each JSON file and displays structure and sample records.
"""

import json
import os
import gzip
from pathlib import Path
from typing import Dict, Any, List

def load_json_file(file_path: Path, max_records: int = 1000) -> List[Dict[Any, Any]]:
    """Load a JSON or JSONL file (compressed or uncompressed) and return list of records."""
    records = []

    try:
        # Determine if file is gzipped
        is_gzipped = file_path.suffix == '.gz'

        # Open with appropriate handler
        if is_gzipped:
            f = gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            f = open(file_path, 'r', encoding='utf-8')

        try:
            # Try loading as regular JSON first
            try:
                data = json.load(f)
                if isinstance(data, list):
                    records = data[:max_records]
                else:
                    records = [data]
            except json.JSONDecodeError:
                # If that fails, try as JSONL (one JSON object per line)
                f.seek(0)
                for i, line in enumerate(f):
                    if i >= max_records:
                        break
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        finally:
            f.close()

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

    return records

def print_record_structure(record: Dict[Any, Any], indent: int = 2) -> None:
    """Print the structure of a record with field names and types."""
    for key, value in record.items():
        value_type = type(value).__name__
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}: dict with {len(value)} keys")
            if len(value) <= 5:  # Show nested structure for small dicts
                print_record_structure(value, indent + 2)
        elif isinstance(value, list):
            list_len = len(value)
            if list_len > 0:
                item_type = type(value[0]).__name__
                print(f"{' ' * indent}{key}: list[{item_type}] ({list_len} items)")
                if isinstance(value[0], dict) and list_len > 0:
                    print(f"{' ' * indent}  Sample item structure:")
                    print_record_structure(value[0], indent + 4)
            else:
                print(f"{' ' * indent}{key}: list (empty)")
        elif isinstance(value, str):
            preview = value[:50] + "..." if len(value) > 50 else value
            print(f"{' ' * indent}{key}: {value_type} = \"{preview}\"")
        else:
            print(f"{' ' * indent}{key}: {value_type} = {value}")

def explore_dataset(data_dir: str, num_examples: int = 3):
    """Explore all JSON files in the dataset directory."""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        return

    # Find all JSON files (including .gz), excluding byGenre folder
    json_files = []
    for pattern in ["*.json", "*.json.gz"]:
        for file in data_path.rglob(pattern):
            if "byGenre" not in file.parts:
                json_files.append(file)

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} JSON files (excluding byGenre folder)\n")
    print("=" * 80)

    for json_file in sorted(json_files):
        relative_path = json_file.relative_to(data_path)
        print(f"\n📁 FILE: {relative_path}")
        print(f"   Full path: {json_file}")
        print(f"   Size: {json_file.stat().st_size / 1024 / 1024:.2f} MB")
        print("-" * 80)

        # Load limited number of records for exploration
        records = load_json_file(json_file, max_records=100)

        if not records:
            print("   ⚠️  No records loaded (empty or invalid JSON)")
            continue

        print(f"   Records loaded for analysis: {len(records)} (limited sample)")
        print(f"\n   STRUCTURE (from first record):")
        print_record_structure(records[0], indent=4)

        print(f"\n   SAMPLE RECORDS (first {min(num_examples, len(records))}):")
        for i, record in enumerate(records[:num_examples]):
            print(f"\n   --- Record {i+1} ---")
            record_str = json.dumps(record, indent=4, ensure_ascii=False)
            if len(record_str) > 800:
                print(record_str[:800] + "\n    ... (truncated)")
            else:
                print(record_str)

        print("\n" + "=" * 80)

if __name__ == "__main__":
    dataset_dir = "/data/user_data/mananaga/goodreads_datasets"
    num_examples = 2  # Number of example records to show per file

    explore_dataset(dataset_dir, num_examples)
