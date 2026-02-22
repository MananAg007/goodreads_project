#!/usr/bin/env python3
"""
Complete Goodreads Database Downloader

This script downloads all datasets from the Goodreads collection hosted at UCSD.
It includes progress tracking, error handling, and the ability to skip already downloaded files.

Usage:
    python download_complete_database.py [--output-dir OUTPUT_DIR] [--skip-existing]
"""

import os
import sys
import argparse
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, Tuple


def create_url_mapping(dataset_names_path: str) -> Dict[str, str]:
    """
    Create a mapping of file names to download URLs based on dataset_names.csv

    Args:
        dataset_names_path: Path to the dataset_names.csv file

    Returns:
        Dictionary mapping file names to their download URLs
    """
    file_names = pd.read_csv(dataset_names_path)
    file_name_type_mapping = dict(zip(file_names['name'].values, file_names['type'].values))
    file_name_url_mapping = {}

    for fname in file_name_type_mapping:
        ftype = file_name_type_mapping[fname]
        if ftype == "complete":
            url = f'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/{fname}'
            file_name_url_mapping[fname] = url
        elif ftype == "byGenre":
            url = f'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/{fname}'
            file_name_url_mapping[fname] = url

    return file_name_url_mapping, file_name_type_mapping


def format_bytes(bytes_size: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def download_file(url: str, local_filename: str, skip_existing: bool = False) -> Tuple[bool, str]:
    """
    Download a file from a URL with progress tracking

    Args:
        url: URL to download from
        local_filename: Local path to save the file
        skip_existing: If True, skip download if file already exists

    Returns:
        Tuple of (success: bool, message: str)
    """
    if skip_existing and os.path.exists(local_filename):
        file_size = os.path.getsize(local_filename)
        return True, f"Skipped (already exists, {format_bytes(file_size)})"

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()

            # Get total file size if available
            total_size = int(r.headers.get('content-length', 0))

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)

            # Download with progress tracking
            downloaded_size = 0
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # Print progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}% ({format_bytes(downloaded_size)} / {format_bytes(total_size)})", end='', flush=True)
                    else:
                        print(f"\r  Downloaded: {format_bytes(downloaded_size)}", end='', flush=True)

            print()  # New line after progress
            return True, f"Success ({format_bytes(downloaded_size)})"

    except requests.exceptions.RequestException as e:
        return False, f"Failed: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def download_all_datasets(
    url_mapping: Dict[str, str],
    type_mapping: Dict[str, str],
    output_dir: str,
    skip_existing: bool = False
):
    """
    Download all datasets from the mapping

    Args:
        url_mapping: Dictionary mapping file names to URLs
        type_mapping: Dictionary mapping file names to types (complete/byGenre)
        output_dir: Base directory to save files
        skip_existing: If True, skip files that already exist
    """
    total_files = len(url_mapping)
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0

    print(f"\n{'='*80}")
    print(f"Goodreads Complete Database Downloader")
    print(f"{'='*80}")
    print(f"Total files to download: {total_files}")
    print(f"Output directory: {output_dir}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'='*80}\n")

    for idx, (fname, url) in enumerate(url_mapping.items(), 1):
        ftype = type_mapping[fname]

        # Organize files by type
        if ftype == "complete":
            local_path = os.path.join(output_dir, fname)
        elif ftype == "byGenre":
            local_path = os.path.join(output_dir, "byGenre", fname)

        print(f"[{idx}/{total_files}] Downloading: {fname}")
        print(f"  URL: {url}")
        print(f"  Path: {local_path}")

        success, message = download_file(url, local_path, skip_existing)

        if success:
            if "Skipped" in message:
                skipped_downloads += 1
                print(f"  ✓ {message}\n")
            else:
                successful_downloads += 1
                print(f"  ✓ {message}\n")
        else:
            failed_downloads += 1
            print(f"  ✗ {message}\n")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Download Summary")
    print(f"{'='*80}")
    print(f"Total files: {total_files}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Skipped (already exists): {skipped_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"{'='*80}\n")

    if failed_downloads > 0:
        print("⚠️  Some downloads failed. You may want to re-run the script to retry failed downloads.")
    else:
        print("✓ All downloads completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Download complete Goodreads database from UCSD servers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets to current directory
  python download_complete_database.py

  # Download to specific directory
  python download_complete_database.py --output-dir ./goodreads_data

  # Skip already downloaded files
  python download_complete_database.py --skip-existing
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./goodreads_datasets',
        help='Directory to save downloaded files (default: ./goodreads_datasets)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist in the output directory'
    )

    parser.add_argument(
        '--dataset-names',
        type=str,
        default='./dataset_names.csv',
        help='Path to dataset_names.csv file (default: ./dataset_names.csv)'
    )

    args = parser.parse_args()

    # Check if dataset_names.csv exists
    if not os.path.exists(args.dataset_names):
        print(f"Error: dataset_names.csv not found at {args.dataset_names}")
        print("Please make sure you're running this script from the goodreads repository directory.")
        sys.exit(1)

    # Create URL mappings
    print("Loading dataset information...")
    url_mapping, type_mapping = create_url_mapping(args.dataset_names)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Download all datasets
    download_all_datasets(url_mapping, type_mapping, args.output_dir, args.skip_existing)


if __name__ == "__main__":
    main()
