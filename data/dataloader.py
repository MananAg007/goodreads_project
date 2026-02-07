"""
Data loader for Goodreads interactions dataset.

Reads CSV with columns: user_id, book_id, is_read, rating, is_reviewed
Filters to only read books, creates train/val/test splits, and provides batching.
"""

import numpy as np
import csv
from pathlib import Path
from typing import Tuple, Iterator, Dict, Optional


class GoodreadsDataLoader:
    """
    Data loader for Goodreads interactions.
    
    Reads the CSV file, filters to is_read=1, shuffles, and splits into
    train (70%), validation (15%), and test (15%) sets.
    """
    
    def __init__(
        self,
        csv_path: str,
        batch_size: int = 512,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 86,
        use_map: bool = False
    ):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the interactions CSV file
            batch_size: Batch size for iteration
            train_ratio: Fraction for training set (default 0.70)
            val_ratio: Fraction for validation set (default 0.15)
            test_ratio: Fraction for test set (default 0.15)
            random_seed: Random seed for shuffling and splitting
            use_map: If True, load user_id_map.json and book_id_map.json from CSV directory
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, val, and test ratios must sum to 1.0"
        
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.use_map = use_map
        self.map_dir = self.csv_path.parent
        
        # Data arrays
        self.user_ids: np.ndarray = np.array([])
        self.book_ids: np.ndarray = np.array([])
        self.ratings: np.ndarray = np.array([])
        
        # Split indices
        self.train_indices: np.ndarray = np.array([])
        self.val_indices: np.ndarray = np.array([])
        self.test_indices: np.ndarray = np.array([])
        
        # Metadata
        self.n_users: int = 0
        self.n_books: int = 0
        self.n_samples: int = 0
        
        # Load data
        self._load_data()
        self._create_splits()
    
    def _load_data(self) -> None:
        """Load and filter data from CSV file."""
        user_ids_raw = []
        book_ids_raw = []
        ratings = []
        
        print(f"Loading data from {self.csv_path}...")
        
        # Load external mappings if use_map is True
        if self.use_map:
            user_map_path = self.map_dir / "user_id_map.csv"
            book_map_path = self.map_dir / "book_id_map.csv"
            
            if not user_map_path.exists():
                raise FileNotFoundError(f"User ID map not found: {user_map_path}")
            if not book_map_path.exists():
                raise FileNotFoundError(f"Book ID map not found: {book_map_path}")
            
            # Load user_id map: CSV with columns (index, original_id)
            print(f"Loading user_id map from {user_map_path}...")
            self.user_id_to_idx = {}
            with open(user_map_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    idx = int(row[0])
                    original_id = row[1]
                    self.user_id_to_idx[original_id] = idx
            
            # Load book_id map: CSV with columns (index, original_id)
            print(f"Loading book_id map from {book_map_path}...")
            self.book_id_to_idx = {}
            with open(book_map_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    idx = int(row[0])
                    original_id = row[1]
                    self.book_id_to_idx[original_id] = idx
            
            print(f"Loaded maps: {len(self.user_id_to_idx):,} users, {len(self.book_id_to_idx):,} books")
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get rating (1-5)
                rating = int(row['rating'])
                if rating < 1 or rating > 5:
                    continue  # Skip invalid ratings
                
                # Get user_id and book_id (as strings for map lookup)
                user_id_str = row['user_id']
                book_id_str = row['book_id']
                
                # Skip if user/book not in map when use_map is True
                if self.use_map:
                    if user_id_str not in self.user_id_to_idx or book_id_str not in self.book_id_to_idx:
                        continue
                    user_ids_raw.append(user_id_str)
                    book_ids_raw.append(book_id_str)
                else:
                    # Filter: only include rows where is_read = 1
                    is_read = int(row['is_read'])
                    if is_read != 1:
                        continue
                    user_ids_raw.append(int(user_id_str))
                    book_ids_raw.append(int(book_id_str))
                
                ratings.append(rating)
        
        self.ratings = np.array(ratings, dtype=np.int32)
        self.n_samples = len(self.ratings)
        
        if self.use_map:
            # Apply pre-loaded mappings (keys are strings) to get intermediate IDs
            user_ids_mapped = np.array([self.user_id_to_idx[uid] for uid in user_ids_raw], dtype=np.int32)
            book_ids_mapped = np.array([self.book_id_to_idx[bid] for bid in book_ids_raw], dtype=np.int32)
        else:
            # Convert to numpy arrays (IDs are already integers)
            user_ids_mapped = np.array(user_ids_raw, dtype=np.int32)
            book_ids_mapped = np.array(book_ids_raw, dtype=np.int32)
        
        unique_users = np.unique(user_ids_mapped)
        unique_books = np.unique(book_ids_mapped)
        
        self.n_users = len(unique_users)
        self.n_books = len(unique_books)
        
        # Create mappings: intermediate_id -> contiguous_index
        user_reindex = {uid: idx for idx, uid in enumerate(unique_users)}
        book_reindex = {bid: idx for idx, bid in enumerate(unique_books)}
        
        # Convert to contiguous indices
        self.user_ids = np.array([user_reindex[uid] for uid in user_ids_mapped], dtype=np.int32)
        self.book_ids = np.array([book_reindex[bid] for bid in book_ids_mapped], dtype=np.int32)
        
        print(f"Loaded {self.n_samples:,} interactions")
        print(f"Number of unique users: {self.n_users:,}")
        print(f"Number of unique books: {self.n_books:,}")
        print(f"Rating distribution: {dict(zip(*np.unique(self.ratings, return_counts=True)))}")
    
    def _create_splits(self) -> None:
        """Create train/val/test splits."""
        np.random.seed(self.random_seed)
        
        # Shuffle indices
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(self.n_samples * self.train_ratio)
        val_end = train_end + int(self.n_samples * self.val_ratio)
        
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]
        
        print(f"\nData splits:")
        print(f"  Train: {len(self.train_indices):,} samples ({100*self.train_ratio:.0f}%)")
        print(f"  Val:   {len(self.val_indices):,} samples ({100*self.val_ratio:.0f}%)")
        print(f"  Test:  {len(self.test_indices):,} samples ({100*self.test_ratio:.0f}%)")
    
    def get_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all data for a specific split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            Tuple of (user_ids, book_ids, ratings) arrays
        """
        if split == 'train':
            indices = self.train_indices
        elif split == 'val':
            indices = self.val_indices
        elif split == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
        
        return (
            self.user_ids[indices],
            self.book_ids[indices],
            self.ratings[indices]
        )
    
    def iterate_batches(
        self,
        split: str,
        shuffle: bool = True,
        repeat: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterate over batches for a specific split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            shuffle: Whether to shuffle the data before iterating
            repeat: If True, cycle through batches indefinitely
            
        Yields:
            Tuples of (user_ids, book_ids, ratings) for each batch
        """
        if split == 'train':
            indices = self.train_indices.copy()
        elif split == 'val':
            indices = self.val_indices.copy()
        elif split == 'test':
            indices = self.test_indices.copy()
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
        
        n_samples = len(indices)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                yield (
                    self.user_ids[batch_indices],
                    self.book_ids[batch_indices],
                    self.ratings[batch_indices]
                )
            
            if not repeat:
                break
    
    def n_batches(self, split: str) -> int:
        """
        Get the number of batches for a specific split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            Number of batches
        """
        if split == 'train':
            n_samples = len(self.train_indices)
        elif split == 'val':
            n_samples = len(self.val_indices)
        elif split == 'test':
            n_samples = len(self.test_indices)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
        
        return (n_samples + self.batch_size - 1) // self.batch_size
    
    def get_stats(self) -> Dict[str, any]:
        """Get dataset statistics."""
        return {
            "n_samples": self.n_samples,
            "n_users": self.n_users,
            "n_books": self.n_books,
            "n_train": len(self.train_indices),
            "n_val": len(self.val_indices),
            "n_test": len(self.test_indices),
            "rating_mean": float(self.ratings.mean()),
            "rating_std": float(self.ratings.std()),
        }


def create_dataloader(
    csv_path: str = "/data/user_data/sheels/Spring2026/10718_mlip/data/goodreads_interactions.csv",
    batch_size: int = 512,
    random_seed: int = 86,
    use_map: bool = False
) -> GoodreadsDataLoader:
    """
    Factory function to create the data loader.
    
    Args:
        csv_path: Path to the interactions CSV file
        batch_size: Batch size for iteration
        random_seed: Random seed for reproducibility
        use_map: If True, load user_id_map.json and book_id_map.json from CSV directory
        
    Returns:
        Configured GoodreadsDataLoader instance
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Interactions file not found: {csv_path}")
    
    return GoodreadsDataLoader(
        csv_path=str(csv_path),
        batch_size=batch_size,
        random_seed=random_seed,
        use_map=use_map
    )
