"""
Collaborative Filtering Model using Matrix Factorization with NumPy.

Learns d-dimensional embeddings for users and books such that their
dot product is predictive of the rating.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class CollaborativeFilteringModel:
    """
    Matrix Factorization model for collaborative filtering using NumPy.
    
    Learns user and book embeddings using alternating gradient descent.
    
    Loss: MSE(predicted_rating, target_rating) + reg_weight * ||user_emb||_2^2 + reg_weight * ||book_emb||_2^2
    
    Target rating is scaled: rating 1->0.2, 2->0.4, 3->0.6, 4->0.8, 5->1.0
    """
    
    def __init__(
        self,
        n_users: int,
        n_books: int,
        embedding_dim: int = 64,
        reg_weight: float = 0.01,
        learning_rate: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the collaborative filtering model.
        
        Args:
            n_users: Number of unique users
            n_books: Number of unique books
            embedding_dim: Dimension of user/book embeddings
            reg_weight: Weight for L2 regularization
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_books = n_books
        self.embedding_dim = embedding_dim
        self.reg_weight = reg_weight
        self.learning_rate = learning_rate
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize embeddings with small random values (scaled by 1/sqrt(d))
        scale = 1.0 / np.sqrt(embedding_dim)
        self.user_embeddings = np.random.randn(n_users, embedding_dim).astype(np.float32) * scale
        self.book_embeddings = np.random.randn(n_books, embedding_dim).astype(np.float32) * scale
        
        # Training history
        self.train_losses: list = []
        self.val_losses: list = []
    
    @staticmethod
    def rating_to_target(rating: np.ndarray) -> np.ndarray:
        """
        Convert rating (1-5) to target (0.2 to 1.0).
        
        Args:
            rating: Array of ratings from 1 to 5
            
        Returns:
            Array of targets from 0.2 to 1.0
        """
        return rating.astype(np.float32) / 5
    
    @staticmethod
    def target_to_rating(target: np.ndarray) -> np.ndarray:
        """
        Convert target (0.2 to 1.0) back to rating (1-5).
        
        Args:
            target: Array of targets from 0.2 to 1.0
            
        Returns:
            Array of ratings from 1 to 5
        """
        return target * 5
    
    def predict(self, user_ids: np.ndarray, book_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for given user-book pairs.
        
        Args:
            user_ids: Array of user IDs
            book_ids: Array of book IDs
            
        Returns:
            Array of predicted targets (dot products)
        """
        user_emb = self.user_embeddings[user_ids]  # (batch, d)
        book_emb = self.book_embeddings[book_ids]  # (batch, d)
        
        # Dot product for each pair
        predictions = np.sum(user_emb * book_emb, axis=1)  # (batch,)
        return predictions

    def compute_loss(
        self,
        user_ids: np.ndarray,
        book_ids: np.ndarray,
        ratings: np.ndarray,
        include_reg: bool = True
    ) -> Tuple[float, float, float]:
        """
        Compute the loss for a batch.
        
        Args:
            user_ids: Array of user IDs
            book_ids: Array of book IDs
            ratings: Array of ratings (1-5)
            include_reg: Whether to include regularization loss
            
        Returns:
            Tuple of (total_loss, mse_loss, reg_loss)
        """
        targets = self.rating_to_target(ratings)
        predictions = self.predict(user_ids, book_ids)
        
        # MSE Loss
        mse_loss = np.mean((predictions - targets) ** 2)
        
        # L2 Regularization: penalize squared norm of embeddings
        reg_loss = 0.0
        if include_reg:
            user_emb = self.user_embeddings[user_ids]
            book_emb = self.book_embeddings[book_ids]
            
            reg_loss = self.reg_weight * (
                np.mean(np.sum(user_emb ** 2, axis=1)) + 
                np.mean(np.sum(book_emb ** 2, axis=1))
            )
        
        total_loss = mse_loss + reg_loss
        return total_loss, mse_loss, reg_loss
    
    def _update_user_embeddings(
        self,
        user_ids: np.ndarray,
        book_ids: np.ndarray,
        ratings: np.ndarray
    ) -> None:
        """
        Update user embeddings while keeping book embeddings fixed.
        
        Args:
            user_ids: Array of user IDs
            book_ids: Array of book IDs
            ratings: Array of ratings (1-5)
        """
        targets = self.rating_to_target(ratings)
        
        user_emb = self.user_embeddings[user_ids]  # (batch, d)
        book_emb = self.book_embeddings[book_ids]  # (batch, d)
        
        # Predictions
        predictions = np.sum(user_emb * book_emb, axis=1)  # (batch,)
        
        # Gradient of MSE loss w.r.t user embeddings
        # d/du (pred - target)^2 = 2 * (pred - target) * book_emb
        errors = predictions - targets  # (batch,)
        grad_mse = 2 * errors[:, np.newaxis] * book_emb  # (batch, d)
        
        # Gradient of L2 regularization loss
        # d/du ||u||_2^2 = 2 * u
        grad_reg = 2 * self.reg_weight * user_emb  # (batch, d)
        
        # Total gradient
        grad = grad_mse + grad_reg  # (batch, d)
        
        # Aggregate gradients for each unique user (in case of duplicates)
        np.add.at(self.user_embeddings, user_ids, -self.learning_rate * grad)
    
    def _update_book_embeddings(
        self,
        user_ids: np.ndarray,
        book_ids: np.ndarray,
        ratings: np.ndarray,
        user_emb_original: Optional[np.ndarray] = None
    ) -> None:
        """
        Update book embeddings while keeping user embeddings fixed.
        
        Args:
            user_ids: Array of user IDs
            book_ids: Array of book IDs
            ratings: Array of ratings (1-5)
            user_emb_original: Optional pre-computed user embeddings to use instead of fetching
        """
        targets = self.rating_to_target(ratings)
        
        # Use provided user embeddings or fetch from current state
        if user_emb_original is not None:
            user_emb = user_emb_original
        else:
            user_emb = self.user_embeddings[user_ids]  # (batch, d)
        
        book_emb = self.book_embeddings[book_ids]  # (batch, d)
        
        # Predictions
        predictions = np.sum(user_emb * book_emb, axis=1)  # (batch,)
        
        # Gradient of MSE loss w.r.t book embeddings
        errors = predictions - targets  # (batch,)
        grad_mse = 2 * errors[:, np.newaxis] * user_emb  # (batch, d)
        
        # Gradient of L2 regularization loss
        # d/db ||b||_2^2 = 2 * b
        grad_reg = 2 * self.reg_weight * book_emb  # (batch, d)
        
        # Total gradient
        grad = grad_mse + grad_reg  # (batch, d)
        
        # Aggregate gradients for each unique book
        np.add.at(self.book_embeddings, book_ids, -self.learning_rate * grad)

    def train_step(
        self,
        user_ids: np.ndarray,
        book_ids: np.ndarray,
        ratings: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Perform one training step with alternating optimization.
        
        Args:
            user_ids: Array of user IDs
            book_ids: Array of book IDs
            ratings: Array of ratings (1-5)
            
        Returns:
            Tuple of (total_loss, mse_loss, reg_loss) as floats
        """
        # Compute loss before update
        loss, mse_loss, reg_loss = self.compute_loss(user_ids, book_ids, ratings)
        
        # Store a copy of the user embeddings for this batch before updating
        # This ensures book embedding updates use the original user embeddings
        user_emb_original = self.user_embeddings[user_ids].copy()
        
        # Alternating optimization
        # Step 1: Update user embeddings (fix book embeddings)
        self._update_user_embeddings(user_ids, book_ids, ratings)
        
        # Step 2: Update book embeddings using original user embeddings
        self._update_book_embeddings(user_ids, book_ids, ratings, user_emb_original)
        
        return loss, mse_loss, reg_loss
    
    def evaluate(
        self,
        user_ids: np.ndarray,
        book_ids: np.ndarray,
        ratings: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            user_ids: Array of user IDs
            book_ids: Array of book IDs
            ratings: Array of ratings (1-5)
            
        Returns:
            Dictionary with evaluation metrics
        """
        targets = self.rating_to_target(ratings)
        predictions = self.predict(user_ids, book_ids)
        
        # Clamp predictions to valid range for rating conversion
        predictions_clamped = np.clip(predictions, 0.2, 1.0)
        predicted_ratings = self.target_to_rating(predictions_clamped)
        
        # MSE on targets
        mse = np.mean((predictions - targets) ** 2)
        
        # RMSE on original rating scale
        rmse = np.sqrt(np.mean((predicted_ratings - ratings) ** 2))
        
        # MAE on original rating scale
        mae = np.mean(np.abs(predicted_ratings - ratings))
        
        # Accuracy: round(prediction / 0.2), clip to [1, 5], compare to actual rating
        rounded_ratings = np.clip(np.round(predictions / 0.2), 1, 5).astype(np.int32)
        accuracy = np.mean(rounded_ratings == ratings)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "accuracy": float(accuracy)
        }
    
    def save(self, path: str) -> None:
        """Save model parameters to a file."""
        np.savez(
            path,
            user_embeddings=self.user_embeddings,
            book_embeddings=self.book_embeddings,
            n_users=self.n_users,
            n_books=self.n_books,
            embedding_dim=self.embedding_dim,
            reg_weight=self.reg_weight,
            learning_rate=self.learning_rate,
            train_losses=np.array(self.train_losses),
            val_losses=np.array(self.val_losses)
        )
    
    @classmethod
    def load(cls, path: str) -> "CollaborativeFilteringModel":
        """Load model parameters from a file."""
        data = np.load(path, allow_pickle=True)
        
        model = cls(
            n_users=int(data['n_users']),
            n_books=int(data['n_books']),
            embedding_dim=int(data['embedding_dim']),
            reg_weight=float(data['reg_weight']),
            learning_rate=float(data['learning_rate'])
        )
        
        model.user_embeddings = data['user_embeddings']
        model.book_embeddings = data['book_embeddings']
        model.train_losses = data['train_losses'].tolist()
        model.val_losses = data['val_losses'].tolist()
        
        return model
