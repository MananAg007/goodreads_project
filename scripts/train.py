#!/usr/bin/env python3
"""
Training script for Collaborative Filtering model on Goodreads dataset.
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CollaborativeFilteringModel
from data.dataloader import create_dataloader, GoodreadsDataLoader


def plot_training_curve(
    train_epochs: List[int],
    train_values: List[float],
    val_epochs: List[int],
    val_values: List[float],
    ylabel: str,
    title: str,
    output_path: Path
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_epochs: List of epoch numbers for training values
        train_values: List of training metric values
        val_epochs: List of epoch numbers when validation was done
        val_values: List of validation metric values
        ylabel: Label for y-axis
        title: Title for the plot
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_epochs, train_values, 'b-', label=f'Train {ylabel}', linewidth=2)
    ax.plot(val_epochs, val_values, 'r-o', label=f'Val {ylabel}', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def train(
    data_dir: str,
    output_dir: str,
    embedding_dim: int = 64,
    reg_weight: float = 0.01,
    learning_rate: float = 0.01,
    batch_size: int = 512,
    n_epochs: int = 20,
    random_seed: int = 86,
    save_every: int = 1,
    verbose: bool = True,
    dataloader: Optional[GoodreadsDataLoader] = None
) -> Tuple[Dict[str, float], Dict[str, float], GoodreadsDataLoader]:
    """
    Train the collaborative filtering model.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to save model checkpoints
        embedding_dim: Dimension of embeddings
        reg_weight: Regularization weight for unit norm constraint
        learning_rate: Learning rate for gradient descent
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        random_seed: Random seed for reproducibility
        save_every: Evaluate and save checkpoint every N epochs
        verbose: Whether to print progress messages
        dataloader: Optional pre-initialized dataloader (to reuse across runs)
        
    Returns:
        Tuple of (val_metrics, test_metrics, dataloader)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data (or reuse provided dataloader)
    if dataloader is None:
        if verbose:
            print("=" * 60)
            print("Loading data...")
            print("=" * 60)
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            random_seed=random_seed
        )
    else:
        if verbose:
            print("=" * 60)
            print("Reusing existing dataloader...")
            print("=" * 60)
    
    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing model...")
    print("=" * 60)
    model = CollaborativeFilteringModel(
        n_users=dataloader.n_users,
        n_books=dataloader.n_books,
        embedding_dim=embedding_dim,
        reg_weight=reg_weight,
        learning_rate=learning_rate,
        random_seed=random_seed
    )
    
    print(f"  Users: {model.n_users:,}")
    print(f"  Books: {model.n_books:,}")
    print(f"  Embedding dim: {model.embedding_dim}")
    print(f"  Regularization weight: {model.reg_weight}")
    print(f"  Learning rate: {model.learning_rate}")
    
    # Get validation data for evaluation
    val_users, val_books, val_ratings = dataloader.get_split_data('val')
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_mae = float('inf')
    
    # Tracking for plots
    train_mse_history = []
    train_mae_history = []
    val_mse_history = []
    val_mae_history = []
    val_epochs = []  # epochs when validation was done
    
    # Get train data for computing train MAE
    train_users, train_books, train_ratings = dataloader.get_split_data('train')
    
    for epoch in range(1, n_epochs + 1):
        epoch_start = time()
        
        # Training
        epoch_losses = []
        epoch_mse_losses = []
        epoch_reg_losses = []
        
        # Use infinite iterator with fixed number of batches per epoch
        n_batches_per_epoch = dataloader.n_batches('train')
        batch_iter = dataloader.iterate_batches('train', shuffle=True, repeat=True)
        
        for _ in range(n_batches_per_epoch):
            user_ids, book_ids, ratings = next(batch_iter)
            loss, mse_loss, reg_loss = model.train_step(user_ids, book_ids, ratings)
            epoch_losses.append(loss)
            epoch_mse_losses.append(mse_loss)
            epoch_reg_losses.append(reg_loss)

            # if verbose and (_ + 1) % 100 == 0:
            #     print(f"  Batch: {len(epoch_losses):,} / {n_batches_per_epoch:,}")
        
        avg_loss = np.mean(epoch_losses)
        avg_mse = np.mean(epoch_mse_losses)
        avg_reg = np.mean(epoch_reg_losses)
        
        model.train_losses.append(avg_loss)
        
        # Compute train metrics for this epoch
        train_metrics = model.evaluate(train_users, train_books, train_ratings)
        train_mse_history.append(train_metrics['mse'])
        train_mae_history.append(train_metrics['mae'])
        
        epoch_time = time() - epoch_start
        
        # Progress output
        print(f"\nEpoch {epoch}/{n_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {avg_loss:.6f} (MSE: {avg_mse:.6f}, Reg: {avg_reg:.6f})")
        
        # Validation and checkpoint saving
        if epoch % save_every == 0:
            val_metrics = model.evaluate(val_users, val_books, val_ratings)
            model.val_losses.append(val_metrics['mse'])
            
            val_mse_history.append(val_metrics['mse'])
            val_mae_history.append(val_metrics['mae'])
            val_epochs.append(epoch)
            
            print(f"  Val MSE: {val_metrics['mse']:.6f}, RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                model.save(str(output_path / "best_model.npz"))
                print(f"  ✓ New best model saved (MAE: {best_val_mae:.4f})")
            
            # Save checkpoint
            model.save(str(output_path / f"checkpoint_epoch_{epoch}.npz"))
            print(f"  ✓ Checkpoint saved")
    
    # Plot training curves
    print("\n" + "=" * 60)
    print("Generating training plots...")
    print("=" * 60)
    
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = list(range(1, n_epochs + 1))
    
    plot_training_curve(
        train_epochs=epochs,
        train_values=train_mse_history,
        val_epochs=val_epochs,
        val_values=val_mse_history,
        ylabel='MSE',
        title='Training and Validation MSE',
        output_path=plots_dir / 'mse_curve.png'
    )
    
    plot_training_curve(
        train_epochs=epochs,
        train_values=train_mae_history,
        val_epochs=val_epochs,
        val_values=val_mae_history,
        ylabel='MAE',
        title='Training and Validation MAE',
        output_path=plots_dir / 'mae_curve.png'
    )
    
    # Final evaluation on test set
    if verbose:
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
    
    # Load best model
    model = CollaborativeFilteringModel.load(str(output_path / "best_model.npz"))
    
    test_users, test_books, test_ratings = dataloader.get_split_data('test')
    test_metrics = model.evaluate(test_users, test_books, test_ratings)
    
    # Get best val metrics
    val_users, val_books, val_ratings = dataloader.get_split_data('val')
    val_metrics = model.evaluate(val_users, val_books, val_ratings)
    
    if verbose:
        print(f"  Test MSE:  {test_metrics['mse']:.6f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test MAE:  {test_metrics['mae']:.4f}")
        print(f"  Test Acc:  {test_metrics['accuracy']:.4f}")
    
    # Save final model
    model.save(str(output_path / "final_model.npz"))
    if verbose:
        print(f"\nFinal model saved to {output_path / 'final_model.npz'}")
    
    return val_metrics, test_metrics, dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Train Collaborative Filtering model on Goodreads dataset"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/user_data/sheels/Spring2026/10718_mlip/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/user_data/saksham3/courses/10-718-project/checkpoints",
        help="Path to save model checkpoints"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Dimension of user/book embeddings"
    )
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=0.01,
        help="Regularization weight for unit norm constraint"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=86,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Evaluate and save checkpoint every N epochs"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling and save results to logs/profiling.txt"
    )
    
    args = parser.parse_args()
    
    def run_training():
        return train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            embedding_dim=args.embedding_dim,
            reg_weight=args.reg_weight,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            random_seed=args.random_seed,
            save_every=args.save_every
        )
    
    if args.profile:
        # Create logs directory
        logs_dir = Path(args.output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        profiling_path = logs_dir / "profiling.txt"
        
        # Run with profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = run_training()
        
        profiler.disable()
        
        # Save profiling results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        
        # Write header and stats
        with open(profiling_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROFILING RESULTS - Sorted by Cumulative Time\n")
            f.write("=" * 80 + "\n\n")
            
            # Get cumulative stats
            stream.truncate(0)
            stream.seek(0)
            stats.print_stats(50)
            f.write(stream.getvalue())
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("PROFILING RESULTS - Sorted by Total Time\n")
            f.write("=" * 80 + "\n\n")
            
            # Get total time stats
            stream.truncate(0)
            stream.seek(0)
            stats.sort_stats('tottime')
            stats.print_stats(50)
            f.write(stream.getvalue())
        
        print(f"\nProfiling results saved to: {profiling_path}")
    else:
        run_training()


if __name__ == "__main__":
    main()
