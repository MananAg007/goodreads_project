#!/usr/bin/env python3
"""
Hyperparameter tuning script for Collaborative Filtering model.

Iterates over different regularization weight values and selects the best model.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import train


def hyperparameter_tuning(
    csv_path: str,
    output_dir: str,
    reg_weights: List[float] = None,
    learning_rates: List[float] = None,
    embedding_dim: int = 64,
    n_epochs: int = 20,
    random_seed: int = 86,
    batch_size: int = 512,
    save_every: int = 5,
    use_map: bool = False
) -> None:
    """
    Run hyperparameter tuning over different regularization weights and learning rates.
    
    Args:
        csv_path: Path to the interactions CSV file
        output_dir: Path to save results
        reg_weights: List of regularization weights to try
        learning_rates: List of learning rates to try
        embedding_dim: Dimension of embeddings
        n_epochs: Number of training epochs per configuration
        random_seed: Random seed for reproducibility
        batch_size: Batch size for training
        save_every: Evaluate and save every N epochs
        use_map: If True, load user_id_map.json and book_id_map.json from CSV directory
    """
    if reg_weights is None:
        reg_weights = [0, 0.01, 0.03, 0.1, 0.3, 1.0]
    
    if learning_rates is None:
        learning_rates = [0.01, 0.1]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = []
    best_overall_val_mae = float('inf')
    best_learning_rate = None
    best_reg_weight = None
    best_test_metrics = None
    best_val_metrics = None
    best_model_dir = None
    
    # Dataloader will be initialized on first run and reused
    dataloader = None
    
    # Try each combination of learning rate and regularization weight
    for learning_rate in learning_rates:
        for reg_weight in reg_weights:
            print(f"\n{'='*60}")
            print(f"Training with lr = {learning_rate}, reg_weight = {reg_weight}")
            print(f"{'='*60}")
            
            # Create output directory for this run
            run_output_dir = output_path / f"lr_{learning_rate}_reg_{reg_weight}"
            
            # Train model using train function from train.py
            # Reuse dataloader across runs
            val_metrics, test_metrics, dataloader = train(
                csv_path=csv_path,
                output_dir=str(run_output_dir),
                embedding_dim=embedding_dim,
                reg_weight=reg_weight,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                random_seed=random_seed,
                save_every=save_every,
                verbose=False,
                dataloader=dataloader,
                use_map=use_map
            )
            
            print(f"  Val MAE: {val_metrics['mae']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Test MAE: {test_metrics['mae']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
            
            results.append({
                'learning_rate': learning_rate,
                'reg_weight': reg_weight,
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_accuracy': val_metrics['accuracy'],
                'test_mse': test_metrics['mse'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_accuracy': test_metrics['accuracy']
            })
            
            if val_metrics['mae'] < best_overall_val_mae:
                best_overall_val_mae = val_metrics['mae']
                best_learning_rate = learning_rate
                best_reg_weight = reg_weight
                best_test_metrics = test_metrics
                best_val_metrics = val_metrics
                best_model_dir = run_output_dir
    
    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 80)
    print(f"\n{'lr':<8} {'reg_weight':<12} {'Val MAE':<10} {'Val Acc':<10} {'Test MAE':<10} {'Test Acc':<10}")
    print("-" * 70)
    
    for r in results:
        marker = " *" if (r['learning_rate'] == best_learning_rate and r['reg_weight'] == best_reg_weight) else ""
        print(f"{r['learning_rate']:<8} {r['reg_weight']:<12} {r['val_mae']:<10.4f} {r['val_accuracy']:<10.4f} "
              f"{r['test_mae']:<10.4f} {r['test_accuracy']:<10.4f}{marker}")
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: lr = {best_learning_rate}, reg_weight = {best_reg_weight}")
    print("=" * 80)
    print(f"  Val MAE:  {best_val_metrics['mae']:.4f}")
    print(f"  Test MSE:  {best_test_metrics['mse']:.6f}")
    print(f"  Test RMSE: {best_test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {best_test_metrics['mae']:.4f}")
    print(f"  Test Acc:  {best_test_metrics['accuracy']:.4f}")
    
    # Copy best model to output root
    if best_model_dir is not None:
        best_model_src = best_model_dir / "best_model.npz"
        best_model_dst = output_path / "best_tuned_model.npz"
        shutil.copy(best_model_src, best_model_dst)
        print(f"\nBest model saved to: {best_model_dst}")
    
    # Save results to CSV
    results_path = output_path / "tuning_results.csv"
    with open(results_path, 'w') as f:
        f.write("learning_rate,reg_weight,val_mse,val_mae,val_rmse,val_accuracy,test_mse,test_mae,test_rmse,test_accuracy\n")
        for r in results:
            f.write(f"{r['learning_rate']},{r['reg_weight']},{r['val_mse']},{r['val_mae']},{r['val_rmse']},"
                    f"{r['val_accuracy']},{r['test_mse']},{r['test_mae']},{r['test_rmse']},{r['test_accuracy']}\n")
    print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Collaborative Filtering model"
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/data/user_data/sheels/Spring2026/10718_mlip/data/goodreads_interactions.csv",
        help="Path to the interactions CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/user_data/saksham3/courses/10-718-project/checkpoints/baseline_no_filter/",
        help="Path to save tuning results"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Dimension of user/book embeddings"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Number of training epochs per configuration"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training"
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
        help="Evaluate and save every N epochs"
    )
    parser.add_argument(
        "--use_map",
        action="store_true",
        help="Load user_id_map.json and book_id_map.json from CSV directory"
    )
    
    args = parser.parse_args()
    
    hyperparameter_tuning(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        n_epochs=args.n_epochs,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        save_every=args.save_every,
        use_map=args.use_map
    )


if __name__ == "__main__":
    main()
