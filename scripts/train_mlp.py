"""
Train MLP model on ChEMBL IC50 data.

Usage:
    python train_mlp.py --split random   # Random 80/10/10
    python train_mlp.py --split scaffold # Scaffold-based split
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    smiles_to_morgan_fp,
    random_split,
    scaffold_split,
    MorganFingerprintDataset,
)
from model import IC50MLP


def load_and_prepare_data(data_path, n_bits=2048, max_samples=None):
    """
    Load cleaned ChEMBL data and convert SMILES to Morgan fingerprints.

    Args:
        data_path: Path to chembl_36_cleaned.parquet
        n_bits: Morgan fingerprint size
        max_samples: Limit number of samples (for testing)

    Returns:
        X (np.array), y (np.array), smiles_list (list)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if max_samples:
        df = df.head(max_samples)
        print(f"  Limited to {max_samples} samples")

    print(f"  Loaded {len(df)} molecules")

    # Deduplicate: average pIC50 for same SMILES
    before = len(df)
    df = df.groupby("canonical_smiles", as_index=False)["pIC50"].mean()
    print(f"  After deduplication: {len(df)} unique molecules (removed {before - len(df)} duplicates)")

    # Extract SMILES and targets
    smiles_list = df["canonical_smiles"].values
    y = df["pIC50"].values

    print(f"\nConverting SMILES to Morgan fingerprints (radius=2, n_bits={n_bits})...")

    # Convert to fingerprints
    fingerprints = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smiles, radius=2, n_bits=n_bits)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)
        else:
            if idx < 5:  # Log first few failures
                print(f"  Warning: Could not parse SMILES: {smiles}")

    valid_indices = np.array(valid_indices)
    X = np.array(fingerprints, dtype=np.float32)
    y = y[valid_indices]
    smiles_list = smiles_list[valid_indices]

    print(f"  Successfully converted {len(X)} molecules")
    print(f"  Dropped {len(df) - len(X)} invalid SMILES")

    return X, y, smiles_list


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, data_loader, criterion, device):
    """Evaluate on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    # Compute metrics
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    r2 = r2_score(targets, preds)

    return total_loss / len(data_loader), rmse, mae, r2, preds, targets


def main(args):
    """Main training pipeline."""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load and prepare data
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path(__file__).parent.parent / "chembl_36" / "chembl_36_cleaned.parquet"
    X, y, smiles_list = load_and_prepare_data(data_path, n_bits=2048)

    print(f"\nData summary:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y mean: {y.mean():.3f}, std: {y.std():.3f}")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")

    # Split data
    print(f"\nPerforming {args.split} split (80/10/10)...")
    if args.split == "random":
        X_train, X_val, X_test, y_train, y_val, y_test = random_split(
            X, y, test_size=0.1, val_size=0.1, random_state=42
        )
    elif args.split == "scaffold":
        X_train, X_val, X_test, y_train, y_val, y_test = scaffold_split(
            X, y, smiles_list, test_size=0.1, val_size=0.1, random_state=42
        )
    else:
        raise ValueError(f"Unknown split: {args.split}")

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Create datasets and loaders
    train_dataset = MorganFingerprintDataset(X_train, y_train)
    val_dataset = MorganFingerprintDataset(X_val, y_val)
    test_dataset = MorganFingerprintDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"\nInitializing model...")
    model = IC50MLP(input_dim=2048, hidden_dims=args.hidden_dims, dropout_rate=args.dropout)
    model.to(device)

    print(f"  Model architecture:")
    print(model)

    # Loss, optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    print(f"\nTraining hyperparameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Loss: MSE")
    print(f"  Optimizer: Adam + ReduceLROnPlateau")

    # Training loop
    print(f"\nStarting training...\n")

    best_val_r2 = float("-inf")
    patience_counter = 0
    patience = 15

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        test_loss, test_rmse, test_mae, test_r2, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Early stopping on val R²
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val (RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}) | "
            f"Test (RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}) | "
            f"lr: {current_lr:.2e}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model and final evaluation
    print(f"\nLoading best model and evaluating...")
    model.load_state_dict(torch.load("best_model.pt"))

    train_loss, train_rmse, train_mae, train_r2, train_preds, train_targets = evaluate(
        model, train_loader, criterion, device
    )
    val_loss, val_rmse, val_mae, val_r2, val_preds, val_targets = evaluate(
        model, val_loader, criterion, device
    )
    test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nFinal Results ({args.split} split):")
    print(f"{'Dataset':<10} {'Loss':>10} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print(f"{'-'*52}")
    print(f"{'Train':<10} {train_loss:>10.6f} {train_rmse:>10.4f} {train_mae:>10.4f} {train_r2:>10.4f}")
    print(f"{'Val':<10} {val_loss:>10.6f} {val_rmse:>10.4f} {val_mae:>10.4f} {val_r2:>10.4f}")
    print(f"{'Test':<10} {test_loss:>10.6f} {test_rmse:>10.4f} {test_mae:>10.4f} {test_r2:>10.4f}")

    # Save results
    results_df = pd.DataFrame({
        "split": [args.split] * len(test_targets),
        "y_true": test_targets,
        "y_pred": test_preds,
        "error": np.abs(test_preds - test_targets),
    })
    results_df.to_csv(f"results_{args.split}_split.csv", index=False)
    print(f"\nResults saved to results_{args.split}_split.csv")

    # Save model
    torch.save(model.state_dict(), f"model_{args.split}_split.pt")
    print(f"Model saved to model_{args.split}_split.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP on ChEMBL IC50 data"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["random", "scaffold"],
        default="random",
        help="Split strategy (random or scaffold)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to parquet data file (default: chembl_36/chembl_36_cleaned.parquet)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 128],
        help="Hidden layer sizes (default: 512 128)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )

    args = parser.parse_args()
    main(args)
