#!/usr/bin/env python3
"""
Train improved GIN model on ChEMBL IC50 data.

Usage:
    python train_gnn_improved.py --split random
    python train_gnn_improved.py --split scaffold
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.featurize import smiles_to_graph, NODE_FEATURE_DIM
from scripts.gnn_model_improved import IC50GIN
from scripts.utils import random_split, scaffold_split


def load_graphs(data_path, max_samples=None):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if max_samples:
        df = df.head(max_samples)

    print(f"  Loaded {len(df)} molecules")

    smiles = df["canonical_smiles"].values
    targets = df["pIC50"].values

    graphs = []
    valid = 0
    for smi, y in zip(smiles, targets):
        g = smiles_to_graph(smi, y=float(y))
        if g is not None:
            graphs.append(g)
            valid += 1

    print(f"  Successfully converted {valid} molecules ({len(df) - valid} dropped)")
    return graphs, smiles


def split_graphs(graphs, smiles, split_type="random", test_size=0.1, val_size=0.1, random_state=42):
    idx = np.arange(len(graphs))
    dummy_y = np.zeros(len(graphs))  # Dummy targets for splitting

    if split_type == "random":
        idx_train, idx_val, idx_test, _, _, _ = random_split(
            idx, dummy_y, test_size=test_size, val_size=val_size, random_state=random_state
        )
    elif split_type == "scaffold":
        idx_train, idx_val, idx_test, _, _, _ = scaffold_split(
            idx, dummy_y, smiles, test_size=test_size, val_size=val_size, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown split: {split_type}")

    train_graphs = [graphs[i] for i in idx_train]
    val_graphs = [graphs[i] for i in idx_val]
    test_graphs = [graphs[i] for i in idx_test]

    print(f"  Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
    return train_graphs, val_graphs, test_graphs


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).squeeze()
        loss = criterion(out, batch.y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()
            loss = criterion(out, batch.y.squeeze())
            total_loss += loss.item()
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.squeeze().cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    r2 = r2_score(targets, preds)
    return total_loss / len(loader), rmse, mae, r2, preds, targets


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data_path = Path(args.data_path) if args.data_path else Path("chembl_36/chembl_36_cleaned_final.parquet")

    graphs, smiles_list = load_graphs(data_path, max_samples=args.max_samples)

    print(f"\nPerforming {args.split} split...")
    train_graphs, val_graphs, test_graphs = split_graphs(
        graphs, smiles_list, args.split
    )

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    print(f"\nInitializing GIN model (node_features={NODE_FEATURE_DIM}, hidden_dim={args.hidden_dim})...")
    model = IC50GIN(node_features=NODE_FEATURE_DIM, hidden_dim=args.hidden_dim, dropout_rate=args.dropout).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.7, min_lr=1e-6, mode='max'
    )

    print(f"\nTraining hyperparameters:")
    print(f"  lr: {args.learning_rate}, batch: {args.batch_size}, hidden_dim: {args.hidden_dim}")
    print(f"  dropout: {args.dropout}, epochs: {args.epochs}")

    best_val_r2 = float("-inf")
    patience_counter = 0
    patience = 20

    print("\nStarting training...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        test_loss, test_rmse, test_mae, test_r2, _, _ = evaluate(model, test_loader, criterion, device)

        scheduler.step(val_r2)
        lr = optimizer.param_groups[0]["lr"]

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), f"best_gnn_model_{args.split}_improved.pt")
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                f"Val (RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}, lr={lr:.2e}) | "
                f"Test (RMSE: {test_rmse:.4f}, R²: {test_r2:.4f})"
            )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model and evaluate
    print("\n" + "="*70)
    model.load_state_dict(torch.load(f"best_gnn_model_{args.split}_improved.pt"))

    train_loss, train_rmse, train_mae, train_r2, train_preds, train_targets = evaluate(
        model, train_loader, criterion, device
    )
    val_loss, val_rmse, val_mae, val_r2, val_preds, val_targets = evaluate(
        model, val_loader, criterion, device
    )
    test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nFinal Results (GIN - {args.split} split):")
    print(f"{'Dataset':<10} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"{'-'*42}")
    print(f"{'Train':<10} {train_rmse:>10.4f} {train_mae:>10.4f} {train_r2:>10.4f}")
    print(f"{'Val':<10} {val_rmse:>10.4f} {val_mae:>10.4f} {val_r2:>10.4f}")
    print(f"{'Test':<10} {test_rmse:>10.4f} {test_mae:>10.4f} {test_r2:>10.4f}")
    print("="*70)

    # Save results
    results_df = pd.DataFrame({
        "split": [args.split] * len(test_targets),
        "y_true": test_targets,
        "y_pred": test_preds,
        "error": np.abs(test_preds - test_targets),
    })
    results_df.to_csv(f"gnn_results_{args.split}_improved.csv", index=False)
    print(f"✅ Results saved to gnn_results_{args.split}_improved.csv")
    print(f"✅ Model saved to best_gnn_model_{args.split}_improved.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved GIN model on ChEMBL IC50 data")
    parser.add_argument("--split", type=str, choices=["random", "scaffold"], default="random")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()
    main(args)
