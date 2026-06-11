"""
Train GCN model on ChEMBL IC50 data.

Usage:
    python scripts/train_gnn.py --split random
    python scripts/train_gnn.py --split scaffold
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

sys.path.insert(0, str(Path(__file__).parent))

from featurize import smiles_to_graph, NODE_FEATURE_DIM
from gnn_model import IC50GCN
from utils import random_split, scaffold_split


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


def split_graphs(graphs, smiles_list, split_type, test_size=0.1, val_size=0.1, random_state=42):
    """Split list of graphs into train/val/test."""
    n = len(graphs)
    indices = np.arange(n)

    if split_type == "random":
        X_dummy = np.zeros((n, 1))
        y_dummy = np.zeros(n)
        idx_train, idx_val, idx_test, _, _, _ = random_split(
            X_dummy, y_dummy, test_size=test_size, val_size=val_size, random_state=random_state
        )
        # random_split returns arrays — we need indices back
        # Use sklearn directly for indices
        from sklearn.model_selection import train_test_split
        idx_temp, idx_test = train_test_split(indices, test_size=test_size, random_state=random_state)
        idx_train, idx_val = train_test_split(idx_temp, test_size=val_size / (1 - test_size), random_state=random_state)

    elif split_type == "scaffold":
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        def get_scaffold(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return ""
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

        scaffolds = [get_scaffold(s) for s in smiles_list]
        unique_scaffolds = list(set(scaffolds))

        rng = np.random.default_rng(random_state)
        rng.shuffle(unique_scaffolds)

        n_test = max(1, int(len(unique_scaffolds) * test_size))
        n_val = max(1, int(len(unique_scaffolds) * val_size))
        test_scaff = set(unique_scaffolds[:n_test])
        val_scaff = set(unique_scaffolds[n_test:n_test + n_val])

        idx_train, idx_val, idx_test = [], [], []
        for i, sc in enumerate(scaffolds):
            if sc in test_scaff:
                idx_test.append(i)
            elif sc in val_scaff:
                idx_val.append(i)
            else:
                idx_train.append(i)
        idx_train = np.array(idx_train)
        idx_val = np.array(idx_val)
        idx_test = np.array(idx_test)
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

    print(f"\nInitializing GCN model (node_features={NODE_FEATURE_DIM})...")
    model = IC50GCN(node_features=NODE_FEATURE_DIM).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    print(f"\nTraining hyperparameters:")
    print(f"  lr: {args.learning_rate}, batch: {args.batch_size}, epochs: {args.epochs}")

    best_val_r2 = float("-inf")
    patience_counter = 0
    patience = 15

    print("\nStarting training...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        test_loss, test_rmse, test_mae, test_r2, _, _ = evaluate(model, test_loader, criterion, device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), "best_gnn_model.pt")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
            f"Val (RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}) | "
            f"Test (RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}) | lr: {lr:.2e}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\nLoading best model and evaluating...")
    model.load_state_dict(torch.load("best_gnn_model.pt"))

    train_loss, train_rmse, train_mae, train_r2, _, _ = evaluate(model, train_loader, criterion, device)
    val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
    test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Results (GCN, {args.split} split):")
    print(f"{'Dataset':<10} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print(f"{'-'*42}")
    print(f"{'Train':<10} {train_rmse:>10.4f} {train_mae:>10.4f} {train_r2:>10.4f}")
    print(f"{'Val':<10} {val_rmse:>10.4f} {val_mae:>10.4f} {val_r2:>10.4f}")
    print(f"{'Test':<10} {test_rmse:>10.4f} {test_mae:>10.4f} {test_r2:>10.4f}")

    results_df = pd.DataFrame({
        "split": [args.split] * len(test_targets),
        "y_true": test_targets,
        "y_pred": test_preds,
        "error": np.abs(test_preds - test_targets),
    })
    results_df.to_csv(f"gnn_results_{args.split}_split.csv", index=False)
    torch.save(model.state_dict(), f"gnn_model_{args.split}_split.pt")
    print(f"\nSaved results and model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GCN on ChEMBL IC50 data")
    parser.add_argument("--split", type=str, choices=["random", "scaffold"], default="random")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
