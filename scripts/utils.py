"""Utility functions for MLP training on ChEMBL IC50 data."""

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import train_test_split
from collections import defaultdict


def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES to Morgan fingerprint (bit vector).

    Args:
        smiles: SMILES string
        radius: Morgan radius (default 2)
        n_bits: Fingerprint size (default 2048)

    Returns:
        numpy array of shape (n_bits,) with dtype float32, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def get_scaffold(smiles):
    """
    Extract Murcko scaffold from SMILES for scaffold split.

    Args:
        smiles: SMILES string

    Returns:
        Scaffold SMILES string, or None if invalid
    """
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def random_split(X, y, test_size=0.1, val_size=0.1, random_state=42):
    """
    Random 80/10/10 train/val/test split.

    Args:
        X: features array
        y: targets array
        test_size: fraction for test set (default 0.1)
        val_size: fraction of remaining for val (default 0.1)
        random_state: seed for reproducibility

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: 80% train+val, 20% test
    # But we want 10% test, so: (1 - 0.1) / (1 - 0.1) for train+val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: split remaining into train/val
    # val_size is fraction of temp, so 0.111... ≈ 10% of original
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scaffold_split(X, y, smiles_list, test_size=0.1, val_size=0.1, random_state=42):
    """
    Scaffold-based split using Murcko scaffolds.
    Ensures molecules with same scaffold don't leak between train/val/test.

    Args:
        X: features array (indices must match smiles_list)
        y: targets array
        smiles_list: list of SMILES strings
        test_size: fraction for test set
        val_size: fraction of remaining for val
        random_state: seed

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Extract scaffolds
    scaffolds = [get_scaffold(smi) for smi in smiles_list]

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)

    # Shuffle scaffolds with fixed seed
    rng = np.random.RandomState(random_state)
    scaffold_list = list(scaffold_to_indices.keys())
    rng.shuffle(scaffold_list)

    # Distribute scaffolds into train/val/test
    n_scaffolds = len(scaffold_list)
    test_idx = int(n_scaffolds * test_size)
    val_idx = test_idx + int(n_scaffolds * val_size / (1 - test_size))

    test_scaffolds = set(scaffold_list[:test_idx])
    val_scaffolds = set(scaffold_list[test_idx:val_idx])
    train_scaffolds = set(scaffold_list[val_idx:])

    # Collect molecule indices by split
    train_indices = []
    val_indices = []
    test_indices = []

    for scaffold in scaffold_to_indices:
        if scaffold in test_scaffolds:
            test_indices.extend(scaffold_to_indices[scaffold])
        elif scaffold in val_scaffolds:
            val_indices.extend(scaffold_to_indices[scaffold])
        elif scaffold in train_scaffolds:
            train_indices.extend(scaffold_to_indices[scaffold])

    # Convert to arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


class MorganFingerprintDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Morgan fingerprints + IC50 targets."""

    def __init__(self, X, y):
        """
        Args:
            X: numpy array of shape (N, 2048) with fingerprints
            y: numpy array of shape (N,) with IC50 values (pIC50)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)  # shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
