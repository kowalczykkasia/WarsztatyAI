"""
Convert SMILES to PyTorch Geometric graph objects.

Node features per atom (18 total):
  - atomic number one-hot: C, N, O, F, P, S, Cl, Br, I, other (10)
  - formal charge: normalized (1)
  - degree: normalized (1)
  - num Hs: normalized (1)
  - is aromatic: binary (1)
  - hybridization one-hot: SP, SP2, SP3, other (4)
"""

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ATOMIC_NUMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]

NODE_FEATURE_DIM = (len(ATOMIC_NUMS) + 1) + 1 + (5 + 1) + 1 + 1 + (len(HYBRIDIZATIONS) + 1)  # = 23


def one_hot(value, categories):
    enc = [0] * (len(categories) + 1)  # last = "other"
    if value in categories:
        enc[categories.index(value)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_features(atom) -> list:
    return (
        one_hot(atom.GetAtomicNum(), ATOMIC_NUMS)          # 10
        + [atom.GetFormalCharge() / 4.0]                   # 1
        + one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4])  # 6
        + [atom.GetTotalNumHs() / 4.0]                     # 1
        + [int(atom.GetIsAromatic())]                      # 1
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS) # 4
    )


def smiles_to_graph(smiles: str, y: float = None) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edges (undirected → add both directions)
    edges = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[u, v], [v, u]]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data
