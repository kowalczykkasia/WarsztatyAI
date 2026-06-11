"""
Targeted ChEMBL join: filter activities FIRST, then join.

Instead of sampling random activities, we pre-filter by:
  - standard_type in (IC50, Ki, Kd, EC50)
  - standard_units in convertible units (nM, uM, mM, pM, fM, M) or null

Then join to get target/molecule/structure info, and apply remaining filters.

Usage:
    python scripts/join_filtered.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PARQUET_DIR = "chembl_36/parquet"
OUTPUT = "chembl_36/chembl_36_joined_filtered.parquet"

print("Loading tables...")
activities = pd.read_parquet(f"{PARQUET_DIR}/activities.parquet")
assays = pd.read_parquet(f"{PARQUET_DIR}/assays.parquet")
target_dict = pd.read_parquet(f"{PARQUET_DIR}/target_dictionary.parquet")
mol_dict = pd.read_parquet(f"{PARQUET_DIR}/molecule_dictionary.parquet")
comp_struct = pd.read_parquet(f"{PARQUET_DIR}/compound_structures.parquet")

comp_struct = comp_struct.drop(columns=["molfile"], errors="ignore")

print(f"Total activities: {len(activities):,}")

# Pre-filter activities
valid_types = ["IC50", "Ki", "Kd", "EC50"]
convertible_units = ["nM", "uM", "mM", "pM", "fM", "M"]

filtered = activities[
    activities["standard_type"].isin(valid_types) &
    (activities["standard_units"].isin(convertible_units) | activities["standard_units"].isna()) &
    activities["standard_value"].notna() &
    (activities["standard_value"] > 0)
].copy()

print(f"After pre-filter (type + units + value > 0): {len(filtered):,}")

# Join assays to get confidence_score and target
print("Joining assays...")
filtered = filtered.merge(
    assays.rename(columns={"chembl_id": "assay_chembl_id", "src_id": "assay_src_id",
                            "doc_id": "assay_doc_id"}),
    on="assay_id", how="left",
)

print("Joining target_dictionary...")
filtered = filtered.merge(
    target_dict.rename(columns={"pref_name": "target_name", "chembl_id": "target_chembl_id",
                                "organism": "target_organism", "tax_id": "target_tax_id"}),
    on="tid", how="left",
)

# Filter Homo sapiens + confidence_score
filtered = filtered[filtered["target_organism"] == "Homo sapiens"]
print(f"After Homo sapiens filter: {len(filtered):,}")

filtered = filtered[filtered["confidence_score"] > 3]
print(f"After confidence_score > 3: {len(filtered):,}")

# Join structures (we need canonical_smiles)
print("Joining compound_structures...")
filtered = filtered.merge(
    mol_dict.rename(columns={"chembl_id": "molecule_chembl_id", "pref_name": "molecule_name"}),
    on="molregno", how="left",
)
filtered = filtered.merge(comp_struct, on="molregno", how="left")

print(f"After joins: {len(filtered):,}")

# Save intermediate joined file
filtered.to_parquet(OUTPUT, engine="pyarrow", index=False)
print(f"\nSaved to {OUTPUT}")
print(f"Shape: {filtered.shape}")
print(f"Columns: {list(filtered.columns)}")
