"""
Clean ChEMBL joined parquet file using standard_value unit conversion.

Pipeline:
1. Filter target_organism == "Homo sapiens"
2. Filter confidence_score > 3
3. Filter standard_type in IC50/Ki/Kd/EC50
4. Impute missing standard_units as nM when value in [0.01, 1e6]
5. Keep only convertible units (nM, uM, mM, pM, fM, M)
6. Convert all to nM
7. Remove outliers using IQR on log10 scale
8. Compute pIC50 = 9 - log10(standard_value_nM)
9. Drop rows with missing canonical_smiles or pIC50
10. Deduplicate: average pIC50 per canonical_smiles

Usage:
    python scripts/clean_chembl.py --input chembl_36/chembl_36_joined_100k.parquet \
                                   --output chembl_36/chembl_36_cleaned_std_100k.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def clean_chembl(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path, engine="pyarrow")
    print(f"  Loaded: {len(df):,} rows")

    # 1. Filter Homo sapiens
    df = df[df["target_organism"] == "Homo sapiens"]
    print(f"  After Homo sapiens filter: {len(df):,}")

    # 2. Filter confidence_score > 3
    df = df[df["confidence_score"] > 3]
    print(f"  After confidence_score > 3: {len(df):,}")

    # 3. Filter standard_type
    valid_types = ["IC50", "Ki", "Kd", "EC50"]
    df = df[df["standard_type"].isin(valid_types)]
    print(f"  After standard_type filter (IC50/Ki/Kd/EC50): {len(df):,}")

    # 4. Impute missing standard_units as nM when value in plausible range
    mask_missing = df["standard_units"].isna() & df["standard_value"].notna()
    mask_range = (df["standard_value"] >= 0.01) & (df["standard_value"] <= 1e6)
    imputed = (mask_missing & mask_range).sum()
    df.loc[mask_missing & mask_range, "standard_units"] = "nM"
    print(f"  Imputed {imputed:,} missing units as nM")

    # 5. Keep only convertible units
    convertible_units = ["nM", "uM", "mM", "pM", "fM", "M"]
    before = len(df)
    df = df[df["standard_units"].isin(convertible_units)]
    print(f"  After unit filter: {len(df):,} (dropped {before - len(df):,} non-convertible)")

    # 6. Convert to nM
    unit_factors = {"nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0,
                    "pM": 0.001, "fM": 0.000001, "M": 1e9}
    df["standard_value_nM"] = df["standard_value"] * df["standard_units"].map(unit_factors)
    df = df[df["standard_value_nM"].notna() & (df["standard_value_nM"] > 0)]
    print(f"  After nM conversion: {len(df):,}")

    # 7. Remove outliers via IQR on log10 scale
    log_vals = np.log10(df["standard_value_nM"].values)
    q1, q3 = np.percentile(log_vals, 25), np.percentile(log_vals, 75)
    iqr = q3 - q1
    lower = 10 ** (q1 - 1.5 * iqr)
    upper = 10 ** (q3 + 1.5 * iqr)
    before = len(df)
    df = df[(df["standard_value_nM"] >= lower) & (df["standard_value_nM"] <= upper)]
    print(f"  After outlier removal (IQR): {len(df):,} (dropped {before - len(df):,})")
    print(f"    Bounds: [{lower:.4f}, {upper:,.0f}] nM")

    # 8. Compute pIC50
    df["pIC50"] = 9 - np.log10(df["standard_value_nM"])
    print(f"  pIC50 range: [{df['pIC50'].min():.2f}, {df['pIC50'].max():.2f}]")

    # 9. Drop rows missing canonical_smiles or pIC50
    before = len(df)
    df = df.dropna(subset=["canonical_smiles", "pIC50"])
    print(f"  After dropping missing SMILES/pIC50: {len(df):,} (dropped {before - len(df):,})")

    # 10. Deduplicate by canonical_smiles
    before = len(df)
    df = df.groupby("canonical_smiles", as_index=False)["pIC50"].mean()
    print(f"  After deduplication: {len(df):,} unique molecules (removed {before - len(df):,} duplicates)")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"\nSaved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"\npIC50 stats:")
    print(f"  mean: {df['pIC50'].mean():.3f}")
    print(f"  std:  {df['pIC50'].std():.3f}")
    print(f"  min:  {df['pIC50'].min():.3f}")
    print(f"  max:  {df['pIC50'].max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean ChEMBL joined parquet using standard_value")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()
    clean_chembl(args.input, args.output)
