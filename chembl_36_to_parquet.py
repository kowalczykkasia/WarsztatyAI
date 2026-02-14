"""
Convert ChEMBL 36 SQLite database to Parquet files.

The DB has some index corruption, so we use PRAGMA writable_schema=ON
to bypass integrity checks and read data directly.
"""

import sqlite3
import os
import sys
import pandas as pd

DB_PATH = "chembl_36/chembl_36_sqlite/chembl_36.db"
OUTPUT_DIR = "chembl_36/parquet"

TABLES = [
    "action_type", "assay_type", "chembl_id_lookup", "confidence_score_lookup",
    "curation_lookup", "chembl_release", "source", "relationship_type",
    "target_type", "variant_sequences", "bioassay_ontology", "data_validity_lookup",
    "activity_smid", "activity_stds_lookup", "assay_classification",
    "atc_classification", "bio_component_sequences", "component_sequences",
    "protein_classification", "domains", "go_classification", "structural_alert_sets",
    "products", "organism_class", "patent_use_codes", "usan_stems", "version",
    "cell_dictionary", "docs", "target_dictionary", "tissue_dictionary",
    "molecule_dictionary", "activity_supp", "component_class", "component_domains",
    "component_go", "component_synonyms", "structural_alerts", "defined_daily_dose",
    "product_patents", "protein_class_synonyms", "assays", "compound_records",
    "binding_sites", "biotherapeutics", "compound_properties",
    "compound_structural_alerts", "compound_structures",
    "molecule_atc_classification", "molecule_hierarchy", "molecule_synonyms",
    "target_components", "target_relations", "activities", "assay_class_map",
    "assay_parameters", "biotherapeutic_components", "drug_indication",
    "drug_mechanism", "drug_warning", "formulations", "metabolism",
    "site_components", "activity_properties", "activity_supp_map",
    "indication_refs", "ligand_eff", "mechanism_refs", "metabolism_refs",
    "pesticide_classification", "predicted_binding_domains", "warning_refs",
    "pesticide_class_mapping",
]

CHUNK_SIZE = 500_000


def convert_table(conn, table_name, output_dir):
    """Read a table in chunks and write to a single Parquet file."""
    out_path = os.path.join(output_dir, f"{table_name}.parquet")
    if os.path.exists(out_path):
        print(f"  SKIP {table_name} (already exists)")
        return True

    try:
        # Use chunks for large tables to avoid memory issues
        chunks = pd.read_sql_query(
            f'SELECT * FROM "{table_name}"', conn, chunksize=CHUNK_SIZE
        )
        first = True
        total_rows = 0
        for chunk in chunks:
            if first:
                chunk.to_parquet(out_path, engine="pyarrow", index=False)
                first = True  # will be overwritten below for multi-chunk
                total_rows += len(chunk)
                # For multi-chunk, we need to accumulate - switch strategy
                remaining = list(chunks)  # exhaust iterator
                if remaining:
                    all_chunks = [chunk] + remaining
                    total_rows = sum(len(c) for c in all_chunks)
                    df = pd.concat(all_chunks, ignore_index=True)
                    df.to_parquet(out_path, engine="pyarrow", index=False)
                break

        print(f"  OK   {table_name}: {total_rows:,} rows -> {out_path}")
        return True
    except Exception as e:
        print(f"  FAIL {table_name}: {e}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA writable_schema=ON")

    ok, fail = 0, 0
    for table in TABLES:
        success = convert_table(conn, table, OUTPUT_DIR)
        if success:
            ok += 1
        else:
            fail += 1

    conn.close()
    print(f"\nDone: {ok} succeeded, {fail} failed out of {len(TABLES)} tables.")


if __name__ == "__main__":
    main()
