"""
Ingest layer — loads raw patient vitals from CSV or generates synthetic data.
In a real DE setting this would pull from an S3 bucket, Kafka stream, or HL7 feed.
"""

import os
import pandas as pd
from data.generate_data import generate_dataset


def ingest(source: str = "auto", csv_path: str = "data/patients.csv") -> pd.DataFrame:
    """
    source='auto'  → use CSV if it exists, otherwise generate synthetic data
    source='csv'   → always load from csv_path
    source='generate' → always generate fresh synthetic data
    """
    if source == "csv" or (source == "auto" and os.path.exists(csv_path)):
        df = pd.read_csv(csv_path)
        print(f"[Ingest] Loaded {len(df)} records from {csv_path}")
    else:
        print("[Ingest] Generating synthetic ICU dataset...")
        df = generate_dataset(n_patients=1200, seed=42)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[Ingest] Saved to {csv_path} ({len(df)} records)")
    return df
