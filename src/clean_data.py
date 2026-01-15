# src/clean_data.py
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/bank.csv")
PROCESSED_PATH = Path("data/processed/bank_clean.csv")


def clean_bank_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalizar texto
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()

    # Reemplazar 'unknown' por NaN
    df = df.replace("unknown", pd.NA)

    return df


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
