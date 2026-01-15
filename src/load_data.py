# src/load_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path = Path("data/raw")
    filename: str = "bank.csv"

    @property
    def raw_file(self) -> Path:
        return self.raw_dir / self.filename


def load_raw_bank_data(paths: DataPaths = DataPaths()) -> pd.DataFrame:
    """
    Load raw Bank Marketing dataset (UCI 'bank' folder).
    Assumes CSV with ';' separator.
    """
    if not paths.raw_file.exists():
        raise FileNotFoundError(
            f"No encontré el archivo: {paths.raw_file}. "
            "Asegúrate de poner el CSV en data/raw/."
        )

    df = pd.read_csv(paths.raw_file, sep=";")
    return df
