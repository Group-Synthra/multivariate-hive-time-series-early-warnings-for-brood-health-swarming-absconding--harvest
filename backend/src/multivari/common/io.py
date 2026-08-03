from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path, *, sheet_name: str = "Common_Dataset") -> pd.DataFrame:
    """Read the immutable source dataset or a processed table."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a compact typed table for reuse by every module."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return output
