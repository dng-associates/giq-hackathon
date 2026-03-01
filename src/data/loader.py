from pathlib import Path
from typing import Union

import pandas as pd


def load_data(
    filename: str = "train.xlsx",
    data_dir: Union[str, Path] = "DATASETS",
    *,
    parse_dates: bool = True,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Load a dataset from DATASETS as CSV or Excel.

    Args:
        filename: Name of the file (e.g. ``train.xlsx``).
        data_dir: Directory that contains the file.
        parse_dates: Whether to parse the date column.
        date_col: Date column name.

    Returns:
        Loaded dataframe.
    """
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    return df


def load_train_data(
    data_dir: Union[str, Path] = "DATASETS",
    *,
    parse_dates: bool = True,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Convenience wrapper to load train.xlsx."""
    return load_data("train.xlsx", data_dir, parse_dates=parse_dates, date_col=date_col)


def load_test_template(
    data_dir: Union[str, Path] = "DATASETS",
    *,
    parse_dates: bool = True,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Convenience wrapper to load test_template.xlsx."""
    return load_data(
        "test_template.xlsx",
        data_dir,
        parse_dates=parse_dates,
        date_col=date_col,
    )
    
