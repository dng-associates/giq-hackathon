import numpy as np
import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Chronological split by date to avoid temporal leakage.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' was not found in dataframe.")
    if not (0 < val_fraction < 1):
        raise ValueError("'val_fraction' must be in (0, 1).")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
    out = out.sort_values(date_col).reset_index(drop=True)

    unique_dates = (
        out[date_col]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if len(unique_dates) < 2:
        raise ValueError("Need at least two distinct dates for time-based split.")

    split_idx = int(len(unique_dates) * (1 - val_fraction))
    split_idx = max(1, min(split_idx, len(unique_dates) - 1))
    split_date = unique_dates.iloc[split_idx]

    train_df = out[out[date_col] < split_date].reset_index(drop=True)
    val_df = out[out[date_col] >= split_date].reset_index(drop=True)
    return train_df, val_df, split_date


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int = 32,
    shuffle_train: bool = False,
) -> tuple[object, object]:
    """
    Create train/validation DataLoaders from already split arrays.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x_train = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train_t)
    val_dataset = TensorDataset(x_val, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
