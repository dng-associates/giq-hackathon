import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    class StandardScaler:  # type: ignore[override]
        """Lightweight fallback when scikit-learn is unavailable."""

        def __init__(self) -> None:
            self.mean_: np.ndarray | None = None
            self.scale_: np.ndarray | None = None

        def fit(self, X: np.ndarray) -> "StandardScaler":
            arr = np.asarray(X, dtype=float)
            self.mean_ = arr.mean(axis=0)
            self.scale_ = arr.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError("StandardScaler must be fitted before transform.")
            arr = np.asarray(X, dtype=float)
            return (arr - self.mean_) / self.scale_

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return self.fit(X).transform(X)

TENOR_MATURITY_PATTERN = re.compile(
    r"Tenor\s*:\s*(?P<tenor>\d+(?:\.\d+)?)\s*;\s*Maturity\s*:\s*(?P<maturity>\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def melt_maturities(
    df: pd.DataFrame,
    date_col: str = "Date",
    value_name: str = "price",
) -> pd.DataFrame:
    """
    Convert wide tenor/maturity columns to long format.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' was not found in dataframe.")

    value_cols = [col for col in df.columns if col != date_col]
    long_df = df.melt(
        id_vars=[date_col],
        value_vars=value_cols,
        var_name="contract",
        value_name=value_name,
    )

    parsed = long_df["contract"].str.extract(TENOR_MATURITY_PATTERN)
    if parsed.isna().all().all():
        raise ValueError(
            "Could not parse tenor/maturity from column names. "
            "Expected format like 'Tenor : 1; Maturity : 0.0833333333'."
        )

    long_df["tenor"] = pd.to_numeric(parsed["tenor"], errors="coerce")
    long_df["maturity"] = pd.to_numeric(parsed["maturity"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df[date_col] = pd.to_datetime(long_df[date_col], errors="coerce", dayfirst=True)

    long_df = long_df.dropna(subset=[date_col, "tenor", "maturity", value_name])
    long_df = long_df.sort_values(["tenor", "maturity", date_col]).reset_index(drop=True)
    return long_df


def _clean_positive_ints(values: Iterable[int], name: str) -> list[int]:
    clean = sorted({int(v) for v in values if int(v) > 0})
    if not clean:
        raise ValueError(f"'{name}' must include at least one positive integer.")
    return clean


def add_temporal_features(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    target_col: str = "price",
    group_cols: Sequence[str] = ("tenor", "maturity"),
    lags: Iterable[int] = (1, 5, 10),
    rolling_windows: Iterable[int] = (5, 20),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Add lag/rolling features so the model captures temporal dynamics.
    """
    required_cols = {date_col, target_col, *group_cols}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    lags = _clean_positive_ints(lags, "lags")
    rolling_windows = _clean_positive_ints(rolling_windows, "rolling_windows")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
    out = out.sort_values([*group_cols, date_col]).reset_index(drop=True)

    grouped = out.groupby(list(group_cols), sort=False)[target_col]
    created_cols: list[str] = []

    for lag in lags:
        col = f"{target_col}_lag_{lag}"
        out[col] = grouped.shift(lag)
        created_cols.append(col)

    out[f"{target_col}_diff_1"] = grouped.diff(1)
    out[f"{target_col}_return_1"] = out[f"{target_col}_diff_1"] / grouped.shift(1)
    created_cols.extend([f"{target_col}_diff_1", f"{target_col}_return_1"])

    for window in rolling_windows:
        mean_col = f"{target_col}_roll_mean_{window}"
        std_col = f"{target_col}_roll_std_{window}"
        out[mean_col] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=window).mean()
        )
        out[std_col] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=window).std()
        )
        created_cols.extend([mean_col, std_col])

    min_date = out[date_col].min()
    out["time_idx_days"] = (out[date_col] - min_date).dt.days.astype(float)
    created_cols.append("time_idx_days")

    if dropna:
        out = out.dropna(subset=created_cols).reset_index(drop=True)

    return out


def normalize_prices(
    df: pd.DataFrame,
    *,
    price_col: str = "price",
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit/Apply StandardScaler on the target price column.
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' was not found in dataframe.")

    out = df.copy()
    if out.empty:
        if scaler is None:
            raise ValueError(
                "Cannot normalize prices from an empty dataframe. "
                "Check if temporal preprocessing removed all rows."
            )
        out[f"{price_col}_norm"] = pd.Series(index=out.index, dtype=float)
        return out, scaler

    if scaler is None:
        scaler = StandardScaler()
        out[f"{price_col}_norm"] = scaler.fit_transform(out[[price_col]])
    else:
        out[f"{price_col}_norm"] = scaler.transform(out[[price_col]])

    return out, scaler


def prepare_features(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    target_col: str = "price_norm",
    return_feature_names: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build model-ready arrays X and y.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' was not found in dataframe.")

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]

    if not feature_cols:
        raise ValueError("No feature columns were provided/found.")

    X = df[list(feature_cols)].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    if return_feature_names:
        return X, y, list(feature_cols)
    return X, y


def build_temporal_dataset(
    df_raw: pd.DataFrame,
    *,
    date_col: str = "Date",
    lags: Iterable[int] = (1, 5, 10),
    rolling_windows: Iterable[int] = (5, 20),
) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    """
    End-to-end helper: melt -> temporal features -> normalize.
    """
    df_long = melt_maturities(df_raw, date_col=date_col, value_name="price")
    df_feat = add_temporal_features(
        df_long,
        date_col=date_col,
        target_col="price",
        lags=lags,
        rolling_windows=rolling_windows,
        dropna=True,
    )
    df_feat, scaler = normalize_prices(df_feat, price_col="price")

    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in {"price", "price_norm"}]
    return df_feat, feature_cols, scaler
