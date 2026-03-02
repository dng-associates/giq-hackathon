from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.classical.mlp import MLP
from src.data.preprocessing import (
    StandardScaler,
    add_temporal_features,
    melt_maturities,
    normalize_prices,
)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Torch is required for model loading and inference. Install torch first."
        ) from exc
    return torch


def _as_int_list(values: Any, default: tuple[int, ...]) -> list[int]:
    if values is None:
        return list(default)
    return [int(v) for v in values]


def _build_scaler_from_checkpoint(checkpoint: dict[str, Any]) -> StandardScaler | None:
    mean = checkpoint.get("target_scaler_mean")
    scale = checkpoint.get("target_scaler_scale")
    if mean is None or scale is None:
        return None

    scaler = StandardScaler()
    scaler.mean_ = np.array([float(mean)], dtype=float)
    scaler.scale_ = np.array([float(scale)], dtype=float)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = 1
    scaler.feature_names_in_ = np.array(["price"], dtype=object)
    return scaler


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    torch = _require_torch()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if "model_state" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'model_state'.")
    if "feature_cols" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'feature_cols'.")
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict[str, Any]):
    model_type = checkpoint.get("model_type", "normal")
    input_dim = len(checkpoint["feature_cols"])

    if model_type == "hybrid":
        from src.hybrid.model import MerlinHybridRegressor

        model = MerlinHybridRegressor(
            input_dim=input_dim,
            n_modes=int(checkpoint.get("n_modes", 4)),
            n_photons=int(checkpoint.get("n_photons", 2)),
            trainable_depth=int(checkpoint.get("quantum_depth", 2)),
            measurement=str(checkpoint.get("measurement", "probs")),
            encoding_type=str(checkpoint.get("encoding_type", "angle")),
            quantum_backend=str(checkpoint.get("quantum_backend", "auto")),
        )
    elif model_type == "normal":
        model = MLP(input_dim=input_dim)
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")

    return model


def load_model(checkpoint_path: str | Path):
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def preprocess_with_checkpoint(
    df_raw: pd.DataFrame,
    checkpoint: dict[str, Any],
    *,
    date_col: str = "Date",
) -> pd.DataFrame:
    lags = _as_int_list(checkpoint.get("lags"), default=(1, 5, 10))
    rolling_windows = _as_int_list(checkpoint.get("rolling_windows"), default=(5, 20))

    df_long = melt_maturities(df_raw, date_col=date_col, value_name="price")
    if df_long.empty:
        raise ValueError(
            "No valid price rows were found after parsing input data. "
            "Check date/contract columns and ensure there are non-null prices."
        )

    df_feat = add_temporal_features(
        df_long,
        date_col=date_col,
        target_col="price",
        lags=lags,
        rolling_windows=rolling_windows,
        dropna=True,
    )
    if df_feat.empty:
        required_history = max(max(lags, default=0), max(rolling_windows, default=0))
        group_sizes = df_long.groupby(["tenor", "maturity"], dropna=False).size()
        max_history_found = int(group_sizes.max()) if not group_sizes.empty else 0
        raise ValueError(
            "Temporal preprocessing produced zero rows for inference. "
            f"This checkpoint needs at least {required_history} observations per contract "
            f"(lags={lags}, rolling_windows={rolling_windows}), but the maximum found was "
            f"{max_history_found}. Provide more historical prices in the input file."
        )

    scaler = _build_scaler_from_checkpoint(checkpoint)
    if scaler is not None:
        df_feat, _ = normalize_prices(df_feat, price_col="price", scaler=scaler)
    else:
        df_feat, _ = normalize_prices(df_feat, price_col="price")

    return df_feat


def make_feature_matrix(df_feat: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    missing = [col for col in feature_cols if col not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for inference: {missing}")
    return df_feat[list(feature_cols)].to_numpy(dtype=np.float32)


def predict_array(model, X: np.ndarray) -> np.ndarray:
    torch = _require_torch()
    with torch.no_grad():
        y_pred_t = model(torch.tensor(X, dtype=torch.float32))
        if y_pred_t.ndim == 2 and y_pred_t.shape[1] == 1:
            y_pred_t = y_pred_t.squeeze(1)
    return y_pred_t.cpu().numpy()


def denormalize_predictions(
    y_pred_norm: np.ndarray,
    checkpoint: dict[str, Any],
) -> np.ndarray | None:
    mean = checkpoint.get("target_scaler_mean")
    scale = checkpoint.get("target_scaler_scale")
    if mean is None or scale is None:
        return None

    return y_pred_norm * float(scale) + float(mean)


def build_predictions_frame(
    df_feat: pd.DataFrame,
    y_pred_norm: np.ndarray,
    checkpoint: dict[str, Any],
) -> pd.DataFrame:
    base_cols = [c for c in ("Date", "contract", "tenor", "maturity") if c in df_feat.columns]
    out = df_feat[base_cols].copy()

    out["pred_price_norm"] = y_pred_norm

    y_pred = denormalize_predictions(y_pred_norm, checkpoint)
    if y_pred is not None:
        out["pred_price"] = y_pred

    if "price" in df_feat.columns:
        out["price"] = df_feat["price"].to_numpy()
    if "price_norm" in df_feat.columns:
        out["price_norm"] = df_feat["price_norm"].to_numpy()

    return out
