from __future__ import annotations

import numpy as np


def _to_1d_numpy(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def evaluate(y_true, y_pred) -> dict[str, float]:
    """
    Minimal metrics template for regression.

    Returns:
        dict with keys: mae, mse, rmse, r2
    """
    y_true_arr = _to_1d_numpy(y_true)
    y_pred_arr = _to_1d_numpy(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )

    error = y_true_arr - y_pred_arr
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))

    denom = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    if denom == 0.0:
        r2 = 0.0
    else:
        r2 = float(1.0 - np.sum(error**2) / denom)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
