from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_data
from src.data.splits import time_based_split
from src.eval.metrics import evaluate
from src.inference.pipeline import (
    load_model,
    make_feature_matrix,
    predict_array,
    preprocess_with_checkpoint,
)


class ModelEvaluator:
    """
    Utility for evaluating a saved checkpoint on a chronological validation split.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.checkpoint = load_model(self.checkpoint_path)

    def evaluate_dataframe(
        self,
        df_raw: pd.DataFrame,
        *,
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        df_feat = preprocess_with_checkpoint(df_raw, self.checkpoint)
        _, val_df, split_date = time_based_split(df_feat, val_fraction=val_fraction)

        feature_cols = list(self.checkpoint["feature_cols"])
        X_val = make_feature_matrix(val_df, feature_cols)
        y_val = val_df["price_norm"].to_numpy(dtype=np.float32)
        y_pred = predict_array(self.model, X_val)

        metrics = evaluate(y_val, y_pred)
        return {
            "metrics": metrics,
            "split_date": str(split_date.date()),
            "val_rows": int(len(val_df)),
            "checkpoint": str(self.checkpoint_path),
            "model_type": str(self.checkpoint.get("model_type", "normal")),
        }

    def evaluate_dataset(
        self,
        *,
        filename: str = "train.xlsx",
        data_dir: str | Path = "DATASETS",
        val_fraction: float = 0.2,
    ) -> dict[str, Any]:
        df_raw = load_data(filename=filename, data_dir=data_dir)
        return self.evaluate_dataframe(df_raw, val_fraction=val_fraction)
