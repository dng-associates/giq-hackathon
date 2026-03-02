from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import load_data
from src.inference.pipeline import (
    build_predictions_frame,
    load_model,
    make_feature_matrix,
    predict_array,
    preprocess_with_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction interface for saved checkpoints.")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoint.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        default="DATASETS",
        help="Dataset directory or S3 prefix (e.g. s3://my-bucket/raw).",
    )
    parser.add_argument(
        "--filename",
        default="test_template.xlsx",
        help="Input dataset filename for predictions.",
    )
    parser.add_argument(
        "--output",
        default="results/predictions.csv",
        help="Path to output predictions CSV.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Number of rows to print in preview.",
    )
    args = parser.parse_args()

    model, checkpoint = load_model(args.checkpoint)
    df_raw = load_data(filename=args.filename, data_dir=args.data_dir)
    try:
        df_feat = preprocess_with_checkpoint(df_raw, checkpoint)
    except ValueError as exc:
        raise SystemExit(f"Inference preprocessing error: {exc}") from exc

    feature_cols = list(checkpoint["feature_cols"])
    X = make_feature_matrix(df_feat, feature_cols)
    y_pred_norm = predict_array(model, X)

    pred_df = build_predictions_frame(df_feat, y_pred_norm, checkpoint)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(f"Rows: {len(pred_df)}")
    print(pred_df.head(max(1, args.preview_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
