from __future__ import annotations

import argparse
from pathlib import Path

from src.data.refined import RefinedDatasetManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate refined dataset from train.xlsx and optionally upload to S3."
    )
    parser.add_argument(
        "--data-dir",
        default="DATASETS",
        help="Local dataset folder or S3 prefix containing train.xlsx.",
    )
    parser.add_argument(
        "--source-filename",
        default="train.xlsx",
        help="Source filename used to build the refined dataset.",
    )
    parser.add_argument(
        "--date-col",
        default="Date",
        help="Date column name.",
    )
    parser.add_argument("--lags", default="1,5,10", help="Lag windows, comma-separated.")
    parser.add_argument(
        "--rolling-windows",
        default="5,20",
        help="Rolling windows, comma-separated.",
    )
    parser.add_argument(
        "--output-local",
        default="results/refined_train.csv",
        help="Local path to save refined dataset (.csv/.xlsx).",
    )
    parser.add_argument(
        "--s3-destination",
        default=None,
        help="Optional destination S3 URI (e.g. s3://my-bucket/refined/refined_train.csv).",
    )
    parser.add_argument(
        "--s3-format",
        choices=("csv", "xlsx"),
        default="csv",
        help="Format to upload to S3 when --s3-destination is provided.",
    )
    args = parser.parse_args()

    manager = RefinedDatasetManager.from_args(
        data_dir=args.data_dir,
        source_filename=args.source_filename,
        date_col=args.date_col,
        lags=args.lags,
        rolling_windows=args.rolling_windows,
    )

    df_refined = manager.build_refined()
    local_path = manager.save_local(df_refined, Path(args.output_local))
    print(f"Refined dataset generated with {len(df_refined)} rows.")
    print(f"Saved locally at: {local_path}")

    if args.s3_destination:
        uploaded_uri = manager.upload_to_s3(
            df_refined,
            args.s3_destination,
            file_format=args.s3_format,
        )
        print(f"Uploaded to S3: {uploaded_uri}")


if __name__ == "__main__":
    main()
