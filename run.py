import argparse

from src.data.loader import load_train_data
from src.data.preprocessing import build_temporal_dataset, prepare_features
from src.data.splits import create_dataloaders, time_based_split


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal preprocessing pipeline for option-pricing data."
    )
    parser.add_argument("--config", default=None, help="Reserved for config integration.")
    parser.add_argument("--data-dir", default="DATASETS", help="Dataset directory.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lags", default="1,5,10", help="Lag windows, comma-separated.")
    parser.add_argument(
        "--rolling-windows",
        default="5,20",
        help="Rolling windows, comma-separated.",
    )
    args = parser.parse_args()

    lags = _parse_int_list(args.lags)
    rolling_windows = _parse_int_list(args.rolling_windows)

    df_raw = load_train_data(args.data_dir)
    df_feat, feature_cols, _ = build_temporal_dataset(
        df_raw,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    train_df, val_df, split_date = time_based_split(
        df_feat,
        val_fraction=args.val_fraction,
    )

    X_train, y_train = prepare_features(
        train_df,
        feature_cols=feature_cols,
        target_col="price_norm",
    )
    X_val, y_val = prepare_features(
        val_df,
        feature_cols=feature_cols,
        target_col="price_norm",
    )

    print(f"Features: {len(feature_cols)} columns")
    print(f"Split date: {split_date.date()}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    try:
        train_loader, val_loader = create_dataloaders(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=args.batch_size,
            shuffle_train=False,
        )
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    except ModuleNotFoundError:
        print("Torch is not installed. Arrays were created, but DataLoaders were skipped.")


if __name__ == "__main__":
    main()
