import argparse
import json
from pathlib import Path

from src.classical.mlp import MLP          # Code to get the model
from src.eval.metrics import evaluate      # Code to get the metrics

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
    parser.add_argument(
        "--data-dir",
        default="DATASETS",
        help="Dataset directory or S3 prefix (e.g. s3://my-bucket/raw).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--model-type",
        default="normal",
        choices=("normal", "hybrid"),
        help="Choose classical MLP ('normal') or MerLin hybrid model ('hybrid').",
    )
    parser.add_argument("--n-modes", type=int, default=4, help="Number of quantum modes.")
    parser.add_argument("--n-photons", type=int, default=2, help="Number of photons.")
    parser.add_argument(
        "--quantum-depth",
        type=int,
        default=2,
        help="Trainable depth for the quantum encoder.",
    )
    parser.add_argument(
        "--encoding-type",
        default="angle",
        choices=("angle", "amplitude"),
        help="Quantum encoding type used by the hybrid model.",
    )
    parser.add_argument(
        "--measurement",
        default="probs",
        help="MerLin measurement strategy (e.g. probs, amplitudes).",
    )
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
        import torch
    except ModuleNotFoundError:
        print(
            "Torch is not installed. Preprocessing completed, but training was skipped. "
            "Install torch to train (`pip install torch`)."
        )
        return

    train_loader, val_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=args.batch_size,
        shuffle_train=False,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        
    ##=======================================
    ## Start saving the model and artifacts
    ##=======================================

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if args.model_type == "hybrid":
        from src.hybrid.model import MerlinHybridRegressor

        model = MerlinHybridRegressor(
            input_dim=len(feature_cols),
            n_modes=args.n_modes,
            n_photons=args.n_photons,
            trainable_depth=args.quantum_depth,
            measurement=args.measurement,
            encoding_type=args.encoding_type,
        )
    else:
        model = MLP(input_dim=len(feature_cols))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = model(X_batch)
            if y_hat.ndim == 2 and y_hat.shape[1] == 1:
                y_hat = y_hat.squeeze(1)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), output_dir / "model.pt")       # pesos do modelo
    torch.save({                                                    # checkpoint completo
        "model_state": model.state_dict(),
        "model_type": args.model_type,
        "feature_cols": feature_cols,
        "split_date": str(split_date.date()),
        "lags": lags,
        "rolling_windows": rolling_windows,
        "n_modes": args.n_modes,
        "n_photons": args.n_photons,
        "quantum_depth": args.quantum_depth,
        "encoding_type": args.encoding_type,
        "measurement": args.measurement,
    }, output_dir / "checkpoint.pt")

    model.eval()
    with torch.no_grad():
        y_pred_t = model(torch.tensor(X_val, dtype=torch.float32))
        if y_pred_t.ndim == 2 and y_pred_t.shape[1] == 1:
            y_pred_t = y_pred_t.squeeze(1)
        y_pred = y_pred_t.numpy()
    metrics = evaluate(y_val, y_pred)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {output_dir}/")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
