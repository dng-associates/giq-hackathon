import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.classical.mlp import MLP          # Code to get the model
from src.eval.metrics import evaluate      # Code to get the metrics

from src.data.loader import load_data
from src.data.preprocessing import build_temporal_dataset, prepare_features
from src.data.splits import create_dataloaders, time_based_split


LAG_COL_PATTERN = re.compile(r"^price_lag_(\d+)$")
ROLL_COL_PATTERN = re.compile(r"^price_roll_(?:mean|std)_(\d+)$")


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _extract_windows_from_refined_columns(columns, pattern: re.Pattern[str]) -> list[int]:
    out = []
    for col in columns:
        match = pattern.match(col)
        if match:
            out.append(int(match.group(1)))
    return sorted(set(out))


def _is_refined_dataframe(columns) -> bool:
    return (
        "price_norm" in columns
        and "price" in columns
        and any(col.startswith("price_lag_") for col in columns)
    )


def _load_training_dataframe(
    train_path: str,
    *,
    date_col: str = "Date",
    data_dir: str = "DATASETS",
):
    path_str = train_path.strip()
    if not path_str:
        raise ValueError("'--train-path' cannot be empty.")

    if path_str.startswith("s3://"):
        s3_prefix, _, filename = path_str.rpartition("/")
        if not s3_prefix or not filename:
            raise ValueError(
                f"Invalid S3 train path: {path_str}. Use s3://bucket/path/file.ext"
            )
        return load_data(
            filename=filename,
            data_dir=s3_prefix,
            parse_dates=True,
            date_col=date_col,
        )

    file_path = Path(path_str)
    if file_path.exists():
        return load_data(
            filename=file_path.name,
            data_dir=file_path.parent,
            parse_dates=True,
            date_col=date_col,
        )

    # Backward-compatible mode: treat train-path as filename inside data_dir.
    if file_path.parent == Path("."):
        try:
            return load_data(
                filename=file_path.name,
                data_dir=data_dir,
                parse_dates=True,
                date_col=date_col,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Training file not found in '{file_path}' or '{Path(data_dir) / file_path.name}'. "
                "Generate it with `python generate_refined.py` or pass another file in --train-path."
            ) from exc

    raise FileNotFoundError(
        f"Training file not found: {file_path}. "
        "Generate it with `python generate_refined.py` or pass another file in --train-path."
    )


def _squeeze_predictions(y_hat):
    if y_hat.ndim == 2 and y_hat.shape[1] == 1:
        return y_hat.squeeze(1)
    return y_hat


def _collect_targets_and_predictions(
    model,
    data_loader,
    torch,
    *,
    device,
    non_blocking: bool = False,
):
    y_true_parts = []
    y_pred_parts = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_hat = _squeeze_predictions(model(x_batch))
            y_true_parts.append(y_batch.detach().cpu().numpy())
            y_pred_parts.append(y_hat.detach().cpu().numpy())

    if not y_true_parts:
        return np.array([]), np.array([])

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return y_true, y_pred


def _resolve_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")

    return torch.device("cpu")


def _extract_model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


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
    parser.add_argument(
        "--train-path",
        default="results/refined_train.csv",
        help=(
            "Training file path (local path or s3://.../file). "
            "Default uses refined dataset generated by generate_refined.py."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (CPU parallelism).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin host memory for faster CPU->GPU transfers.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (requires --num-workers > 0).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched by each worker (used only when --num-workers > 0).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use torch.nn.DataParallel on multiple CUDA GPUs.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="Set torch CPU threads (>0). Keeps default when 0.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. If omitted, uses all available training batches as epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Epoch interval for logging train/val metrics.",
    )
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
    parser.add_argument(
        "--quantum-backend",
        default="merlin",
        choices=("merlin", "simulated", "auto"),
        help=(
            "Quantum backend for hybrid runs: "
            "'merlin' requires MerLin installed, "
            "'simulated' forces fallback, "
            "'auto' tries MerLin then falls back."
        ),
    )
    parser.add_argument("--lags", default="1,5,10", help="Lag windows, comma-separated.")
    parser.add_argument(
        "--rolling-windows",
        default="5,20",
        help="Rolling windows, comma-separated.",
    )
    args = parser.parse_args()
    if args.log_every <= 0:
        raise ValueError("'--log-every' must be > 0.")
    if args.num_workers < 0:
        raise ValueError("'--num-workers' must be >= 0.")
    if args.prefetch_factor <= 0:
        raise ValueError("'--prefetch-factor' must be > 0.")
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("'--persistent-workers' requires '--num-workers' > 0.")
    if args.torch_num_threads < 0:
        raise ValueError("'--torch-num-threads' must be >= 0.")

    lags = _parse_int_list(args.lags)
    rolling_windows = _parse_int_list(args.rolling_windows)

    df_loaded = _load_training_dataframe(
        args.train_path,
        date_col="Date",
        data_dir=args.data_dir,
    )
    if _is_refined_dataframe(df_loaded.columns):
        df_feat = df_loaded.copy()

        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in {"price", "price_norm"}]
        if not feature_cols:
            raise ValueError(
                "Refined dataset has no feature columns. "
                "Expected numeric feature columns plus 'price_norm'."
            )

        inferred_lags = _extract_windows_from_refined_columns(df_feat.columns, LAG_COL_PATTERN)
        inferred_rolling = _extract_windows_from_refined_columns(
            df_feat.columns,
            ROLL_COL_PATTERN,
        )
        if inferred_lags:
            lags = inferred_lags
        if inferred_rolling:
            rolling_windows = inferred_rolling

        scaler_mean = (
            float(df_feat["price"].mean())
            if "price" in df_feat.columns
            else None
        )
        scaler_scale = (
            float(df_feat["price"].std(ddof=0))
            if "price" in df_feat.columns
            else None
        )
        print(f"Using refined dataset: {args.train_path}")
    else:
        df_raw = df_loaded
        df_feat, feature_cols, scaler = build_temporal_dataset(
            df_raw,
            lags=lags,
            rolling_windows=rolling_windows,
        )
        scaler_mean = (
            float(scaler.mean_[0]) if getattr(scaler, "mean_", None) is not None else None
        )
        scaler_scale = (
            float(scaler.scale_[0]) if getattr(scaler, "scale_", None) is not None else None
        )
        print(f"Using raw dataset: {args.train_path}")

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

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    device = _resolve_device(torch, args.device)
    if args.data_parallel and device.type != "cuda":
        raise ValueError("'--data-parallel' is only supported with CUDA.")
    if args.data_parallel and torch.cuda.device_count() < 2:
        raise ValueError("'--data-parallel' requires at least 2 CUDA GPUs.")

    train_loader, val_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=args.batch_size,
        shuffle_train=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(
        "Parallel settings: "
        f"device={device} | workers={args.num_workers} | "
        f"pin_memory={args.pin_memory} | persistent_workers={args.persistent_workers}"
    )

    total_epochs = args.epochs if args.epochs is not None else len(train_loader)
    if total_epochs <= 0:
        raise ValueError(
            "Computed '--epochs' is <= 0. Check training data size or pass a positive --epochs."
        )
    if args.epochs is None:
        print(f"No --epochs provided. Using all available epochs: {total_epochs}")
        
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
            quantum_backend=args.quantum_backend,
        )
        print(f"Hybrid backend: requested={args.quantum_backend} active={model.quantum.backend}")
    else:
        model = MLP(input_dim=len(feature_cols))

    model = model.to(device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel enabled on {torch.cuda.device_count()} GPUs.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    history: list[dict[str, Any]] = []
    use_non_blocking = args.pin_memory and device.type == "cuda"

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=use_non_blocking)
            y_batch = y_batch.to(device, non_blocking=use_non_blocking)

            optimizer.zero_grad()
            y_hat = _squeeze_predictions(model(X_batch))
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(y_batch.shape[0])
            epoch_loss_sum += float(loss.item()) * batch_size
            epoch_count += batch_size

        should_log = (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == total_epochs
        )
        if should_log:
            y_train_true, y_train_pred = _collect_targets_and_predictions(
                model,
                train_loader,
                torch,
                device=device,
                non_blocking=use_non_blocking,
            )
            y_val_true, y_val_pred = _collect_targets_and_predictions(
                model,
                val_loader,
                torch,
                device=device,
                non_blocking=use_non_blocking,
            )
            train_metrics = evaluate(y_train_true, y_train_pred)
            val_metrics = evaluate(y_val_true, y_val_pred)
            epoch_avg_loss = epoch_loss_sum / max(1, epoch_count)

            epoch_info = {
                "epoch": epoch,
                "train_loss": epoch_avg_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            history.append(epoch_info)
            print(
                "Epoch "
                f"{epoch:03d}/{total_epochs} | "
                f"train_loss={epoch_avg_loss:.6f} | "
                f"train_rmse={train_metrics['rmse']:.6f} | "
                f"val_rmse={val_metrics['rmse']:.6f} | "
                f"train_r2={train_metrics['r2']:.6f} | "
                f"val_r2={val_metrics['r2']:.6f}"
            )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    model_state = _extract_model_state_dict(model)
    torch.save(model_state, output_dir / "model.pt")       # pesos do modelo
    torch.save({                                                    # checkpoint completo
        "model_state": model_state,
        "model_type": args.model_type,
        "train_path": args.train_path,
        "feature_cols": feature_cols,
        "split_date": str(split_date.date()),
        "val_fraction": args.val_fraction,
        "epochs": total_epochs,
        "lr": args.lr,
        "log_every": args.log_every,
        "lags": lags,
        "rolling_windows": rolling_windows,
        "target_scaler_mean": scaler_mean,
        "target_scaler_scale": scaler_scale,
        "n_modes": args.n_modes,
        "n_photons": args.n_photons,
        "quantum_depth": args.quantum_depth,
        "encoding_type": args.encoding_type,
        "measurement": args.measurement,
        "quantum_backend": args.quantum_backend,
    }, output_dir / "checkpoint.pt")

    model.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_pred_t = _squeeze_predictions(model(x_val_t))
        y_pred = y_pred_t.detach().cpu().numpy()
    metrics = evaluate(y_val, y_pred)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {output_dir}/")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
