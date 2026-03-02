from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Allow running as a script: `python src/data/technical_report.py`
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.data.loader import load_data
from src.data.preprocessing import (
    add_temporal_features,
    melt_maturities,
    normalize_prices,
    prepare_features,
)
from src.data.splits import create_dataloaders, time_based_split
from src.eval.metrics import evaluate
from src.inference.pipeline import (
    load_model,
    make_feature_matrix,
    predict_array,
    preprocess_with_checkpoint,
)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required to generate the technical report charts. "
        "Install it with `pip install matplotlib`."
    ) from exc


@dataclass
class BenchmarkResult:
    name: str
    repeats: int
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass
class TechnicalReportConfig:
    data_dir: str = "DATASETS"
    train_filename: str = "train.xlsx"
    checkpoint_path: str = "results/checkpoint.pt"
    training_history_path: str = "results/training_history.json"
    output_markdown: str = "docs/technical_report.md"
    assets_dir: str = "docs/assets/technical_report"
    benchmark_repeats: int = 3
    val_fraction: float = 0.2
    batch_size: int = 256


def _safe_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def _benchmark(name: str, fn: Callable[[], Any], repeats: int) -> BenchmarkResult:
    if repeats <= 0:
        raise ValueError("'repeats' must be > 0.")

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    return BenchmarkResult(
        name=name,
        repeats=repeats,
        mean_ms=float(np.mean(times_ms)),
        min_ms=float(np.min(times_ms)),
        max_ms=float(np.max(times_ms)),
        std_ms=float(statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0),
    )


def _to_markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _plot_benchmark_chart(results: list[BenchmarkResult], output_path: Path) -> None:
    ordered = sorted(results, key=lambda item: item.mean_ms, reverse=True)
    names = [item.name for item in ordered]
    means = [item.mean_ms for item in ordered]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    ax.barh(names, means, color="#1f77b4")
    ax.set_title("Execution Time by Function")
    ax.set_xlabel("Mean time (ms)")
    ax.set_ylabel("Function")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_data_profile(df_long: pd.DataFrame, output_path: Path) -> dict[str, float]:
    counts = df_long.groupby(["tenor", "maturity"]).size()
    mean_price_by_date = (
        df_long.groupby("Date")["price"]
        .mean()
        .sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(counts.to_numpy(), bins=min(30, max(5, len(counts) // 8)), color="#2ca02c")
    axes[0].set_title("History Size per Contract")
    axes[0].set_xlabel("Rows per (tenor, maturity)")
    axes[0].set_ylabel("Frequency")

    axes[1].plot(mean_price_by_date.index, mean_price_by_date.to_numpy(), color="#ff7f0e")
    axes[1].set_title("Average Price Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Mean price")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "contracts_count": float(len(counts)),
        "contract_history_min": float(counts.min()),
        "contract_history_max": float(counts.max()),
        "contract_history_mean": float(counts.mean()),
    }


def _plot_training_history(history: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    train_rmse = [float(item["train_metrics"]["rmse"]) for item in history]
    val_rmse = [float(item["val_metrics"]["rmse"]) for item in history]
    train_r2 = [float(item["train_metrics"]["r2"]) for item in history]
    val_r2 = [float(item["val_metrics"]["r2"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_rmse, marker="o", label="Train RMSE", color="#1f77b4")
    axes[0].plot(epochs, val_rmse, marker="o", label="Val RMSE", color="#d62728")
    axes[0].set_title("RMSE by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].plot(epochs, train_r2, marker="o", label="Train R2", color="#1f77b4")
    axes[1].plot(epochs, val_r2, marker="o", label="Val R2", color="#d62728")
    axes[1].set_title("R2 by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R2")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_prediction_quality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_scatter: Path,
    output_residuals: Path,
) -> dict[str, float]:
    min_v = float(np.min([y_true.min(), y_pred.min()]))
    max_v = float(np.max([y_true.max(), y_pred.max()]))
    residuals = y_true - y_pred

    fig_scatter, ax_scatter = plt.subplots(figsize=(6.5, 6))
    ax_scatter.scatter(y_true, y_pred, s=8, alpha=0.35, color="#1f77b4", edgecolors="none")
    ax_scatter.plot([min_v, max_v], [min_v, max_v], color="#d62728", linestyle="--", linewidth=2)
    ax_scatter.set_title("Predicted vs True (price_norm)")
    ax_scatter.set_xlabel("True")
    ax_scatter.set_ylabel("Predicted")
    fig_scatter.tight_layout()
    fig_scatter.savefig(output_scatter, dpi=160)
    plt.close(fig_scatter)

    fig_res, ax_res = plt.subplots(figsize=(8, 4.8))
    ax_res.hist(residuals, bins=40, color="#9467bd", alpha=0.85)
    ax_res.set_title("Residual Distribution (true - pred)")
    ax_res.set_xlabel("Residual")
    ax_res.set_ylabel("Frequency")
    fig_res.tight_layout()
    fig_res.savefig(output_residuals, dpi=160)
    plt.close(fig_res)

    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_abs_p95": float(np.percentile(np.abs(residuals), 95)),
    }


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if isinstance(content, list):
        return content
    return []


def _best_history_snapshot(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return min(history, key=lambda item: float(item["val_metrics"]["rmse"]))


def generate_technical_report(config: TechnicalReportConfig) -> Path:
    if config.benchmark_repeats <= 0:
        raise ValueError("'benchmark_repeats' must be > 0.")
    if not (0.0 < config.val_fraction < 1.0):
        raise ValueError("'val_fraction' must be in (0, 1).")

    _safe_style()

    output_md = Path(config.output_markdown)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = Path(config.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    chart_benchmark = assets_dir / "benchmark_times.png"
    chart_data_profile = assets_dir / "data_profile.png"
    chart_training = assets_dir / "training_quality.png"
    chart_scatter = assets_dir / "prediction_scatter.png"
    chart_residuals = assets_dir / "prediction_residuals.png"
    data_json_path = assets_dir / "technical_report_data.json"

    repeats = config.benchmark_repeats

    benchmark_rows: list[BenchmarkResult] = []
    benchmark_rows.append(
        _benchmark(
            "load_data",
            lambda: load_data(
                filename=config.train_filename,
                data_dir=config.data_dir,
                parse_dates=True,
                date_col="Date",
            ),
            repeats,
        )
    )

    df_raw = load_data(
        filename=config.train_filename,
        data_dir=config.data_dir,
        parse_dates=True,
        date_col="Date",
    )
    benchmark_rows.append(
        _benchmark("melt_maturities", lambda: melt_maturities(df_raw, date_col="Date"), repeats)
    )

    df_long = melt_maturities(df_raw, date_col="Date")
    benchmark_rows.append(
        _benchmark(
            "add_temporal_features",
            lambda: add_temporal_features(
                df_long,
                date_col="Date",
                target_col="price",
                lags=(1, 5, 10),
                rolling_windows=(5, 20),
                dropna=True,
            ),
            repeats,
        )
    )

    df_feat = add_temporal_features(
        df_long,
        date_col="Date",
        target_col="price",
        lags=(1, 5, 10),
        rolling_windows=(5, 20),
        dropna=True,
    )
    benchmark_rows.append(_benchmark("normalize_prices", lambda: normalize_prices(df_feat), repeats))

    df_norm, _ = normalize_prices(df_feat)
    benchmark_rows.append(
        _benchmark(
            "time_based_split",
            lambda: time_based_split(df_norm, val_fraction=config.val_fraction),
            repeats,
        )
    )
    train_df, val_df, split_date = time_based_split(df_norm, val_fraction=config.val_fraction)

    benchmark_rows.append(
        _benchmark(
            "prepare_features",
            lambda: prepare_features(train_df, target_col="price_norm"),
            repeats,
        )
    )
    X_train, y_train, feature_cols = prepare_features(
        train_df,
        target_col="price_norm",
        return_feature_names=True,
    )
    X_val, y_val = prepare_features(val_df, feature_cols=feature_cols, target_col="price_norm")

    dataloader_status = "ok"
    try:
        benchmark_rows.append(
            _benchmark(
                "create_dataloaders",
                lambda: create_dataloaders(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    batch_size=config.batch_size,
                    shuffle_train=False,
                    num_workers=0,
                    pin_memory=False,
                ),
                repeats,
            )
        )
    except ModuleNotFoundError as exc:
        dataloader_status = f"skipped ({exc})"

    ckpt_path = Path(config.checkpoint_path)
    model_metrics: dict[str, float] | None = None
    residual_summary: dict[str, float] | None = None
    inference_summary: dict[str, Any] = {}
    if ckpt_path.exists():
        benchmark_rows.append(_benchmark("load_model", lambda: load_model(ckpt_path), 1))
        model, checkpoint = load_model(ckpt_path)

        benchmark_rows.append(
            _benchmark(
                "preprocess_with_checkpoint",
                lambda: preprocess_with_checkpoint(df_raw, checkpoint),
                repeats,
            )
        )
        df_ckpt = preprocess_with_checkpoint(df_raw, checkpoint)
        _, val_ckpt, split_date_ckpt = time_based_split(df_ckpt, val_fraction=config.val_fraction)
        ckpt_feature_cols = list(checkpoint["feature_cols"])

        benchmark_rows.append(
            _benchmark(
                "make_feature_matrix",
                lambda: make_feature_matrix(val_ckpt, ckpt_feature_cols),
                repeats,
            )
        )
        X_val_ckpt = make_feature_matrix(val_ckpt, ckpt_feature_cols)
        y_val_ckpt = val_ckpt["price_norm"].to_numpy(dtype=np.float32)

        benchmark_rows.append(
            _benchmark("predict_array", lambda: predict_array(model, X_val_ckpt), repeats)
        )
        y_pred_ckpt = predict_array(model, X_val_ckpt)
        model_metrics = evaluate(y_val_ckpt, y_pred_ckpt)
        residual_summary = _plot_prediction_quality(
            y_val_ckpt,
            y_pred_ckpt,
            output_scatter=chart_scatter,
            output_residuals=chart_residuals,
        )

        inference_summary = {
            "checkpoint": str(ckpt_path),
            "checkpoint_model_type": str(checkpoint.get("model_type", "unknown")),
            "checkpoint_split_date": str(split_date_ckpt.date()),
            "validation_rows": int(len(val_ckpt)),
        }

    data_profile_summary = _plot_data_profile(df_long, chart_data_profile)
    _plot_benchmark_chart(benchmark_rows, chart_benchmark)

    history = _read_history(Path(config.training_history_path))
    best_epoch = _best_history_snapshot(history)
    if history:
        _plot_training_history(history, chart_training)

    report_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "dataset": {
            "raw_rows": int(len(df_raw)),
            "raw_columns": int(len(df_raw.columns)),
            "long_rows": int(len(df_long)),
            "feature_rows": int(len(df_norm)),
            "feature_columns": int(len(feature_cols)),
            "split_date": str(split_date.date()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            **data_profile_summary,
        },
        "benchmark": [asdict(item) for item in benchmark_rows],
        "training_history_points": len(history),
        "best_epoch": best_epoch,
        "model_metrics": model_metrics,
        "residual_summary": residual_summary,
        "inference_summary": inference_summary,
        "dataloader_status": dataloader_status,
    }
    with open(data_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    benchmark_table_rows = [
        [
            item.name,
            f"{item.mean_ms:.2f}",
            f"{item.min_ms:.2f}",
            f"{item.max_ms:.2f}",
            f"{item.std_ms:.2f}",
            str(item.repeats),
        ]
        for item in sorted(benchmark_rows, key=lambda x: x.mean_ms, reverse=True)
    ]

    rel_benchmark = chart_benchmark.relative_to(output_md.parent).as_posix()
    rel_data_profile = chart_data_profile.relative_to(output_md.parent).as_posix()
    rel_training = chart_training.relative_to(output_md.parent).as_posix()
    rel_scatter = chart_scatter.relative_to(output_md.parent).as_posix()
    rel_residuals = chart_residuals.relative_to(output_md.parent).as_posix()
    rel_data_json = data_json_path.relative_to(output_md.parent).as_posix()

    lines: list[str] = []
    lines.append("# Technical Report")
    lines.append("")
    lines.append(
        f"Generated on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"with Python {sys.version.split()[0]} on {platform.platform()}."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report profiles key data/inference functions, summarizes dataset quality, "
        "and tracks model quality using current project artifacts."
    )
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(
        _to_markdown_table(
            ["Metric", "Value"],
            [
                ["Raw rows", str(report_payload["dataset"]["raw_rows"])],
                ["Raw columns", str(report_payload["dataset"]["raw_columns"])],
                ["Long-format rows", str(report_payload["dataset"]["long_rows"])],
                ["Feature rows", str(report_payload["dataset"]["feature_rows"])],
                ["Feature columns", str(report_payload["dataset"]["feature_columns"])],
                ["Split date", str(report_payload["dataset"]["split_date"])],
                ["Train rows", str(report_payload["dataset"]["train_rows"])],
                ["Validation rows", str(report_payload["dataset"]["val_rows"])],
                ["Contracts", f"{report_payload['dataset']['contracts_count']:.0f}"],
                ["Contract history min", f"{report_payload['dataset']['contract_history_min']:.0f}"],
                ["Contract history max", f"{report_payload['dataset']['contract_history_max']:.0f}"],
                ["Contract history mean", f"{report_payload['dataset']['contract_history_mean']:.2f}"],
            ],
        )
    )
    lines.append("")
    lines.append(f"![Data profile]({rel_data_profile})")
    lines.append("")
    lines.append("## Function Execution Time")
    lines.append("")
    lines.append(
        _to_markdown_table(
            ["Function", "Mean (ms)", "Min (ms)", "Max (ms)", "Std (ms)", "Repeats"],
            benchmark_table_rows,
        )
    )
    lines.append("")
    lines.append(f"![Benchmark chart]({rel_benchmark})")
    lines.append("")
    lines.append("## Training Quality")
    lines.append("")
    if history and best_epoch is not None:
        lines.append(
            _to_markdown_table(
                ["Metric", "Value"],
                [
                    ["History points", str(len(history))],
                    ["Best epoch (val RMSE)", str(best_epoch["epoch"])],
                    ["Best val RMSE", f"{float(best_epoch['val_metrics']['rmse']):.6f}"],
                    ["Best val R2", f"{float(best_epoch['val_metrics']['r2']):.6f}"],
                ],
            )
        )
        lines.append("")
        lines.append(f"![Training quality]({rel_training})")
    else:
        lines.append("No `training_history.json` data found for plotting.")
    lines.append("")
    lines.append("## Inference Quality")
    lines.append("")
    if model_metrics is not None:
        inference_rows = [
            ["MAE", f"{model_metrics['mae']:.6f}"],
            ["MSE", f"{model_metrics['mse']:.6f}"],
            ["RMSE", f"{model_metrics['rmse']:.6f}"],
            ["R2", f"{model_metrics['r2']:.6f}"],
        ]
        if residual_summary is not None:
            inference_rows.extend(
                [
                    ["Residual mean", f"{residual_summary['residual_mean']:.6f}"],
                    ["Residual std", f"{residual_summary['residual_std']:.6f}"],
                    ["|Residual| p95", f"{residual_summary['residual_abs_p95']:.6f}"],
                ]
            )
        lines.append(_to_markdown_table(["Metric", "Value"], inference_rows))
        lines.append("")
        lines.append(f"![Prediction scatter]({rel_scatter})")
        lines.append("")
        lines.append(f"![Prediction residuals]({rel_residuals})")
        lines.append("")
        lines.append(
            _to_markdown_table(
                ["Inference context", "Value"],
                [
                    ["Checkpoint", inference_summary.get("checkpoint", "n/a")],
                    ["Model type", inference_summary.get("checkpoint_model_type", "n/a")],
                    ["Split date", inference_summary.get("checkpoint_split_date", "n/a")],
                    ["Validation rows", str(inference_summary.get("validation_rows", "n/a"))],
                ],
            )
        )
    else:
        lines.append(
            "Checkpoint metrics were not computed. Ensure `results/checkpoint.pt` exists "
            "and supports preprocessing for the selected dataset."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- DataLoader benchmark status: "
        f"`{dataloader_status}`."
    )
    lines.append(
        "- Machine-readable payload: "
        f"`{rel_data_json}`."
    )
    lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_md


def _parse_args() -> TechnicalReportConfig:
    parser = argparse.ArgumentParser(
        description="Generate technical performance report with charts."
    )
    parser.add_argument("--data-dir", default="DATASETS", help="Dataset folder or S3 prefix.")
    parser.add_argument("--train-filename", default="train.xlsx", help="Raw training file name.")
    parser.add_argument(
        "--checkpoint-path",
        default="results/checkpoint.pt",
        help="Model checkpoint used for inference-quality analysis.",
    )
    parser.add_argument(
        "--training-history-path",
        default="results/training_history.json",
        help="Training history JSON path.",
    )
    parser.add_argument(
        "--output-markdown",
        default="docs/technical_report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--assets-dir",
        default="docs/assets/technical_report",
        help="Output directory for generated report charts.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="How many times each function benchmark is repeated.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction used in split and inference quality checks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for DataLoader benchmark.",
    )
    args = parser.parse_args()
    return TechnicalReportConfig(
        data_dir=args.data_dir,
        train_filename=args.train_filename,
        checkpoint_path=args.checkpoint_path,
        training_history_path=args.training_history_path,
        output_markdown=args.output_markdown,
        assets_dir=args.assets_dir,
        benchmark_repeats=args.benchmark_repeats,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
    )


def main() -> None:
    config = _parse_args()
    output_path = generate_technical_report(config)
    print(f"Technical report generated: {output_path}")


if __name__ == "__main__":
    main()
