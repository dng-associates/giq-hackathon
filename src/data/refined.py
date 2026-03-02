from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.loader import load_data
from src.data.preprocessing import build_temporal_dataset


def _parse_int_list(values: Iterable[int] | str) -> list[int]:
    if isinstance(values, str):
        return [int(part.strip()) for part in values.split(",") if part.strip()]
    return [int(v) for v in values]


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    no_scheme = s3_uri[5:]
    bucket, _, key = no_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(
            f"Invalid S3 URI: {s3_uri}. Expected format: s3://<bucket>/<key>"
        )
    return bucket, key


@dataclass
class RefinedDatasetConfig:
    data_dir: str | Path = "DATASETS"
    source_filename: str = "train.xlsx"
    date_col: str = "Date"
    lags: tuple[int, ...] = (1, 5, 10)
    rolling_windows: tuple[int, ...] = (5, 20)


class RefinedDatasetManager:
    """
    Build a refined dataset from raw train data and optionally publish it to S3.
    """

    def __init__(self, config: RefinedDatasetConfig | None = None) -> None:
        self.config = config or RefinedDatasetConfig()

    def build_refined(self) -> pd.DataFrame:
        df_raw = load_data(
            filename=self.config.source_filename,
            data_dir=self.config.data_dir,
            parse_dates=True,
            date_col=self.config.date_col,
        )
        df_refined, _, _ = build_temporal_dataset(
            df_raw,
            date_col=self.config.date_col,
            lags=self.config.lags,
            rolling_windows=self.config.rolling_windows,
        )
        return df_refined

    def save_local(
        self,
        df_refined: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()

        if suffix == ".csv":
            df_refined.to_csv(output, index=False)
        elif suffix in {".xlsx", ".xls"}:
            df_refined.to_excel(output, index=False)
        else:
            raise ValueError(
                f"Unsupported output format: {suffix}. Use .csv or .xlsx/.xls."
            )
        return output

    def upload_to_s3(
        self,
        df_refined: pd.DataFrame,
        s3_uri: str,
        *,
        file_format: str = "csv",
    ) -> str:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "boto3 is required for S3 upload. Install it with `pip install boto3`."
            ) from exc

        bucket, key = _parse_s3_uri(s3_uri)
        fmt = file_format.lower()

        if fmt == "csv":
            body = df_refined.to_csv(index=False).encode("utf-8")
            content_type = "text/csv"
        elif fmt in {"xlsx", "xls"}:
            buffer = io.BytesIO()
            df_refined.to_excel(buffer, index=False)
            body = buffer.getvalue()
            content_type = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            raise ValueError("Unsupported S3 file format. Use 'csv' or 'xlsx'.")

        s3 = boto3.client("s3")
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        except (NoCredentialsError, ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"Failed to upload refined dataset to {s3_uri}") from exc

        return s3_uri

    @classmethod
    def from_args(
        cls,
        *,
        data_dir: str | Path = "DATASETS",
        source_filename: str = "train.xlsx",
        date_col: str = "Date",
        lags: Iterable[int] | str = (1, 5, 10),
        rolling_windows: Iterable[int] | str = (5, 20),
    ) -> "RefinedDatasetManager":
        cfg = RefinedDatasetConfig(
            data_dir=data_dir,
            source_filename=source_filename,
            date_col=date_col,
            lags=tuple(_parse_int_list(lags)),
            rolling_windows=tuple(_parse_int_list(rolling_windows)),
        )
        return cls(cfg)
