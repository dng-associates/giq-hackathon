import io
from pathlib import Path
from typing import Union
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _join_s3_uri(base_uri: str, filename: str) -> str:
    base_clean = base_uri.rstrip("/")
    return f"{base_clean}/{filename}"


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


def _read_s3_bytes(s3_uri: str) -> bytes:
    bucket, key = _parse_s3_uri(s3_uri)

    def _read_public_object() -> bytes:
        encoded_key = quote(key, safe="/")
        url = f"https://{bucket}.s3.amazonaws.com/{encoded_key}"
        try:
            with urlopen(url) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(f"The file {s3_uri} does not exist.") from exc
            raise RuntimeError(f"Could not download public S3 object: {s3_uri}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not download public S3 object: {s3_uri}") from exc

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ModuleNotFoundError:
        return _read_public_object()

    s3_client = boto3.client("s3")

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"NoSuchKey", "404", "NotFound"}:
            raise FileNotFoundError(f"The file {s3_uri} does not exist.") from exc
    except (NoCredentialsError, BotoCoreError):
        # Fallback for public buckets.
        pass

    try:
        # Fallback for public buckets.
        unsigned_client = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
        )
        response = unsigned_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"NoSuchKey", "404", "NotFound"}:
            raise FileNotFoundError(f"The file {s3_uri} does not exist.") from exc
        return _read_public_object()
    except BotoCoreError:
        return _read_public_object()


def _read_dataframe_from_bytes(file_bytes: bytes, *, suffix: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_data(
    filename: str = "train.xlsx",
    data_dir: Union[str, Path] = "DATASETS",
    *,
    parse_dates: bool = True,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Load a dataset as CSV or Excel from local folder or S3.

    Args:
        filename: Name of the file (e.g. ``train.xlsx``).
        data_dir: Local directory path or an S3 URI prefix (e.g. ``s3://my-bucket/raw``).
        parse_dates: Whether to parse the date column.
        date_col: Date column name.

    Returns:
        Loaded dataframe.
    """
    data_dir_str = str(data_dir)

    if _is_s3_path(data_dir_str):
        s3_uri = _join_s3_uri(data_dir_str, filename)
        suffix = Path(filename).suffix.lower()
        file_bytes = _read_s3_bytes(s3_uri)
        df = _read_dataframe_from_bytes(file_bytes, suffix=suffix)
    else:
        file_path = Path(data_dir) / filename
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        suffix = file_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        elif suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    return df


def load_train_data(
    data_dir: Union[str, Path] = "DATASETS",
    *,
    parse_dates: bool = True,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Convenience wrapper to load train.xlsx."""
    return load_data("train.xlsx", data_dir, parse_dates=parse_dates, date_col=date_col)


def load_test_template(
    data_dir: Union[str, Path] = "DATASETS",
    *,
    parse_dates: bool = True,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Convenience wrapper to load test_template.xlsx."""
    return load_data(
        "test_template.xlsx",
        data_dir,
        parse_dates=parse_dates,
        date_col=date_col,
    )
