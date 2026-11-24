import hashlib
import io
import os
from collections.abc import Callable, Mapping
from re import Pattern

import pandas as pd
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContainerClient

from f1.helpers.constants import (
    DATA_START_YEAR,
    INT_GROUPS,
    RAW_CONTAINER,
    BlobSource,
)


def get_blob_container_client(container_name: str) -> ContainerClient:
    """
    Return an Azurite container client, creating the container if it doesn't exist yet
    """
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    )
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass
    return container_client


def upload_df_to_blob(df: pd.DataFrame, container_name: str, blob_name: str) -> str:
    """
    Serialize a DataFrame to parquet in memory and upload it as a blob
    """
    container_client = get_blob_container_client(container_name)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    parquet_bytes = buffer.getvalue()

    container_client.upload_blob(name=blob_name, data=parquet_bytes, overwrite=True)
    return hashlib.sha256(parquet_bytes).hexdigest()


def download_df_from_blob(container_name: str, blob_name: str) -> pd.DataFrame:
    """
    Download a parquet blob and load it back into a DataFrame
    """
    container_client = get_blob_container_client(container_name)

    blob_bytes = container_client.download_blob(blob_name).readall()
    return pd.read_parquet(io.BytesIO(blob_bytes))


def get_blob(blob_name: str) -> pd.DataFrame:
    data = download_df_from_blob(RAW_CONTAINER, blob_name)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Blob {blob_name!r} did not contain a DataFrame")
    return data


def list_blobs_matching(
    container_client: ContainerClient,
    pattern: Pattern[str],
    prefixes: list[str] | None = None,
    min_year: int | None = None,
) -> pd.DataFrame:
    rows = []
    for prefix in [None] if prefixes is None else prefixes:
        for blob in container_client.list_blobs(name_starts_with=prefix):
            parsed_name = pattern.match(blob.name)
            if parsed_name is None:
                continue
            row = {"blob_name": blob.name, **parsed_name.groupdict()}
            for key in INT_GROUPS:
                if row.get(key) is not None:
                    row[key] = int(row[key])
            if (
                min_year is not None
                and row.get("year") is not None
                and row["year"] < min_year
            ):
                continue
            rows.append(row)

    columns = ["blob_name", *pattern.groupindex]
    return pd.DataFrame(rows, columns=columns)


def list_blobs(
    container_client: ContainerClient,
    source: BlobSource,
    folders: tuple[str, ...] | None = None,
    min_year: int = DATA_START_YEAR,
) -> pd.DataFrame:
    selected_folders = source.folders if folders is None else folders
    return list_blobs_matching(
        container_client,
        source.pattern,
        prefixes=[f"{folder}/" for folder in selected_folders],
        min_year=min_year,
    )


def require_columns(data: pd.DataFrame, columns: set, source: str) -> None:
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def filter_blob_index(
    blob_index: pd.DataFrame, filters: Mapping[str, object] | None = None
) -> pd.DataFrame:
    filters = filters or {}
    if blob_index.empty:
        raise ValueError("No blobs found for the requested source and folders")

    mask = pd.Series(True, index=blob_index.index)
    for column, value in filters.items():
        if column not in blob_index.columns:
            raise ValueError(f"Blob index has no {column!r} field to filter")
        mask &= blob_index[column] == value
    results = blob_index[mask]
    if results.empty:
        described = ", ".join(f"{key}={value}" for key, value in filters.items())
        raise ValueError(f"No F1 blobs found for {described}")
    return results


def concat_blob_frames(
    blob_index: pd.DataFrame,
    required_columns: set,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    add_round: bool = False,
) -> pd.DataFrame:
    frames = []
    for row in blob_index.itertuples(index=False):
        df = get_blob(row.blob_name)
        if transform is not None:
            df = transform(df)
        if add_round:
            df["round"] = row.round
        require_columns(df, required_columns, row.blob_name)
        frames.append(df)
    if not frames:
        raise ValueError("No blobs to load; check the pattern and filters")

    frames = [f.dropna(axis=1, how="all") for f in frames]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("All blobs were empty after dropping all-NA columns")
    out = pd.concat(frames, ignore_index=True)

    for col in required_columns - set(out.columns):
        out[col] = pd.NA

    return out.drop_duplicates()


def load_blobs(
    source: BlobSource,
    required_columns: set,
    *,
    folders: tuple[str, ...] | None = None,
    filters: Mapping[str, object] | None = None,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    add_round: bool = False,
    min_year: int | None = DATA_START_YEAR,
    container_client: ContainerClient | None = None,
) -> pd.DataFrame:
    if container_client is None:
        container_client = get_blob_container_client(RAW_CONTAINER)
    blob_index = list_blobs(container_client, source, folders, min_year=min_year)
    blob_index = filter_blob_index(blob_index, filters)
    return concat_blob_frames(
        blob_index,
        required_columns,
        transform=transform,
        add_round=add_round,
    )
