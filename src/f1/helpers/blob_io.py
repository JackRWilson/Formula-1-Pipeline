import hashlib
import io
import os
import re
from collections.abc import Callable

import pandas as pd
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContainerClient

from f1.helpers.constants import (
    DATA_START_YEAR,
    FASTEST_LAP_BLOB_PATTERN,
    FASTF1_BLOB_PATTERN,
    GRID_BLOB_PATTERN,
    GRIDS,
    INT_GROUPS,
    JOLPI_DRIVERS_BLOB_PREFIX,
    JOLPI_DRIVERS_PATTERN,
    RAW_CONTAINER,
    SESSION_BLOB_PATTERN,
    SESSIONS,
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


def load_blobs(blob_index: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in blob_index.itertuples(index=False):
        df = get_blob(row.blob_name)
        meta = {
            key: value for key, value in row._asdict().items() if key != "blob_name"
        }
        frames.append(df.assign(**meta))
    if not frames:
        raise ValueError("No blobs to load; check the pattern and prefixes")
    return pd.concat(frames, ignore_index=True)


def list_blobs_matching(
    container_client: ContainerClient,
    pattern: re.Pattern,
    prefixes: list | None = None,
    min_year: int | None = None,
) -> pd.DataFrame:
    rows = []
    for prefix in prefixes or [None]:
        for blob in container_client.list_blobs(name_starts_with=prefix):
            match = pattern.match(blob.name)
            if match is None:
                continue
            row = {"blob_name": blob.name, **match.groupdict()}
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


def list_session_blobs(container_client: ContainerClient) -> pd.DataFrame:
    return list_blobs_matching(
        container_client,
        SESSION_BLOB_PATTERN,
        prefixes=[f"{session}/" for session in SESSIONS],
        min_year=DATA_START_YEAR,
    )


def list_grid_blobs(container_client: ContainerClient) -> pd.DataFrame:
    return list_blobs_matching(
        container_client,
        GRID_BLOB_PATTERN,
        prefixes=[f"{grid}/" for grid in GRIDS],
        min_year=DATA_START_YEAR,
    )


def list_fastest_lap_blobs(container_client: ContainerClient) -> pd.DataFrame:
    return list_blobs_matching(
        container_client,
        FASTEST_LAP_BLOB_PATTERN,
        prefixes=["fastest-laps/"],
        min_year=DATA_START_YEAR,
    )


def list_jolpi_driver_blobs(container_client: ContainerClient) -> pd.DataFrame:
    return list_blobs_matching(
        container_client,
        JOLPI_DRIVERS_PATTERN,
        prefixes=[f"{JOLPI_DRIVERS_BLOB_PREFIX}/"],
        min_year=DATA_START_YEAR,
    )


def list_fastf1_blobs(container_client: ContainerClient) -> pd.DataFrame:
    return list_blobs_matching(
        container_client,
        FASTF1_BLOB_PATTERN,
        prefixes=["fastf1/"],
        min_year=DATA_START_YEAR,
    )


def require_columns(data: pd.DataFrame, columns: set, source: str) -> None:
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def filter_blob_index(
    container_client: ContainerClient,
    list_blobs: Callable[[ContainerClient], pd.DataFrame],
    **filters: object,
) -> pd.DataFrame:
    blob_index = list_blobs(container_client)

    mask = pd.Series(True, index=blob_index.index)
    for column, value in filters.items():
        mask &= blob_index[column] == value

    results = blob_index[mask]
    if results.empty:
        described = ", ".join(f"{key}={value}" for key, value in filters.items())
        raise ValueError(f"No F1 blobs found for {described}")
    return results


def concat_blob_frames(
    blob_index: pd.DataFrame,
    required_columns: set,
    add_round: bool = False,
) -> pd.DataFrame:
    frames = []
    for row in blob_index.itertuples(index=False):
        df = get_blob(row.blob_name)
        if add_round:
            df["round"] = row.round
        require_columns(df, required_columns, row.blob_name)
        frames.append(df)
    if not frames:
        raise ValueError("No blobs to load; check the pattern and filters")
    return pd.concat(frames, ignore_index=True).drop_duplicates()
