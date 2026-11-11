import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text

if __package__:
    from .extract import (
        DATA_START_YEAR,
        F1SITE_MANIFEST_TABLE,
        MAX_ROUNDS_BLOB,
        PHOTON_LOCATIONS_BLOB,
        RAW_CONTAINER,
        WIKIPEDIA_CIRCUITS_BLOB,
    )
    from .project_functions import clean_circuit_name
    from .utils import download_df_from_blob
else:
    from extract import (
        DATA_START_YEAR,
        F1SITE_MANIFEST_TABLE,
        MAX_ROUNDS_BLOB,
        PHOTON_LOCATIONS_BLOB,
        RAW_CONTAINER,
        WIKIPEDIA_CIRCUITS_BLOB,
    )
    from project_functions import clean_circuit_name
    from utils import download_df_from_blob


LOCATION_COLUMNS = [
    "city",
    "country",
    "latitude",
    "longitude",
    "elevation_m",
]

CURATED_CIRCUITS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "curated_circuits.csv"
)


def get_blob(blob_name):
    data = download_df_from_blob(RAW_CONTAINER, blob_name)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Blob {blob_name!r} did not contain a DataFrame")
    return data


def _require_columns(data, columns, source):
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _normalize_text(value):
    if pd.isna(value):
        return None
    value = " ".join(str(value).split())
    return value or None


def _normalized_key(value):
    normalized = clean_circuit_name(_normalize_text(value))
    return normalized.casefold() if normalized else None


class TransformSeason:
    def __init__(self, conn):
        self._conn = conn

    def transform(self):
        data = get_blob(MAX_ROUNDS_BLOB)
        _require_columns(data, {"year", "max_rounds"}, MAX_ROUNDS_BLOB)
        return self._get_complete_bool(data)

    def _get_complete_bool(self, data):
        completed = (
            self._conn.execute(
                text(f"""
                select year,
                    count(distinct round_number) as completed_rounds,
                    max(round_number) as last_completed_round
                from {F1SITE_MANIFEST_TABLE}
                where status = 'success'
                    and blob_name is not null
                    and session_type = 'race-result'
                    and year >= :start_year
                group by year
                """),
                {"start_year": DATA_START_YEAR},
            )
            .mappings()
            .all()
        )
        completed = pd.DataFrame(
            completed,
            columns=["year", "completed_rounds", "last_completed_round"],
        )
        season_data = data.merge(completed, on="year", how="left")
        season_data["is_complete"] = (
            season_data["max_rounds"].gt(0)
            & season_data["completed_rounds"].eq(season_data["max_rounds"])
            & season_data["last_completed_round"].eq(season_data["max_rounds"])
        )
        return season_data[["year", "is_complete"]].rename(
            columns={"year": "season_year"}
        )


class TransformLocation:
    def transform(self, circuits=None):
        data = get_blob(PHOTON_LOCATIONS_BLOB)
        _require_columns(
            data,
            {
                "cleaned_name",
                "city",
                "country",
                "latitude",
                "longitude",
                "elevation",
            },
            PHOTON_LOCATIONS_BLOB,
        )
        data = data.copy()
        data["_venue_key"] = data["cleaned_name"].map(_normalized_key)

        if circuits is not None:
            _require_columns(circuits, {"location_source_name"}, "circuit data")
            used_venues = {
                key
                for key in circuits["location_source_name"].map(_normalized_key)
                if key is not None
            }
            data = data[data["_venue_key"].isin(used_venues)]

        data["city"] = data["city"].map(_normalize_text)
        data["country"] = data["country"].map(_normalize_text)
        data["latitude"] = pd.to_numeric(data["latitude"], errors="raise")
        data["longitude"] = pd.to_numeric(data["longitude"], errors="raise")
        data["elevation_m"] = pd.to_numeric(data["elevation"], errors="coerce")
        data = data.dropna(subset=["city", "country", "latitude", "longitude"])

        if not data["latitude"].between(-90, 90).all():
            raise ValueError("Location latitude must be between -90 and 90")
        if not data["longitude"].between(-180, 180).all():
            raise ValueError("Location longitude must be between -180 and 180")
        data[["latitude", "longitude"]] = data[["latitude", "longitude"]].round(6)

        key_columns = ["city", "country"]
        conflicts = data.groupby(key_columns, dropna=False).agg(
            latitude_count=("latitude", "nunique"),
            longitude_count=("longitude", "nunique"),
            elevation_count=("elevation_m", "nunique"),
        )
        conflicts = conflicts[
            (conflicts["latitude_count"] > 1)
            | (conflicts["longitude_count"] > 1)
            | (conflicts["elevation_count"] > 1)
        ]
        if not conflicts.empty:
            raise ValueError(
                "Multiple coordinates found for the same city and country: "
                f"{conflicts.index.tolist()}"
            )

        return data.drop_duplicates(key_columns)[LOCATION_COLUMNS].reset_index(
            drop=True
        )


class TransformCircuit:
    def transform(self):
        data = get_blob(WIKIPEDIA_CIRCUITS_BLOB)
        required_columns = {
            "name",
            "type",
            "direction",
            "location",
            "country",
            "length",
            "turns",
            "gp",
            "seasons",
        }
        _require_columns(data, required_columns, WIKIPEDIA_CIRCUITS_BLOB)

        data = data.copy()
        curated = pd.read_csv(CURATED_CIRCUITS_PATH)
        _require_columns(
            curated,
            required_columns | {"location_source_name"},
            str(CURATED_CIRCUITS_PATH),
        )
        data = pd.concat([data, curated], ignore_index=True, sort=False)

        data["_source_name"] = data["name"].map(
            lambda name: clean_circuit_name(_normalize_text(name))
        )
        data["_circuit_key"] = data["_source_name"].map(
            lambda name: name.casefold() if isinstance(name, str) and name else None
        )
        data = data.dropna(subset=["_circuit_key"])
        data = data.drop_duplicates("_circuit_key", keep="last")

        if "location_source_name" not in data:
            data["location_source_name"] = None
        data["location_source_name"] = data["location_source_name"].where(
            data["location_source_name"].notna(), data["_source_name"]
        )
        data = data[data["seasons"].map(self._has_recent_season)].copy()
        turns = pd.to_numeric(data["turns"], errors="coerce")
        invalid_turns = data["turns"].notna() & turns.isna()
        if invalid_turns.any():
            raise ValueError(
                "Could not parse circuit turn counts: "
                f"{data.loc[invalid_turns, 'turns'].tolist()}"
            )
        if (turns.dropna() < 0).any():
            raise ValueError("Circuit turn counts cannot be negative")

        circuit_data = pd.DataFrame(
            {
                "circuit_name": data["_source_name"],
                "circuit_type": data["type"].map(_normalize_text),
                "direction": data["direction"].map(self._normalize_direction),
                "location_city": data["location"].map(_normalize_text),
                "location_country": data["country"].map(_normalize_text),
                "location_source_name": data["location_source_name"].map(
                    clean_circuit_name
                ),
                "length_km": data["length"].map(self._length_in_km),
                "turns": turns.astype("Int64"),
            }
        )
        circuit_data = circuit_data.drop_duplicates(
            subset=["circuit_name"], keep="last"
        )
        circuit_data = circuit_data.sort_values(
            "circuit_name", kind="stable"
        ).reset_index(drop=True)
        return circuit_data

    @staticmethod
    def _has_recent_season(seasons):
        if pd.isna(seasons) or not str(seasons).strip():
            return True
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", str(seasons))]
        return not years or max(years) >= DATA_START_YEAR

    @staticmethod
    def _length_in_km(value):
        if pd.isna(value):
            return None
        match = re.search(r"\d+(?:[.,]\d+)?", str(value))
        if match is None:
            raise ValueError(f"Could not parse circuit length from {value!r}")
        length = float(match.group().replace(",", "."))
        if length <= 0:
            raise ValueError(f"Circuit length must be positive, got {value!r}")
        return length

    @staticmethod
    def _normalize_direction(direction):
        direction = _normalize_text(direction)
        if direction is None:
            return None
        normalized_directions = {
            "clockwise": "Clockwise",
            "anti-clockwise": "Anti-clockwise",
            "part clockwise and part anti-clockwise (figure of eight)": "Figure eight",
        }
        try:
            return normalized_directions[direction.casefold()]
        except KeyError as exc:
            raise ValueError(f"Unknown circuit direction: {direction!r}") from exc
