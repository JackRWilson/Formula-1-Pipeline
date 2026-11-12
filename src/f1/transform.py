import re
import unicodedata
from pathlib import Path

import pandas as pd
from azure.storage.blob import ContainerClient
from sqlalchemy import text
from sqlalchemy.engine import Connection

if __package__:
    from .extract import (
        DATA_START_YEAR,
        F1SITE_MANIFEST_TABLE,
        JOLPI_DRIVERS_BLOB_PREFIX,
        MAX_ROUNDS_BLOB,
        PHOTON_LOCATIONS_BLOB,
        RAW_CONTAINER,
        WIKIPEDIA_CIRCUITS_BLOB,
    )
    from .utils import (
        clean_circuit_name,
        constructor_mapping,
        download_df_from_blob,
        get_blob_container_client,
    )
else:
    from extract import (
        DATA_START_YEAR,
        F1SITE_MANIFEST_TABLE,
        JOLPI_DRIVERS_BLOB_PREFIX,
        MAX_ROUNDS_BLOB,
        PHOTON_LOCATIONS_BLOB,
        RAW_CONTAINER,
        WIKIPEDIA_CIRCUITS_BLOB,
    )
    from utils import (
        clean_circuit_name,
        constructor_mapping,
        download_df_from_blob,
        get_blob_container_client,
    )


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

SESSIONS = [
    "practice",
    "qualifying",
    "sprint-qualifying",
    "race-result",
    "sprint-result",
]
SESSION_BLOB_PATTERN = re.compile(
    rf"^(?P<session>{'|'.join(SESSIONS)})"
    r"/(?:(?P<session_number>\d+)/)?"
    r"(?P<year>\d{4})/(?P<round>\d+)\.parquet$"
)
JOLPI_DRIVERS_PATTERN = re.compile(
    rf"^{re.escape(JOLPI_DRIVERS_BLOB_PREFIX)}/(?P<year>\d{{4}})\.parquet$"
)
INT_GROUPS = ("year", "round")


def list_blobs_matching(
    container_client: ContainerClient,
    pattern: re.Pattern,
    prefixes: list | None = None,
    min_year: int | None = None,
) -> pd.DataFrame:
    rows = []
    for prefix in prefixes or [None]:
        for blob in container_client.list_blobs(name_starts_with=prefix):
            m = pattern.match(blob.name)
            if m is None:
                continue
            row = {"blob_name": blob.name, **m.groupdict()}
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


def get_blob(blob_name: str) -> pd.DataFrame:
    data = download_df_from_blob(RAW_CONTAINER, blob_name)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Blob {blob_name!r} did not contain a DataFrame")
    return data


def load_blobs(blob_index: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in blob_index.itertuples(index=False):
        df = get_blob(row.blob_name)
        meta = {k: v for k, v in row._asdict().items() if k != "blob_name"}
        frames.append(df.assign(**meta))
    if not frames:
        raise ValueError("No blobs to load; check the pattern and prefixes")
    return pd.concat(frames, ignore_index=True)


def list_session_blobs(container_client: ContainerClient) -> pd.DataFrame:
    return list_blobs_matching(
        container_client,
        SESSION_BLOB_PATTERN,
        prefixes=[f"{session}/" for session in SESSIONS],
        min_year=DATA_START_YEAR,
    )


def list_jolpi_driver_blobs(container_client: ContainerClient) -> pd.DataFrame:
    return list_blobs_matching(
        container_client,
        JOLPI_DRIVERS_PATTERN,
        prefixes=[f"{JOLPI_DRIVERS_BLOB_PREFIX}/"],
        min_year=DATA_START_YEAR,
    )


def _require_columns(data: pd.DataFrame, columns: set, source: str) -> None:
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return None
    value = " ".join(str(value).split())
    return value or None


def _normalized_key(value: object) -> str:
    normalized = clean_circuit_name(_normalize_text(value))
    return normalized.casefold() if isinstance(normalized, str) and normalized else None


class TransformSeason:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def transform(self) -> pd.DataFrame:
        data = get_blob(MAX_ROUNDS_BLOB)
        _require_columns(data, {"year", "max_rounds"}, MAX_ROUNDS_BLOB)
        return self._get_complete_bool(data)

    def _get_complete_bool(self, data: pd.DataFrame) -> pd.DataFrame:
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
    def transform(self, circuits: pd.DataFrame = None) -> pd.DataFrame:
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
    def transform(self) -> pd.DataFrame:
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
    def _has_recent_season(seasons: object) -> bool:
        if pd.isna(seasons) or not str(seasons).strip():
            return True
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", str(seasons))]
        return not years or max(years) >= DATA_START_YEAR

    @staticmethod
    def _length_in_km(value: object) -> float:
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
    def _normalize_direction(direction: object) -> str:
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


class TransformConstructor:
    def transform(self) -> pd.DataFrame:
        teams = self._extract_teams()
        team_aliases = {
            _normalize_text(alias).casefold(): name
            for alias, name in constructor_mapping["team_id"].items()
        }
        teams["constructor_name"] = teams["constructor_name"].map(
            lambda name: team_aliases.get(name.casefold(), name)
        )
        return (
            teams.drop_duplicates(subset=["constructor_name"])
            .sort_values("constructor_name", kind="stable")
            .reset_index(drop=True)
        )

    @staticmethod
    def _extract_teams() -> pd.DataFrame:
        container = get_blob_container_client(RAW_CONTAINER)
        blob_index = list_session_blobs(container)
        if blob_index.empty:
            raise ValueError("No F1 session blobs found for constructor extraction")
        race_results = blob_index[blob_index["session"].eq("race-result")]
        if race_results.empty:
            raise ValueError("No race-result blobs found for constructor extraction")

        team_names = set()
        for row in race_results.itertuples():
            df = get_blob(row.blob_name)
            _require_columns(df, {"team_id"}, row.blob_name)
            team_names.update(
                name for name in df["team_id"].map(_normalize_text).dropna().unique()
            )
        if not team_names:
            raise ValueError("Race-result blobs contained no constructor names")
        return pd.DataFrame({"constructor_name": sorted(team_names)})


class TransformDriver:
    def __init__(self):
        self.DRIVER_COLUMNS = [
            "source_driver_id",
            "first_name",
            "last_name",
            "driver_code",
            "nationality",
            "country_code",
            "date_of_birth",
        ]

    def transform(self) -> pd.DataFrame:
        jolpi_drivers = self._extract_jolpi_drivers()
        session_pairs = self._extract_session_pairs()

        session_pairs = self._clean_session_pairs(session_pairs)
        self._check_session_name_conflicts(session_pairs)
        drivers = self._latest_jolpi_drivers(jolpi_drivers)

        sessions = self._match_by_name(drivers, session_pairs)
        sessions = self._match_by_code(drivers, sessions)

        drivers = self._attach_session_codes(drivers, sessions)
        self._check_code_conflicts(drivers)

        session_only = self._session_only_drivers(sessions)
        return self._combine(drivers, session_only)

    @staticmethod
    def _name_key(series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype("string")
            .map(lambda s: unicodedata.normalize("NFKD", s) if pd.notna(s) else s)
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii")
            .str.casefold()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return cleaned.map(lambda s: " ".join(sorted(s.split())) if pd.notna(s) else s)

    @staticmethod
    def _extract_jolpi_drivers() -> pd.DataFrame:
        container = get_blob_container_client(RAW_CONTAINER)
        blob_index = list_jolpi_driver_blobs(container)
        if blob_index.empty:
            raise ValueError("No jolpi driver blobs found for driver extraction")
        return load_blobs(blob_index)

    @staticmethod
    def _extract_session_pairs() -> pd.DataFrame:
        container = get_blob_container_client(RAW_CONTAINER)
        blob_index = list_session_blobs(container)
        if blob_index.empty:
            raise ValueError("No F1 session blobs found for driver extraction")

        frames = []
        for row in blob_index.itertuples():
            df = get_blob(row.blob_name)
            _require_columns(df, {"driver_name", "driver_code"}, row.blob_name)
            frames.append(df[["driver_name", "driver_code"]].drop_duplicates())
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    def _clean_session_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        pairs = pairs.copy()
        pairs["driver_code"] = pairs["driver_code"].map(_normalize_text)
        pairs["driver_name"] = pairs["driver_name"].map(_normalize_text)
        pairs = pairs.dropna()
        pairs["_name_key"] = self._name_key(pairs["driver_name"])
        return pairs

    @staticmethod
    def _check_session_name_conflicts(pairs: pd.DataFrame) -> None:
        codes_per_name = pairs.groupby("_name_key")["driver_code"].nunique()
        conflicts = codes_per_name[codes_per_name > 1].index.tolist()
        if conflicts:
            raise ValueError(f"Driver names with several codes: {conflicts}")

    def _latest_jolpi_drivers(self, jolpi_drivers: pd.DataFrame) -> pd.DataFrame:
        data = jolpi_drivers.copy()
        for column in ("first_name", "last_name", "driver_code"):
            data[column] = data[column].map(_normalize_text)
        data = data.sort_values("season_year", kind="stable")

        drivers = data.groupby("source_driver_id", as_index=False).last()[
            self.DRIVER_COLUMNS
        ]
        drivers["_name_key"] = self._name_key(
            drivers["first_name"] + " " + drivers["last_name"]
        )
        return drivers

    @staticmethod
    def _match_by_name(
        drivers: pd.DataFrame, session_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        sessions = session_pairs.drop_duplicates("_name_key").reset_index(drop=True)
        name_to_id = drivers.set_index("_name_key")["source_driver_id"]
        sessions["source_driver_id"] = sessions["_name_key"].map(name_to_id)
        return sessions

    @staticmethod
    def _match_by_code(drivers: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
        sessions = sessions.copy()
        claimed = sessions["source_driver_id"].dropna()
        free = drivers.loc[
            ~drivers["source_driver_id"].isin(claimed),
            ["source_driver_id", "driver_code", "_name_key"],
        ]
        unmatched = (
            sessions.loc[
                sessions["source_driver_id"].isna(), ["driver_code", "_name_key"]
            ]
            .rename_axis("session_idx")
            .reset_index()
        )
        candidates = free.merge(unmatched, on="driver_code", suffixes=("_j", "_s"))
        if candidates.empty:
            return sessions

        shares_word = candidates.apply(
            lambda r: bool(
                set(r["_name_key_j"].split()) & set(r["_name_key_s"].split())
            ),
            axis=1,
        )
        candidates = candidates[shares_word].drop_duplicates("session_idx", keep=False)
        sessions.loc[candidates["session_idx"], "source_driver_id"] = candidates[
            "source_driver_id"
        ].to_numpy()
        return sessions

    @staticmethod
    def _attach_session_codes(
        drivers: pd.DataFrame, sessions: pd.DataFrame
    ) -> pd.DataFrame:
        matched = (
            sessions.dropna(subset=["source_driver_id"])
            .drop_duplicates("source_driver_id")[["source_driver_id", "driver_code"]]
            .rename(columns={"driver_code": "driver_code_session"})
        )
        drivers = drivers.merge(matched, on="source_driver_id", how="left")
        drivers["driver_code"] = drivers["driver_code"].fillna(
            drivers["driver_code_session"]
        )
        return drivers

    @staticmethod
    def _check_code_conflicts(drivers: pd.DataFrame) -> None:
        clash = drivers[
            drivers["driver_code_session"].notna()
            & (drivers["driver_code"] != drivers["driver_code_session"])
        ]
        if not clash.empty:
            raise ValueError(f"Jolpi and session codes disagree:\n{clash}")

    def _session_only_drivers(self, sessions: pd.DataFrame) -> pd.DataFrame:
        only = sessions[sessions["source_driver_id"].isna()].copy()
        if only.empty:
            return only.reindex(columns=self.DRIVER_COLUMNS)

        parts = only["driver_name"].str.split(" ", n=1, expand=True)
        only["first_name"], only["last_name"] = parts[0], parts[1]
        return only.reindex(columns=self.DRIVER_COLUMNS)

    def _combine(
        self, drivers: pd.DataFrame, session_only: pd.DataFrame
    ) -> pd.DataFrame:
        out = pd.concat(
            [drivers.reindex(columns=self.DRIVER_COLUMNS), session_only],
            ignore_index=True,
        )
        out["date_of_birth"] = pd.to_datetime(out["date_of_birth"], errors="coerce")
        out = out.drop(columns=["source_driver_id"])
        return out.sort_values(["last_name", "first_name"]).reset_index(drop=True)
