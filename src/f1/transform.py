import logging
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process
from sqlalchemy import text
from sqlalchemy.engine import Connection

from f1.helpers.blob_io import (
    concat_blob_frames,
    filter_blob_index,
    get_blob,
    get_blob_container_client,
    list_fastest_lap_blobs,
    list_grid_blobs,
    list_jolpi_driver_blobs,
    list_session_blobs,
    load_blobs,
    require_columns,
)
from f1.helpers.constants import (
    CIRCUIT_FUZZY_SCORE_CUTOFF,
    CIRCUIT_GENERIC_WORDS,
    CURATED_CIRCUITS_PATH,
    DATA_START_YEAR,
    F1SITE_MANIFEST_TABLE,
    MAX_ROUNDS_BLOB,
    PHOTON_LOCATIONS_BLOB,
    RAW_CONTAINER,
    WIKIPEDIA_CIRCUITS_BLOB,
)
from f1.helpers.constructor_mapping import constructor_mapping
from f1.helpers.lap_times import SESSION_MAP, parse_lap_strings
from f1.helpers.text import normalize_text, normalized_key
from f1.helpers.utils import clean_circuit_name

logger = logging.getLogger(__name__)

LOCATION_COLUMNS = [
    "city",
    "country",
    "latitude",
    "longitude",
    "elevation_m",
]


class TransformSeason:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def transform(self) -> pd.DataFrame:
        data = get_blob(MAX_ROUNDS_BLOB)
        require_columns(data, {"year", "max_rounds"}, MAX_ROUNDS_BLOB)
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
        require_columns(
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
        data["_venue_key"] = data["cleaned_name"].map(normalized_key)

        if circuits is not None:
            require_columns(circuits, {"location_source_name"}, "circuit data")
            used_venues = {
                key
                for key in circuits["location_source_name"].map(normalized_key)
                if key is not None
            }
            data = data[data["_venue_key"].isin(used_venues)]

        data["city"] = data["city"].map(normalize_text)
        data["country"] = data["country"].map(normalize_text)
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
        require_columns(data, required_columns, WIKIPEDIA_CIRCUITS_BLOB)

        data = data.copy()
        curated = pd.read_csv(CURATED_CIRCUITS_PATH)
        require_columns(
            curated,
            required_columns | {"location_source_name"},
            str(CURATED_CIRCUITS_PATH),
        )
        data = pd.concat([data, curated], ignore_index=True, sort=False)

        data["_source_name"] = data["name"].map(
            lambda name: clean_circuit_name(normalize_text(name))
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
                "circuit_type": data["type"].map(normalize_text),
                "direction": data["direction"].map(self._normalize_direction),
                "location_city": data["location"].map(normalize_text),
                "location_country": data["country"].map(normalize_text),
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
        direction = normalize_text(direction)
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
            normalize_text(alias).casefold(): name
            for alias, name in constructor_mapping["team_name"].items()
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
            require_columns(df, {"team_name"}, row.blob_name)
            team_names.update(
                name for name in df["team_name"].map(normalize_text).dropna().unique()
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
            require_columns(df, {"driver_name", "driver_code"}, row.blob_name)
            frames.append(df[["driver_name", "driver_code"]].drop_duplicates())
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    def _clean_session_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        pairs = pairs.copy()
        pairs["driver_code"] = pairs["driver_code"].map(normalize_text)
        pairs["driver_name"] = pairs["driver_name"].map(normalize_text)
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
            data[column] = data[column].map(normalize_text)
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


class TransformRace:
    def transform(self, circuit_df: pd.DataFrame) -> pd.DataFrame:
        races_df = self._extract_races()
        races_df = self._group_race_date(races_df)
        races_df = self._clean_date(races_df)
        races_df = self._standardize_circuit_names(races_df, circuit_df)
        return races_df

    def _extract_races(self) -> pd.DataFrame:
        blob_index = filter_blob_index(
            get_blob_container_client(RAW_CONTAINER),
            list_session_blobs,
            session="race-result",
        )
        return concat_blob_frames(
            blob_index, {"race_id", "year", "round"}, add_round=True
        )

    def _group_race_date(self, races_df: pd.DataFrame) -> pd.DataFrame:
        return (
            races_df.groupby("race_id")[
                [
                    "year",
                    "round",
                    "circuit_name",
                    "city",
                    "race_name",
                    "date",
                    "race_url",
                ]
            ]
            .first()
            .reset_index()
        )

    def _clean_date(self, races_df: pd.DataFrame) -> pd.DataFrame:
        has_range = races_df["date"].str.contains("-", na=False)
        races_df.loc[has_range, "date"] = (
            races_df.loc[has_range, "date"].str.split("-").str[1].str.strip()
        )
        races_df["date"] = pd.to_datetime(
            races_df["date"], format="mixed", errors="coerce"
        )
        return (
            races_df.rename(
                columns={
                    "year": "season_year",
                    "date": "race_date",
                    "race_url": "source_url",
                }
            )
            .sort_values(by=["season_year", "round"])
            .reset_index(drop=True)
        )

    def _standardize_circuit_names(
        self, races_df: pd.DataFrame, circuit_df: pd.DataFrame
    ) -> pd.DataFrame:
        circuit_name_map = self._build_circuit_name_map(races_df, circuit_df)

        races_df["circuit_name"] = (
            races_df["circuit_name"]
            .map(circuit_name_map)
            .fillna(races_df["circuit_name"])
        )
        return races_df

    def _build_circuit_name_map(
        self, races_df: pd.DataFrame, circuit_df: pd.DataFrame
    ) -> dict:
        circuit_df_keyed = self._add_circuit_match_keys(circuit_df)

        raw_names = races_df["circuit_name"].dropna().unique().tolist()
        circuit_name_map = {}
        claimed_circuit_keys = set()

        # Exact match on normalized key
        self._match_circuit_names_by_exact_key(
            raw_names, circuit_df_keyed, circuit_name_map, claimed_circuit_keys
        )

        # Fuzzy match, leftovers only
        unresolved_names = [n for n in raw_names if n not in circuit_name_map]
        self._match_circuit_names_by_fuzzy_key(
            unresolved_names, circuit_df_keyed, circuit_name_map, claimed_circuit_keys
        )

        # City fallback, leftovers only
        unresolved_names = [n for n in raw_names if n not in circuit_name_map]
        self._match_circuit_names_by_city(
            unresolved_names,
            races_df,
            circuit_df_keyed,
            circuit_name_map,
            claimed_circuit_keys,
        )

        for name in raw_names:
            if name not in circuit_name_map:
                logger.warning("No circuit match found for race circuit name: %r", name)

        return circuit_name_map

    def _add_circuit_match_keys(self, circuit_df: pd.DataFrame) -> pd.DataFrame:
        circuit_df_keyed = circuit_df.copy()
        circuit_df_keyed["circuit_key"] = circuit_df_keyed["circuit_name"].map(
            self._normalize_circuit_name
        )
        circuit_df_keyed["city_key"] = circuit_df_keyed["location_city"].map(
            self._normalize_city
        )

        duplicated_keys = circuit_df_keyed[
            circuit_df_keyed["circuit_key"].duplicated(keep=False)
        ]
        if not duplicated_keys.empty:
            raise ValueError(
                "Circuit names collapse to the same key; adjust CIRCUIT_GENERIC_WORDS:\n"
                f"{duplicated_keys[['circuit_name', 'circuit_key']]}"
            )
        return circuit_df_keyed

    def _match_circuit_names_by_exact_key(
        self, raw_names, circuit_df_keyed, circuit_name_map, claimed_circuit_keys
    ) -> None:
        canonical_name_by_key = dict(
            zip(circuit_df_keyed["circuit_key"], circuit_df_keyed["circuit_name"])
        )
        for raw_name in raw_names:
            circuit_key = self._normalize_circuit_name(raw_name)
            if circuit_key in canonical_name_by_key:
                circuit_name_map[raw_name] = canonical_name_by_key[circuit_key]
                claimed_circuit_keys.add(circuit_key)

    def _match_circuit_names_by_fuzzy_key(
        self, unresolved_names, circuit_df_keyed, circuit_name_map, claimed_circuit_keys
    ) -> None:
        canonical_name_by_key = dict(
            zip(circuit_df_keyed["circuit_key"], circuit_df_keyed["circuit_name"])
        )
        unclaimed_circuit_keys = set(canonical_name_by_key) - claimed_circuit_keys

        for raw_name in sorted(unresolved_names):
            if not unclaimed_circuit_keys:
                break
            best_match = process.extractOne(
                self._normalize_circuit_name(raw_name),
                list(unclaimed_circuit_keys),
                scorer=fuzz.token_set_ratio,
                score_cutoff=CIRCUIT_FUZZY_SCORE_CUTOFF,
            )
            if best_match is None:
                continue
            matched_key, score, _ = best_match
            circuit_name_map[raw_name] = canonical_name_by_key[matched_key]
            claimed_circuit_keys.add(matched_key)
            unclaimed_circuit_keys.discard(matched_key)
            logger.info(
                "Fuzzy circuit match: %r -> %r (score %.0f)",
                raw_name,
                canonical_name_by_key[matched_key],
                score,
            )

    def _match_circuit_names_by_city(
        self,
        unresolved_names,
        races_df,
        circuit_df_keyed,
        circuit_name_map,
        claimed_circuit_keys,
    ) -> None:
        if not unresolved_names:
            return

        df_unclaimed_circuits = circuit_df_keyed[
            ~circuit_df_keyed["circuit_key"].isin(claimed_circuit_keys)
        ]
        unclaimed_names_by_city_key = df_unclaimed_circuits.groupby("city_key")[
            "circuit_name"
        ].agg(list)

        city_by_raw_name = (
            races_df.dropna(subset=["circuit_name"])
            .groupby("circuit_name")["city"]
            .first()
        )

        for raw_name in unresolved_names:
            city_key = self._normalize_city(city_by_raw_name.get(raw_name))
            candidate_names = unclaimed_names_by_city_key.get(city_key, [])
            if len(candidate_names) == 1:
                circuit_name_map[raw_name] = candidate_names[0]
                claimed_circuit_keys.add(
                    self._normalize_circuit_name(candidate_names[0])
                )
                logger.info(
                    "City circuit match: %r -> %r", raw_name, candidate_names[0]
                )

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    def _normalize_circuit_name(self, name: str) -> str:
        text = self._strip_accents(name).lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens = [t for t in text.split() if t not in CIRCUIT_GENERIC_WORDS]
        return " ".join(tokens)

    def _normalize_city(self, city) -> str:
        if pd.isna(city):
            return ""
        return self._strip_accents(str(city)).lower().strip()


class TransformRaceResult:
    def transform(self) -> pd.DataFrame:
        results_df = self._extract_race_results()
        grid_df = self._extract_starting_grid()
        fastest_df = self._extract_fastest_laps()

        results_df, grid_df = self._consolidate_columns(results_df, grid_df)
        results_df = self._join_starting_grid(results_df, grid_df)
        results_df = self._classify_end_position(results_df)
        results_df = self._normalize_team_names(results_df)
        results_df = self._join_fastest_laps(results_df, fastest_df)

        return results_df

    def _extract_race_results(self) -> pd.DataFrame:
        blob_index = filter_blob_index(
            get_blob_container_client(RAW_CONTAINER),
            list_session_blobs,
            session="race-result",
        )
        return concat_blob_frames(
            blob_index,
            {
                "race_id",
                "year",
                "round",
                "driver_name",
                "driver_code",
                "team_name",
                "end_position",
                "laps_completed",
            },
            add_round=True,
        )

    def _extract_starting_grid(self) -> pd.DataFrame:
        blob_index = filter_blob_index(
            get_blob_container_client(RAW_CONTAINER),
            list_grid_blobs,
            session="starting-grid",
        )
        return concat_blob_frames(
            blob_index,
            {
                "race_id",
                "driver_name",
                "team_name",
                "start_position",
                "round",
                "driver_code",
            },
            add_round=True,
        )

    def _extract_fastest_laps(self) -> pd.DataFrame:
        blob_index = filter_blob_index(
            get_blob_container_client(RAW_CONTAINER),
            list_fastest_lap_blobs,
            session="fastest-laps",
        )
        df = concat_blob_frames(
            blob_index,
            {"race_id", "fastest_lap_time", "lap_number", "driver_code", "driver_name"},
        )
        df["fastest_lap_seconds"] = parse_lap_strings(df["fastest_lap_time"])[
            "seconds"
        ]
        df["fastest_lap_number"] = pd.to_numeric(df["lap_number"]).astype("Int64")
        return df[
            [
                "race_id",
                "driver_name",
                "driver_code",
                "fastest_lap_seconds",
                "fastest_lap_number",
            ]
        ]

    def _consolidate_columns(
        self, results_df: pd.DataFrame, grid_df: pd.DataFrame
    ) -> pd.DataFrame:
        results_df = results_df[
            [
                "race_id",
                "year",
                "round",
                "driver_name",
                "driver_code",
                "team_name",
                "end_position",
                "points",
                "laps_completed",
            ]
        ]
        results_df = results_df.rename(
            columns={"end_position": "finish_position", "team_name": "constructor_name"}
        )

        grid_df = grid_df.drop(columns=["session_type"])
        grid_df = grid_df.rename(columns={"team_name": "constructor_name"})

        return results_df, grid_df

    def _join_starting_grid(
        self, results_df: pd.DataFrame, grid_df: pd.DataFrame
    ) -> pd.DataFrame:
        return results_df.merge(
            grid_df,
            on=["race_id", "round", "driver_name", "driver_code", "constructor_name"],
            how="left",
        )

    @staticmethod
    def _classify_end_position(results_df: pd.DataFrame) -> pd.DataFrame:
        df = results_df.copy()
        raw = df["finish_position"].astype("string").str.strip().str.upper()
        numeric = pd.to_numeric(raw, errors="coerce")

        status = pd.Series("DNF", index=df.index, dtype="string")
        status.loc[raw.isin(["NC", "DQ"])] = raw
        status.loc[numeric.notna()] = "CLAS"

        df["finish_position"] = numeric.astype("Int64")
        df["result_status"] = status
        return df

    def _normalize_team_names(self, result_df: pd.DataFrame) -> pd.DataFrame:
        team_aliases = {
            normalize_text(alias).casefold(): name
            for alias, name in constructor_mapping["team_name"].items()
        }
        result_df["constructor_name"] = result_df["constructor_name"].map(
            lambda name: team_aliases.get(name.casefold(), name)
        )
        return result_df

    def _join_fastest_laps(
        self, results_df: pd.DataFrame, fastest_df: pd.DataFrame
    ) -> pd.DataFrame:
        return results_df.merge(
            fastest_df, on=["race_id", "driver_name", "driver_code"]
        )


class TransformPracticeResult:
    def transform(self) -> pd.DataFrame:
        practice_df = self._extract_practice_results()
        practice_df = self._best_practice_laps(practice_df)
        practice_df = self._rename_columns(practice_df)
        practice_df = self._normalize_team_names(practice_df)
        return practice_df

    def _extract_practice_results(self) -> pd.DataFrame:
        blob_index = filter_blob_index(
            get_blob_container_client(RAW_CONTAINER),
            list_session_blobs,
            session="practice",
        )
        return concat_blob_frames(
            blob_index,
            {
                "race_id",
                "driver_name",
                "team_name",
                "session_type",
                "lap_time",
                "lap_count",
                "position",
                "round",
                "driver_code",
            },
            add_round=True,
        )

    @staticmethod
    def _best_practice_laps(practice_df: pd.DataFrame) -> pd.DataFrame:
        df = practice_df.copy()
        parsed = parse_lap_strings(df["lap_time"])
        keys = ["race_id", "session_type"]

        absolute = parsed["seconds"].where(~parsed["is_gap"])
        base = absolute.groupby([df[k] for k in keys]).transform("min")
        df["lap_seconds"] = (
            parsed["seconds"]
            .where(~parsed["is_gap"], base + parsed["seconds"])
            .round(3)
        )

        base_counts = absolute.notna().groupby([df[k] for k in keys]).sum()
        bad = base_counts[base_counts != 1]
        if not bad.empty:
            raise ValueError(f"Sessions without exactly one base time:\n{bad}")

        df["session_type"] = df["session_type"].map(SESSION_MAP)
        df = df.rename(columns={"lap_seconds": "best_lap_seconds"})
        return df.drop(columns=["lap_time", "round"])

    def _rename_columns(self, practice_df: pd.DataFrame) -> pd.DataFrame:
        df = practice_df.rename(
            columns={"team_name": "constructor_name", "position": "session_position"}
        )
        return df

    def _normalize_team_names(self, practice_df: pd.DataFrame) -> pd.DataFrame:
        team_aliases = {
            normalize_text(alias).casefold(): name
            for alias, name in constructor_mapping["team_name"].items()
        }
        practice_df["constructor_name"] = practice_df["constructor_name"].map(
            lambda name: team_aliases.get(name.casefold(), name)
        )
        return practice_df


