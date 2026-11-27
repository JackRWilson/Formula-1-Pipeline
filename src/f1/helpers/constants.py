import re
from pathlib import Path
from re import Pattern
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_START_YEAR = 2018
RAW_CONTAINER = "f1-raw"

SESSION_SUFFIXES_BY_SESSION_TYPE = {
    "race-result": 2001,
    "pit-stop-summary": 2016,
}
SPRINT_START_YEAR = 2021
SPRINT_SESSION_TYPES = (
    "sprint-qualifying",
    "sprint-grid",
    "sprint-results",
)

F1SITE_MANIFEST_TABLE = "formula_one.f1site_scrape_manifest"
FASTF1_START_YEAR = 2018
FASTF1_SESSION_TABLE = "formula_one.fastf1_scrape_manifest"
FASTF1_EVENT_TABLE = "formula_one.fastf1_event_schedule"
FASTF1_CACHE_PATH = PROJECT_ROOT / "data" / "cache"

WIKIPEDIA_CIRCUITS_URL = "https://en.wikipedia.org/wiki/List_of_Formula_One_circuits"
MAX_ROUNDS_BLOB = "max_rounds.parquet"
WIKIPEDIA_CIRCUITS_BLOB = "circuits_raw.parquet"
PHOTON_LOCATIONS_BLOB = "locations_raw.parquet"
JOLPI_DRIVERS_URL = "https://api.jolpi.ca/f1/alpha/core/drivers/"
JOLPI_DRIVERS_BLOB_PREFIX = "jolpi/drivers"
JOLPI_DRIVER_COLUMNS = [
    "season_year",
    "source_driver_id",
    "driver_code",
    "first_name",
    "last_name",
    "nationality",
    "country_code",
    "date_of_birth",
]

CURATED_CIRCUITS_PATH = PROJECT_ROOT / "data" / "reference" / "curated_circuits.csv"
CIRCUIT_FUZZY_SCORE_CUTOFF = 90
CIRCUIT_GENERIC_WORDS = {
    "circuit",
    "circuito",
    "autodromo",
    "autodrom",
    "international",
    "internacional",
    "internazionale",
    "grand",
    "prix",
    "street",
    "track",
    "racing",
    "course",
    "de",
    "di",
    "del",
    "do",
    "of",
    "the",
    "e",
}

FASTF1_SESSION_KEYS = ["season_year", "round_number", "session_name"]
BASIC_FLAGS = ["blue", "yellow", "double_yellow", "red"]
SC_COLS = ["safety_car_count", "vsc_count", "safety_car_laps", "vsc_laps"]

TIRE_GROUP_KEYS = [*FASTF1_SESSION_KEYS, "driver_code", "tire_compound"]
TIRE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_MAPPING_2018 = {
    "SUPERSOFT": "SOFT",
    "HYPERSOFT": "SOFT",
    "ULTRASOFT": "SOFT",
    "SOFT": "MEDIUM",
    "MEDIUM": "HARD",
    "HARD": "HARD",
}
MIN_LAPS_FOR_DEGRADATION = 4
LAP_TIME_QUANTILE_CUTOFF = 0.95

# Blob layouts used by the shared blob-index helpers in blob_io.py.
SESSION_RESULT_FOLDERS = (
    "practice",
    "qualifying",
    "sprint-qualifying",
    "race-result",
    "sprint-result",
    "sprint-results",
)
SESSION_BLOB_FOLDERS = SESSION_RESULT_FOLDERS + (
    "starting-grid",
    "sprint-grid",
    "fastest-laps",
    "pit-stop-summary",
)
RACE_RESULT_FOLDERS = ("race-result",)
SPRINT_RESULT_FOLDERS = ("sprint-results", "sprint-result")
STARTING_GRID_FOLDERS = ("starting-grid",)
SPRINT_GRID_FOLDERS = ("sprint-grid",)
FASTEST_LAP_FOLDERS = ("fastest-laps",)
PRACTICE_FOLDERS = ("practice",)
QUALIFYING_FOLDERS = ("qualifying",)
SPRINT_QUALIFYING_FOLDERS = ("sprint-qualifying",)
PIT_STOP_FOLDERS = ("pit-stop-summary",)

SESSION_BLOB_PATTERN = re.compile(
    rf"^(?P<session>{'|'.join(map(re.escape, SESSION_BLOB_FOLDERS))})"
    r"/(?:(?P<session_number>\d+)/)?"
    r"(?P<year>\d{4})/(?P<round>\d+)\.parquet$"
)

JOLPI_DRIVERS_PATTERN = re.compile(
    rf"^{re.escape(JOLPI_DRIVERS_BLOB_PREFIX)}/(?P<year>\d{{4}})\.parquet$"
)
FASTF1_SESSIONS = [
    "practice-1",
    "practice-2",
    "practice-3",
    "qualifying",
    "race",
    "sprint-qualifying",
    "sprint-shootout",
    "sprint",
]
FASTF1_KINDS = ["laps", "weather", "messages"]
FASTF1_BLOB_PATTERN = re.compile(
    r"^fastf1/"
    r"(?P<year>\d{4})/"
    r"(?P<round>\d+)-(?P<grand_prix>[^/]+)/"
    rf"(?P<session>{'|'.join(map(re.escape, FASTF1_SESSIONS))})"
    rf"_(?P<kind>{'|'.join(FASTF1_KINDS)})"
    r"\.parquet$"
)
INT_GROUPS = ("year", "round", "session_number")


class BlobSource(NamedTuple):
    pattern: Pattern[str]
    folders: tuple[str, ...]

SESSION_SOURCE = BlobSource(SESSION_BLOB_PATTERN, SESSION_BLOB_FOLDERS)
JOLPI_DRIVERS_SOURCE = BlobSource(
    JOLPI_DRIVERS_PATTERN, (JOLPI_DRIVERS_BLOB_PREFIX,)
)
FASTF1_SOURCE = BlobSource(FASTF1_BLOB_PATTERN, ("fastf1",))
