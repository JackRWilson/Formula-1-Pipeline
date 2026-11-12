import hashlib
import io
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
from numpy.typing import NDArray
from selenium import webdriver
from selenium.common.exceptions import (
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
SRC_ROOT = str(Path(PROJECT_ROOT) / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
MAX_ATTEMPTS = 3

load_dotenv()


def init_col_map(col_map: dict) -> dict:
    """
    Takes a {column_name: column_index} dictionary as an input and returns a new dictionary with
    indexes and empty lists
    """
    return {col: {"index": index, "values": []} for col, index in col_map.items()}


def create_browser() -> WebDriver:
    """
    Creates a headless Chrome browser with logging suppressed
    """
    chrome_options = Options()
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    browser = webdriver.Chrome(options=chrome_options)
    return browser


_progress_start_time = None


def print_progress_bar(
    current: int,
    total: int,
    bar_length: int = 40,
    start_time: float = None,
) -> None:
    """
    Print a progress bar with ETA that updates on the same lines
    """
    global _progress_start_time
    if start_time is not None:
        _progress_start_time = start_time
    elif _progress_start_time is None or current == 1:
        _progress_start_time = time.time()

    percent = current * 100 // total
    filled = bar_length * current // total
    bar = "█" * filled + "-" * (bar_length - filled)

    eta_str = "Calculating..."
    if _progress_start_time and current > 0:
        elapsed = time.time() - _progress_start_time
        if elapsed > 0:
            rate = current / elapsed
            if rate > 0:
                remaining = (total - current) / rate

                if remaining < 60:
                    eta_str = f"{int(remaining)}s"
                elif remaining < 3600:
                    mins = int(remaining // 60)
                    secs = int(remaining % 60)
                    eta_str = f"{mins}m {secs}s"
                else:
                    hours = int(remaining // 3600)
                    mins = int((remaining % 3600) // 60)
                    eta_str = f"{hours}h {mins}m"

    print(
        f"\r   Progress: [{bar}] {current}/{total} ({percent}%) "
        f"ETA: {eta_str}            ",
        end="",
        flush=True,
    )

    if current >= total:
        print()
        _progress_start_time = None


def _extract_matching_rows(browser: WebDriver, min_col: int, max_col: int) -> list:
    """
    Return the cell texts of every table row on the current page whose
    <td> count falls within [min_col, max_col]. Each table's header row
    (the first <tr>) is skipped.
    """
    matching_rows = []
    for table in browser.find_elements(By.TAG_NAME, "table"):
        for row in table.find_elements(By.TAG_NAME, "tr")[1:]:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if not (min_col <= len(cells) <= max_col):
                    continue

                cell_texts = []
                for cell in cells:
                    try:
                        cell_texts.append(cell.text.strip())
                    except (StaleElementReferenceException, WebDriverException):
                        cell_texts.append(None)
            except (StaleElementReferenceException, WebDriverException):
                logger.warning(
                    "Skipping stale row while scraping table data.", exc_info=True
                )
                continue

            matching_rows.append(cell_texts)
    return matching_rows


def scrape_url_table(
    urls: list,
    min_col: int,
    max_col: int,
    col_idx_map: dict,
    page_lvl_cols: list | None = None,
    browser: WebDriver = None,
    show_progress: bool = True,
    raise_on_error: bool = False,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Scrapes table rows from one or more webpages and returns a DataFrame.

    Parameters
    ----------
    urls : list
        The webpage URL(s) to scrape
    min_col : int
        Minimum number of columns to accept in a table row
    max_col : int
        Maximum number of columns to accept in a table row
    col_idx_map : dict
        Maps desired column names to cell indices. An index of None produces
        a column of None values (to be filled in later); a callable index is
        a page-level column (see page_lvl_cols).
        Example: {'race_id': None, 'start_pos': 1, 'driver_name': 3, ...}
    page_lvl_cols : list, optional
        Names of columns whose index is a callable taking the browser and
        returning one value shared by every row on that page
    browser : optional
        An existing Selenium browser to reuse. If None, one is created
        and closed by this function.
    show_progress : bool, optional
        Whether to print a progress bar. Default: True.
    raise_on_error : bool, optional
        Re-raise scraping errors after the final attempt instead of logging
        and skipping the URL. Default: False.
    retry_delay : float, optional
        Seconds to wait between attempts on the same URL. Default: 1.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the scraped rows
    """
    # Validate inputs up front
    if min_col > max_col:
        raise ValueError(
            f"min_col ({min_col}) cannot be greater than max_col ({max_col})"
        )

    page_lvl_cols = page_lvl_cols or []
    col_data = init_col_map(col_idx_map)

    for col_name, col_info in col_data.items():
        idx = col_info["index"]
        if idx is None or isinstance(idx, int):
            continue
        if col_name in page_lvl_cols and callable(idx):
            continue
        raise ValueError(f"Unsupported index type for {col_name}: {type(idx)}")

    # Create a browser if not supplied
    owns_browser = browser is None
    if owns_browser:
        browser = create_browser()

    total_urls = len(urls)
    if show_progress:
        print(f"\n   Scraping {total_urls} URLs...")

    try:
        # Visit each URL
        for i, url in enumerate(urls, start=1):
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    # Load page and collect every row that fits the column range
                    browser.get(url)
                    rows = _extract_matching_rows(browser, min_col, max_col)

                    # Fetch page-level values once per page
                    page_level_data = {}
                    if rows:
                        page_level_data = {
                            name: col_data[name]["index"](browser)
                            for name in page_lvl_cols
                        }

                    # Map cells to output columns
                    buffer = {name: [] for name in col_data}
                    for cell_texts in rows:
                        for name, info in col_data.items():
                            idx = info["index"]
                            if idx is None:
                                value = None
                            elif isinstance(idx, int):
                                value = (
                                    cell_texts[idx] if idx < len(cell_texts) else None
                                )
                            else:
                                value = page_level_data[name]
                            buffer[name].append(value)

                    # Commit only after the whole page succeeded
                    for name, values in buffer.items():
                        col_data[name]["values"].extend(values)
                    break

                except Exception:
                    if attempt == MAX_ATTEMPTS:
                        if raise_on_error:
                            raise
                        logger.warning("ERROR with %s", url, exc_info=True)
                    else:
                        time.sleep(retry_delay)

            if show_progress:
                print_progress_bar(i, total_urls)
    finally:
        if owns_browser:
            browser.quit()

    if show_progress:
        print()

    # Assemble DataFrame
    return pd.DataFrame({name: info["values"] for name, info in col_data.items()})


def get_date(browser: WebDriver) -> str:
    """
    Extract date with wait and retry
    """
    try:
        # Wait for the element to be present
        element = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[@class='Container-module_container__0e4ac']//p[contains(@class, 'display-s-bold')]",
                )
            )
        )

        # Wait a bit for any re-rendering to finish
        time.sleep(0.5)

        # Get fresh reference to avoid stale element
        element = browser.find_element(
            By.XPATH,
            "//div[@class='Container-module_container__0e4ac']//p[contains(@class, 'display-s-bold')]",
        )
        return element.text
    except Exception as e:
        print(f"Failed to extract date: {e}")
        return None


def aggregate_columns(
    df: pd.DataFrame,
    columns: list | None = None,
    boolean_columns: list | None = None,
    string_columns: list | None = None,
) -> pd.DataFrame:
    """
    Aggregation function that returns mean, min, max, and std values for numeric columns,
    boolean aggregation for True/False columns, and the first value for string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of numeric column names to aggregate. If None, aggregates all numeric columns.
    boolean_columns : list
        List of boolean column names to check for any True values.
    string_columns : list
        List of string column names to extract (just takes first value).

    Returns
    -------
    pd.DataFrame
        Aggregated statistics for the specified columns with a single row
    """
    agg = {}

    # Handle numeric columns
    if columns is None:
        # Get all numeric columns if none specified
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            agg[f"{col}_mean"] = df[col].mean()
            agg[f"{col}_min"] = df[col].min()
            agg[f"{col}_max"] = df[col].max()
            agg[f"{col}_std"] = df[col].std()

    # Handle boolean columns
    if boolean_columns is not None:
        for col in boolean_columns:
            if col in df.columns:
                agg[f"{col}_any"] = bool(df[col].any())
                agg[f"{col}_mean"] = df[col].mean()

    # Handle string columns
    if string_columns is not None:
        for col in string_columns:
            if col in df.columns:
                agg[col] = df[col].iloc[0]

    return pd.DataFrame([agg])


def get_location_data(place: str, city: str, country: str) -> dict:
    """
    Get latitude, longitude, and elevation for a location.

    Parameters
    ----------
    place : str
        The name of the place/landmark
    city : str
        The city name
    country : str
        The country name

    Returns
    -------
    dict
        A dictionary with keys 'latitude', 'longitude', and 'elevation'
        Returns 'no_coords' if location is not found
        Returns 'no_elevation' if elevation is not found
        Returns 'error' if an error occurs

    """
    try:
        # Combine inputs into a single query string
        query = f"{place}, {city}, {country}"

        # Get coordinates from Photon API
        url = f"https://photon.komoot.io/api/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        response_raw = requests.get(url, headers=headers)
        if response_raw.status_code != 200:
            return "error"
        try:
            response = response_raw.json()
        except ValueError:
            return "error"

        if not response.get("features"):
            return "no_coords"

        coords = response["features"][0].get("geometry", {}).get("coordinates")
        if not coords or len(coords) < 2:
            return "no_coords"
        lat, lon = coords[1], coords[0]

        # Get elevation from Open Elevation API
        elevation_url = (
            f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        )
        elevation_raw = requests.get(elevation_url)
        if elevation_raw.status_code != 200:
            return "error"
        try:
            elevation_response = elevation_raw.json()
        except ValueError:
            return "error"
        try:
            elevation = elevation_response["results"][0]["elevation"]
        except (KeyError, IndexError, TypeError):
            return "no_elevation"

        return {"latitude": lat, "longitude": lon, "elevation": elevation}

    except requests.exceptions.RequestException:
        return "error"
    except (KeyError, IndexError):
        return "error"


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


def get_degradation_rate(lap_times: NDArray, lap_numbers: NDArray) -> float:
    """
    Calculates the degradation rate of tyres based on a linear regression
    """
    valid_mask = ~np.isnan(lap_times)
    clean_lap_times = lap_times[valid_mask]
    clean_lap_numbers = lap_numbers[valid_mask]

    if len(clean_lap_times) > 3:
        X = np.array(clean_lap_numbers).reshape(-1, 1)
        y = np.array(clean_lap_times)
        model = LinearRegression().fit(X, y)
        return model.coef_[0]
    return np.nan


def compute_compound_stats(comp_data: pd.DataFrame, compound: str) -> dict:
    """
    Calculate tyre compound statistics for a specific compount
    """
    if comp_data.empty:
        return {
            f"avg_pace_{compound.lower()}": np.nan,
            f"std_pace_{compound.lower()}": np.nan,
            f"laps_on_{compound.lower()}": 0,
            f"deg_rate_{compound.lower()}": np.nan,
        }
    lap_times = comp_data["LapTime"].values
    lap_numbers = comp_data["LapNumber"].values
    return {
        f"avg_pace_{compound.lower()}": np.nanmean(lap_times),
        f"std_pace_{compound.lower()}": np.nanstd(lap_times),
        f"laps_on_{compound.lower()}": len(lap_times),
        f"deg_rate_{compound.lower()}": get_degradation_rate(lap_times, lap_numbers),
    }


def process_lap_file(
    filepath: Path,
    code_to_name_map: dict,
    driver_id_map: dict,
) -> pd.DataFrame:
    """
    Process a single laps file and return aggregated driver stats
    """
    fh = pd.read_parquet(filepath)

    # Filter rows
    filtered_data = fh[
        (fh["TrackStatus"] == "1")  # No flags
        & fh["PitOutTime"].isna()  # Not an OUT lap
        & fh["PitInTime"].isna()  # Not an IN lap
        & fh["IsAccurate"]
        == True  # Full lap completed and is accurate
        & (fh["LapTime"] < fh["LapTime"].quantile(0.95))  # Get rid of outliers
    ].copy()

    if len(filtered_data) == 0:
        print("      No data after filtering")
        return pd.DataFrame()

    # Convert lap times
    filtered_data["LapTime"] = filtered_data["LapTime"].dt.total_seconds()
    filtered_data = filtered_data[filtered_data["LapTime"].notna()].copy()

    # Map driver codes and add metadata
    filtered_data["driver_id"] = (
        filtered_data["Driver"].map(code_to_name_map).map(driver_id_map)
    )
    if "race_id" in fh.columns:
        filtered_data["race_id"] = fh.loc[filtered_data.index, "race_id"]
    if "session" in fh.columns:
        filtered_data["session"] = fh.loc[filtered_data.index, "session"]

    # Extract year from filename for compound mapping
    year = int(filepath.stem.split("_")[0])

    # Apply compound mapping for 2018
    if year == 2018:
        compound_mapping = {
            "SUPERSOFT": "SOFT",
            "HYPERSOFT": "SOFT",
            "ULTRASOFT": "SOFT",
            "SOFT": "MEDIUM",
            "MEDIUM": "HARD",
            "HARD": "HARD",
        }
        filtered_data["Compound"] = (
            filtered_data["Compound"]
            .map(compound_mapping)
            .fillna(filtered_data["Compound"])
        )

    # Aggregate by driver
    compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    summaries = []

    for driver, group in filtered_data.groupby("Driver"):
        summary = {"driver_name": driver}

        # Add metadata from first row
        for col in ["driver_id", "race_id", "session"]:
            if col in group.columns:
                summary[col] = group[col].iloc[0]

        # Add compound stats
        for comp in compounds:
            comp_data = group[group["Compound"] == comp]
            summary.update(compute_compound_stats(comp_data, comp))

        summaries.append(summary)

    return pd.DataFrame(summaries)


def process_flag_file(
    filepath: Path, basic_count_flags: list, string_columns: list
) -> pd.DataFrame:
    """
    Process a single flag file and return aggregated stats
    """

    fh = pd.read_parquet(filepath)

    # Initialize aggregation dictionary
    agg = {}

    # Count each flag type
    if "Flag" in fh.columns:
        for flag in basic_count_flags:
            agg[f'flag_{flag.lower().replace(" ", "_")}_count'] = (
                fh["Flag"] == flag
            ).sum()

    # Count safety car deployments
    if all(col in fh.columns for col in ["Category", "Status", "Message"]):
        sc_deployments = fh[
            (fh["Category"] == "SafetyCar")
            & (fh["Status"] == "DEPLOYED")
            & (~fh["Message"].str.contains("VIRTUAL", case=False, na=False))
        ]
        agg["safety_car_deployments"] = len(sc_deployments)

    # Count virtual safety Car deployments
    if all(col in fh.columns for col in ["Category", "Status", "Message"]):
        vsc_deployments = fh[
            (fh["Category"] == "SafetyCar")
            & (fh["Status"] == "DEPLOYED")
            & (fh["Message"].str.contains("VIRTUAL", case=False, na=False))
        ]
        agg["virtual_safety_car_deployments"] = len(vsc_deployments)

    # Calculate safety car lap percentages
    if "Lap" in fh.columns:
        total_laps = fh["Lap"].max()
        if total_laps > 0:
            # Find all SC laps
            sc_deployed_laps = fh[
                (fh["Category"] == "SafetyCar")
                & (fh["Status"] == "DEPLOYED")
                & (~fh["Message"].str.contains("VIRTUAL", case=False, na=False))
            ]["Lap"].values
            sc_in_laps = fh[
                (fh["Category"] == "SafetyCar")
                & (fh["Status"] == "IN THIS LAP")
                & (~fh["Message"].str.contains("VIRTUAL", case=False, na=False))
            ]["Lap"].values

            # Find all VSC laps
            vsc_deployed_laps = fh[
                (fh["Category"] == "SafetyCar")
                & (fh["Status"] == "DEPLOYED")
                & (fh["Message"].str.contains("VIRTUAL", case=False, na=False))
            ]["Lap"].values
            vsc_ending_laps = fh[
                (fh["Category"] == "SafetyCar")
                & (fh["Status"] == "ENDING")
                & (fh["Message"].str.contains("VIRTUAL", case=False, na=False))
            ]["Lap"].values

            # Sum up all the safety car periods
            total_sc_laps = sum(
                sc_in_laps[i] - sc_deployed_laps[i] + 1
                for i in range(min(len(sc_deployed_laps), len(sc_in_laps)))
            )
            total_vsc_laps = sum(
                vsc_ending_laps[i] - vsc_deployed_laps[i] + 1
                for i in range(min(len(vsc_deployed_laps), len(vsc_ending_laps)))
            )
            agg["total_sc_laps"] = total_sc_laps
            agg["total_vsc_laps"] = total_vsc_laps

    # Add string columns
    for col in string_columns:
        if col in fh.columns and not fh[col].empty:
            agg[col] = fh[col].iloc[0]

    return pd.DataFrame([agg])


def convert_position(row: pd.Series, prev_position: int | None = None) -> int:
    """
    Converts position to numeric, or changes classification to a number
    """
    try:
        return int(row["position"])
    except (ValueError, TypeError):
        if prev_position is not None:
            return prev_position + 1
        else:
            return 1


def clean_qualifying_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans qualifying session times and adds no_lap_time flags for Q1, Q2, Q3.
    Uses qualifying position to determine advancement and flags drivers with DNF/DNS/NA times.

    Assumes df contains: ['race_id', 'driver_id', 'q1_time', 'q2_time', 'q3_time', 'qual_position']
    """

    df = df.copy()

    # Define cutoffs
    Q2_CUTOFF = 15  # Top 15 advance to Q2
    Q3_CUTOFF = 10  # Top 10 advance to Q3

    # Ensure qualifying position is numeric and sort
    df["qual_position"] = pd.to_numeric(df["qual_position"], errors="coerce")
    df = df.sort_values(["race_id", "qual_position"])

    # Determine who advances to Q2 and Q3
    df["advanced_to_q2"] = df["qual_position"] <= Q2_CUTOFF
    df["advanced_to_q3"] = df["qual_position"] <= Q3_CUTOFF

    # Convert time strings to seconds
    for session in ["q1", "q2", "q3"]:
        time_col = f"{session}_time"

        # Convert to numeric
        df[time_col] = df[time_col].apply(
            lambda x: (
                None
                if pd.isna(x)
                or (isinstance(x, str) and x.upper() in ["DNF", "DNS", ""])
                else (
                    float(x.split(":")[0]) * 60 + float(x.split(":")[1])
                    if isinstance(x, str) and ":" in x
                    else float(x) if isinstance(x, str) else x
                )
            )
        )

    # Add no_lap_time flags for each session
    df["q1_no_lap_time_flag"] = df["q1_time"].isna().astype(int)
    df["q2_no_lap_time_flag"] = (
        (df["advanced_to_q2"]) & (df["q2_time"].isna())
    ).astype(int)
    df["q3_no_lap_time_flag"] = (
        (df["advanced_to_q3"]) & (df["q3_time"].isna())
    ).astype(int)

    # Get max times per session for imputation
    df["max_q1_time"] = df.groupby("race_id")["q1_time"].transform("max")
    df["max_q2_time"] = df.groupby("race_id")["q2_time"].transform("max")
    df["max_q3_time"] = df.groupby("race_id")["q3_time"].transform("max")

    # Q1 imputation
    df.loc[df["q1_time"].isna(), "q1_time"] = df["max_q1_time"] + 5

    # Q2 imputation
    q2_missing = df["q2_time"].isna()
    df.loc[q2_missing & df["advanced_to_q2"], "q2_time"] = (
        df["max_q2_time"] + 5
    )  # +5s for drivers who advanced but didnt set a time
    df.loc[q2_missing & ~df["advanced_to_q2"], "q2_time"] = (
        df["max_q2_time"] + 10
    )  # +10s for drivers who didnt advance

    # Q3 imputation
    q3_missing = df["q3_time"].isna()
    df.loc[q3_missing & df["advanced_to_q3"], "q3_time"] = df["max_q3_time"] + 5
    df.loc[q3_missing & ~df["advanced_to_q3"], "q3_time"] = df["max_q3_time"] + 10

    # Drop helper columns
    df.drop(
        columns=[
            "advanced_to_q2",
            "advanced_to_q3",
            "max_q1_time",
            "max_q2_time",
            "max_q3_time",
        ],
        inplace=True,
    )

    return df


def clean_pit_time(time_str: str) -> float:
    """
    Convert pit stop time from min:sec.millisec format to total seconds
    """
    if ":" in time_str:
        minutes, seconds = time_str.split(":")
        return float(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


def impute_pit_times(row: pd.Series, pit_stops_bound: pd.DataFrame) -> float:
    current_race_id = row["race_id"]
    driver_id = row["driver_id"]
    team_id = row["team_id"]

    # Filter pit_stops_bound to only include stops before current race
    past_stops = pit_stops_bound[pit_stops_bound["race_id"] < current_race_id]

    # No historical data, return 0
    if past_stops.empty:
        return 0

    # Try driver average (last 5 pits)
    driver_stops = past_stops[past_stops["driver_id"] == driver_id][
        "pits_time_seconds"
    ].tail(5)
    if len(driver_stops) > 0:
        return driver_stops.mean()

    # Try driver average (all pits)
    driver_stops = past_stops[past_stops["driver_id"] == driver_id]["pits_time_seconds"]
    if len(driver_stops) > 0:
        return driver_stops.mean()

    # Try team average (last 5 pits)
    team_stops = past_stops[past_stops["team_id"] == team_id]["pits_time_seconds"].tail(
        5
    )
    if len(team_stops) > 0:
        return team_stops.mean()

    # Try team average (last 10 pits)
    team_stops = past_stops[past_stops["team_id"] == team_id]["pits_time_seconds"].tail(
        10
    )
    if len(team_stops) > 0:
        return team_stops.mean()

    # Global team average (all pits)
    team_stops = past_stops[past_stops["team_id"] == team_id]["pits_time_seconds"]
    if len(team_stops) > 0:
        return team_stops.mean()

    # Fallback to 0
    return 0


def clean_circuit_name(name: str) -> str:
    """
    Clean circuit name to remove special characters while keeping letters, spaces, hyphens, and international letters with accents
    """
    if not isinstance(name, str):
        return name

    # Normalize to NFKC to standardize Unicode characters
    name = unicodedata.normalize("NFKC", name)

    # Remove daggers and asterisks
    name = name.replace("†", "").replace("*", "")

    # Keep only letters (including international), spaces, and hyphens
    cleaned = "".join(char for char in name if char.isalpha() or char in (" ", "-"))

    # Replace multiple spaces or hyphens with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    return cleaned


constructor_mapping = {
    "team_id": {
        # Red Bull
        "Red Bull Racing Renault": "Red Bull",
        "Red Bull Renault": "Red Bull",
        "RBR Renault": "Red Bull",
        "RBR Cosworth": "Red Bull",
        "RBR Ferrari": "Red Bull",
        "Red Bull Racing TAG Heuer": "Red Bull",
        "Red Bull Racing Honda": "Red Bull",
        "Red Bull Racing RBPT": "Red Bull",
        "Red Bull Racing Honda RBPT": "Red Bull",
        "Red Bull Racing": "Red Bull",
        # AlphaTauri/Toro Rosso
        "Toro Rosso": "Toro Rosso",
        "STR Ferrari": "Toro Rosso",
        "STR Renault": "Toro Rosso",
        "STR Cosworth": "Toro Rosso",
        "Toro Rosso Ferrari": "Toro Rosso",
        "Scuderia Toro Rosso Honda": "Toro Rosso",
        "AlphaTauri Honda": "AlphaTauri",
        "AlphaTauri RBPT": "AlphaTauri",
        "AlphaTauri Honda RBPT": "AlphaTauri",
        # Racing Bulls
        "RB Honda RBPT": "Racing Bulls",
        # Ferrari
        "Ferrari": "Ferrari",
        "Ferrari Jaguar": "Ferrari",
        "Thin Wall Ferrari": "Ferrari",
        # Mercedes
        "Mercedes": "Mercedes",
        "Mercedes-Benz": "Mercedes",
        # Aston Martin
        "Aston Martin Mercedes": "Aston Martin",
        "Aston Martin Aramco Mercedes": "Aston Martin",
        "Aston Butterworth": "Aston Martin",
        "Aston Martin": "Aston Martin",
        # McLaren
        "McLaren Ford": "McLaren",
        "McLaren TAG": "McLaren",
        "McLaren Honda": "McLaren",
        "McLaren Peugeot": "McLaren",
        "McLaren Renault": "McLaren",
        "McLaren BRM": "McLaren",
        "McLaren Mercedes": "McLaren",
        "McLaren Serenissima": "McLaren",
        "Mclaren BRM": "McLaren",
        "McLaren Alfa Romeo": "McLaren",
        # Williams
        "Williams Ford": "Williams",
        "Williams Renault": "Williams",
        "Williams Honda": "Williams",
        "Williams Judd": "Williams",
        "Williams BMW": "Williams",
        "Williams Toyota": "Williams",
        "Williams Cosworth": "Williams",
        "Williams Mecachrome": "Williams",
        "Williams Supertec": "Williams",
        "Williams Mercedes": "Williams",
        "Frank Williams Racing Cars/Williams": "Williams",
        # Renault
        "Renault": "Renault",
        # Alpine
        "Alpine Renault": "Alpine",
        # Lotus
        "Lotus Renault": "Lotus",
        "Lotus Ford": "Lotus",
        "Lotus Climax": "Lotus",
        "Lotus BRM": "Lotus",
        "Lotus Honda": "Lotus",
        "Lotus Judd": "Lotus",
        "Lotus Lamborghini": "Lotus",
        "Lotus Mugen Honda": "Lotus",
        "Lotus Mercedes": "Lotus",
        "Lotus Cosworth": "Lotus",
        "Lotus Maserati": "Lotus",
        "Lotus Pratt & Whitney": "Lotus",
        # Force India
        "Force India Ferrari": "Force India",
        "Force India Mercedes": "Force India",
        # Racing Point
        "Racing Point BWT Mercedes": "Racing Point",
        # Sauber
        "Sauber": "Sauber",
        "Sauber Ferrari": "Sauber",
        "Sauber Petronas": "Sauber",
        "Sauber BMW": "Sauber",
        "Sauber Mercedes": "Sauber",
        "Sauber Ford": "Sauber",
        "Kick Sauber Ferrari": "Sauber",
        # Alfa Romeo
        "Alfa Romeo Racing Ferrari": "Alfa Romeo",
        "Alfa Romeo Ferrari": "Alfa Romeo",
        "Alfa Romeo": "Alfa Romeo",
        # Haas
        "Haas Ferrari": "Haas",
        "Haas F1 Team": "Haas",
        # Jordan
        "Jordan Ford": "Jordan",
        "Jordan Peugeot": "Jordan",
        "Jordan Hart": "Jordan",
        "Jordan Honda": "Jordan",
        "Jordan Yamaha": "Jordan",
        "Jordan Toyota": "Jordan",
        "Jordan Mugen Honda": "Jordan",
        # BAR
        "BAR Honda": "BAR",
        "BAR Supertec": "BAR",
        # Honda
        "Honda": "Honda",
        # Benetton
        "Benetton Ford": "Benetton",
        "Benetton BMW": "Benetton",
        "Benetton Renault": "Benetton",
        "Benetton Playlife": "Benetton",
        # Toyota
        "Toyota": "Toyota",
        # Jaguar
        "Jaguar Cosworth": "Jaguar",
        # Stewart
        "Stewart Ford": "Stewart",
        # BRM
        "BRM": "BRM",
        "BRM Climax": "BRM",
        # JBW
        "JBW Maserati": "JBW",
        "JBW Climax": "JBW",
        # Cooper
        "Cooper Climax": "Cooper",
        "Cooper Maserati": "Cooper",
        "Cooper Bristol": "Cooper",
        "Cooper Castellotti": "Cooper",
        "Cooper BRM": "Cooper",
        "Cooper JAP": "Cooper",
        "Cooper Alta": "Cooper",
        "Cooper Borgward": "Cooper",
        "Cooper Alfa Romeo": "Cooper",
        "Cooper Ferrari": "Cooper",
        "Cooper ATS": "Cooper",
        "Cooper Ford": "Cooper",
        "Cooper OSCA": "Cooper",
        # Brabham
        "Brabham Climax": "Brabham",
        "Brabham Repco": "Brabham",
        "Brabham Ford": "Brabham",
        "Brabham Alfa Romeo": "Brabham",
        "Brabham BMW": "Brabham",
        "Brabham BRM": "Brabham",
        "Brabham Judd": "Brabham",
        "Brabham Yamaha": "Brabham",
        # Maserati
        "Maserati": "Maserati",
        "Maserati Offenhauser": "Maserati",
        "Maserati Milano": "Maserati",
        "Maserati-Offenhauser": "Maserati",
        "Maserati OSCA": "Maserati",
        "Maserati Plate": "Maserati",
        # Ligier
        "Ligier Matra": "Ligier",
        "Ligier Ford": "Ligier",
        "Ligier Renault": "Ligier",
        "Ligier Megatron": "Ligier",
        "Ligier Mugen Honda": "Ligier",
        # Tyrrell
        "Tyrrell Ford": "Tyrrell",
        "Tyrrell Renault": "Tyrrell",
        "Tyrrell Honda": "Tyrrell",
        "Tyrrell Yamaha": "Tyrrell",
        "Tyrrell Ilmor": "Tyrrell",
        # Arrows/Footwork
        "Arrows Ford": "Arrows",
        "Arrows BMW": "Arrows",
        "Arrows Megatron": "Arrows",
        "Arrows Yamaha": "Arrows",
        "Arrows Supertec": "Arrows",
        "Arrows Asiatech": "Arrows",
        "Arrows Cosworth": "Arrows",
        "Arrows": "Arrows",
        "Footwork Ford": "Footwork",
        "Footwork Hart": "Footwork",
        "Footwork Mugen Honda": "Footwork",
        "Footwork Porsche": "Footwork",
        # Vanwall
        "Vanwall": "Vanwall",
        # Wolf
        "Wolf Ford": "Wolf",
        "Wolf-Williams": "Wolf",
        # Lola
        "Lola Ford": "Lola",
        "Lola Lamborghini": "Lola",
        "Lola Climax": "Lola",
        "Lola BMW": "Lola",
        "Lola Hart": "Lola",
        "Lola Ferrari": "Lola",
        # March
        "March Ford": "March",
        "March Judd": "March",
        "March Ilmor": "March",
        "March Alfa Romeo": "March",
        # Minardi
        "Minardi Ford": "Minardi",
        "Minardi Ferrari": "Minardi",
        "Minardi Lamborghini": "Minardi",
        "Minardi Asiatech": "Minardi",
        "Minardi Cosworth": "Minardi",
        "Minardi Fondmetal": "Minardi",
        "Minardi European": "Minardi",
        "Minardi Hart": "Minardi",
        "Minardi Motori Moderni": "Minardi",
        # LDS
        "LDS Alfa Romeo": "LDS",
        "LDS Climax": "LDS",
        "LDS Repco": "LDS",
        # Porche
        "Porsche (F2)": "Porsche",
        "Porsche": "Porsche",
        "Behra-Porsche": "Porsche",
        # Scirocco
        "Scirocco BRM": "Scirocco",
        "Scirocco Climax": "Scirocco",
        # AFM
        "AFM Kuchen": "AFM",
        "AFM BMW": "AFM",
        "AFM Bristol": "AFM",
        # ATS
        "ATS Ford": "ATS",
        "ATS": "ATS",
        "ATS BMW": "ATS",
        "Derrington-Francis ATS": "ATS",
        # Leyton House
        "Leyton House Judd": "Leyton House",
        "Leyton House Ilmor": "Leyton House",
        # Prost
        "Prost Mugen Honda": "Prost",
        "Prost Peugeot": "Prost",
        "Prost Acer": "Prost",
        # Dallara
        "Dallara Judd": "Dallara",
        "Dallara Ferrari": "Dallara",
        "Dallara Ford": "Dallara",
        # Larrousse
        "Larrousse Lamborghini": "Larrousse",
        "Larrousse Ford": "Larrousse",
        # Osella
        "Osella Ford": "Osella",
        "Osella Alfa Romeo": "Osella",
        "Osella": "Osella",
        "Osella Hart": "Osella",
        # Kurtis Kraft
        "Kurtis Kraft Offenhauser": "Kurtis Kraft",
        "Kurtis Kraft Novi": "Kurtis Kraft",
        "Kurtis Kraft Cummins": "Kurtis Kraft",
        # Marussia
        "Marussia Cosworth": "Marussia",
        "Marussia Ferrari": "Marussia",
        # Gordini
        "Simca-Gordini": "Gordini",
        "Gordini": "Gordini",
        # Connaught
        "Connaught Lea Francis": "Connaught",
        "Connaught Alta": "Connaught",
        # Eagle
        "Eagle Climax": "Eagle",
        "Eagle Weslake": "Eagle",
        # RAM
        "RAM Ford": "RAM",
        "RAM Hart": "RAM",
        # Shadow
        "Shadow Ford": "Shadow",
        "Shadow Matra": "Shadow",
        # Matra
        "Matra Ford": "Matra",
        "Matra": "Matra",
        "Matra Cosworth": "Matra",
        "Matra BRM": "Matra",
        # ERA
        "ERA": "ERA",
        "ERA Bristol": "ERA",
        # Spirit
        "Spirit Honda": "Spirit",
        "Spirit Hart": "Spirit",
        # Frazer Nash
        "Frazer Nash": "Frazer Nash",
        "Frazer Nash Bristol": "Frazer Nash",
        # Emeryson
        "Emeryson Alta": "Emeryson",
        "Emeryson Climax": "Emeryson",
        # De Tomaso
        "De Tomaso OSCA": "De Tomaso",
        "De Tomaso Alfa Romeo": "De Tomaso",
        "De Tomaso Ford": "De Tomaso",
        # Gilby
        "Gilby Climax": "Gilby",
        "Gilby BRM": "Gilby",
        # Tecno
        "Tecno": "Tecno",
        "Tecno Cosworth": "Tecno",
        # Ligier
        "Ligier Judd": "Ligier",
        "Ligier Lamborghini": "Ligier",
        # Euro Brun
        "Euro Brun Judd": "Euro Brun",
        "Euro Brun Ford": "Euro Brun",
        # Other
        "No Team": "Privateer",
        "Toleman Hart": "Toleman",
        "Venturi Lamborghini": "Venturi",
        "Onyx Ford": "Onyx",
        "AGS Ford": "AGS",
        "Rial Ford": "Rial",
        "Zakspeed": "Zakspeed",
        "Theodore Ford": "Theodore",
        "Deidt Offenhauser": "Deidt",
        "Sherman Offenhauser": "Sherman",
        "Schroeder Offenhauser": "Schroeder",
        "Kuzma Offenhauser": "Kuzma",
        "Lesovsky Offenhauser": "Lesovsky",
        "Watson Offenhauser": "Watson",
        "Phillips Offenhauser": "Phillips",
        "Epperly Offenhauser": "Epperly",
        "Trevis Offenhauser": "Trevis",
        "HRT Cosworth": "HRT",
        "Virgin Cosworth": "Virgin",
        "Caterham Renault": "Caterham",
        "Milano Speluzzi": "Milano",
        "Turner Offenhauser": "Turner",
        "Alta": "Alta",
        "Moore Offenhauser": "Moore",
        "Nichels Offenhauser": "Nichels",
        "Marchese Offenhauser": "Marchese",
        "Stevens Offenhauser": "Stevens",
        "Langley Offenhauser": "Langley",
        "Ewing Offenhauser": "Ewing",
        "Rae Offenhauser": "Rae",
        "Olson Offenhauser": "Olson",
        "Wetteroth Offerhauser": "Wetteroth",
        "Snowberger Offenhauser": "Snowberger",
        "Adams Offenhauser": "Adams",
        "HWM Alta": "HWM",
        "Lancia": "Lancia",
        "Talbot-Lago": "Talbot-Lago",
        "BRP BRM": "BRP",
        "Hesketh Ford": "Hesketh",
        "Hill Ford": "Hill",
        "Ensign Ford": "Ensign",
        "Penske Ford": "Penske",
        "Fittipaldi Ford": "Fittipaldi",
        "ISO Marlboro Ford": "ISO Marlboro",
        "Iso Marlboro Ford": "ISO Marlboro",
        "Surtees Ford": "Surtees",
        "Parnelli Ford": "Parnelli",
        "Super Aguri Honda": "Super Aguri",
        "MRT Mercedes": "Manor",
        "Brawn Mercedes": "Brawn",
        "Spyker Ferrari": "Spyker",
        "MF1 Toyota": "Midland",
        "Veritas": "Veritas",
        "Pawl Offenhauser": "Pawl",
        "Hall Offenhauser": "Hall",
        "Bromme Offenhauser": "Bromme",
        "OSCA": "OSCA",
        "BMW": "BMW",
        "EMW": "EMW",
        "Pankratz Offenhauser": "Pankratz",
        "Bugatti": "Bugatti",
        "Klenk BMW": "Klenk",
        "Dunn Offenhauser": "Dunn",
        "Elder Offenhauser": "Elder",
        "Christensen Offenhauser": "Christensen",
        "Sutton Offenhauser": "Sutton",
        "Tec-Mec Maserati": "Tec-Mec",
        "Meskowski Offenhauser": "Meskowski",
        "Scarab": "Scarab",
        "Ferguson Climax": "Ferguson",
        "ENB Maserati": "ENB",
        "Stebro Ford": "Stebro",
        "Shannon Climax": "Shannon",
        "Protos Cosworth": "Protos",
        "Bellasi Ford": "Bellasi",
        "Eifelland Ford": "Eifelland",
        "Politoys Ford": "Politoys",
        "Connew Ford": "Connew",
        "Trojan Ford": "Trojan",
        "Amon Ford": "Amon",
        "Token Ford": "Token",
        "Lyncar Ford": "Lyncar",
        "Boro Ford": "Boro",
        "Kojima Ford": "Kojima",
        "LEC Ford": "LEC",
        "Merzario Ford": "Merzario",
        "Martini Ford": "Martini",
        "Rebaque Ford": "Rebaque",
        "AGS Motori Moderni": "AGS",
        "Coloni Ford": "Coloni",
        "Zakspeed Yamaha": "Zakspeed",
        "Fondmetal Ford": "Fondmetal",
        "Moda Judd": "Moda",
        "Simtek Ford": "Simtek",
        "Pacific Ilmor": "Pacific",
        "Forti Ford": "Forti",
        "Lambo Lamborghini": "Modena",
    }
}
