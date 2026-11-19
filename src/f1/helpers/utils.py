
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


load_dotenv()


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
            agg[f"flag_{flag.lower().replace(' ', '_')}_count"] = (
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
                    else float(x)
                    if isinstance(x, str)
                    else x
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
