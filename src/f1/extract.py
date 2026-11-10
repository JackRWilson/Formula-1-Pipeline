import logging
import re
from datetime import date, datetime
from pathlib import Path

import fastf1
import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from sqlalchemy import text

if __package__:
    from .project_functions import clean_circuit_name, get_date
    from .utils import (
        create_browser,
        download_df_from_blob,
        get_location_data,
        print_progress_bar,
        scrape_url_table,
        upload_df_to_blob,
    )
else:
    from project_functions import clean_circuit_name, get_date
    from utils import (
        create_browser,
        download_df_from_blob,
        get_location_data,
        print_progress_bar,
        scrape_url_table,
        upload_df_to_blob,
    )

SESSION_SUFFIXES_BY_SESSION_TYPE = {
    "race-result": 2001,
    "pit-stop-summary": 2016,
}
DATA_START_YEAR = 2018
SPRINT_START_YEAR = 2021
SPRINT_SESSION_TYPES = (
    "sprint-qualifying",
    "sprint-grid",
    "sprint-results",
)
F1SITE_MANIFEST_TABLE = "formula_one.f1site_scrape_manifest"
RAW_CONTAINER = "f1-raw"
FASTF1_START_YEAR = 2018
FASTF1_SESSION_TABLE = "formula_one.fastf1_scrape_manifest"
FASTF1_EVENT_TABLE = "formula_one.fastf1_event_schedule"
FASTF1_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "cache"
WIKIPEDIA_CIRCUITS_URL = "https://en.wikipedia.org/wiki/List_of_Formula_One_circuits"
MAX_ROUNDS_BLOB = "max_rounds.parquet"
WIKIPEDIA_CIRCUITS_BLOB = "circuits_raw.parquet"
PHOTON_LOCATIONS_BLOB = "locations_raw.parquet"
logger = logging.getLogger(__name__)


class NoSessionDataError(ValueError):
    pass


def expand_race_url_to_session_urls(race_url, year):
    base_url = race_url.rsplit("/", 1)[0]
    session_types = [
        session_type
        for session_type, first_year in SESSION_SUFFIXES_BY_SESSION_TYPE.items()
        if year >= first_year
    ]
    if year >= DATA_START_YEAR:
        if year == 2020 and "/emilia-romagna/" in race_url:
            session_types.append("practice/0")
        else:
            session_types.extend(("practice/1", "practice/2", "practice/3"))
        session_types.extend(("qualifying", "starting-grid", "fastest-laps"))
    if year >= SPRINT_START_YEAR:
        session_types.extend(SPRINT_SESSION_TYPES)

    sessions = [
        {
            "url": f"{base_url}/{session_type}",
            "year": year,
            "session_type": session_type,
        }
        for session_type in session_types
    ]
    return sessions


class ExtractF1Site:
    def __init__(self, conn):
        self._conn = conn

    def extract(self):
        self.discover_sessions()
        self.extract_pending_sessions()
        self.extract_max_rounds()

    def discover_sessions(self):
        print("\nDiscovering F1 session URLs...")
        browser = create_browser()
        sessions = []
        newest_year = datetime.now().year

        try:
            for year in range(2001, newest_year + 1):
                index_url = f"https://www.formula1.com/en/results/{year}/races"
                browser.get(index_url)

                round_number = 0
                for table in browser.find_elements(By.TAG_NAME, "table"):
                    rows = table.find_elements(By.TAG_NAME, "tr")[1:]
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if not cells:
                            continue
                        try:
                            race_url = (
                                cells[0]
                                .find_element(By.TAG_NAME, "a")
                                .get_attribute("href")
                            )
                        except NoSuchElementException:
                            continue
                        if not race_url:
                            continue

                        round_number += 1
                        for session in expand_race_url_to_session_urls(race_url, year):
                            sessions.append({**session, "round_number": round_number})

                print_progress_bar(year - 2000, newest_year - 2000)
        finally:
            browser.quit()

        if sessions:
            self._conn.execute(
                text(f"""
                    insert into {F1SITE_MANIFEST_TABLE}
                        (url, year, round_number, session_type, status)
                    values
                        (:url, :year, :round_number, :session_type, 'pending')
                    on conflict (url) do nothing
                    """),
                sessions,
            )
            self._conn.commit()
        print(f"\nDiscovered {len(sessions)} session URLs.\n")

    def extract_pending_sessions(self):
        sessions = self._conn.execute(text(f"""
                select url, year, round_number, session_type
                from {F1SITE_MANIFEST_TABLE}
                where status in ('pending', 'failed')
                order by year, round_number, session_type
                """)).mappings().all()
        if not sessions:
            print("No pending session pages to scrape.\n")
            return

        browser = create_browser()
        failures = []
        unavailable = 0
        try:
            for index, session in enumerate(sessions, start=1):
                try:
                    data = self._scrape_session(session, browser)
                    blob_name = self._blob_name(session)
                    content_sha256 = upload_df_to_blob(data, RAW_CONTAINER, blob_name)
                    self._conn.execute(
                        text(f"""
                            update {F1SITE_MANIFEST_TABLE}
                            set status = 'success',
                                blob_name = :blob_name,
                                content_sha256 = :content_sha256,
                                scraped_at = now()
                            where url = :url
                            """),
                        {
                            "url": session["url"],
                            "blob_name": blob_name,
                            "content_sha256": content_sha256,
                        },
                    )
                    self._conn.commit()
                except NoSessionDataError:
                    self._conn.rollback()
                    unavailable += 1
                    self._conn.execute(
                        text(f"""
                            update {F1SITE_MANIFEST_TABLE}
                            set status = 'unavailable',
                                blob_name = null,
                                content_sha256 = null,
                                scraped_at = null
                            where url = :url
                            """),
                        {"url": session["url"]},
                    )
                    self._conn.commit()
                except Exception as error:
                    self._conn.rollback()
                    failures.append((session["url"], error))
                    self._conn.execute(
                        text(f"""
                            update {F1SITE_MANIFEST_TABLE}
                            set status = 'failed',
                                blob_name = null,
                                content_sha256 = null,
                                scraped_at = null
                            where url = :url
                            """),
                        {"url": session["url"]},
                    )
                    self._conn.commit()
                print_progress_bar(index, len(sessions))
        finally:
            browser.quit()

        if unavailable:
            print(
                f"{unavailable} session page(s) had no data and were marked unavailable."
            )
        if failures:
            print(
                f"\n{len(failures)} session page(s) failed and remain retryable "
                "in the manifest.\n"
            )
            for url, error in failures:
                logger.error("Failed to scrape %s: %s", url, error)

    def _scrape_session(self, session, browser):
        session_type = session["session_type"]
        year = session["year"]

        if session_type == "race-result" and year < 2018:
            min_col = max_col = 7
            col_idx_map = {
                "date": get_date,
                "driver_id": 2,
                "team_id": 3,
                "position": 0,
                "driver_name": 2,
                "points": 6,
            }
            page_lvl_cols = ["date"]
        elif session_type == "race-result":
            min_col = max_col = 7
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "circuit_id": lambda current_browser: current_browser.find_element(
                    By.ID, "content-dropdown"
                ).text,
                "team_id": 3,
                "year": lambda _: year,
                "race_url": lambda current_browser: current_browser.current_url,
                "circuit_name": lambda current_browser: current_browser.find_element(
                    By.ID, "content-dropdown"
                ).text,
                "driver_name": 2,
                "team_name": 3,
                "end_position": 0,
                "points": 6,
                "laps_completed": 4,
            }
            page_lvl_cols = [
                "race_id",
                "circuit_id",
                "year",
                "race_url",
                "circuit_name",
            ]
        elif session_type == "pit-stop-summary" and 2016 <= year <= 2017:
            min_col = max_col = 8
            col_idx_map = {
                "date": get_date,
                "driver_id": 2,
                "team_id": 3,
                "stop_number": 0,
                "stop_lap": 4,
                "pits_time": 6,
            }
            page_lvl_cols = ["date"]
        elif session_type == "pit-stop-summary" and year >= 2018:
            min_col = max_col = 8
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "team_id": 3,
                "stop_number": 0,
                "stop_lap": 4,
                "pits_time": 6,
            }
            page_lvl_cols = ["race_id"]
        elif session_type == "practice/0" and year == 2020:
            min_col = max_col = 6
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "team_id": 3,
                "session_type": lambda _: "practice/0",
                "lap_time": 4,
                "lap_count": 5,
                "position": 0,
            }
            page_lvl_cols = ["race_id", "session_type"]
        elif (
            session_type in ("practice/1", "practice/2", "practice/3")
            and year >= DATA_START_YEAR
        ):
            min_col = max_col = 6
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "team_id": 3,
                "session_type": lambda current_browser: "/".join(
                    current_browser.current_url.split("/")[9:11]
                ),
                "lap_time": 4,
                "lap_count": 5,
                "position": 0,
            }
            page_lvl_cols = ["race_id", "session_type"]
        elif session_type in ("qualifying", "sprint-qualifying") and year >= (
            SPRINT_START_YEAR
            if session_type == "sprint-qualifying"
            else DATA_START_YEAR
        ):
            min_col = max_col = 8
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "team_id": 3,
                "q1_time": 4,
                "q2_time": 5,
                "q3_time": 6,
                "qual_position": 0,
                "qual_laps": 7,
                "session_type": lambda _: session_type,
            }
            page_lvl_cols = ["race_id", "session_type"]
        elif session_type in ("starting-grid", "sprint-grid") and year >= (
            SPRINT_START_YEAR if session_type == "sprint-grid" else DATA_START_YEAR
        ):
            min_col = 4
            max_col = 5
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "team_id": 3,
                "start_position": 0,
                "session_type": lambda _: session_type,
            }
            page_lvl_cols = ["race_id", "session_type"]
        elif session_type == "sprint-results" and year >= SPRINT_START_YEAR:
            min_col = max_col = 7
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "team_id": 3,
                "session_type": lambda _: "sprint",
                "sprint_position": 0,
                "laps_completed": 4,
                "time": 5,
                "points": 6,
            }
            page_lvl_cols = ["race_id", "session_type"]
        elif session_type == "fastest-laps" and year >= DATA_START_YEAR:
            min_col = max_col = 8
            col_idx_map = {
                "race_id": lambda current_browser: (
                    current_browser.find_element(By.ID, "content-dropdown").text
                    + "_"
                    + current_browser.current_url.split("/")[5]
                ),
                "driver_id": 2,
                "team_id": 3,
                "fastest_lap_time": 6,
                "lap_number": 4,
            }
            page_lvl_cols = ["race_id"]
        else:
            raise ValueError(f"No parser configured for {session_type!r} in {year}")

        data = scrape_url_table(
            urls=[session["url"]],
            min_col=min_col,
            max_col=max_col,
            col_idx_map=col_idx_map,
            data_folder="",
            page_lvl_cols=page_lvl_cols,
            browser=browser,
            show_progress=False,
            raise_on_error=True,
        )
        if not isinstance(data, pd.DataFrame) or data.empty:
            if not self._page_has_result_rows(browser):
                if session_type != "race-result":
                    raise NoSessionDataError(
                        f"No result rows are listed for {session_type}"
                    )
                raise ValueError("The required race-result page has no result rows")
            raise ValueError("The page did not contain any rows matching its parser")
        driver_codes = self.extract_driver_codes(
            browser, min_col=min_col, max_col=max_col
        )
        if len(driver_codes) != len(data):
            raise ValueError(
                "Driver-code rows do not match parsed session rows "
                f"({len(driver_codes)} != {len(data)})"
            )
        data["driver_code"] = driver_codes
        return data

    @staticmethod
    def _page_has_result_rows(browser):
        for table in browser.find_elements(By.TAG_NAME, "table"):
            for row in table.find_elements(By.TAG_NAME, "tr")[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 3 and any(cell.text.strip() for cell in cells):
                    return True
        return False

    @staticmethod
    def _blob_name(session):
        return (
            f"{session['session_type']}/{session['year']}/"
            f"{session['round_number']:02d}.parquet"
        )

    @staticmethod
    def extract_driver_codes(browser, min_col, max_col):
        driver_codes = []
        for table in browser.find_elements(By.TAG_NAME, "table"):
            for row in table.find_elements(By.TAG_NAME, "tr")[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                if not min_col <= len(cells) <= max_col:
                    continue
                code = None
                for cell in cells:
                    for span in cell.find_elements(By.TAG_NAME, "span"):
                        classes = span.get_attribute("class") or ""
                        text_content = (span.get_attribute("textContent") or "").strip()
                        if "md:hidden" in classes.split() and text_content:
                            code = text_content
                            break
                    if code:
                        break
                driver_codes.append(code)
        return driver_codes

    def extract_max_rounds(self):
        years = range(DATA_START_YEAR, date.today().year + 1)
        browser = create_browser()
        max_rounds = []
        try:
            for index, year in enumerate(years, start=1):
                browser.get(f"https://www.formula1.com/en/racing/{year}")
                data = browser.find_elements(
                    By.XPATH, "//*[contains(text(), 'ROUND ')]"
                )
                valid_rounds = []

                for item in data:
                    number = "".join(filter(str.isdigit, item.text))
                    if not number:
                        continue
                    number = int(number)

                    try:
                        card = item.find_element(
                            By.XPATH, "./ancestor::div[contains(@class, 'flex-col')]"
                        )
                        if "CALLED OFF" in card.text.upper():
                            continue
                    except Exception:
                        print(f"Error finding {year} round {number} card")

                    valid_rounds.append(number)

                max_rounds.append({"year": year, "max_rounds": len(valid_rounds)})
                print_progress_bar(index, len(years))
        finally:
            browser.quit()

        upload_df_to_blob(
            pd.DataFrame(max_rounds),
            container_name=RAW_CONTAINER,
            blob_name=MAX_ROUNDS_BLOB,
        )


class ExtractFastF1:
    def __init__(self, conn):
        self._conn = conn

    def extract(self):
        FASTF1_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.configure(cache_dir=str(FASTF1_CACHE_PATH))
        logging.getLogger("fastf1").setLevel(logging.CRITICAL)
        sessions = self.discover_sessions()
        self.extract_pending_sessions()
        return sessions

    def discover_sessions(self):
        print("\nDiscovering FastF1 event schedules...")
        event_records = []
        session_records = []
        newest_year = date.today().year

        for year in range(FASTF1_START_YEAR, newest_year + 1):
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            events, sessions = self._schedule_records(schedule, year)
            event_records.extend(events)
            session_records.extend(sessions)
            print_progress_bar(
                year - FASTF1_START_YEAR + 1, newest_year - FASTF1_START_YEAR + 1
            )

        if event_records:
            self._conn.execute(
                text(f"""
                    insert into {FASTF1_EVENT_TABLE} (
                        season_year, round_number, event_name, official_event_name,
                        country, location, event_date, event_format, f1_api_support
                    )
                    values (
                        :season_year, :round_number, :event_name, :official_event_name,
                        :country, :location, :event_date, :event_format, :f1_api_support
                    )
                    on conflict (season_year, round_number) do update set
                        event_name = excluded.event_name,
                        official_event_name = excluded.official_event_name,
                        country = excluded.country,
                        location = excluded.location,
                        event_date = excluded.event_date,
                        event_format = excluded.event_format,
                        f1_api_support = excluded.f1_api_support,
                        updated_at = now()
                    """),
                event_records,
            )
        if session_records:
            self._conn.execute(
                text(f"""
                    insert into {FASTF1_SESSION_TABLE} (
                        season_year, round_number, session_number, session_name,
                        session_start_local, session_start_utc, session_timezone,
                        status
                    )
                    values (
                        :season_year, :round_number, :session_number, :session_name,
                        :session_start_local, :session_start_utc, :session_timezone,
                        'pending'
                    )
                    on conflict (season_year, round_number, session_number) do update set
                        session_name = excluded.session_name,
                        session_start_local = excluded.session_start_local,
                        session_start_utc = excluded.session_start_utc,
                        session_timezone = excluded.session_timezone
                    """),
                session_records,
            )
            self._conn.execute(text(f"""
                update {FASTF1_SESSION_TABLE} s
                set status = case
                    when e.f1_api_support then 'pending'
                    else 'unavailable'
                end
                from {FASTF1_EVENT_TABLE} e
                where e.season_year = s.season_year
                    and e.round_number = s.round_number
                    and s.status in ('pending', 'failed', 'unavailable')
                    and (
                        (e.f1_api_support and s.status = 'unavailable')
                        or (not e.f1_api_support and s.status in ('pending', 'failed'))
                    )
                """))
        self._conn.commit()
        print(
            f"\nDiscovered {len(event_records)} FastF1 events and {len(session_records)} sessions.\n"
        )
        return session_records

    @classmethod
    def _schedule_records(cls, schedule, year):
        event_records = []
        session_records = []
        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            if round_number <= 0:
                continue
            event_name = cls._string_or_none(event.get("EventName"))
            if not event_name:
                raise ValueError(
                    f"FastF1 schedule has no event name for {year} round {round_number}"
                )
            api_support = event.get("F1ApiSupport", False)
            event_records.append(
                {
                    "season_year": year,
                    "round_number": round_number,
                    "event_name": event_name,
                    "official_event_name": cls._string_or_none(
                        event.get("OfficialEventName")
                    ),
                    "country": cls._string_or_none(event.get("Country")),
                    "location": cls._string_or_none(event.get("Location")),
                    "event_date": cls._datetime_or_none(event.get("EventDate")),
                    "event_format": cls._string_or_none(event.get("EventFormat")),
                    "f1_api_support": (
                        bool(api_support) if not cls._is_missing(api_support) else False
                    ),
                }
            )
            for session_number in range(1, 6):
                session_name = cls._string_or_none(
                    event.get(f"Session{session_number}")
                )
                if not session_name:
                    continue
                local_start = event.get(f"Session{session_number}Date")
                session_records.append(
                    {
                        "season_year": year,
                        "round_number": round_number,
                        "session_number": session_number,
                        "session_name": session_name,
                        "session_start_local": cls._datetime_or_none(local_start),
                        "session_start_utc": cls._datetime_or_none(
                            event.get(f"Session{session_number}DateUtc")
                        ),
                        "session_timezone": cls._timezone_or_none(local_start),
                    }
                )
        return event_records, session_records

    @staticmethod
    def _is_missing(value):
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _string_or_none(cls, value):
        if cls._is_missing(value):
            return None
        value = str(value).strip()
        return value or None

    @classmethod
    def _datetime_or_none(cls, value):
        if cls._is_missing(value):
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.to_pydatetime()

    @classmethod
    def _timezone_or_none(cls, value):
        if cls._is_missing(value):
            return None
        timestamp = pd.Timestamp(value)
        return str(timestamp.tzinfo) if timestamp.tzinfo else None

    def extract_pending_sessions(self):
        sessions = self._conn.execute(text(f"""
                select s.season_year, s.round_number, s.session_number,
                    s.session_name, e.event_name, e.f1_api_support
                from {FASTF1_SESSION_TABLE} s
                join {FASTF1_EVENT_TABLE} e using (season_year, round_number)
                where s.status in ('pending', 'failed')
                    and e.f1_api_support
                    and (
                        coalesce(
                            s.session_start_utc,
                            s.session_start_local,
                            e.event_date
                        ) is null
                        or coalesce(
                            s.session_start_utc,
                            s.session_start_local,
                            e.event_date
                        ) <= now()
                    )
                order by s.season_year, s.round_number, s.session_number
                """)).mappings().all()
        if not sessions:
            print("No pending FastF1 sessions to extract.\n")
            return

        failures = []
        rate_limited = False
        for index, session_info in enumerate(sessions, start=1):
            try:
                self._extract_session(session_info)
            except RateLimitExceededError as error:
                self._conn.rollback()
                rate_limited = True
                logger.warning(
                    "FastF1 rate limit reached while extracting %s %s %s; "
                    "leaving this and remaining sessions pending.",
                    session_info["season_year"],
                    session_info["event_name"],
                    session_info["session_name"],
                )
                logger.debug("FastF1 rate-limit details: %s", error)
                print_progress_bar(index, len(sessions))
                break
            except Exception as error:
                self._conn.rollback()
                failures.append((session_info, error))
                self._conn.execute(
                    text(f"""
                        update {FASTF1_SESSION_TABLE}
                        set status = 'failed',
                            laps_blob_name = null,
                            weather_blob_name = null,
                            messages_blob_name = null,
                            scraped_at = null
                        where season_year = :season_year
                            and round_number = :round_number
                            and session_number = :session_number
                        """),
                    {
                        "season_year": session_info["season_year"],
                        "round_number": session_info["round_number"],
                        "session_number": session_info["session_number"],
                    },
                )
                self._conn.commit()
            print_progress_bar(index, len(sessions))

        if rate_limited:
            print(
                "\nFastF1 rate limit reached. Remaining sessions are still pending "
                "and can be resumed on a later run.\n"
            )
        if failures:
            print(f"{len(failures)} FastF1 session(s) failed; they will be retried.")
            for session_info, error in failures:
                logger.error(
                    "FastF1 extraction failed for %s %s %s: %s",
                    session_info["season_year"],
                    session_info["event_name"],
                    session_info["session_name"],
                    error,
                )

    def _extract_session(self, session_info):
        session = fastf1.get_session(
            session_info["season_year"],
            session_info["event_name"],
            session_info["session_name"],
        )
        session.load(laps=True, telemetry=False, weather=True, messages=True)

        race_id = self._conn.execute(
            text("""
                select race_id
                from formula_one.race
                where season_year = :season_year and round = :round_number
                """),
            {
                "season_year": session_info["season_year"],
                "round_number": session_info["round_number"],
            },
        ).scalar()
        frames = {
            data_type: self._get_loaded_frame(
                session, property_name, session_info, data_type
            )
            for data_type, property_name in (
                ("laps", "laps"),
                ("weather", "weather_data"),
                ("messages", "race_control_messages"),
            )
        }
        blob_names = {}
        for data_type, frame in frames.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                blob_names[f"{data_type}_blob_name"] = None
                continue
            data = frame.copy()
            data["race_id"] = race_id
            data["season_year"] = session_info["season_year"]
            data["round_number"] = session_info["round_number"]
            data["event_name"] = session_info["event_name"]
            data["session_name"] = session_info["session_name"]
            blob_name = self._blob_name(session_info, data_type)
            upload_df_to_blob(data, RAW_CONTAINER, blob_name)
            blob_names[f"{data_type}_blob_name"] = blob_name

        has_data = any(blob_names.values())
        self._conn.execute(
            text(f"""
                update {FASTF1_SESSION_TABLE}
                set status = :status,
                    laps_blob_name = :laps_blob_name,
                    weather_blob_name = :weather_blob_name,
                    messages_blob_name = :messages_blob_name,
                    scraped_at = now()
                where season_year = :season_year
                    and round_number = :round_number
                    and session_number = :session_number
                """),
            {
                **session_info,
                "season_year": session_info["season_year"],
                "round_number": session_info["round_number"],
                "session_number": session_info["session_number"],
                **blob_names,
                "status": "success" if has_data else "no_data",
            },
        )
        self._conn.commit()
        del session

    @staticmethod
    def _get_loaded_frame(session, property_name):
        try:
            return getattr(session, property_name)
        except DataNotLoadedError:
            return pd.DataFrame()

    @staticmethod
    def _blob_name(session_info, data_type):
        event_slug = re.sub(
            r"[^a-z0-9]+", "-", session_info["event_name"].lower()
        ).strip("-")
        session_slug = re.sub(
            r"[^a-z0-9]+", "-", session_info["session_name"].lower()
        ).strip("-")
        return (
            f"fastf1/{session_info['season_year']}/"
            f"{session_info['round_number']:02d}-{event_slug}/"
            f"{session_slug}_{data_type}.parquet"
        )


class ExtractWikipedia:
    def extract(self):
        print("\nScraping circuits from Wikipedia...")
        circuits = scrape_url_table(
            urls=[WIKIPEDIA_CIRCUITS_URL],
            min_col=11,
            max_col=11,
            col_idx_map={
                "name": 0,
                "type": 2,
                "direction": 3,
                "location": 4,
                "country": 5,
                "length": 6,
                "turns": 7,
                "gp": 8,
                "seasons": 9,
                "gps_held": 10,
            },
            data_folder="",
            raise_on_error=True,
        )
        if not isinstance(circuits, pd.DataFrame) or circuits.empty:
            raise ValueError("Wikipedia returned no circuit rows")

        upload_df_to_blob(circuits, RAW_CONTAINER, WIKIPEDIA_CIRCUITS_BLOB)
        print(f"Scraped {len(circuits)} circuit rows from Wikipedia.\n")
        return circuits


class ExtractPhoton:
    def extract(self, circuits):
        required_columns = {"name", "location", "country"}
        missing_columns = required_columns.difference(circuits.columns)
        if missing_columns:
            raise ValueError(
                "Circuit data is missing columns required for location lookup: "
                f"{sorted(missing_columns)}"
            )

        circuits = circuits.copy()
        circuits["cleaned_place"] = circuits["name"].apply(clean_circuit_name)
        circuits = circuits.drop_duplicates(subset=["cleaned_place"])
        existing_locations = self._load_existing_locations()
        known_places = set(
            existing_locations.get("cleaned_name", pd.Series(dtype=str)).dropna()
        )
        pending = circuits[~circuits["cleaned_place"].isin(known_places)]

        print(f"\nLooking up locations for {len(pending)} new circuits...")
        new_locations = []
        failures = {"no_coords": [], "no_elev": [], "error": []}
        for index, (_, circuit) in enumerate(pending.iterrows(), start=1):
            place = circuit["cleaned_place"]
            result = get_location_data(
                place,
                circuit["location"],
                circuit["country"],
            )
            if isinstance(result, dict):
                new_locations.append(
                    {
                        "original_name": circuit["name"],
                        "cleaned_name": place,
                        "city": circuit["location"],
                        "country": circuit["country"],
                        "latitude": result.get("latitude"),
                        "longitude": result.get("longitude"),
                        "elevation": result.get("elevation"),
                    }
                )
            elif result in failures:
                failures[result].append(place)
            else:
                raise ValueError(
                    f"Location lookup returned an unexpected result for {place!r}: "
                    f"{result!r}"
                )
            print_progress_bar(index, len(pending))

        locations = pd.concat(
            [existing_locations, pd.DataFrame(new_locations)],
            ignore_index=True,
        )
        if not locations.empty:
            locations = locations.drop_duplicates(subset=["cleaned_name"], keep="last")
        upload_df_to_blob(locations, RAW_CONTAINER, PHOTON_LOCATIONS_BLOB)
        for failure, places in failures.items():
            if places:
                logger.warning(
                    "%s location lookup(s) returned %s: %s",
                    len(places),
                    failure,
                    ", ".join(places),
                )
        print(
            f"Stored {len(new_locations)} new location rows "
            f"({len(locations)} total).\n"
        )
        return locations

    @staticmethod
    def _load_existing_locations():
        try:
            locations = download_df_from_blob(RAW_CONTAINER, PHOTON_LOCATIONS_BLOB)
        except ResourceNotFoundError:
            return pd.DataFrame(
                columns=[
                    "original_name",
                    "cleaned_name",
                    "city",
                    "country",
                    "latitude",
                    "longitude",
                    "elevation",
                ]
            )
        if "cleaned_name" not in locations.columns:
            raise ValueError(
                f"Existing {PHOTON_LOCATIONS_BLOB} is missing the cleaned_name column"
            )
        return locations
