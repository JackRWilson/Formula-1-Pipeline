import logging
from datetime import date, datetime

import pandas as pd
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from sqlalchemy import Connection, text

if __package__:
    from .project_functions import get_date
    from .utils import (
        create_browser,
        download_df_from_blob,
        print_progress_bar,
        scrape_url_table,
        upload_df_to_blob,
    )
else:
    from project_functions import get_date
    from utils import (
        create_browser,
        download_df_from_blob,
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
MANIFEST_TABLE = "formula_one.scrape_manifest"
RAW_CONTAINER = "f1-raw"
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
                    insert into {MANIFEST_TABLE}
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
                from {MANIFEST_TABLE}
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
                            update {MANIFEST_TABLE}
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
                            update {MANIFEST_TABLE}
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
                            update {MANIFEST_TABLE}
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
                rounds = [
                    int(number)
                    for item in data
                    if (number := "".join(filter(str.isdigit, item.text)))
                ]
                max_rounds.append({"year": year, "max_rounds": max(rounds, default=0)})
                print_progress_bar(index, len(years))
        finally:
            browser.quit()

        upload_df_to_blob(
            pd.DataFrame(max_rounds),
            container_name=RAW_CONTAINER,
            blob_name="max_rounds.parquet",
        )


class ETLProcessor:
    def __init__(self, conn: Connection):
        self._conn = conn

    def process(self):
        print("ETL started")
        extractor = ExtractF1Site(self._conn)
        extractor.extract()
        self.print_blob_heads()
        print("ETL complete")

    def print_blob_heads(self):
        print("\nSample rows from each successful F1 session blob:")
        sessions = self._conn.execute(text(f"""
                select distinct on (session_type)
                    session_type, blob_name
                from {MANIFEST_TABLE}
                where status = 'success' and blob_name is not null
                order by session_type, year desc, round_number desc
                """)).mappings().all()
        if not sessions:
            print("No successful session blobs are available yet.")
        for session in sessions:
            self._print_blob_head(session["session_type"], session["blob_name"])

        self._print_blob_head("max_rounds", "max_rounds.parquet")

    @staticmethod
    def _print_blob_head(label: str, blob_name: str):
        try:
            data = download_df_from_blob(RAW_CONTAINER, blob_name)
        except Exception as exc:
            print(f"\n{label}: {blob_name} could not be read ({exc})")
            return
        print(f"\n{label}: {blob_name}")
        print(data.head())
