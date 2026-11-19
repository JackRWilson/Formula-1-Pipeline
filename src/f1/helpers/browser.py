import logging
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


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
            cells = row.find_elements(By.TAG_NAME, "td")
            if not (min_col <= len(cells) <= max_col):
                continue

            cell_texts = [cell.text.strip() for cell in cells]
            matching_rows.append(cell_texts)
    return matching_rows


MAX_ATTEMPTS = 3  # Max attempts to scrape url


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

                except Exception as error:
                    if attempt == MAX_ATTEMPTS:
                        if raise_on_error:
                            raise
                        logger.warning(
                            "Failed to scrape %s after %d attempts (%s)",
                            url,
                            MAX_ATTEMPTS,
                            type(error).__name__,
                        )
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


def _wait_for_text(browser: WebDriver, xpath: str, timeout: int = 10) -> str:
    """
    Poll until the element exists and has non-empty text.
    Finding the element and reading .text happen in the same poll, so a
    re-render between the two just triggers another poll instead of a stale element.
    """

    def _read_text(driver: WebDriver):
        try:
            text = driver.find_element(By.XPATH, xpath).text.strip()
        except (NoSuchElementException, StaleElementReferenceException):
            return False
        return text or False

    return WebDriverWait(browser, timeout).until(_read_text)


DATE_XPATH = (
    "//p[contains(@class, 'display-s-bold')]"
    "[following-sibling::p[contains(@class, 'body-xs-semibold')]]"
)


def get_date(browser: WebDriver) -> str | None:
    """
    Extract the date range, e.g. '06 - 08 Mar 2026'
    """
    try:
        return _wait_for_text(browser, DATE_XPATH)
    except TimeoutException:
        print("Failed to extract date: element not found or empty")
        return None


TRACK_XPATH = f"{DATE_XPATH}/following-sibling::p[contains(@class, 'body-xs-semibold')]"


def get_track(browser: WebDriver) -> tuple[str | None, str | None]:
    """
    Extract (circuit, city), e.g. ('Albert Park Grand Prix Circuit', 'Melbourne')
    """
    try:
        text = _wait_for_text(browser, TRACK_XPATH)
    except TimeoutException:
        print("Failed to extract track: element not found or empty")
        return None, None

    circuit, separator, city = text.rpartition(", ")
    if not separator:
        return text, None
    return circuit, city
