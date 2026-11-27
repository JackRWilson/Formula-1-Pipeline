import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from f1.extract import (
    ExtractF1Site,
    ExtractFastF1,
    ExtractJolpiDrivers,
    ExtractPhoton,
    ExtractWikipedia,
)
from f1.helpers.blob_io import download_df_from_blob
from f1.helpers.constants import (
    F1SITE_MANIFEST_TABLE,
    FASTF1_SESSION_TABLE,
    MAX_ROUNDS_BLOB,
    PHOTON_LOCATIONS_BLOB,
    RAW_CONTAINER,
    WIKIPEDIA_CIRCUITS_BLOB,
)
from f1.transform import (
    TransformCircuit,
    TransformConstructor,
    TransformDriver,
    TransformDriverSessionTirePerformance,
    TransformLocation,
    TransformPitStops,
    TransformPracticeResult,
    TransformQualResult,
    TransformRace,
    TransformRaceControlSummary,
    TransformRaceResult,
    TransformSeason,
    TransformWeatherSummary,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


class ETLProcessor:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def process(self) -> None:
        print("ETL started")
        # ExtractF1Site(self._conn).extract()
        # circuits = ExtractWikipedia().extract()
        # ExtractPhoton().extract(circuits)
        # ExtractFastF1(self._conn).extract()
        # ExtractJolpiDrivers().extract()
        # self.print_blob_heads()

        season_df = TransformSeason(self._conn).transform()
        circuit_df = TransformCircuit().transform()
        location_df = TransformLocation().transform(circuit_df)
        constructor_df = TransformConstructor().transform()
        driver_df = TransformDriver().transform()
        race_df = TransformRace().transform(circuit_df)
        race_result_df = TransformRaceResult().transform()
        practice_result_df = TransformPracticeResult().transform()
        qual_result_df = TransformQualResult().transform()
        pit_stop_df = TransformPitStops().transform()
        weather_sum_df = TransformWeatherSummary().transform()
        race_control_df = TransformRaceControlSummary().transform()
        tire_perf_df = TransformDriverSessionTirePerformance().transform()

        print("ETL complete")

    def print_blob_heads(self) -> None:
        print("\nSample rows from each successful F1 session blob:")
        sessions = (
            self._conn.execute(
                text(f"""
                select distinct on (session_type)
                    session_type, blob_name
                from {F1SITE_MANIFEST_TABLE}
                where status = 'success' and blob_name is not null
                order by session_type, year desc, round_number desc
                """)
            )
            .mappings()
            .all()
        )
        if not sessions:
            print("No successful session blobs are available yet.")
        for session in sessions:
            self._print_blob_head(session["session_type"], session["blob_name"])

        self._print_blob_head("max_rounds", MAX_ROUNDS_BLOB)
        self._print_blob_head("circuits", WIKIPEDIA_CIRCUITS_BLOB)
        self._print_blob_head("locations", PHOTON_LOCATIONS_BLOB)
        self._print_fastf1_blob_heads()

    def _print_fastf1_blob_heads(self) -> None:
        sessions = (
            self._conn.execute(
                text(f"""
                select distinct on (s.session_name)
                    s.session_name, s.laps_blob_name, s.weather_blob_name,
                    s.messages_blob_name
                from {FASTF1_SESSION_TABLE} s
                where s.status = 'success'
                order by s.session_name, s.season_year desc,
                    s.round_number desc, s.session_number desc
                """)
            )
            .mappings()
            .all()
        )
        for session in sessions:
            for data_type in ("laps", "weather", "messages"):
                blob_name = session[f"{data_type}_blob_name"]
                if blob_name:
                    self._print_blob_head(
                        f"FastF1 {session['session_name']} {data_type}",
                        blob_name,
                    )

    @staticmethod
    def _print_blob_head(label: str, blob_name: str) -> None:
        try:
            data = download_df_from_blob(RAW_CONTAINER, blob_name)
        except Exception as exc:
            print(f"\n{label}: {blob_name} could not be read ({exc})")
            return
        print(f"\n{label}: {blob_name}")
        print(data.head())
