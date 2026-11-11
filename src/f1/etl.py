from sqlalchemy import text

if __package__:
    from .extract import (
        F1SITE_MANIFEST_TABLE,
        FASTF1_SESSION_TABLE,
        RAW_CONTAINER,
        ExtractF1Site,
        ExtractFastF1,
        ExtractPhoton,
        ExtractWikipedia,
    )
    from .transform import TransformCircuit, TransformLocation, TransformSeason
    from .utils import download_df_from_blob
else:
    from extract import (
        F1SITE_MANIFEST_TABLE,
        FASTF1_SESSION_TABLE,
        RAW_CONTAINER,
        ExtractF1Site,
        ExtractFastF1,
        ExtractPhoton,
        ExtractWikipedia,
    )
    from transform import TransformCircuit, TransformLocation, TransformSeason
    from utils import download_df_from_blob

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


class ETLProcessor:
    def __init__(self, conn):
        self._conn = conn

    def process(self):
        print("ETL started")
        # ExtractF1Site(self._conn).extract()
        # circuits = ExtractWikipedia().extract()
        # ExtractPhoton().extract(circuits)
        # ExtractFastF1(self._conn).extract()
        # self.print_blob_heads()

        # season_df = TransformSeason(self._conn).transform()
        circuit_df = TransformCircuit().transform()
        location_df = TransformLocation().transform(circuit_df)
        print("\nTransformed circuits:")
        print(circuit_df.head())
        print("\nTransformed locations:")
        print(location_df.head())

        print("ETL complete")

    def print_blob_heads(self):
        print("\nSample rows from each successful F1 session blob:")
        sessions = self._conn.execute(text(f"""
                select distinct on (session_type)
                    session_type, blob_name
                from {F1SITE_MANIFEST_TABLE}
                where status = 'success' and blob_name is not null
                order by session_type, year desc, round_number desc
                """)).mappings().all()
        if not sessions:
            print("No successful session blobs are available yet.")
        for session in sessions:
            self._print_blob_head(session["session_type"], session["blob_name"])

        self._print_blob_head("max_rounds", "max_rounds.parquet")
        self._print_blob_head("circuits", "circuits_raw.parquet")
        self._print_blob_head("locations", "locations_raw.parquet")
        self._print_fastf1_blob_heads()

    def _print_fastf1_blob_heads(self):
        sessions = self._conn.execute(text(f"""
                select distinct on (s.session_name)
                    s.session_name, s.laps_blob_name, s.weather_blob_name,
                    s.messages_blob_name
                from {FASTF1_SESSION_TABLE} s
                where s.status = 'success'
                order by s.session_name, s.season_year desc,
                    s.round_number desc, s.session_number desc
                """)).mappings().all()
        for session in sessions:
            for data_type in ("laps", "weather", "messages"):
                blob_name = session[f"{data_type}_blob_name"]
                if blob_name:
                    self._print_blob_head(
                        f"FastF1 {session['session_name']} {data_type}",
                        blob_name,
                    )

    @staticmethod
    def _print_blob_head(label, blob_name):
        try:
            data = download_df_from_blob(RAW_CONTAINER, blob_name)
        except Exception as exc:
            print(f"\n{label}: {blob_name} could not be read ({exc})")
            return
        print(f"\n{label}: {blob_name}")
        print(data.head())
