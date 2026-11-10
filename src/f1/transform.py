import pandas as pd
from extract import (
    DATA_START_YEAR,
    F1SITE_MANIFEST_TABLE,
    MAX_ROUNDS_BLOB,
    RAW_CONTAINER,
)
from sqlalchemy import text
from utils import download_df_from_blob


class TransformSeason:
    def __init__(self, conn):
        self._conn = conn

    def transform(self):
        try:
            df = download_df_from_blob(RAW_CONTAINER, MAX_ROUNDS_BLOB)
        except Exception as exc:
            print(f"\n{MAX_ROUNDS_BLOB} could not be read ({exc})")
        df = self._get_complete_bool(df)
        return df

    def _get_complete_bool(self, df):
        rounds = self._conn.execute(text(f"""
                select distinct
                    year,
                    max(round_number) as completed_rounds
                from {F1SITE_MANIFEST_TABLE}
                where status = 'success'
                    and blob_name is not null
                    and year >= {DATA_START_YEAR}
                group by year
                """)).mappings().all()
        if not rounds:
            print("No successful session blobs are available yet.")
        rounds = pd.DataFrame(rounds)
        season_df = df.merge(rounds, on="year", how="left")
        season_df["is_complete"] = season_df["completed_rounds"].ge(
            season_df["max_rounds"]
        )
        return season_df[["year", "is_complete"]]
