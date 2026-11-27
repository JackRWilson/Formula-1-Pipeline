import re

import pandas as pd

LAP_TIME_PATTERN = re.compile(
    r"^(?P<gap>\+)?(?:(?P<minutes>\d+):)?(?P<seconds>\d+)\.(?P<fraction>\d+)s?$"
)


def parse_lap_strings(raw: pd.Series) -> pd.DataFrame:
    parts = raw.astype("string").str.strip().str.extract(LAP_TIME_PATTERN)
    minutes = pd.to_numeric(parts["minutes"]).astype("float64")
    seconds = pd.to_numeric(parts["seconds"]).astype("float64")
    fraction = pd.to_numeric("0." + parts["fraction"]).astype("float64")
    return pd.DataFrame(
        {
            "is_gap": parts["gap"].notna(),
            "seconds": minutes.fillna(0) * 60 + seconds + fraction,
        },
        index=raw.index,
    )
