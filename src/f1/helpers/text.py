import pandas as pd

from f1.helpers.utils import clean_circuit_name


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = " ".join(str(value).split())
    return value or None


def normalized_key(value: object) -> str | None:
    normalized = clean_circuit_name(normalize_text(value))
    return normalized.casefold() if isinstance(normalized, str) and normalized else None
