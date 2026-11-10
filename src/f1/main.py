import sys
from pathlib import Path

from dotenv import load_dotenv
from dynaconf import Dynaconf
from sqlalchemy import create_engine

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f1.etl import ETLProcessor


def run():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    def build_engine():
        settings = Dynaconf(envvar_prefix="DB", load_dotenv=True, dotenv_override=True)
        return create_engine(settings.ENGINE_URL, echo=False)

    engine = build_engine()
    try:
        with engine.connect() as conn:
            processor = ETLProcessor(conn)
            processor.process()
    finally:
        engine.dispose()


if __name__ == "__main__":
    run()
