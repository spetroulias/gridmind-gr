import os
from functools import lru_cache
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    from sqlalchemy.engine import URL
    return URL.create(
        "postgresql+psycopg",
        username=os.getenv("POSTGRES_USER", "gridmind"),
        password=os.getenv("POSTGRES_PASSWORD", "gridmind"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "gridmind"),
    )


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True, connect_args={"connect_timeout": 5})


def get_connection():
    """Return a psycopg connection; its context manager commits and closes."""
    import psycopg
    url = get_database_url()
    return psycopg.connect(user=url.username, password=url.password,
                           host=url.host, port=url.port, dbname=url.database,
                           connect_timeout=5)


def save_system_load(df: pd.DataFrame) -> None:
    if df.empty:
        return
    engine = get_engine()
    
    # Μετατροπή ημερομηνιών σε string/date format
    df_to_save = df.copy()
    if 'date' in df_to_save.columns:
        df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.date

    query = text("""
        INSERT INTO system_load (date, period, net_load_mwh, crete_flow_mwh)
        VALUES (:date, :period, :net_load_mwh, :crete_flow_mwh)
        ON CONFLICT (date, period)
        DO UPDATE SET
            net_load_mwh = EXCLUDED.net_load_mwh,
            crete_flow_mwh = EXCLUDED.crete_flow_mwh,
            updated_at = NOW();
    """)

    records = df_to_save.to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(query, records)

def save_res(df: pd.DataFrame) -> None:
    if df.empty:
        return
    engine = get_engine()
    df_to_save = df.copy()
    if 'date' in df_to_save.columns:
        df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.date

    query = text("""
        INSERT INTO res_production (date, period, res_mwh)
        VALUES (:date, :period, :res_mwh)
        ON CONFLICT (date, period)
        DO UPDATE SET
            res_mwh = EXCLUDED.res_mwh,
            updated_at = NOW();
    """)

    records = df_to_save.to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(query, records)

def save_generation(df: pd.DataFrame) -> None:
    if df.empty:
        return
    engine = get_engine()
    df_to_save = df.copy()
    if 'date' in df_to_save.columns:
        df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.date

    query = text("""
        INSERT INTO generation_actual (date, period, unit_name, technology, production_mwh)
        VALUES (:date, :period, :unit_name, :technology, :production_mwh)
        ON CONFLICT (date, period, unit_name)
        DO UPDATE SET
            technology = EXCLUDED.technology,
            production_mwh = EXCLUDED.production_mwh,
            updated_at = NOW();
    """)

    records = df_to_save.to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(query, records)
