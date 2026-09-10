import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    user = os.getenv("POSTGRES_USER", "gridmind")
    password = os.getenv("POSTGRES_PASSWORD", "gridmind")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "gridmind")
    
    db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(db_url)

def save_system_load(df: pd.DataFrame) -> None:
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