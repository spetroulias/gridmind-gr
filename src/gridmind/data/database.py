import os

import pandas as pd
import psycopg
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def save_system_load(df: pd.DataFrame) -> None:
    query = """
        INSERT INTO system_load (
            date,
            period,
            net_load_mwh,
            crete_flow_mwh
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (date, period)
        DO UPDATE SET
            net_load_mwh = EXCLUDED.net_load_mwh,
            crete_flow_mwh = EXCLUDED.crete_flow_mwh,
            updated_at = NOW();
    """

    rows = [
        (
            row.date.date(),
            int(row.period),
            float(row.net_load_mwh),
            float(row.crete_flow_mwh),
        )
        for row in df.itertuples(index=False)
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)
        connection.commit()


def save_res(df: pd.DataFrame) -> None:
    query = """
        INSERT INTO res_production (
            date,
            period,
            res_mwh
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (date, period)
        DO UPDATE SET
            res_mwh = EXCLUDED.res_mwh,
            updated_at = NOW();
    """

    rows = [
        (
            row.date.date(),
            int(row.period),
            float(row.res_mwh),
        )
        for row in df.itertuples(index=False)
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)
        connection.commit()


def save_generation(df: pd.DataFrame) -> None:
    query = """
        INSERT INTO generation_actual (
            date,
            period,
            unit_name,
            technology,
            production_mwh
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (date, period, unit_name)
        DO UPDATE SET
            technology = EXCLUDED.technology,
            production_mwh = EXCLUDED.production_mwh,
            updated_at = NOW();
    """

    rows = [
        (
            row.date.date(),
            int(row.period),
            str(row.unit_name),
            str(row.technology),
            float(row.production_mwh),
        )
        for row in df.itertuples(index=False)
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)
        connection.commit()
