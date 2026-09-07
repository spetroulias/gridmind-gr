import os
from datetime import timedelta

import pandas as pd
from sqlalchemy import create_engine, text


# ============================================================
# PostgreSQL connection
# ============================================================

def get_engine():
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "gridmind")
    db_user = os.getenv("DB_USER", "gridmind")
    db_password = os.getenv("DB_PASSWORD", "gridmind")

    database_url = (
        f"postgresql+psycopg2://"
        f"{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    return create_engine(database_url)


# ============================================================
# Load historical LOAD data from PostgreSQL
# ============================================================
def load_historical_data():
    query = text("""
        SELECT
            date,
            period,
            net_load_mwh
        FROM system_load
        WHERE period BETWEEN 1 AND 24
        ORDER BY date, period
    """)

    engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    df["timestamp"] = (
        pd.to_datetime(df["date"])
        + pd.to_timedelta(df["period"] - 1, unit="h")
    )

    df = df.rename(
        columns={
            "net_load_mwh": "load_mwh"
        }
    )

    return df[
        [
            "timestamp",
            "load_mwh",
        ]
    ]

# ============================================================
# Historical LOAD query
# ============================================================

def get_historical_load(
    start_date,
    end_date,
    hour=None,
):
    df = load_historical_data()

    start = pd.to_datetime(start_date)
    end_exclusive = pd.to_datetime(end_date) + timedelta(days=1)

    result = df[
        (df["timestamp"] >= start)
        & (df["timestamp"] < end_exclusive)
    ].copy()

    if hour is not None:
        if not 0 <= hour <= 23:
            raise ValueError("hour must be between 0 and 23")

        result = result[
            result["timestamp"].dt.hour == hour
        ]

    return result.reset_index(drop=True)


# ============================================================
# Historical LOAD summary
# ============================================================

def get_load_summary(
    start_date,
    end_date,
    hour=None,
):
    result = get_historical_load(
        start_date=start_date,
        end_date=end_date,
        hour=hour,
    )

    if result.empty:
        return {
            "count": 0,
            "mean_mwh": None,
            "min_mwh": None,
            "max_mwh": None,
        }

    return {
        "count": len(result),
        "mean_mwh": float(result["load_mwh"].mean()),
        "min_mwh": float(result["load_mwh"].min()),
        "max_mwh": float(result["load_mwh"].max()),
    }


# ============================================================
# Historical LOAD peak
# ============================================================

def get_peak_load(
    start_date,
    end_date,
):
    result = get_historical_load(
        start_date=start_date,
        end_date=end_date,
    )

    if result.empty:
        return None

    peak_row = result.loc[
        result["load_mwh"].idxmax()
    ]

    return {
        "timestamp": peak_row["timestamp"],
        "load_mwh": float(peak_row["load_mwh"]),
    }