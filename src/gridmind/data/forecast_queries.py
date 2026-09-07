import pandas as pd
from sqlalchemy import text

from gridmind.data.queries import get_engine


# --------------------------------------------------
# Load forecast data from PostgreSQL
# --------------------------------------------------

def load_forecast_data():
    query = text("""
        SELECT
            timestamp,
            forecast_mwh
        FROM load_forecasts
        ORDER BY timestamp
    """)

    engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError(
            "No forecast data found in PostgreSQL. "
            "Run the forecasting pipeline first."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# --------------------------------------------------
# Forecast query
# --------------------------------------------------

def get_forecast(
    start_date,
    end_date,
    hour=None
):
    """
    Return forecast load between start_date and end_date.

    Optional:
    filter by hour of day.
    """

    df = load_forecast_data()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Include the entire end date
    end_exclusive = end_date + pd.Timedelta(days=1)

    mask = (
        (df["timestamp"] >= start_date)
        & (df["timestamp"] < end_exclusive)
    )

    result = df.loc[
        mask,
        ["timestamp", "forecast_mwh"]
    ].copy()

    if hour is not None:
        if not 0 <= hour <= 23:
            raise ValueError(
                "hour must be between 0 and 23"
            )

        result = result[
            result["timestamp"].dt.hour == hour
        ]

    return result.reset_index(drop=True)


# --------------------------------------------------
# Forecast summary
# --------------------------------------------------

def get_forecast_summary(
    start_date,
    end_date,
    hour=None
):
    """
    Return summary statistics for forecast load.
    """

    result = get_forecast(
        start_date=start_date,
        end_date=end_date,
        hour=hour
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
        "mean_mwh": float(result["forecast_mwh"].mean()),
        "min_mwh": float(result["forecast_mwh"].min()),
        "max_mwh": float(result["forecast_mwh"].max()),
    }


# --------------------------------------------------
# Peak forecast
# --------------------------------------------------

def get_peak_forecast(
    start_date,
    end_date
):
    """
    Return timestamp and value of maximum forecast load.
    """

    result = get_forecast(
        start_date=start_date,
        end_date=end_date
    )

    if result.empty:
        return None

    peak_row = result.loc[
        result["forecast_mwh"].idxmax()
    ]

    return {
        "timestamp": peak_row["timestamp"],
        "forecast_mwh": float(
            peak_row["forecast_mwh"]
        ),
    }