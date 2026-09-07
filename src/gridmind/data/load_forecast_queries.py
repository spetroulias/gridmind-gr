import pandas as pd

from gridmind.data.database import get_connection
from gridmind.data.queries import get_historical_load


def load_forecast_data() -> pd.DataFrame:
    """Load all saved load forecasts from PostgreSQL."""
    query = """
        SELECT
            timestamp,
            forecast_mwh
        FROM load_forecasts
        ORDER BY timestamp;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]

    if not rows:
        return pd.DataFrame(columns=["timestamp", "forecast_mwh"])

    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def get_forecast(
    start_date,
    end_date,
    hour=None,
) -> pd.DataFrame:
    """Return saved load forecasts for a date range."""
    if hour is not None and not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")

    df = load_forecast_data()

    if df.empty:
        return df

    start = pd.to_datetime(start_date)
    end_exclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    result = df[
        (df["timestamp"] >= start)
        & (df["timestamp"] < end_exclusive)
    ].copy()

    if hour is not None:
        result = result[result["timestamp"].dt.hour == hour]

    return result.reset_index(drop=True)


def get_forecast_summary(
    start_date,
    end_date,
    hour=None,
) -> dict:
    result = get_forecast(start_date, end_date, hour)

    if result.empty:
        return {
            "count": 0,
            "mean_mwh": None,
            "min_mwh": None,
            "max_mwh": None,
        }

    values = result["forecast_mwh"]

    return {
        "count": len(result),
        "mean_mwh": float(values.mean()),
        "min_mwh": float(values.min()),
        "max_mwh": float(values.max()),
    }


def get_peak_forecast(start_date, end_date):
    result = get_forecast(start_date, end_date)

    if result.empty:
        return None

    peak = result.loc[result["forecast_mwh"].idxmax()]

    return {
        "timestamp": peak["timestamp"],
        "forecast_mwh": float(peak["forecast_mwh"]),
    }


def get_forecast_comparison(
    start_date,
    end_date,
    hour=None,
) -> pd.DataFrame:
    """Compare saved load forecasts with actual system load."""
    actual = get_historical_load(start_date, end_date, hour)
    forecast = get_forecast(start_date, end_date, hour)

    if actual.empty or forecast.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "actual_mwh",
                "forecast_mwh",
                "error_mwh",
                "abs_error_mwh",
            ]
        )

    actual = actual.rename(columns={"load_mwh": "actual_mwh"})

    comparison = actual.merge(
        forecast,
        on="timestamp",
        how="inner",
    )

    comparison["error_mwh"] = (
        comparison["forecast_mwh"] - comparison["actual_mwh"]
    )
    comparison["abs_error_mwh"] = comparison["error_mwh"].abs()

    return comparison.reset_index(drop=True)


def get_forecast_metrics(
    start_date,
    end_date,
    hour=None,
) -> dict:
    comparison = get_forecast_comparison(
        start_date,
        end_date,
        hour,
    )

    if comparison.empty:
        return {
            "count": 0,
            "mae_mwh": None,
            "rmse_mwh": None,
            "wape": None,
        }

    mae = comparison["abs_error_mwh"].mean()
    rmse = (comparison["error_mwh"].pow(2).mean()) ** 0.5

    actual_sum = comparison["actual_mwh"].abs().sum()
    wape = (
        None
        if actual_sum == 0
        else comparison["abs_error_mwh"].sum() / actual_sum * 100
    )

    return {
        "count": len(comparison),
        "mae_mwh": float(mae),
        "rmse_mwh": float(rmse),
        "wape": float(wape) if wape is not None else None,
    }
