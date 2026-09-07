from datetime import timedelta

import pandas as pd

from gridmind.data.database import get_connection


VALID_TECHNOLOGIES = {
    "lignite",
    "petroleum",
    "natural_gas",
    "hydro",
    "res",
}


def _validate_hour(hour) -> None:
    if hour is not None and not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")


def _validate_technology(technology) -> None:
    if technology is not None and technology not in VALID_TECHNOLOGIES:
        raise ValueError(
            f"technology must be one of: {sorted(VALID_TECHNOLOGIES)}"
        )


def _build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = (
        pd.to_datetime(df["date"])
        + pd.to_timedelta(df["period"] - 1, unit="h")
    )
    return df


def _date_range_mask(df: pd.DataFrame, start_date, end_date):
    start = pd.to_datetime(start_date)
    end_exclusive = pd.to_datetime(end_date) + timedelta(days=1)

    return (
        (df["timestamp"] >= start)
        & (df["timestamp"] < end_exclusive)
    )


# Load


def load_historical_data() -> pd.DataFrame:
    """Load the standard 24-hour system load time series."""
    query = """
        SELECT
            date,
            period,
            net_load_mwh
        FROM system_load
        WHERE period BETWEEN 1 AND 24
        ORDER BY date, period;
    """

    with get_connection() as connection:
        df = pd.read_sql_query(query, connection)

    df = _build_timestamp(df)
    df = df.rename(columns={"net_load_mwh": "load_mwh"})

    return df[["timestamp", "load_mwh"]].reset_index(drop=True)


def get_historical_load(
    start_date,
    end_date,
    hour=None,
) -> pd.DataFrame:
    """Return actual system load for a date range."""
    _validate_hour(hour)

    df = load_historical_data()
    result = df[_date_range_mask(df, start_date, end_date)].copy()

    if hour is not None:
        result = result[result["timestamp"].dt.hour == hour]

    return result.reset_index(drop=True)


def get_load_summary(
    start_date,
    end_date,
    hour=None,
) -> dict:
    result = get_historical_load(start_date, end_date, hour)

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


def get_peak_load(start_date, end_date):
    result = get_historical_load(start_date, end_date)

    if result.empty:
        return None

    peak = result.loc[result["load_mwh"].idxmax()]

    return {
        "timestamp": peak["timestamp"],
        "load_mwh": float(peak["load_mwh"]),
    }


# RES


def load_historical_res_data() -> pd.DataFrame:
    """Load the standard 24-hour RES production time series."""
    query = """
        SELECT
            date,
            period,
            res_mwh
        FROM res_production
        WHERE period BETWEEN 1 AND 24
        ORDER BY date, period;
    """

    with get_connection() as connection:
        df = pd.read_sql_query(query, connection)

    df = _build_timestamp(df)

    return df[["timestamp", "res_mwh"]].reset_index(drop=True)


def get_historical_res(
    start_date,
    end_date,
    hour=None,
) -> pd.DataFrame:
    """Return actual RES production for a date range."""
    _validate_hour(hour)

    df = load_historical_res_data()
    result = df[_date_range_mask(df, start_date, end_date)].copy()

    if hour is not None:
        result = result[result["timestamp"].dt.hour == hour]

    return result.reset_index(drop=True)


def get_res_summary(
    start_date,
    end_date,
    hour=None,
) -> dict:
    result = get_historical_res(start_date, end_date, hour)

    if result.empty:
        return {
            "count": 0,
            "mean_mwh": None,
            "min_mwh": None,
            "max_mwh": None,
        }

    return {
        "count": len(result),
        "mean_mwh": float(result["res_mwh"].mean()),
        "min_mwh": float(result["res_mwh"].min()),
        "max_mwh": float(result["res_mwh"].max()),
    }


def get_peak_res(start_date, end_date):
    result = get_historical_res(start_date, end_date)

    if result.empty:
        return None

    peak = result.loc[result["res_mwh"].idxmax()]

    return {
        "timestamp": peak["timestamp"],
        "res_mwh": float(peak["res_mwh"]),
    }


# Generation


def load_generation_data() -> pd.DataFrame:
    """Load unit-level generation actuals for periods 1-24."""
    query = """
        SELECT
            date,
            period,
            unit_name,
            technology,
            production_mwh
        FROM generation_actual
        WHERE period BETWEEN 1 AND 24
        ORDER BY date, period, technology, unit_name;
    """

    with get_connection() as connection:
        df = pd.read_sql_query(query, connection)

    df = _build_timestamp(df)

    return df[
        ["timestamp", "unit_name", "technology", "production_mwh"]
    ].reset_index(drop=True)


def get_historical_generation(
    start_date,
    end_date,
    technology=None,
    hour=None,
) -> pd.DataFrame:
    """Return actual unit-level generation for a date range."""
    _validate_hour(hour)
    _validate_technology(technology)

    df = load_generation_data()
    result = df[_date_range_mask(df, start_date, end_date)].copy()

    if technology is not None:
        result = result[result["technology"] == technology]

    if hour is not None:
        result = result[result["timestamp"].dt.hour == hour]

    return result.reset_index(drop=True)


def get_generation_by_technology(
    start_date,
    end_date,
) -> pd.DataFrame:
    result = get_historical_generation(start_date, end_date)

    if result.empty:
        return pd.DataFrame(
            columns=["technology", "production_mwh"]
        )

    return (
        result.groupby("technology", as_index=False)["production_mwh"]
        .sum()
        .sort_values("production_mwh", ascending=False)
        .reset_index(drop=True)
    )


def get_generation_mix(
    start_date,
    end_date,
) -> pd.DataFrame:
    generation = get_generation_by_technology(start_date, end_date)

    if generation.empty:
        return pd.DataFrame(
            columns=["technology", "production_mwh", "share_pct"]
        )

    total = generation["production_mwh"].sum()

    if total == 0:
        generation["share_pct"] = 0.0
    else:
        generation["share_pct"] = (
            generation["production_mwh"] / total * 100
        ).round(2)

    return generation


def get_hourly_generation_mix(
    start_date,
    end_date,
) -> pd.DataFrame:
    result = get_historical_generation(start_date, end_date)

    if result.empty:
        return pd.DataFrame(
            columns=["timestamp", "technology", "production_mwh"]
        )

    return (
        result.groupby(
            ["timestamp", "technology"],
            as_index=False,
        )["production_mwh"]
        .sum()
        .sort_values(["timestamp", "technology"])
        .reset_index(drop=True)
    )


def get_daily_generation_mix(
    start_date,
    end_date,
) -> pd.DataFrame:
    result = get_historical_generation(start_date, end_date)

    if result.empty:
        return pd.DataFrame(
            columns=["date", "technology", "production_mwh"]
        )

    result["date"] = result["timestamp"].dt.date

    return (
        result.groupby(
            ["date", "technology"],
            as_index=False,
        )["production_mwh"]
        .sum()
        .sort_values(["date", "technology"])
        .reset_index(drop=True)
    )


def get_top_generating_units(
    start_date,
    end_date,
    technology=None,
    limit=10,
) -> pd.DataFrame:
    _validate_technology(technology)

    if limit < 1:
        raise ValueError("limit must be at least 1")

    result = get_historical_generation(
        start_date,
        end_date,
        technology=technology,
    )

    if result.empty:
        return pd.DataFrame(
            columns=["unit_name", "technology", "production_mwh"]
        )

    return (
        result.groupby(
            ["unit_name", "technology"],
            as_index=False,
        )["production_mwh"]
        .sum()
        .sort_values("production_mwh", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def get_generation_summary(
    start_date,
    end_date,
    technology=None,
) -> dict:
    _validate_technology(technology)

    result = get_historical_generation(
        start_date,
        end_date,
        technology=technology,
    )

    if result.empty:
        return {
            "count": 0,
            "total_mwh": None,
            "mean_mwh": None,
            "min_mwh": None,
            "max_mwh": None,
        }

    values = result["production_mwh"]

    return {
        "count": len(result),
        "total_mwh": float(values.sum()),
        "mean_mwh": float(values.mean()),
        "min_mwh": float(values.min()),
        "max_mwh": float(values.max()),
    }


def get_peak_generation(
    start_date,
    end_date,
    technology=None,
):
    _validate_technology(technology)

    result = get_historical_generation(
        start_date,
        end_date,
        technology=technology,
    )

    if result.empty:
        return None

    peak = result.loc[result["production_mwh"].idxmax()]

    return {
        "timestamp": peak["timestamp"],
        "unit_name": peak["unit_name"],
        "technology": peak["technology"],
        "production_mwh": float(peak["production_mwh"]),
    }
