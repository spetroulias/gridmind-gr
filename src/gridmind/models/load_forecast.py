import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gridmind.data.database import get_connection
from gridmind.data.queries import load_historical_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_load_forecast.joblib"

MIN_VALID_LOAD_MWH = 500

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "load_lag_24",
    "load_lag_48",
    "load_lag_168",
]


def load_history() -> pd.DataFrame:
    """Load and clean the hourly system load history used by the model."""
    print("Loading historical load data from PostgreSQL...")

    df = load_historical_data()

    if df.empty:
        raise ValueError("No historical load data found in PostgreSQL.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in load history.")

    df.loc[df["load_mwh"] < MIN_VALID_LOAD_MWH, "load_mwh"] = np.nan

    print(f"Historical rows: {len(df)}")
    print(
        f"Historical period: {df['timestamp'].min()} "
        f"-> {df['timestamp'].max()}"
    )

    return df


def load_model_data() -> pd.DataFrame:
    """Create the model-ready load dataset used by backtesting."""
    df = load_history().copy()

    df["target"] = df["load_mwh"]
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["load_lag_24"] = df["load_mwh"].shift(24)
    df["load_lag_48"] = df["load_mwh"].shift(48)
    df["load_lag_168"] = df["load_mwh"].shift(168)

    model_data = (
        df[["timestamp", "target", *FEATURE_COLS]]
        .dropna()
        .reset_index(drop=True)
    )

    if model_data.empty:
        raise ValueError("No model-ready load rows available.")

    print(f"Model-ready rows: {len(model_data)}")
    print(
        f"Model period: {model_data['timestamp'].min()} "
        f"-> {model_data['timestamp'].max()}"
    )

    return model_data


def load_saved_model():
    """Load the production model and verify its feature contract."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        return artifact

    model = artifact["model"]
    saved_features = artifact.get("features", FEATURE_COLS)

    if list(saved_features) != FEATURE_COLS:
        raise ValueError(
            "Saved model feature list does not match production FEATURE_COLS."
        )

    return model


def build_features(timestamp, load_history: dict) -> pd.DataFrame:
    timestamp = pd.Timestamp(timestamp)
    day_of_week = timestamp.dayofweek

    lag_values = {
        "load_lag_24": load_history.get(
            timestamp - pd.Timedelta(hours=24)
        ),
        "load_lag_48": load_history.get(
            timestamp - pd.Timedelta(hours=48)
        ),
        "load_lag_168": load_history.get(
            timestamp - pd.Timedelta(hours=168)
        ),
    }

    for name, value in lag_values.items():
        if value is None or pd.isna(value):
            raise ValueError(f"Missing or invalid {name} for {timestamp}")

    row = {
        "hour": timestamp.hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        **lag_values,
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


def make_historical_forecast(
    model,
    historical: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    """Simulate a historical day using actual lag values."""
    target_start = pd.Timestamp(target_date).normalize()
    target_end = target_start + pd.Timedelta(days=1)

    first_available = historical["timestamp"].min()

    if target_start - pd.Timedelta(hours=168) < first_available:
        raise ValueError(
            "Not enough historical data to build 168-hour lag features."
        )

    actual_day = historical[
        (historical["timestamp"] >= target_start)
        & (historical["timestamp"] < target_end)
    ]

    if actual_day.empty:
        raise ValueError(
            f"No historical observations found for {target_start.date()}."
        )

    load_history = historical.set_index("timestamp")["load_mwh"].to_dict()
    rows = []

    for hour in range(24):
        timestamp = target_start + pd.Timedelta(hours=hour)
        features = build_features(timestamp, load_history)
        prediction = float(model.predict(features)[0])

        actual = load_history.get(timestamp)
        error = None

        if actual is not None and not pd.isna(actual):
            error = prediction - float(actual)

        rows.append(
            {
                "timestamp": timestamp,
                "forecast_mwh": prediction,
                "actual_mwh": actual,
                "error_mwh": error,
                "forecast_type": "historical",
            }
        )

    return pd.DataFrame(rows)


def make_future_forecast(
    model,
    historical: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    """Forecast a future day, recursively filling missing future lag values."""
    target_start = pd.Timestamp(target_date).normalize()
    target_end = target_start + pd.Timedelta(days=1)

    last_actual_timestamp = historical["timestamp"].max()
    last_actual_date = last_actual_timestamp.normalize()

    if target_start <= last_actual_date:
        raise ValueError("Target date is not in the future.")

    load_history = historical.set_index("timestamp")["load_mwh"].to_dict()

    forecast_range = pd.date_range(
        start=last_actual_timestamp + pd.Timedelta(hours=1),
        end=target_end - pd.Timedelta(hours=1),
        freq="h",
    )

    print(f"Recursive steps required: {len(forecast_range)}")

    rows = []

    for timestamp in forecast_range:
        features = build_features(timestamp, load_history)
        prediction = float(model.predict(features)[0])

        load_history[timestamp] = prediction

        if target_start <= timestamp < target_end:
            rows.append(
                {
                    "timestamp": timestamp,
                    "forecast_mwh": prediction,
                }
            )

    result = pd.DataFrame(rows)

    if len(result) != 24:
        raise ValueError(
            f"Expected 24 forecast rows, got {len(result)}."
        )

    result["actual_mwh"] = np.nan
    result["error_mwh"] = np.nan
    result["forecast_type"] = (
        "day_ahead"
        if target_start == last_actual_date + pd.Timedelta(days=1)
        else "recursive_future"
    )

    return result


def forecast_date(
    model,
    historical: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    target_date = pd.Timestamp(target_date).normalize()

    first_date = historical["timestamp"].min().normalize()
    last_date = historical["timestamp"].max().normalize()

    if target_date < first_date:
        raise ValueError(
            "Requested date is before the available historical dataset."
        )

    if target_date <= last_date:
        print(f"Mode: historical simulation ({target_date.date()})")
        return make_historical_forecast(
            model=model,
            historical=historical,
            target_date=target_date,
        )

    print(f"Mode: future forecast ({target_date.date()})")
    return make_future_forecast(
        model=model,
        historical=historical,
        target_date=target_date,
    )


def save_forecast_to_db(forecast: pd.DataFrame) -> None:
    """Persist only future forecasts."""
    future = forecast[
        forecast["forecast_type"].isin(["day_ahead", "recursive_future"])
    ].copy()

    if future.empty:
        print("Historical simulation was not saved to load_forecasts.")
        return

    query = """
        INSERT INTO load_forecasts (timestamp, forecast_mwh)
        VALUES (%s, %s)
        ON CONFLICT (timestamp)
        DO UPDATE SET
            forecast_mwh = EXCLUDED.forecast_mwh,
            created_at = NOW();
    """

    rows = [
        (row.timestamp.to_pydatetime(), float(row.forecast_mwh))
        for row in future.itertuples(index=False)
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)
        connection.commit()

    print(f"Saved {len(rows)} load forecast rows to PostgreSQL.")


def print_historical_metrics(forecast: pd.DataFrame) -> None:
    historical = forecast[
        forecast["forecast_type"] == "historical"
    ].dropna(subset=["actual_mwh", "forecast_mwh"])

    if historical.empty:
        return

    actual = historical["actual_mwh"].to_numpy(dtype=float)
    prediction = historical["forecast_mwh"].to_numpy(dtype=float)

    error = actual - prediction
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    denominator = np.abs(actual).sum()
    wape = np.nan if denominator == 0 else np.abs(error).sum() / denominator * 100

    print("\nHistorical load performance")
    print(f"MAE:  {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")
    print("WAPE: undefined" if pd.isna(wape) else f"WAPE: {wape:.2f}%")


def _parse_target_date(args, last_date: pd.Timestamp) -> pd.Timestamp:
    if args.date is not None:
        try:
            return pd.to_datetime(args.date, format="%Y-%m-%d").normalize()
        except ValueError as exc:
            raise ValueError("--date must use YYYY-MM-DD format") from exc

    if args.days is not None:
        if args.days < 1:
            raise ValueError("--days must be at least 1")
        return last_date + pd.Timedelta(days=args.days)

    return last_date + pd.Timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GridMind GR electricity load forecast"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    group.add_argument(
        "--days",
        type=int,
        help="Forecast N days after the latest historical date",
    )

    args = parser.parse_args()

    model = load_saved_model()
    historical = load_history()

    last_date = historical["timestamp"].max().normalize()
    target_date = _parse_target_date(args, last_date)

    print(f"Requested date: {target_date.date()}")

    forecast = forecast_date(
        model=model,
        historical=historical,
        target_date=target_date,
    )

    print_historical_metrics(forecast)
    save_forecast_to_db(forecast)

    print()
    print(forecast.to_string(index=False))
    print(
        f"\nRows: {len(forecast)} | "
        f"Type: {forecast['forecast_type'].iloc[0]}"
    )


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gridmind.data.database import get_connection
from gridmind.data.queries import load_historical_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_load_forecast.joblib"

MIN_VALID_LOAD_MWH = 500

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "load_lag_24",
    "load_lag_48",
    "load_lag_168",
]


def load_history() -> pd.DataFrame:
    """Load and clean the hourly system load history used by the model."""
    print("Loading historical load data from PostgreSQL...")

    df = load_historical_data()

    if df.empty:
        raise ValueError("No historical load data found in PostgreSQL.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in load history.")

    df.loc[df["load_mwh"] < MIN_VALID_LOAD_MWH, "load_mwh"] = np.nan

    print(f"Historical rows: {len(df)}")
    print(
        f"Historical period: {df['timestamp'].min()} "
        f"-> {df['timestamp'].max()}"
    )

    return df


def load_model_data() -> pd.DataFrame:
    """Create the model-ready load dataset used by backtesting."""
    df = load_history().copy()

    df["target"] = df["load_mwh"]
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["load_lag_24"] = df["load_mwh"].shift(24)
    df["load_lag_48"] = df["load_mwh"].shift(48)
    df["load_lag_168"] = df["load_mwh"].shift(168)

    model_data = (
        df[["timestamp", "target", *FEATURE_COLS]]
        .dropna()
        .reset_index(drop=True)
    )

    if model_data.empty:
        raise ValueError("No model-ready load rows available.")

    print(f"Model-ready rows: {len(model_data)}")
    print(
        f"Model period: {model_data['timestamp'].min()} "
        f"-> {model_data['timestamp'].max()}"
    )

    return model_data


def load_saved_model():
    """Load the production model and verify its feature contract."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        return artifact

    model = artifact["model"]
    saved_features = artifact.get("features", FEATURE_COLS)

    if list(saved_features) != FEATURE_COLS:
        raise ValueError(
            "Saved model feature list does not match production FEATURE_COLS."
        )

    return model


def build_features(timestamp, load_history: dict) -> pd.DataFrame:
    timestamp = pd.Timestamp(timestamp)
    day_of_week = timestamp.dayofweek

    lag_values = {
        "load_lag_24": load_history.get(
            timestamp - pd.Timedelta(hours=24)
        ),
        "load_lag_48": load_history.get(
            timestamp - pd.Timedelta(hours=48)
        ),
        "load_lag_168": load_history.get(
            timestamp - pd.Timedelta(hours=168)
        ),
    }

    for name, value in lag_values.items():
        if value is None or pd.isna(value):
            raise ValueError(f"Missing or invalid {name} for {timestamp}")

    row = {
        "hour": timestamp.hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        **lag_values,
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


def make_historical_forecast(
    model,
    historical: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    """Simulate a historical day using actual lag values."""
    target_start = pd.Timestamp(target_date).normalize()
    target_end = target_start + pd.Timedelta(days=1)

    first_available = historical["timestamp"].min()

    if target_start - pd.Timedelta(hours=168) < first_available:
        raise ValueError(
            "Not enough historical data to build 168-hour lag features."
        )

    actual_day = historical[
        (historical["timestamp"] >= target_start)
        & (historical["timestamp"] < target_end)
    ]

    if actual_day.empty:
        raise ValueError(
            f"No historical observations found for {target_start.date()}."
        )

    load_history = historical.set_index("timestamp")["load_mwh"].to_dict()
    rows = []

    for hour in range(24):
        timestamp = target_start + pd.Timedelta(hours=hour)
        features = build_features(timestamp, load_history)
        prediction = float(model.predict(features)[0])

        actual = load_history.get(timestamp)
        error = None

        if actual is not None and not pd.isna(actual):
            error = prediction - float(actual)

        rows.append(
            {
                "timestamp": timestamp,
                "forecast_mwh": prediction,
                "actual_mwh": actual,
                "error_mwh": error,
                "forecast_type": "historical",
            }
        )

    return pd.DataFrame(rows)


def make_future_forecast(
    model,
    historical: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    """Forecast a future day, recursively filling missing future lag values."""
    target_start = pd.Timestamp(target_date).normalize()
    target_end = target_start + pd.Timedelta(days=1)

    last_actual_timestamp = historical["timestamp"].max()
    last_actual_date = last_actual_timestamp.normalize()

    if target_start <= last_actual_date:
        raise ValueError("Target date is not in the future.")

    load_history = historical.set_index("timestamp")["load_mwh"].to_dict()

    forecast_range = pd.date_range(
        start=last_actual_timestamp + pd.Timedelta(hours=1),
        end=target_end - pd.Timedelta(hours=1),
        freq="h",
    )

    print(f"Recursive steps required: {len(forecast_range)}")

    rows = []

    for timestamp in forecast_range:
        features = build_features(timestamp, load_history)
        prediction = float(model.predict(features)[0])

        load_history[timestamp] = prediction

        if target_start <= timestamp < target_end:
            rows.append(
                {
                    "timestamp": timestamp,
                    "forecast_mwh": prediction,
                }
            )

    result = pd.DataFrame(rows)

    if len(result) != 24:
        raise ValueError(
            f"Expected 24 forecast rows, got {len(result)}."
        )

    result["actual_mwh"] = np.nan
    result["error_mwh"] = np.nan
    result["forecast_type"] = (
        "day_ahead"
        if target_start == last_actual_date + pd.Timedelta(days=1)
        else "recursive_future"
    )

    return result


def forecast_date(
    model,
    historical: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    target_date = pd.Timestamp(target_date).normalize()

    first_date = historical["timestamp"].min().normalize()
    last_date = historical["timestamp"].max().normalize()

    if target_date < first_date:
        raise ValueError(
            "Requested date is before the available historical dataset."
        )

    if target_date <= last_date:
        print(f"Mode: historical simulation ({target_date.date()})")
        return make_historical_forecast(
            model=model,
            historical=historical,
            target_date=target_date,
        )

    print(f"Mode: future forecast ({target_date.date()})")
    return make_future_forecast(
        model=model,
        historical=historical,
        target_date=target_date,
    )


def save_forecast_to_db(forecast: pd.DataFrame) -> None:
    """Persist only future forecasts."""
    future = forecast[
        forecast["forecast_type"].isin(["day_ahead", "recursive_future"])
    ].copy()

    if future.empty:
        print("Historical simulation was not saved to load_forecasts.")
        return

    query = """
        INSERT INTO load_forecasts (timestamp, forecast_mwh)
        VALUES (%s, %s)
        ON CONFLICT (timestamp)
        DO UPDATE SET
            forecast_mwh = EXCLUDED.forecast_mwh,
            created_at = NOW();
    """

    rows = [
        (row.timestamp.to_pydatetime(), float(row.forecast_mwh))
        for row in future.itertuples(index=False)
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)
        connection.commit()

    print(f"Saved {len(rows)} load forecast rows to PostgreSQL.")


def print_historical_metrics(forecast: pd.DataFrame) -> None:
    historical = forecast[
        forecast["forecast_type"] == "historical"
    ].dropna(subset=["actual_mwh", "forecast_mwh"])

    if historical.empty:
        return

    actual = historical["actual_mwh"].to_numpy(dtype=float)
    prediction = historical["forecast_mwh"].to_numpy(dtype=float)

    error = actual - prediction
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    denominator = np.abs(actual).sum()
    wape = np.nan if denominator == 0 else np.abs(error).sum() / denominator * 100

    print("\nHistorical load performance")
    print(f"MAE:  {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")
    print("WAPE: undefined" if pd.isna(wape) else f"WAPE: {wape:.2f}%")


def _parse_target_date(args, last_date: pd.Timestamp) -> pd.Timestamp:
    if args.date is not None:
        try:
            return pd.to_datetime(args.date, format="%Y-%m-%d").normalize()
        except ValueError as exc:
            raise ValueError("--date must use YYYY-MM-DD format") from exc

    if args.days is not None:
        if args.days < 1:
            raise ValueError("--days must be at least 1")
        return last_date + pd.Timedelta(days=args.days)

    return last_date + pd.Timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GridMind GR electricity load forecast"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    group.add_argument(
        "--days",
        type=int,
        help="Forecast N days after the latest historical date",
    )

    args = parser.parse_args()

    model = load_saved_model()
    historical = load_history()

    last_date = historical["timestamp"].max().normalize()
    target_date = _parse_target_date(args, last_date)

    print(f"Requested date: {target_date.date()}")

    forecast = forecast_date(
        model=model,
        historical=historical,
        target_date=target_date,
    )

    print_historical_metrics(forecast)
    save_forecast_to_db(forecast)

    print()
    print(forecast.to_string(index=False))
    print(
        f"\nRows: {len(forecast)} | "
        f"Type: {forecast['forecast_type'].iloc[0]}"
    )


if __name__ == "__main__":
    main()
