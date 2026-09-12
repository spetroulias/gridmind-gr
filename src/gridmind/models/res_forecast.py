import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from gridmind.data.database import get_connection
from gridmind.data.queries import load_historical_res_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"
FORECAST_PATH = FORECAST_DIR / "res_forecast.csv"

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "res_lag_24",
    "res_lag_48",
    "res_lag_168",
]

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


def load_res_history():
    print("Loading historical RES data from PostgreSQL...")

    df = load_historical_res_data()

    if df.empty:
        raise ValueError("No historical RES data found in PostgreSQL.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in RES history.")

    print(f"Historical rows: {len(df)}")
    print(
        "Historical period:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    return df


def build_model_data(historical=None):
    if historical is None:
        historical = load_res_history()

    df = historical.copy()

    df["target"] = df["res_mwh"]
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["res_lag_24"] = (df["timestamp"] - pd.Timedelta(hours=24)).map(df.set_index("timestamp")["res_mwh"])
    df["res_lag_48"] = (df["timestamp"] - pd.Timedelta(hours=48)).map(df.set_index("timestamp")["res_mwh"])
    df["res_lag_168"] = (df["timestamp"] - pd.Timedelta(hours=168)).map(df.set_index("timestamp")["res_mwh"])

    model_data = (
        df[["timestamp", "target", *FEATURE_COLS]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if model_data.empty:
        raise ValueError("No model-ready RES rows available.")

    print(f"Model-ready rows: {len(model_data)}")
    print(
        "Model period:",
        model_data["timestamp"].min(),
        "->",
        model_data["timestamp"].max(),
    )

    return model_data


def build_model():
    return RandomForestRegressor(**RF_PARAMS)


def train_model(training_data):
    if training_data.empty:
        raise ValueError("RES training dataset is empty.")

    model = build_model()

    X_train = training_data[FEATURE_COLS]
    y_train = training_data["target"]

    print(f"Training RES Random Forest on {len(training_data)} rows...")

    model.fit(X_train, y_train)

    return model


def _get_history_value(res_history, timestamp):
    value = res_history.get(timestamp)

    if value is None or pd.isna(value):
        raise ValueError(f"Missing RES history for {timestamp}")

    return float(value)


def build_features(timestamp, res_history):
    timestamp = pd.Timestamp(timestamp)
    day_of_week = timestamp.dayofweek

    row = {
        "hour": timestamp.hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "res_lag_24": _get_history_value(
            res_history,
            timestamp - pd.Timedelta(hours=24),
        ),
        "res_lag_48": _get_history_value(
            res_history,
            timestamp - pd.Timedelta(hours=48),
        ),
        "res_lag_168": _get_history_value(
            res_history,
            timestamp - pd.Timedelta(hours=168),
        ),
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


def make_historical_forecast(historical, target_date):
    target_start = pd.Timestamp(target_date).normalize()
    target_end = target_start + pd.Timedelta(days=1)

    first_timestamp = historical["timestamp"].min()

    if target_start - pd.Timedelta(hours=168) < first_timestamp:
        raise ValueError(
            "Not enough historical RES data for 168-hour lag features."
        )

    model_data = build_model_data(historical)

    # Leakage-free historical simulation:
    # only observations before the requested day are used for training.
    training_data = model_data[
        model_data["timestamp"] < target_start
    ].copy()

    if training_data.empty:
        raise ValueError(
            f"No training data exists before {target_start.date()}."
        )

    model = train_model(training_data)

    res_history = (
        historical
        .set_index("timestamp")["res_mwh"]
        .to_dict()
    )

    actual_day = historical[
        (historical["timestamp"] >= target_start)
        & (historical["timestamp"] < target_end)
    ]

    if actual_day.empty:
        raise ValueError(
            f"No historical RES observations found for {target_start.date()}."
        )

    forecasts = []

    for hour in range(24):
        timestamp = target_start + pd.Timedelta(hours=hour)

        X = build_features(
            timestamp=timestamp,
            res_history=res_history,
        )

        prediction = max(float(model.predict(X)[0]), 0.0)

        actual = res_history.get(timestamp)
        error = np.nan
        abs_error = np.nan

        if actual is not None and not pd.isna(actual):
            actual = float(actual)
            error = prediction - actual
            abs_error = abs(error)

        forecasts.append(
            {
                "timestamp": timestamp,
                "forecast_mwh": prediction,
                "actual_mwh": actual,
                "error_mwh": error,
                "abs_error_mwh": abs_error,
                "forecast_type": "historical",
            }
        )

    return pd.DataFrame(forecasts)


def make_future_forecast(historical, target_date):
    target_start = pd.Timestamp(target_date).normalize()
    target_end = target_start + pd.Timedelta(days=1)

    last_actual_timestamp = historical["timestamp"].max()
    last_actual_date = last_actual_timestamp.normalize()

    if target_start <= last_actual_date:
        raise ValueError(
            "Future RES forecast requires a date after latest actual data."
        )

    model_data = build_model_data(historical)

    # Production model uses all currently available actual RES history.
    model = train_model(model_data)

    res_history = (
        historical
        .set_index("timestamp")["res_mwh"]
        .to_dict()
    )

    first_future_timestamp = (
        last_actual_timestamp + pd.Timedelta(hours=1)
    )

    final_timestamp = target_end - pd.Timedelta(hours=1)

    forecast_range = pd.date_range(
        start=first_future_timestamp,
        end=final_timestamp,
        freq="h",
    )

    print(f"Recursive steps required: {len(forecast_range)}")

    forecasts = []

    for timestamp in forecast_range:
        X = build_features(
            timestamp=timestamp,
            res_history=res_history,
        )

        prediction = max(float(model.predict(X)[0]), 0.0)

        # Recursive future predictions become history for later lag features.
        res_history[timestamp] = prediction

        if target_start <= timestamp < target_end:
            forecasts.append(
                {
                    "timestamp": timestamp,
                    "forecast_mwh": prediction,
                }
            )

    result = pd.DataFrame(forecasts)

    if len(result) != 24:
        raise ValueError(
            f"Expected 24 RES forecasts, got {len(result)}."
        )

    next_day = last_actual_date + pd.Timedelta(days=1)

    if target_start == next_day:
        forecast_type = "day_ahead"
    else:
        forecast_type = "recursive_future"

    result["actual_mwh"] = np.nan
    result["error_mwh"] = np.nan
    result["abs_error_mwh"] = np.nan
    result["forecast_type"] = forecast_type

    return result


def forecast_date(historical, target_date):
    target_date = pd.Timestamp(target_date).normalize()

    first_date = historical["timestamp"].min().normalize()
    last_date = historical["timestamp"].max().normalize()

    if target_date < first_date:
        raise ValueError(
            "Requested date is before available RES history."
        )

    if target_date <= last_date:
        print(
            f"Mode: HISTORICAL RES SIMULATION ({target_date.date()})"
        )
        return make_historical_forecast(
            historical=historical,
            target_date=target_date,
        )

    print(
        f"Mode: FUTURE RES FORECAST ({target_date.date()})"
    )

    return make_future_forecast(
        historical=historical,
        target_date=target_date,
    )


def save_forecast_to_db(forecast):
    future = forecast[
        forecast["forecast_type"].isin(
            ["day_ahead", "recursive_future"]
        )
    ].copy()

    if future.empty:
        print(
            "Historical RES simulation: not saved to res_forecasts."
        )
        return

    query = """
        INSERT INTO res_forecasts (
            timestamp,
            forecast_mwh
        )
        VALUES (%s, %s)
        ON CONFLICT (timestamp)
        DO UPDATE SET
            forecast_mwh = EXCLUDED.forecast_mwh,
            created_at = NOW();
    """

    rows = [
        (
            row.timestamp.to_pydatetime(),
            float(row.forecast_mwh),
        )
        for row in future.itertuples(index=False)
    ]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, rows)

        connection.commit()

    print(
        f"Saved {len(rows)} RES forecast rows to PostgreSQL."
    )


def print_historical_metrics(forecast):
    historical = forecast[
        forecast["forecast_type"] == "historical"
    ].dropna(
        subset=["actual_mwh", "forecast_mwh"]
    )

    if historical.empty:
        return

    actual = historical["actual_mwh"].to_numpy(dtype=float)
    prediction = historical["forecast_mwh"].to_numpy(dtype=float)

    error = actual - prediction

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    denominator = np.abs(actual).sum()

    if denominator == 0:
        wape = np.nan
    else:
        wape = (
            np.abs(error).sum()
            / denominator
        ) * 100

    print()
    print("======================================")
    print("      HISTORICAL RES PERFORMANCE")
    print("======================================")
    print(f"MAE:  {mae:.2f} MWh")
    print(f"RMSE: {rmse:.2f} MWh")

    if pd.isna(wape):
        print("WAPE: undefined")
    else:
        print(f"WAPE: {wape:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="GridMind GR RES production forecast"
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--date",
        type=str,
        help="Target date YYYY-MM-DD",
    )

    group.add_argument(
        "--days",
        type=int,
        help="Forecast N days after latest RES actual data",
    )

    args = parser.parse_args()

    print()
    print("======================================")
    print("      GRIDMIND GR RES FORECAST")
    print("======================================")
    print()

    historical = load_res_history()

    last_date = (
        historical["timestamp"]
        .max()
        .normalize()
    )

    if args.date is not None:
        try:
            target_date = (
                pd.to_datetime(
                    args.date,
                    format="%Y-%m-%d",
                )
                .normalize()
            )
        except ValueError as exc:
            raise ValueError(
                "--date must use YYYY-MM-DD format"
            ) from exc

    elif args.days is not None:
        if args.days < 1:
            raise ValueError(
                "--days must be at least 1"
            )

        target_date = (
            last_date
            + pd.Timedelta(days=args.days)
        )

    else:
        target_date = (
            last_date
            + pd.Timedelta(days=1)
        )

    print(f"Requested date: {target_date.date()}")

    forecast = forecast_date(
        historical=historical,
        target_date=target_date,
    )

    print_historical_metrics(forecast)
    save_forecast_to_db(forecast)

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        FORECAST_PATH,
        index=False,
    )

    print()
    print(
        forecast.to_string(index=False)
    )

    print()
    print("======================================")
    print("RES FORECAST COMPLETED")
    print("======================================")
    print(f"Rows: {len(forecast)}")
    print(
        "Period:",
        forecast["timestamp"].min(),
        "->",
        forecast["timestamp"].max(),
    )
    print(
        "Forecast type:",
        forecast["forecast_type"].iloc[0],
    )
    print(f"CSV saved to: {FORECAST_PATH}")


if __name__ == "__main__":
    main()
