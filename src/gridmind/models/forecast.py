import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import text

from gridmind.data.queries import (
    load_historical_data,
    get_engine,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"
FORECAST_PATH = FORECAST_DIR / "forecast.csv"


# --------------------------------------------------
# Features
# --------------------------------------------------

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "load_lag_24",
    "load_lag_48",
    "load_lag_168",
]


# --------------------------------------------------
# Load data from PostgreSQL + feature engineering
# --------------------------------------------------

def load_data():
    print("Loading historical LOAD data from PostgreSQL...")

    df = load_historical_data()

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No historical LOAD data found in PostgreSQL")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in system_load")

    df = df.rename(
        columns={
            "load_mwh": "target"
        }
    )

    # Calendar features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Lag features
    load_by_timestamp = df.set_index("timestamp")["target"]

    df["load_lag_24"] = (
        df["timestamp"] - pd.Timedelta(hours=24)
    ).map(load_by_timestamp)

    df["load_lag_48"] = (
        df["timestamp"] - pd.Timedelta(hours=48)
    ).map(load_by_timestamp)

    df["load_lag_168"] = (
        df["timestamp"] - pd.Timedelta(hours=168)
    ).map(load_by_timestamp)

    df = df.dropna(
        subset=[
            "target",
            "load_lag_24",
            "load_lag_48",
            "load_lag_168",
        ]
    ).reset_index(drop=True)

    print(f"Model-ready rows: {len(df)}")
    print(
        f"Model data period: "
        f"{df['timestamp'].min()} -> "
        f"{df['timestamp'].max()}"
    )

    return df


# --------------------------------------------------
# Train model
# --------------------------------------------------

def train_model(df):
    X = df[FEATURE_COLS]
    y = df["target"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    print(f"\nTraining rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Training data until: {df['timestamp'].max()}")

    return model


# --------------------------------------------------
# Recursive multi-day forecast
# --------------------------------------------------

def make_multiday_forecast(model, df, days):
    hours_to_forecast = days * 24

    load_history = (
        df.set_index("timestamp")["target"]
        .to_dict()
    )

    last_timestamp = df["timestamp"].max()

    forecasts = []

    for step in range(1, hours_to_forecast + 1):

        timestamp = (
            last_timestamp
            + pd.Timedelta(hours=step)
        )

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_weekend = int(day_of_week >= 5)

        lag_24 = load_history.get(
            timestamp - pd.Timedelta(hours=24)
        )

        lag_48 = load_history.get(
            timestamp - pd.Timedelta(hours=48)
        )

        lag_168 = load_history.get(
            timestamp - pd.Timedelta(hours=168)
        )

        if (
            lag_24 is None
            or lag_48 is None
            or lag_168 is None
        ):
            raise ValueError(
                f"Missing lag value for {timestamp}"
            )

        X_future = pd.DataFrame(
            [
                {
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "load_lag_24": lag_24,
                    "load_lag_48": lag_48,
                    "load_lag_168": lag_168,
                }
            ]
        )

        prediction = float(
            model.predict(X_future)[0]
        )

        load_history[timestamp] = prediction

        forecasts.append(
            {
                "timestamp": timestamp,
                "forecast_mwh": prediction,
            }
        )

    return pd.DataFrame(forecasts)


# --------------------------------------------------
# Save forecast to PostgreSQL
# --------------------------------------------------

def save_forecast_to_db(forecast):
    query = text("""
        INSERT INTO load_forecasts (
            timestamp,
            forecast_mwh
        )
        VALUES (
            :timestamp,
            :forecast_mwh
        )
        ON CONFLICT (timestamp)
        DO UPDATE SET
            forecast_mwh = EXCLUDED.forecast_mwh,
            created_at = NOW()
    """)

    records = forecast[
        [
            "timestamp",
            "forecast_mwh",
        ]
    ].to_dict(orient="records")

    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(query, records)

    print(
        f"Saved {len(forecast)} forecast rows "
        f"to PostgreSQL."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GridMind-GR electricity load forecast"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to forecast",
    )

    args = parser.parse_args()

    if args.days < 1:
        raise ValueError("--days must be at least 1")

    print("\n======================================")
    print("       GRIDMIND-GR FORECAST")
    print("======================================\n")

    print(f"Forecast horizon: {args.days} day(s)")

    # PostgreSQL -> feature engineering
    df = load_data()

    # Train
    model = train_model(df)

    # Forecast
    forecast = make_multiday_forecast(
        model=model,
        df=df,
        days=args.days,
    )

    # Save to PostgreSQL
    save_forecast_to_db(forecast)

    # Save CSV backup/export
    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        FORECAST_PATH,
        index=False,
    )

    print("\nForecast:")
    print(
        forecast.to_string(index=False)
    )

    print("\n======================================")
    print("FORECAST COMPLETED")
    print("======================================")

    print(f"\nForecast rows: {len(forecast)}")

    print(
        f"Forecast period: "
        f"{forecast['timestamp'].min()} -> "
        f"{forecast['timestamp'].max()}"
    )

    print(
        f"\nSaved CSV to:\n{FORECAST_PATH}"
    )


if __name__ == "__main__":
    main()