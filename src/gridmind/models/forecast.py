from pathlib import Path
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_data.csv"
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
# Load data
# --------------------------------------------------

def load_data():
    df = pd.read_csv(
        MODEL_DATA_PATH,
        parse_dates=["timestamp"]
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

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
        n_jobs=-1
    )

    model.fit(X, y)

    print(f"Training rows: {len(df)}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Training data until: {df['timestamp'].max()}")

    return model


# --------------------------------------------------
# Recursive multi-day forecast
# --------------------------------------------------

def make_multiday_forecast(model, df, days):
    hours_to_forecast = days * 24

    # Historical actual loads
    load_history = df.set_index("timestamp")["target"].to_dict()

    last_timestamp = df["timestamp"].max()

    forecasts = []

    for step in range(1, hours_to_forecast + 1):

        timestamp = last_timestamp + pd.Timedelta(hours=step)

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_weekend = int(day_of_week >= 5)

        # Historical actual values OR previous predictions
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
            [{
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "load_lag_24": lag_24,
                "load_lag_48": lag_48,
                "load_lag_168": lag_168,
            }]
        )

        prediction = model.predict(X_future)[0]

        # Store prediction so future hours can use it as a lag
        load_history[timestamp] = prediction

        forecasts.append(
            {
                "timestamp": timestamp,
                "forecast_mwh": prediction
            }
        )

    return pd.DataFrame(forecasts)


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
        help="Number of days to forecast"
    )

    args = parser.parse_args()

    if args.days < 1:
        raise ValueError("--days must be at least 1")

    print("\n======================================")
    print("       GRIDMIND-GR FORECAST")
    print("======================================\n")

    print(f"Forecast horizon: {args.days} day(s)")

    # Load data
    df = load_data()

    # Train
    model = train_model(df)

    # Forecast
    forecast = make_multiday_forecast(
        model=model,
        df=df,
        days=args.days
    )

    # Save
    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast.to_csv(
        FORECAST_PATH,
        index=False
    )

    print("\nForecast:")
    print(forecast.to_string(index=False))

    print("\n======================================")
    print("FORECAST COMPLETED")
    print("======================================")

    print(f"\nForecast rows: {len(forecast)}")
    print(
        f"Forecast period: "
        f"{forecast['timestamp'].min()} -> "
        f"{forecast['timestamp'].max()}"
    )

    print(f"\nSaved to:\n{FORECAST_PATH}")


if __name__ == "__main__":
    main()