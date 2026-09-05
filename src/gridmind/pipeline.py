"""
GridMind-GR
End-to-End Multi-Day Electricity Load Forecasting Pipeline
"""

import argparse

from gridmind.models.forecast import (
    load_data,
    train_model,
    make_multiday_forecast,
    FORECAST_DIR,
    FORECAST_PATH,
)


def main():

    parser = argparse.ArgumentParser(
        description="GridMind-GR forecasting pipeline"
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
    print("       GRIDMIND-GR PIPELINE")
    print("======================================\n")

    print(f"Forecast horizon: {args.days} day(s)")

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    print("\n[1/3] Loading model data...")

    df = load_data()

    print(f"Rows loaded: {len(df)}")
    print(f"Data until: {df['timestamp'].max()}")

    # --------------------------------------------------
    # 2. Train model
    # --------------------------------------------------

    print("\n[2/3] Training Random Forest...")

    model = train_model(df)

    # --------------------------------------------------
    # 3. Forecast
    # --------------------------------------------------

    print("\n[3/3] Generating forecast...")

    forecast = make_multiday_forecast(
        model=model,
        df=df,
        days=args.days
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast.to_csv(
        FORECAST_PATH,
        index=False
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\n======================================")
    print("            FORECAST")
    print("======================================\n")

    print(forecast.to_string(index=False))

    print("\n======================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("======================================")

    print(f"\nForecast rows: {len(forecast)}")

    print(
        f"Forecast period: "
        f"{forecast['timestamp'].min()} "
        f"-> "
        f"{forecast['timestamp'].max()}"
    )

    print(f"\nSaved forecast to:\n{FORECAST_PATH}\n")


if __name__ == "__main__":
    main()