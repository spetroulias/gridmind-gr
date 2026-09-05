from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_data.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "backtest_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "load_lag_24",
    "load_lag_48",
    "load_lag_168",
]


# ============================================================
# MODEL
# ============================================================

def build_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df):

    # Monthly test periods during 2026
    test_months = pd.date_range(
        "2026-01-01",
        "2026-08-01",
        freq="MS",
    )

    results = []

    for test_start in test_months:

        test_end = (
            test_start
            + pd.offsets.MonthBegin(1)
        )

        train = df[
            df["timestamp"] < test_start
        ].copy()

        test = df[
            (df["timestamp"] >= test_start)
            & (df["timestamp"] < test_end)
        ].copy()

        if train.empty or test.empty:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train["target"]

        X_test = test[FEATURE_COLS]
        y_test = test["target"]

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred,
            )
        )

        mape = np.mean(
            np.abs(
                (y_test.values - pred)
                / y_test.values
            )
        ) * 100

        results.append({
            "month": test_start.strftime("%Y-%m"),
            "train_rows": len(train),
            "test_rows": len(test),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })

        print(
            test_start.strftime("%Y-%m"),
            f"| MAE: {mae:.2f}",
            f"| RMSE: {rmse:.2f}",
            f"| MAPE: {mape:.2f}%",
        )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "Data:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    print("\nRunning monthly backtest...\n")

    results = run_backtest(df)

    print("\n==========================")
    print("BACKTEST SUMMARY")
    print("==========================")

    print(
        "Average MAE:",
        round(results["MAE"].mean(), 2),
        "MWh",
    )

    print(
        "Average RMSE:",
        round(results["RMSE"].mean(), 2),
        "MWh",
    )

    print(
        "Average MAPE:",
        round(results["MAPE"].mean(), 2),
        "%",
    )

    print(
        "Best month:",
        results.loc[
            results["MAPE"].idxmin(),
            "month"
        ],
    )

    print(
        "Worst month:",
        results.loc[
            results["MAPE"].idxmax(),
            "month"
        ],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()